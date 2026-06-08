"""
pastis_velib.py — Pastis.ai base library for velib-predict-dispo

Imported both by students (notebook / my_model.py) and by the Pastis
scoring sandbox. Versions must stay in sync.
Do NOT modify this file — it is part of the grading contract.
"""

from abc import ABC, abstractmethod

import ast
import pickle

import numpy as np
import pandas as pd


# --- Features contract v1 — matches velib_dataset_dev.csv ---

AVAILABLE_FEATURES = [
    "stationcode",
    "name",
    "snapshot_at",
    "capacity",
    "numbikesavailable",
    "mechanical",
    "ebike",
    "numdocksavailable",
    "is_renting",
    "nom_arrondissement_communes",
    "coordonnees_geo",
    "has_bike",
]

# Local dev dataset — used as fallback in verify_submission()
SAMPLE_DATA_PATH = "velib_dataset_dev.csv"


# --- Abstract base class ---

class PastisBaseModel(ABC):
    """
    Base class for all Pastis.ai submissions.

    Subclass this and implement fit() and predict().
    The sandbox calls these two methods with the raw DataFrame.
    Do not change the signature or the output contract.
    """

    @abstractmethod
    def fit(self, df_raw: pd.DataFrame) -> None:
        """
        Train the model on the raw input DataFrame.

        Parameters
        ----------
        df_raw : pd.DataFrame
            Raw data with columns listed in AVAILABLE_FEATURES.
        """

    @abstractmethod
    def predict(self, df_raw: pd.DataFrame) -> np.ndarray:
        """
        Return predicted taux_remplissage for each row.

        Parameters
        ----------
        df_raw : pd.DataFrame
            Raw data with columns listed in AVAILABLE_FEATURES.

        Returns
        -------
        np.ndarray
            1D array of floats in [0.0, 1.0], shape (len(df_raw),).
            Immutable output contract — the sandbox enforces it strictly.
        """


# --- Target computation ---

def compute_target(df: pd.DataFrame) -> pd.Series:
    """
    Compute the prediction target: fill rate for each station snapshot.

    taux_remplissage = numbikesavailable / capacity

    Returns values in [0.0, 1.0]:
    - 0.0 = station completely empty
    - 1.0 = station completely full
    - Rows with capacity=0 are filled with 0.0

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame with numbikesavailable and capacity columns.

    Returns
    -------
    pd.Series
        Float series in [0.0, 1.0], same index as df.
    """
    capacity = df["capacity"].replace(0, np.nan)
    rate = df["numbikesavailable"] / capacity
    return rate.fillna(0.0).clip(0.0, 1.0)


# --- Feature engineering helper ---

# --- Why these features and not others? ---
#
# INCLUDED:
#   hour, dayofweek, is_weekend, month
#     → Time of day and day of week are the strongest temporal signals.
#       Rush hours drain stations. Weekends have different patterns than weekdays.
#
#   stationcode
#     → Each station has its own structural fill rate.
#       A station near a metro hub empties fast; one on a quiet street stays full.
#       This is the most important spatial feature.
#
#   capacity
#     → Larger stations buffer demand differently — a 40-dock station reacts
#       more slowly to demand shocks than a 10-dock station.
#
#   arrondissement (derived from stationcode numeric prefix)
#     → Neighborhood-level signal on top of individual station identity.
#       Derived from stationcode (e.g. 18001 → arrondissement 18) rather than
#       nom_arrondissement_communes, which returns "Paris" for all rows in this dataset.
#
# INTENTIONALLY EXCLUDED (data leakage):
#   numbikesavailable, mechanical, ebike, numdocksavailable
#     → These directly reveal taux_remplissage = numbikesavailable / capacity.
#       Using them as features means predicting from the answer, not prior info.
#       A model trained with these would score perfectly but fail in real
#       deployment where you forecast FUTURE availability.
#
# EXCLUDED (zero variance in current dataset):
#   is_renting → almost always "OUI" — contributes nothing to the model.
#              Left as a commented-out example for students to explore.

def make_basic_features(df: pd.DataFrame, extra_df=None) -> pd.DataFrame:
    """
    Build baseline features from the raw Vélib DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame with AVAILABLE_FEATURES columns.
    extra_df : pd.DataFrame, optional
        Reserved for external data sources (weather, events, etc.).
        Currently ignored. Future use: merge on snapshot_at + location.

    Returns
    -------
    pd.DataFrame
        Numeric DataFrame ready for sklearn. Does NOT include the target column.
    """
    out = pd.DataFrame(index=df.index)

    # Temporal features — strong predictors of station demand
    ts = pd.to_datetime(df["snapshot_at"])
    out["hour"] = ts.dt.hour.astype(int)
    out["dayofweek"] = ts.dt.dayofweek.astype(int)   # 0=Monday, 6=Sunday
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["month"] = ts.dt.month.astype(int)

    # Station identity — each station has its own demand profile
    out["stationcode"] = (
        pd.to_numeric(df["stationcode"], errors="coerce").fillna(0).astype(int)
    )

    # Station capacity — larger stations buffer demand differently
    out["capacity"] = df["capacity"].fillna(0).astype(int)

    # Arrondissement derived from stationcode numeric prefix
    # e.g. 20001 → 20, 18005 → 18, 9101 → 9
    # More reliable than nom_arrondissement_communes, which returns "Paris" for all rows
    out["arrondissement"] = (
        pd.to_numeric(df["stationcode"], errors="coerce")
        .fillna(0)
        .astype(int) // 1000
    )

    # is_renting is excluded — almost always "OUI", contributes zero information.
    # Uncomment to experiment:
    # out["is_renting"] = (df["is_renting"] == "OUI").astype(int)

    # NOTE: numbikesavailable, mechanical, ebike, numdocksavailable are
    # intentionally excluded — they directly reveal taux_remplissage.
    # Students may add lagged values of these if they understand the
    # leakage implications.

    # extra_df reserved for future external features (weather, etc.) — no-op
    _ = extra_df

    return out


def _parse_coordonnees(geo_str) -> tuple:
    """
    Parse coordonnees_geo field into (lat, lon) floats.

    Handles both formats found in Pastis data:
      - Dict-like string: "{'lon': 2.341, 'lat': 48.857}"
      - Comma-separated: "48.857,2.341"
    Returns (nan, nan) on failure.
    """
    try:
        if isinstance(geo_str, str) and geo_str.strip().startswith("{"):
            d = ast.literal_eval(geo_str)
            return float(d["lat"]), float(d["lon"])
        parts = str(geo_str).split(",")
        return float(parts[0]), float(parts[1])
    except Exception:
        return float("nan"), float("nan")


# --- Submission verifier ---

def verify_submission(pkl_path: str, sample_df: pd.DataFrame = None) -> dict:
    """
    Load a .pkl file and verify that its predict() output is valid.

    Parameters
    ----------
    pkl_path : str
        Path to the submission .pkl file.
    sample_df : pd.DataFrame, optional
        A sample DataFrame with AVAILABLE_FEATURES columns.
        Defaults to loading SAMPLE_DATA_PATH (first 50 rows).

    Returns
    -------
    dict
        {"valid": bool, "message": str, "sample_output": list}
    """
    if sample_df is None:
        try:
            sample_df = pd.read_csv(SAMPLE_DATA_PATH, nrows=50)
        except FileNotFoundError:
            return {
                "valid": False,
                "message": f"Sample file not found: {SAMPLE_DATA_PATH}",
                "sample_output": [],
            }

    try:
        with open(pkl_path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        return {"valid": False, "message": f"Could not load .pkl: {e}", "sample_output": []}

    if not hasattr(model, "predict"):
        return {"valid": False, "message": "Object has no predict() method.", "sample_output": []}

    try:
        output = model.predict(sample_df)
    except Exception as e:
        return {"valid": False, "message": f"predict() raised an error: {e}", "sample_output": []}

    if not isinstance(output, np.ndarray):
        return {
            "valid": False,
            "message": f"predict() must return np.ndarray, got {type(output).__name__}.",
            "sample_output": [],
        }

    if output.shape != (len(sample_df),):
        return {
            "valid": False,
            "message": f"Shape mismatch: expected ({len(sample_df)},), got {output.shape}.",
            "sample_output": [],
        }

    if not np.all((output >= 0.0) & (output <= 1.0)):
        bad = output[(output < 0.0) | (output > 1.0)]
        return {
            "valid": False,
            "message": f"Values out of [0, 1]: {bad[:5]}",
            "sample_output": output.tolist()[:5],
        }

    return {
        "valid": True,
        "message": "Submission is valid. Ready to upload to pastis.ai.",
        "sample_output": output.tolist()[:5],
    }


# --- Brier Score helper ---

def brier_score_local(y_true, y_proba, label: str = "") -> float:
    """
    Compute Brier Score = MSE(y_true, y_pred) for values in [0, 1].

    Works for both:
    - Binary classification: y_true in {0, 1}, y_pred in [0, 1]
    - Regression on [0, 1]: y_true = taux_remplissage, y_pred in [0, 1]

    Lower is better. Baseline (always predict mean) → ~0.05–0.15.

    Parameters
    ----------
    y_true : array-like
        Ground truth values in [0, 1].
    y_proba : array-like
        Predicted values in [0, 1].
    label : str, optional
        Label prefix for the printed output lines.

    Returns
    -------
    float
        The Brier Score.
    """
    y_true = np.array(y_true, dtype=float)
    y_proba = np.array(y_proba, dtype=float)

    bs = float(np.mean((y_proba - y_true) ** 2))
    leaderboard = (1 - bs) * 100
    prefix = f"{label} — " if label else ""
    print(f"{prefix}Brier Score      : {bs:.4f}  (lower is better)")
    print(f"{prefix}Leaderboard score: {leaderboard:.2f} (higher is better)")
    return bs
