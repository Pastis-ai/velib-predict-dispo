"""
pastis_velib.py — Pastis.ai base library for velib-predict-dispo

Imported both by students (notebook / my_model.py) and by the Pastis
scoring sandbox. Versions must stay in sync.
Do NOT modify this file — it is part of the grading contract.
"""

from abc import ABC, abstractmethod

import ast
import inspect
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

        Notes
        -----
        The sandbox never calls fit() — it receives an already-trained
        model. Your fit() signature is therefore free: adding optional
        parameters (e.g. ``fit(self, df_raw, extra_df=None)`` for symmetry
        with contract v2 predict) is safe.
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

        Notes — contract v2 (optional: auxiliary data)
        ----------------------------------------------
        The signature above is contract v1, the default. It stays fully
        valid forever — nothing forces you to change it.

        If the scenario declares auxiliary data sources (weather, station
        geography, ...), your model can opt in to receive them by declaring
        an explicit ``extra_df`` parameter:

            def predict(self, df_raw, extra_df=None) -> np.ndarray: ...

        How it works:

        - The sandbox inspects ``inspect.signature(model.predict)``. If an
          explicitly named ``extra_df`` parameter is present, it calls
          ``model.predict(df, extra_df=extra)``. A bare ``**kwargs`` does
          NOT count — it is excluded on purpose (a raw sklearn Pipeline has
          ``predict(X, **predict_params)`` and would forward ``extra_df``
          to its final estimator, crashing it).
        - ``extra_df`` is a dict ``{source_key: pd.DataFrame}`` — one entry
          per auxiliary source declared by the scenario, each DataFrame at
          the natural granularity of its source (e.g. hourly weather = one
          row per hour for the city; station geography = one row per
          station). It is NOT a single DataFrame.
        - Datetime columns in these DataFrames arrive already parsed as
          datetime dtypes.
        - The platform never joins the sources to the main DataFrame —
          merging them in your own make_features() is your job (and where
          the value is).
        - If the scenario declares no source, the sandbox always calls
          ``predict(df)`` v1-style, even if your model accepts ``extra_df``.
          Defaulting ``extra_df=None`` is therefore always safe.

        The output contract is unchanged in v2: np.ndarray of floats in
        [0.0, 1.0], shape (len(df_raw),).
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
    extra_df : dict[str, pd.DataFrame], optional
        Auxiliary data sources keyed by source key — the same dict the
        scoring sandbox passes to a contract-v2 predict() (and that
        load_aux_data() in my_model.py returns locally). One entry per
        source declared by the scenario, each DataFrame at the natural
        granularity of its source (e.g. hourly weather = one row per hour;
        station geography = one row per station).
        This baseline helper ignores it on purpose: joining auxiliary data
        to the main DataFrame is student territory — do it in your own
        make_features().

    Returns
    -------
    pd.DataFrame
        Numeric DataFrame ready for sklearn. Does NOT include the target column.
    """
    out = pd.DataFrame(index=df.index)

    # Temporal features — strong predictors of station demand.
    #
    # snapshot_at is raw UTC (the CSV and the API both serve UTC). But Vélib
    # demand follows the human wall clock, not UTC: the morning commute peaks at
    # 8am *Paris time* all year round. Deriving `hour` straight from UTC would
    # smear that peak across two different UTC hours depending on the season,
    # because Paris is UTC+1 in winter and UTC+2 in summer (daylight saving).
    # That adds noise to the single most predictive signal.
    #
    # So we convert UTC -> Europe/Paris before extracting hour / dayofweek.
    # The IANA zone "Europe/Paris" handles DST automatically — never use a fixed
    # offset, which would be wrong for ~half the year. `utc=True` makes this
    # robust whether snapshot_at is tz-naive (UTC implied) or already tz-aware.
    #
    # NOTE: changing how features are derived here changes what the model was
    # trained on. Any existing submission.pkl must be retrained after this edit.
    ts = pd.to_datetime(df["snapshot_at"], utc=True).dt.tz_convert("Europe/Paris")
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

    # extra_df is intentionally unused here — merging auxiliary sources is
    # the student's job, in their own make_features() (see my_model.py).
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

def verify_submission(
    pkl_path: str,
    sample_df: pd.DataFrame = None,
    extra_df: dict = None,
) -> dict:
    """
    Load a .pkl file and verify that its predict() output is valid.

    Parameters
    ----------
    pkl_path : str
        Path to the submission .pkl file.
    sample_df : pd.DataFrame, optional
        A sample DataFrame with AVAILABLE_FEATURES columns.
        Defaults to loading SAMPLE_DATA_PATH (first 50 rows).
    extra_df : dict[str, pd.DataFrame], optional
        Auxiliary data sources keyed by source key (contract v2).
        If provided AND the loaded model's predict() declares an explicit
        ``extra_df`` parameter, predict is called with it — exactly like
        the scoring sandbox. Otherwise predict(sample_df) is called v1-style.

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

    # Contract-v2 sniffing — mirror of what the scoring sandbox does:
    # pass extra_df only if predict() declares it as an explicitly named
    # parameter. A bare **kwargs does NOT count (a raw sklearn Pipeline has
    # predict(X, **predict_params) and would forward extra_df to its final
    # estimator, crashing it — the sandbox excludes it on purpose).
    use_extra = False
    if extra_df is not None:
        try:
            params = inspect.signature(model.predict).parameters
            use_extra = (
                "extra_df" in params
                and params["extra_df"].kind != inspect.Parameter.VAR_KEYWORD
            )
        except (TypeError, ValueError):
            use_extra = False

    try:
        if use_extra:
            output = model.predict(sample_df, extra_df=extra_df)
        else:
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

def brier_score_local(y_true, y_pred, label: str = "") -> float:
    """
    Compute the leaderboard metric: Brier Score = MSE(y_true, y_pred).

    This is a REGRESSION metric on the continuous fill rate — the mean
    squared error between the predicted taux_remplissage and the true one,
    both in [0.0, 1.0]. It is NOT sklearn.metrics.brier_score_loss, which is
    reserved for binary-classification probabilities: here y_true is a
    continuous rate, not a 0/1 label, so we compute the squared error
    directly.

    Lower is better. On the dev dataset the naive baseline (always predict
    the training mean) scores ≈ 0.037 — roughly the variance of the target.

    Parameters
    ----------
    y_true : array-like
        True taux_remplissage, values in [0, 1].
    y_pred : array-like
        Predicted taux_remplissage, values in [0, 1].
    label : str, optional
        Label prefix for the printed output lines.

    Returns
    -------
    float
        The Brier Score (mean squared error).
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    bs = float(np.mean((y_pred - y_true) ** 2))
    leaderboard = (1 - bs) * 100
    prefix = f"{label} — " if label else ""
    print(f"{prefix}Brier Score      : {bs:.4f}  (lower is better)")
    print(f"{prefix}Leaderboard score: {leaderboard:.2f} (higher is better)")
    return bs
