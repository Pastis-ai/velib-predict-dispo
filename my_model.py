"""
my_model.py — Your Pastis.ai submission for velib-predict-dispo

OBJECTIVE: predict taux_remplissage = numbikesavailable / capacity
           for a given station at a given time.
           Output is a float in [0.0, 1.0].

Quickstart:
    python my_model.py          # train, evaluate, export submission.pkl
    python my_model.py --verify # verify an existing submission.pkl
"""

import os
import sys
import cloudpickle
import argparse
import time

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingRegressor

from pastis_velib import (
    PastisBaseModel,
    AVAILABLE_FEATURES,
    make_basic_features,
    compute_target,
    verify_submission,
    brier_score_local,
    SAMPLE_DATA_PATH,
)

# --- Configuration ---
DATA_PATH  = "velib_dataset_dev.csv"
OUTPUT_PKL = "submission.pkl"
API_BASE   = "https://pastis.ai/api/scenarios/velib-predict-dispo"


# ---------------------------------------------------------------------------
# MyModel — implement your best model here
# ---------------------------------------------------------------------------

class MyModel(PastisBaseModel):
    """
    Your ML pipeline for the velib-predict-dispo challenge.

    The default model is HistGradientBoostingRegressor from sklearn.
    Override make_features() to improve your score.

    The predict() method clips output to [0.0, 1.0] — regressors can
    occasionally produce values just outside this range.
    """

    def __init__(self):
        # HistGradientBoostingRegressor — sklearn's fastest gradient boosting.
        # It builds decision trees sequentially, each one correcting the errors
        # of the previous one. This is called "boosting" vs RandomForest's
        # "bagging" (parallel trees trained independently).
        #
        # Key hyperparameters to tune (see Going Further → Level 3):
        #   max_iter     : number of boosting rounds (more = potentially better, slower)
        #   learning_rate: how much each tree corrects errors (lower = more conservative)
        #   max_depth    : maximum depth of each tree (controls overfitting)
        #
        # Why not RandomForestRegressor?
        #   RandomForest averages its leaf node values → predictions cluster around
        #   the training mean, rarely reaching 0.0 or 1.0.
        #   Gradient boosting corrects residuals iteratively → better at extremes.
        self.model = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
        )
        # Per-station mean fill rate, computed from training data in fit()
        # Used as a feature in make_features() — gives the model a per-station anchor
        self.station_means: dict = {}

    def make_features(self, df: pd.DataFrame, extra_df=None) -> pd.DataFrame:
        """
        Feature engineering — THIS IS WHERE YOU ADD VALUE.

        Start with make_basic_features() then add your own features.
        The goal: give the model information it can use to distinguish
        "this station will be empty at 8am on Monday" from
        "this station will be half-full at 3pm on Sunday".
        """
        features = make_basic_features(df, extra_df=extra_df)

        # --- Station historical mean fill rate ---
        # Each station has a different structural fill rate.
        # By providing this per-station average (computed from training data),
        # we encode the structural difference between stations as a feature.
        #
        # IMPORTANT: self.station_means is computed in fit() from training data only.
        # We never use test data to compute this — that would be data leakage.
        if self.station_means:
            features["station_mean_rate"] = (
                df["stationcode"]
                .astype(str)
                .map(self.station_means)
                .fillna(0.2)   # fallback for stations not seen during training
            )

        # --- TODO: add your own features below ---
        # Uncomment and experiment:
        #
        # Interaction term: "8am on Friday" is very different from "8am on Sunday"
        # features["hour_x_dow"] = features["hour"] * features["dayofweek"]
        #
        # Rush hour flag: simplified commute signal
        # features["is_rush_hour"] = features["hour"].isin([7,8,9,17,18,19]).astype(int)
        #
        # Capacity bins: small/medium/large station behavior differs
        # features["capacity_bin"] = pd.cut(
        #     df["capacity"], bins=[0, 15, 25, 999], labels=[0, 1, 2]
        # ).astype(int)

        return features

    def fit(self, df_raw: pd.DataFrame) -> None:
        """
        Train the model on raw data.

        Step 1: compute per-station statistics from training data
        Step 2: build features (station_mean_rate uses the stats from step 1)
        Step 3: compute target (taux_remplissage)
        Step 4: fit the sklearn estimator
        """
        target = compute_target(df_raw)
        # Compute per-station mean fill rate — stored in self for make_features()
        self.station_means = (
            df_raw.assign(_target=target)
            .groupby(df_raw["stationcode"].astype(str))["_target"]
            .mean()
            .to_dict()
        )
        X = self.make_features(df_raw)
        self.model.fit(X, target)

    def predict(self, df_raw: pd.DataFrame) -> np.ndarray:
        """
        Return predicted taux_remplissage for each row.
        Output clipped to [0.0, 1.0] — regressors can exceed this range.
        """
        X = self.make_features(df_raw)
        return np.clip(self.model.predict(X), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """
    Load data for training.

    Priority:
    1. Local CSV (velib_dataset_dev.csv) — fastest for development
    2. Download from Pastis API — for larger or newer datasets
    """
    if os.path.exists(DATA_PATH):
        print(f"Loading local dataset: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH, parse_dates=["snapshot_at"])
    else:
        print("Local file not found — downloading from API...")
        params = {
            "date_from": "2026-04-14",
            "date_to": "2026-04-30",
            "arrondissements": ["Paris 18e", "Paris 19e", "Paris 20e"],
        }
        with requests.get(
            f"{API_BASE}/download",
            params=params,
            stream=True,
            timeout=300,
        ) as r:
            r.raise_for_status()
            with open(DATA_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        df = pd.read_csv(DATA_PATH, parse_dates=["snapshot_at"])

    fill_rate = compute_target(df).mean()
    print(f"  {len(df):,} rows | {df['stationcode'].nunique()} stations")
    print(f"  Date range   : {df['snapshot_at'].min()} → {df['snapshot_at'].max()}")
    print(f"  Mean fill rate: {fill_rate:.1%}")
    return df


def temporal_split(df: pd.DataFrame, test_fraction: float = 0.2):
    """
    Temporal split — NEVER random split on time series data.

    Random split leaks future information into the training set,
    giving inflated scores that do not reflect real-world performance.
    """
    df = df.sort_values("snapshot_at").reset_index(drop=True)
    idx = int(len(df) * (1 - test_fraction))
    df_train = df.iloc[:idx].copy()
    df_test  = df.iloc[idx:].copy()
    print(f"  Train: {len(df_train):,} rows (up to {df_train['snapshot_at'].max()})")
    print(f"  Test : {len(df_test):,}  rows (from {df_test['snapshot_at'].min()})")
    return df_train, df_test


# ---------------------------------------------------------------------------
# Main: train → evaluate → export
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train and export a Pastis.ai submission.")
    parser.add_argument("--verify", action="store_true", help="Only verify an existing submission.pkl")
    args = parser.parse_args()

    # --- Verify-only mode ---
    if args.verify:
        print(f"\nVerifying {OUTPUT_PKL}...")
        result = verify_submission(OUTPUT_PKL)
        print(f"  Valid   : {result['valid']}")
        print(f"  Message : {result['message']}")
        print(f"  Sample  : {result['sample_output']}")
        return

    # --- [1/5] Load data ---
    print("\n[1/5] Loading data...")
    df = load_data()

    # --- [2/5] Temporal train / test split ---
    print("\n[2/5] Splitting data (temporal, not random)...")
    df_train, df_test = temporal_split(df, test_fraction=0.2)

    # --- [3/5] Naive baseline: always predict mean fill rate from train set ---
    print("\n[3/5] Evaluating naive baseline (always predict mean fill rate)...")
    y_test = compute_target(df_test).values
    baseline_pred_value = compute_target(df_train).mean()
    baseline_preds = np.full(len(df_test), baseline_pred_value)
    print(f"  Baseline always predicts: {baseline_pred_value:.4f}")
    brier_score_local(y_test, baseline_preds, label="Naive baseline")

    # --- [4/5] Train your model ---
    print("\n[4/5] Training MyModel...")
    t0 = time.time()
    model = MyModel()
    model.fit(df_train)
    print(f"  Done in {time.time() - t0:.1f}s")

    # --- [5/5] Evaluate, interpret, export ---
    print("\n[5/5] Evaluating MyModel on test set...")
    y_pred = model.predict(df_test)
    bs_model = brier_score_local(y_test, y_pred, label="MyModel")

    bs_baseline = float(np.mean((baseline_preds - y_test) ** 2))
    # Leakage guard: the naive baseline already sits near 0.037 on this dataset,
    # so a legitimate feature-engineered model lands around 0.02-0.03. Only a
    # near-perfect score (BS < 0.005, leaderboard > 99.5) is implausible without
    # leaking the target — that is what this branch flags.
    if bs_model >= bs_baseline:
        print("  ⚠  Model does not beat baseline — check your features")
    elif bs_model < 0.005:
        print("  ⚠  Suspiciously low Brier Score — verify there is no data leakage")
    else:
        improvement_pct = (bs_baseline - bs_model) / bs_baseline * 100
        print(f"  ✓  {improvement_pct:.1f}% improvement over baseline")

    # Retrain on full dataset before export
    print(f"\nRetraining on full dataset before export...")
    t0 = time.time()
    model.fit(df)
    print(f"  Done in {time.time() - t0:.1f}s")

    print(f"\nExporting to {OUTPUT_PKL}...")
    import my_model as _my_model_module
    cloudpickle.register_pickle_by_value(_my_model_module)
    with open(OUTPUT_PKL, "wb") as f:
        cloudpickle.dump(model, f)
    print(f"  Saved: {OUTPUT_PKL}")

    print(f"\nVerifying {OUTPUT_PKL}...")
    result = verify_submission(OUTPUT_PKL)
    print(f"  Valid   : {result['valid']}")
    print(f"  Message : {result['message']}")
    print(f"  Sample  : {result['sample_output']}")

    if result["valid"]:
        print("\nReady to upload submission.pkl to https://pastis.ai")
    else:
        print("\nFix the issues above before submitting.")
        sys.exit(1)

    # --- LEVEL 3: Hyperparameter tuning (uncomment to run) ---
    #
    # TimeSeriesSplit preserves temporal order during cross-validation.
    # Never use KFold or train_test_split(shuffle=True) on time series —
    # that would leak future data into the training folds.
    #
    # from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    # X_train = model.make_features(df_train)
    # y_train = compute_target(df_train).values
    # param_grid = {
    #     "max_iter"      : [100, 200, 400],
    #     "learning_rate" : [0.02, 0.05, 0.1],
    #     "max_depth"     : [4, 6, 8],
    # }
    # tscv = TimeSeriesSplit(n_splits=3)
    # gs = GridSearchCV(
    #     HistGradientBoostingRegressor(random_state=42),
    #     param_grid,
    #     cv=tscv,
    #     scoring="neg_mean_squared_error",
    #     verbose=1,
    # )
    # gs.fit(X_train, y_train)
    # print(f"Best params : {gs.best_params_}")
    # print(f"Best CV MSE : {-gs.best_score_:.4f}")


if __name__ == "__main__":
    main()
