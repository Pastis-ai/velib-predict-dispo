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
    ANSWER_COLUMNS,
    make_basic_features,
    compute_target,
    station_key,
    as_serve_frame,
    preflight_check,
    verify_submission,
    brier_score_local,
    SAMPLE_DATA_PATH,
)

# Raw-column dtypes, pinned so they never depend on what pandas infers.
# stationcode is the one that matters: it is used as a dict key, and the
# full API export contains a few rows with an empty stationcode, which is
# enough for pandas to read the whole column as float64 ("20001.0" instead
# of "20001"). Pin it at every read_csv — see station_key() for the rest.
RAW_DTYPES = {"stationcode": "string"}

# --- Configuration ---
DATA_PATH  = "velib_dataset_dev.csv"
OUTPUT_PKL = "submission.pkl"
API_BASE   = "https://pastis.ai/api/scenarios/velib-predict-dispo"


def export_submission(model, path=OUTPUT_PKL):
    """Single authorized way to write a submission .pkl.

    The scoring sandbox imports ``pastis_velib`` but does NOT have access to
    your local ``my_model`` module. cloudpickle must therefore serialize these
    local modules BY VALUE (embedding their bytecode) so the pickle is
    self-contained. ``register_pickle_by_value`` is a per-session cloudpickle
    setting that must be active at dump time — it is not stored on the model —
    so every export path must go through this function.

    If you add a new local module imported by your model (e.g. ``features.py``),
    register it here too, otherwise scoring fails with "No module named ...".
    """
    import my_model as _my_model_module
    import pastis_velib as _pastis_velib_module
    cloudpickle.register_pickle_by_value(_my_model_module)
    cloudpickle.register_pickle_by_value(_pastis_velib_module)
    with open(path, "wb") as f:
        cloudpickle.dump(model, f)
    return path


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

        The boundary you are working inside
        -----------------------------------
        This method runs twice on two different frames, and that asymmetry is
        the whole game:

        - from fit(), on the training CSV — every column, answers included;
        - from predict(), on a live snapshot cut down to AVAILABLE_FEATURES —
          twelve columns, and none of the five ANSWER_COLUMNS.

        So anything you read here must come from those twelve, or from a
        statistic your model computed in fit() and carries inside itself
        (self.station_means below is the reference example). Reading an answer
        column gets the submission rejected, not silently rewarded.
        """
        features = make_basic_features(df, extra_df=extra_df)

        # --- Station historical mean fill rate ---
        # Each station has a different structural fill rate.
        # By providing this per-station average (computed from training data),
        # we encode the structural difference between stations as a feature.
        #
        # IMPORTANT: self.station_means is computed in fit() from training data only.
        # We never use test data to compute this — that would be data leakage.
        #
        # station_key() — not .astype(str) — on BOTH sides of the lookup.
        # fit() and this line must produce byte-identical keys, or every
        # station falls back to 0.2 and this feature silently dies. See
        # station_key() in pastis_velib.py for why that is not theoretical.
        if self.station_means:
            features["station_mean_rate"] = (
                station_key(df["stationcode"])
                .map(self.station_means)
                .astype(float)
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
        #
        # What NOT to add, whatever it does to your local score:
        # numbikesavailable, numdocksavailable, mechanical, ebike, has_bike.
        # They are the target in five spellings, they are absent from the
        # snapshot at scoring, and the platform probes for them at submission.
        # run `python my_model.py` — the preflight at the end catches this
        # before the upload page does.

        # --- TODO (advanced): auxiliary data — contract v2 ---
        # No auxiliary source is declared for this scenario today, so
        # load_aux_data() returns {} and this whole block stays commented.
        # It shows the upgrade path for the day a source appears
        # (see notebook "Going Further" -> Level 5).
        #
        # Step 1 — switch predict() to the v2 signature so the sandbox
        # passes the auxiliary dict in (declaring extra_df explicitly is
        # what opts you in — a **kwargs would NOT be detected):
        #
        #     def predict(self, df_raw, extra_df=None):
        #         X = self.make_features(df_raw, extra_df=extra_df)
        #         return np.clip(self.model.predict(X), 0.0, 1.0)
        #
        # At training time, pass the same dict: model.fit(df, extra_df=aux)
        # with aux = load_aux_data() — train/serve symmetry is guaranteed.
        #
        # Step 2 — join each source yourself. The platform NEVER pre-joins:
        # picking the join key and handling granularity/missing rows is
        # where you add value.
        #
        # Example A — static per-station source (one row per station),
        # e.g. a "station_geo" source with elevation:
        #
        # if extra_df and "station_geo" in extra_df:
        #     geo = extra_df["station_geo"].copy()
        #     geo["stationcode"] = pd.to_numeric(
        #         geo["stationcode"], errors="coerce"
        #     ).fillna(0).astype(int)
        #     features = features.merge(
        #         geo[["stationcode", "elevation"]],
        #         on="stationcode", how="left",
        #     )
        #
        # Example B — hourly city-wide source (one row per hour),
        # e.g. a "weather" source with a parsed datetime column "hour_utc":
        # snapshot_at is UTC — floor it to the hour to build the join key.
        # (For wall-clock FEATURES like hour-of-day, convert UTC ->
        # Europe/Paris exactly like make_basic_features() does; for a
        # UTC-to-UTC join key, no conversion is needed.)
        #
        # if extra_df and "weather" in extra_df:
        #     hour_utc = pd.to_datetime(df["snapshot_at"], utc=True).dt.floor("h")
        #     weather = extra_df["weather"].rename(columns={"hour_utc": "_hour"})
        #     features = (
        #         features.assign(_hour=hour_utc.values)
        #         .merge(weather, on="_hour", how="left")
        #         .drop(columns="_hour")
        #     )
        #
        # LEAKAGE WARNING: never join a source that contains (or derives
        # from) the target, and compute any join statistics (means, counts)
        # on the train set only — same rule as station_means above.
        #
        # NETWORK WARNING: the scoring sandbox runs with NO network access.
        # Never call requests/urllib inside predict() or make_features() —
        # it works locally, then fails at scoring. Auxiliary data must come
        # in through extra_df, nothing else.

        return features

    def fit(self, df_raw: pd.DataFrame, extra_df=None) -> None:
        """
        Train the model on raw data.

        Step 1: compute per-station statistics from training data
        Step 2: build features (station_mean_rate uses the stats from step 1)
        Step 3: compute target (taux_remplissage)
        Step 4: fit the sklearn estimator

        extra_df (optional) mirrors the contract-v2 predict() signature:
        a dict {source_key: DataFrame} of auxiliary sources, as returned by
        load_aux_data(). The sandbox never calls fit() — it receives an
        already-trained model — so this parameter is purely for your local
        training symmetry. Ignored unless your make_features() uses it.
        """
        target = compute_target(df_raw)
        # Compute per-station mean fill rate — stored in self for make_features()
        self.station_means = (
            df_raw.assign(_target=target)
            .groupby(station_key(df_raw["stationcode"]))["_target"]
            .mean()
            .to_dict()
        )
        X = self.make_features(df_raw, extra_df=extra_df)
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
        df = pd.read_csv(DATA_PATH, parse_dates=["snapshot_at"], dtype=RAW_DTYPES)
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
        df = pd.read_csv(DATA_PATH, parse_dates=["snapshot_at"], dtype=RAW_DTYPES)

    fill_rate = compute_target(df).mean()
    print(f"  {len(df):,} rows | {df['stationcode'].nunique()} stations")
    print(f"  Date range   : {df['snapshot_at'].min()} → {df['snapshot_at'].max()}")
    print(f"  Mean fill rate: {fill_rate:.1%}")
    return df


def load_full_dataset(
    date_from: str | None = None,
    date_to: str | None = None,
    arrondissements: list[str] | None = None,
    cache_path: str = "velib_full.csv",
) -> pd.DataFrame:
    """
    Download the FULL training dataset from the Pastis API.

    WARNING — this can be very large and slow:
    - All arrondissements over the full history = millions of rows
    - The dataset grows every day as new snapshots are scraped
    - Expect long download times and high memory usage on a laptop

    For heavy training, prefer Google Colab (more RAM, faster network).
    See the notebook "Going Further" section for Colab/TPU guidance.

    Parameters
    ----------
    date_from, date_to : str, optional
        Date range (YYYY-MM-DD). Default None/None = no bound: the API
        returns everything it has collected so far. Because the dataset is
        scraped continuously, "no date limit" always means "all of history
        up to right now" — it grows a little every day, it is not a fixed
        window. Pass explicit values to scope down to a smaller/faster
        download. An out-of-range value (e.g. a date_to in the future, or a
        date_from before data collection started) is not an error — the API
        silently clamps it to the real availability window, so you do not
        need to know the exact collection start date yourself.
    arrondissements : list[str], optional
        Filter to specific arrondissements (e.g. ["Paris 18e", "Paris 19e"]).
        Use the EXACT names returned by the /arrondissements endpoint:
            "Paris 1er", "Paris 18e", "Boulogne-Billancourt", ...
        NOT "Paris 18e Arrondissement" — that returns zero rows.
        None = ALL arrondissements (largest possible download).
    cache_path : str
        Local file to cache the download. If it exists, it is reused
        instead of re-downloading (delete it to force a fresh pull) —
        including when you widen the date range or the live dataset has
        grown since the cache was made. Delete the file to pick up either.

    Returns
    -------
    pd.DataFrame with parsed snapshot_at.
    """
    if date_from is not None and date_to is not None and date_from > date_to:
        raise ValueError(f"date_from ({date_from}) is after date_to ({date_to})")

    # Reuse cached file if present — avoids re-downloading on every run
    if os.path.exists(cache_path):
        print(f"Using cached full dataset: {cache_path}")
        print(f"  (delete {cache_path} to force a fresh download — e.g. if you widened the date range, or the live dataset has grown since this cache was made)")
        return pd.read_csv(cache_path, parse_dates=["snapshot_at"], dtype=RAW_DTYPES)

    params = {}
    if date_from is not None:
        params["date_from"] = date_from
    if date_to is not None:
        params["date_to"] = date_to
    if arrondissements:
        params["arrondissements"] = arrondissements

    print(f"Downloading full dataset from API...")
    print(f"  Period          : {date_from or 'earliest available'} -> {date_to or 'latest available (now)'}")
    print(f"  Arrondissements : {arrondissements or 'ALL (this is large!)'}")
    print(f"  This may take several minutes. Be patient.")

    with requests.get(
        f"{API_BASE}/download",
        params=params,
        stream=True,
        timeout=600,  # 10 min — full dataset can be slow
    ) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(cache_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                # Lightweight progress without tqdm dependency in the loop
                if total and downloaded % (5 * 1024 * 1024) < 8192:
                    pct = downloaded / total * 100
                    print(f"    {downloaded/1e6:.0f} MB ({pct:.0f}%)", end="\r")

    df = pd.read_csv(cache_path, parse_dates=["snapshot_at"], dtype=RAW_DTYPES)
    print(f"\n  Downloaded: {len(df):,} rows, {df['stationcode'].nunique()} stations")
    print(f"  Date range: {df['snapshot_at'].min()} -> {df['snapshot_at'].max()}")
    return df


def load_aux_data(cache_dir: str = "aux_data") -> dict:
    """
    Download the auxiliary data sources declared for this scenario.

    Contract-v2 companion to load_data():
    1. GET {API_BASE}/aux lists the declared sources
    2. each source is downloaded from {API_BASE}/aux/{key}/download and
       cached in cache_dir/ (delete a file to force a fresh download)
    3. returns {source_key: DataFrame} — the exact dict shape the sandbox
       passes to a v2 predict(df, extra_df=...) at scoring time.
       Train/serve symmetry is guaranteed by the platform: what you
       download here is what your model will receive.

    No auxiliary source is declared for this scenario today — this returns
    {} with a clear message. The code is future-proof: the day a source
    appears, the same call picks it up.

    NOTE: the sandbox delivers datetime columns already parsed. When
    loading from CSV here, parse them yourself (pd.to_datetime) — or parse
    defensively inside make_features() so both paths behave the same.
    """
    try:
        r = requests.get(f"{API_BASE}/aux", timeout=30)
        r.raise_for_status()
        sources = r.json().get("sources", [])
    except (requests.RequestException, ValueError):
        print("Could not list auxiliary sources (endpoint unreachable) — continuing without.")
        return {}

    if not sources:
        print("No auxiliary data source is declared for this scenario today.")
        print("That is expected — see the notebook 'Going Further' -> Level 5 for ideas.")
        return {}

    os.makedirs(cache_dir, exist_ok=True)
    extra = {}
    for src in sources:
        key = src["key"]
        path = os.path.join(cache_dir, f"{key}.csv")
        if os.path.exists(path):
            print(f"  {key}: using cached {path} (delete it to force a fresh download)")
        else:
            print(f"  {key}: downloading — {src.get('description', 'no description')}")
            with requests.get(
                f"{API_BASE}/aux/{key}/download",
                stream=True,
                timeout=300,
            ) as resp:
                resp.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
        extra[key] = pd.read_csv(path)
        print(f"  {key}: {len(extra[key]):,} rows | columns: {list(extra[key].columns)}")
    return extra


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
    # Two training modes:
    #   python my_model.py          → fast: local dev CSV (~42k rows, 3 arrondissements)
    #   python my_model.py --full   → heavy: full API dataset (millions of rows)
    #
    # Use the fast mode for development and debugging.
    # Use --full only when you are ready for a final training run,
    # ideally on Google Colab (see notebook for Colab/TPU guidance).
    parser = argparse.ArgumentParser(description="Train and export a Pastis.ai submission.")
    parser.add_argument("--verify", action="store_true", help="Only verify an existing submission.pkl")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Train on the full dataset from the API (large, slow — prefer Colab)",
    )
    parser.add_argument(
        "--date-from",
        default=None,
        help="With --full: restrict to snapshots from this date (YYYY-MM-DD). Default: earliest available.",
    )
    parser.add_argument(
        "--date-to",
        default=None,
        help="With --full: restrict to snapshots up to this date (YYYY-MM-DD). Default: latest available (now).",
    )
    args = parser.parse_args()

    # --- Verify-only mode ---
    if args.verify:
        print(f"\nVerifying {OUTPUT_PKL}...")
        result = verify_submission(OUTPUT_PKL)
        print(f"  Valid   : {result['valid']}")
        print(f"  Message : {result['message']}")
        print(f"  Sample  : {result['sample_output']}")
        for w in result["warnings"]:
            print(f"  ⚠  {w}")
        if result["valid"]:
            with open(OUTPUT_PKL, "rb") as f:
                preflight_check(cloudpickle.load(f), pkl_path=OUTPUT_PKL)
        return

    # --- [1/5] Load data ---
    if args.full:
        print("\n[1/5] Loading FULL dataset from API (this is large)...")
        df = load_full_dataset(date_from=args.date_from, date_to=args.date_to)
    else:
        print("\n[1/5] Loading local dev dataset...")
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

    # --- Serve-shape check ---
    # The score above cannot detect a train/serve mismatch: y_pred was
    # computed on df_test, which came out of the same loader, with the same
    # columns and dtypes, as the training data. The sandbox serves something
    # narrower — the twelve AVAILABLE_FEATURES, with its own dtypes — so a
    # model that depends on your loader scores perfectly here and collapses
    # there. Re-score the exact same rows in serve shape; the two numbers must
    # be identical.
    try:
        y_pred_serve = model.predict(as_serve_frame(df_test))
    except KeyError as missing:
        print(f"  ⚠  SERVE-SHAPE FAILURE: predict() needs {missing}, which the")
        print("     sandbox does not serve. Available at scoring:")
        print(f"     {', '.join(AVAILABLE_FEATURES)}")
        print(f"     Never served: {', '.join(ANSWER_COLUMNS)} — they are the target.")
        sys.exit(1)

    bs_serve = float(np.mean((y_pred_serve - y_test) ** 2))
    if np.isclose(bs_serve, bs_model, atol=1e-9):
        print(f"  ✓  Serve-shape check passed (BS unchanged: {bs_serve:.4f})")
    else:
        print(f"  ⚠  SERVE-SHAPE MISMATCH: {bs_model:.4f} here -> {bs_serve:.4f} at scoring")
        print("     Your model reads something the sandbox will not reproduce.")
        print("     Most common cause: a per-station dict keyed on a dtype your")
        print("     loader produced and the sandbox does not — use station_key().")

    # Retrain on full dataset before export
    print(f"\nRetraining on full dataset before export...")
    t0 = time.time()
    model.fit(df)
    print(f"  Done in {time.time() - t0:.1f}s")

    print(f"\nExporting to {OUTPUT_PKL}...")
    export_submission(model, OUTPUT_PKL)
    print(f"  Saved: {OUTPUT_PKL}")

    print(f"\nVerifying {OUTPUT_PKL}...")
    result = verify_submission(OUTPUT_PKL)
    print(f"  Valid   : {result['valid']}")
    print(f"  Message : {result['message']}")
    print(f"  Sample  : {result['sample_output']}")
    for w in result["warnings"]:
        print(f"  ⚠  {w}")

    if not result["valid"]:
        print("\nFix the issues above before submitting.")
        sys.exit(1)

    # --- Preflight ---
    # verify_submission() answers "is this a well-formed model?". The preflight
    # answers "would the platform accept it?" — determinism, no target leakage,
    # inside the 60s budget. Same checks the submission page runs, one minute
    # earlier. pkl_path adds the file's sha256: re-uploading a byte-identical
    # file is refused (409), so the fingerprint tells you which file this is.
    preflight = preflight_check(model, df, pkl_path=OUTPUT_PKL)

    if preflight["passed"]:
        print("\nReady to upload submission.pkl to https://pastis.ai")
    else:
        print("\nThis submission would be rejected. Fix the failures above.")
        sys.exit(1)

    # --- LEVEL 3: Hyperparameter tuning (uncomment to run) ---
    #
    # TimeSeriesSplit preserves temporal order during cross-validation.
    # Never use KFold or train_test_split(shuffle=True) on time series —
    # that would leak future data into the training folds.
    #
    # Tuning leaves you with several candidate models. The "Bonus — Pick Your
    # Champion" section at the end of starter.ipynb shows how to compare them
    # side by side with skore before choosing the one to submit.
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
