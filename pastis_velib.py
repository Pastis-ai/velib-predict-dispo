"""
pastis_velib.py — Pastis.ai base library for velib-predict-dispo

Imported both by students (notebook / my_model.py) and by the Pastis
scoring sandbox. Versions must stay in sync.
Do NOT modify this file — it is part of the grading contract.
"""

from abc import ABC, abstractmethod

import ast
import hashlib
import inspect
import os
import pickle
import time

import numpy as np
import pandas as pd


# --- Column contracts ---
#
# Two lists, two different jobs. Confusing them is the most expensive mistake
# you can make in this kit: it produces a model that works perfectly on your
# machine and is REFUSED at submission.
#
#   AVAILABLE_FEATURES → what predict() receives at scoring time.
#   ANSWER_COLUMNS     → what your training CSV also carries, and predict() never sees.

# The scoring sandbox applies a whitelist to the snapshot just before running
# your model: every column outside this list is removed, at scoring AND at
# submission validation. These twelve are all predict() ever gets.
AVAILABLE_FEATURES = [
    "stationcode",
    "name",
    "capacity",
    "snapshot_at",
    "coordonnees_geo",
    "nom_arrondissement_communes",
    "nom_arrondissement_communes_raw",
    "code_insee_commune",
    "station_opening_hours",
    "is_installed",
    "is_renting",
    "is_returning",
]

# The answer, in five spellings. Present in the training data — compute_target()
# needs numbikesavailable to build y, and there is no supervised learning
# without the answer — and NEVER served to predict().
#
#   numbikesavailable  the target, literally
#   mechanical + ebike sums to numbikesavailable exactly
#   numdocksavailable  capacity - numdocksavailable is the target, near-exactly
#   has_bike           target > 0
#
# The name for putting one of these in X is TARGET LEAKAGE. numdocksavailable
# is the instructive case: it looks like an ordinary feature, and it is
# capacity - answer. Train on it and you get a spectacular local score from a
# model that has learned nothing — at prediction time nobody knows how many
# bikes are at the station, which is the entire reason the scenario exists.
#
# So these are not "banned columns" in the abstract. They are banned AT SERVE.
# The platform runs a counterfactual probe on every submission: it calls your
# model twice on two frames identical over AVAILABLE_FEATURES and opposite over
# these five. If your predictions move on more than 5% of the rows, the model
# is reading the answer and the submission is rejected. Run preflight_check()
# to catch this before you upload.
ANSWER_COLUMNS = [
    "numbikesavailable",
    "numdocksavailable",
    "mechanical",
    "ebike",
    "has_bike",
]

# Shape of the training data — both velib_dataset_dev.csv and the /download
# API serve exactly these twelve columns: seven trainable features + the five
# answers.
#
# There are THREE data surfaces in this scenario, and they are not meant to
# match:
#
#   1. velib_dataset_dev.csv — an EXPLORATION fixture. Small, fixed, fully
#      known, so the notebook's commentary and charts stay true. You look at
#      it; you do not have to train on it.
#   2. the /download API — the real TRAINING set: complete, current, 24h
#      embargo. This is what you train on.
#   3. the live snapshot — the twelve AVAILABLE_FEATURES, at scoring time.
#
# The contract that must hold is 2 <-> 3. And note DATASET_COLUMNS is NOT
# AVAILABLE_FEATURES + ANSWER_COLUMNS: five served columns
# (nom_arrondissement_communes_raw, code_insee_commune, station_opening_hours,
# is_installed, is_returning) exist in the live snapshot only. A model trained
# on the seven receives twelve at serve time and ignores the extra five —
# no error, no leak. If you want to use one, read it defensively:
# df.get("is_returning").
DATASET_COLUMNS = [
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

# --- Submission limits enforced by the platform ---

# One scoring cycle = one live snapshot of all Paris, ~1500 rows.
SNAPSHOT_ROWS = 1500
# predict() must return within this budget on a full snapshot — at validation
# AND on every scoring cycle afterwards.
PREDICT_TIME_BUDGET_S = 60.0
# Hard cap on the uploaded .pkl. Empty or truncated files are refused too.
MAX_SUBMISSION_MB = 50
# Two runs on the same input must agree to this tolerance.
DETERMINISM_TOLERANCE = 1e-9


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
            Raw training data — the DATASET_COLUMNS shape. It carries the
            ANSWER_COLUMNS (that is how compute_target() builds y); predict()
            will not.

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
            One live snapshot, restricted to AVAILABLE_FEATURES — twelve
            columns, nothing else. The ANSWER_COLUMNS are stripped before
            your model runs, so reading one is not "leakage you got away
            with", it is a KeyError at best and a rejected submission at
            worst. Build your features from as_serve_frame() locally and you
            are working against the real boundary.

        Returns
        -------
        np.ndarray
            1D array of floats in [0.0, 1.0], shape (len(df_raw),).
            Immutable output contract — the sandbox enforces it strictly.

        Three rules the platform checks at submission
        ---------------------------------------------
        - Deterministic: two calls on the same frame must agree to 1e-9.
          Set random_state / np.random.seed everywhere it exists.
        - Fast: under 60 s on a full ~1500-row snapshot — the budget of
          every scoring cycle, not just of validation.
        - Offline: the sandbox runs with no network (``--network none``).
          No requests, no urllib, no download inside predict().

        ``preflight_check()`` replays all three on your machine.

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


# --- Station key normalisation ---

def station_key(codes) -> pd.Series:
    """
    Canonical station key — identical whatever dtype stationcode arrives as.

    Use this every time you build a per-station dict / groupby key, and
    every time you look one up. Never use ``stationcode.astype(str)``
    directly.

    Why this exists
    ---------------
    ``stationcode`` does not always parse to the same dtype. The committed
    dev CSV is clean, so pandas infers int64 and ``astype(str)`` gives
    "20001". The full API export contains a handful of rows with an empty
    stationcode, so pandas infers float64 instead — and the very same
    ``astype(str)`` now gives "20001.0".

    Build a dict with one form, look it up with the other, and *nothing*
    matches. There is no error: the lookup simply returns NaN, your
    ``fillna(default)`` turns it into a plausible number, and the feature
    silently becomes a constant. A model trained that way validates fine
    locally and collapses at scoring time.

    Returns
    -------
    pd.Series
        StringDtype series, same index as the input. Missing codes are
        <NA> — not the string "nan" — so groupby() drops them from the
        statistics and map() leaves them unmatched, letting them follow
        the normal fallback path instead of forming a bogus bucket.
    """
    s = pd.Series(codes).astype("string").str.strip()
    # "20001.0" -> "20001": undo float inference, whatever the source dtype.
    s = s.str.replace(r"^(\d+)\.0+$", r"\1", regex=True)
    return s.replace("", pd.NA)


def as_serve_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rebuild a DataFrame the way the scoring sandbox delivers it.

    The sandbox does not hand you the frame your loader produced. It applies
    a whitelist to the live snapshot — AVAILABLE_FEATURES, twelve columns —
    and builds its own frame from those, with whatever dtypes *its* reader
    inferred. Two things are therefore invisible to a plain holdout, because
    the holdout reuses your training frame:

    1. The ANSWER_COLUMNS are gone. Your training CSV has them, the snapshot
       your model is scored on does not.
    2. The dtypes are not yours. ``stationcode`` in particular has no stable
       dtype across readers (see ``station_key``).

    This helper reproduces that boundary:
    - drops every ANSWER_COLUMN — the five columns that spell out the target
    - keeps only AVAILABLE_FEATURES columns, in contract order
    - stationcode as plain strings, no float artefact ("20001", not "20001.0")
    - snapshot_at as the raw timestamp strings the CSV and API serve
    - a fresh RangeIndex

    Use it as a second evaluation, not a replacement::

        bs_holdout = brier_score_local(y_test, model.predict(df_test))
        bs_serve   = brier_score_local(y_test, model.predict(as_serve_frame(df_test)))

    The two numbers must match. If ``model.predict`` now raises a KeyError,
    that is the point of the helper: the feature it wants does not exist at
    scoring time. If both run but the scores differ, the model depends on
    something the sandbox will not reproduce — most often a per-station dict
    keyed on a dtype that only exists in your loader.

    Note on coverage: the training data carries seven of the twelve served
    columns, so locally you get seven. The five snapshot-only columns
    (``is_installed``, ``is_returning``, ``code_insee_commune``,
    ``station_opening_hours``, ``nom_arrondissement_communes_raw``) are
    served at scoring but absent here — read them defensively
    (``df.get("is_returning")``) if you use them at all. This is not a bug
    and not a leak: a model trained on the seven simply ignores the extra
    five it is handed at scoring.
    """
    out = df.drop(columns=[c for c in ANSWER_COLUMNS if c in df.columns])
    out["stationcode"] = station_key(out["stationcode"])
    out["snapshot_at"] = (
        pd.to_datetime(out["snapshot_at"], utc=True).dt.tz_convert(None).astype(str)
    )
    cols = [c for c in AVAILABLE_FEATURES if c in out.columns]
    return out[cols].reset_index(drop=True)


def serve_snapshot_frame(df: pd.DataFrame, n_rows: int = SNAPSHOT_ROWS) -> pd.DataFrame:
    """
    Build a serve-shaped frame the size of one real scoring cycle.

    A scoring cycle is one snapshot of all Paris — every station at a single
    timestamp, ~1500 rows. The dev CSV holds 67 stations, so one of its
    snapshots is 67 rows: timing a model on it says nothing about the 60 s
    budget. This repeats the most recent snapshot until it reaches n_rows,
    which gets the *scale* right even though the dev data cannot get the
    station count right.

    Used by preflight_check(). Also handy on its own::

        frame = serve_snapshot_frame(df)
        %timeit model.predict(frame)
    """
    frame = as_serve_frame(df)
    if len(frame) == 0:
        return frame

    latest = frame[frame["snapshot_at"] == frame["snapshot_at"].max()]
    if len(latest) > 0:
        frame = latest

    repeats = int(np.ceil(n_rows / len(frame)))
    return pd.concat([frame] * repeats, ignore_index=True).head(n_rows)


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
# NOT AVAILABLE AT ALL (the answer — see ANSWER_COLUMNS):
#   numbikesavailable, mechanical, ebike, numdocksavailable, has_bike
#     → These directly reveal taux_remplissage = numbikesavailable / capacity.
#       They are in your training CSV so you can build the target from them,
#       and they are stripped from the snapshot before predict() runs. So this
#       is not a judgement call about leakage you might make differently — the
#       columns are simply not there at scoring, and a model that reaches for
#       one is rejected at submission by the counterfactual probe.
#       If you want history, derive it from the training data and store it in
#       your model (self.station_means is exactly that pattern): a statistic
#       computed in fit() travels inside the .pkl, a column does not.
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

    # NOTE: the ANSWER_COLUMNS (numbikesavailable, numdocksavailable,
    # mechanical, ebike, has_bike) are not read here — and could not be.
    # They are stripped from the snapshot before predict() runs.

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


# --- Preflight: replay the platform's submission checks locally ---

def _inject_answer_columns(frame: pd.DataFrame, bikes, cap) -> pd.DataFrame:
    """Add the five ANSWER_COLUMNS to a serve frame, consistent with `bikes`."""
    bikes = np.asarray(bikes)
    return frame.assign(
        numbikesavailable=bikes,
        numdocksavailable=np.asarray(cap) - bikes,
        mechanical=bikes // 2,
        ebike=bikes - bikes // 2,
        has_bike=(bikes > 0).astype(int),
    )


def _time_check(elapsed: float, n_rows: int, budget_s: float) -> dict:
    """Format check 3 — predict() must fit the per-cycle time budget."""
    if elapsed > budget_s:
        return {
            "name": "time budget", "status": "fail",
            "message": (
                f"predict() took {elapsed:.1f}s on {n_rows} rows, over the "
                f"{budget_s:g}s budget — which applies to every scoring cycle, "
                "not just to this validation."
            ),
        }
    return {
        "name": "time budget", "status": "pass",
        "message": f"{elapsed:.2f}s on {n_rows} rows (budget {budget_s:g}s).",
    }


def preflight_check(
    model,
    df_raw: pd.DataFrame = None,
    *,
    pkl_path: str = None,
    seed: int = 0,
    n_rows: int = SNAPSHOT_ROWS,
    time_budget_s: float = PREDICT_TIME_BUDGET_S,
    verbose: bool = True,
) -> dict:
    """
    Replay the platform's submission checks on your own machine.

    Every upload is validated before it reaches the leaderboard. The checks
    below are the ones that reject a model, and they are all cheap to run
    locally — which is the whole point: a rejection you discover here costs a
    minute, one you discover on the submission page costs a session.

    1. Determinism — predict() twice on the same frame, agree to 1e-9.
       Fails when a model leaves an unseeded RandomState somewhere.
    2. Target leakage — the counterfactual probe. Two frames identical over
       the twelve served columns and opposite over the five ANSWER_COLUMNS.
       Predictions must not move on 5% or more of the rows.
    3. Time — predict() on a full ~1500-row snapshot, under 60 s. That is the
       budget of every scoring cycle, not a one-off validation allowance.

    A fourth refusal cannot be tested locally, only surfaced: re-uploading a
    file that is **byte-identical** to one you already submitted is refused
    with a 409, naming your previous submission and its date. Pass pkl_path
    and the preflight prints the file's sha256 — the exact quantity the
    platform compares.

    Measured, not assumed: cloudpickle is not byte-reproducible, so two
    exports of the *same* model give two different hashes. A 409 therefore
    does not mean "your model has not changed" — it means you uploaded a file
    left over from a previous export instead of the one you just wrote. This
    check is also scoped to your own account: your submission is never blocked
    because it resembles someone else's.

    Check 2 FAILS OPEN, deliberately. If predict() raises on the probe frames
    (a strict pipeline that refuses two extra columns, for instance), no
    conclusion is drawn: you get a warning and a pass. The platform does the
    same, for the same reason — a false accusation costs far more than a
    missed one. Same logic one step further: a non-deterministic model moves
    under the probe on its own, so when check 1 fails, check 2 is skipped
    rather than run and misreported.

    Parameters
    ----------
    model : object
        Anything with a predict() method — your live MyModel instance, or the
        object you just loaded back from submission.pkl.
    df_raw : pd.DataFrame, optional
        Raw data to build the probe frame from. Defaults to SAMPLE_DATA_PATH.
    pkl_path : str, optional
        Path to the exported .pkl. Only used to report its sha256 fingerprint
        (see the 409 note above); the model itself is the one you passed in.
    seed : int
        Seed of the probe's random draw. Only changes which counterfactual
        values are tried, never whether a clean model passes.
    n_rows, time_budget_s : int, float
        Snapshot size and time budget. The defaults mirror the platform.
    verbose : bool
        Print a readable report. Set False to only use the returned dict.

    Returns
    -------
    dict
        {"passed": bool, "checks": [{"name", "status", "message"}, ...]}
        status is "pass", "fail", "skip" (could not conclude) or "info"
        (reported, never blocking). "passed" is False only if some check
        actually failed.
    """
    fingerprint = None
    if pkl_path is not None:
        try:
            with open(pkl_path, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()
            fingerprint = {
                "name": "fingerprint", "status": "info",
                "message": (
                    f"sha256 {digest[:16]}… — re-uploading a byte-identical file "
                    "is refused (409). Every export produces a new hash, even "
                    "from an identical model, so a 409 means you picked the old "
                    "file rather than this fresh one."
                ),
            }
        except OSError as e:
            fingerprint = {
                "name": "fingerprint", "status": "info",
                "message": f"could not read {pkl_path} ({e}).",
            }

    def report(checks):
        if fingerprint is not None:
            checks = checks + [fingerprint]
        passed = not any(c["status"] == "fail" for c in checks)
        if verbose:
            symbol = {"pass": "✓", "fail": "✗", "skip": "⚠", "info": "ℹ"}
            print("\nPreflight — replaying the platform's submission checks:")
            for c in checks:
                print(f"  {symbol[c['status']]}  {c['name']}: {c['message']}")
            if passed:
                print("  → Ready to upload.")
                print(
                    "     Reminder: your local score is NOT your leaderboard score. The\n"
                    "     leaderboard scores a live snapshot every 30 min, on all of Paris.\n"
                    "     Expect a gap of several points — that is normal, not a bug."
                )
            else:
                print("  → This submission would be REJECTED. Fix the ✗ above.")
        return {"passed": passed, "checks": checks}

    if df_raw is None:
        try:
            df_raw = pd.read_csv(SAMPLE_DATA_PATH, dtype={"stationcode": "string"})
        except FileNotFoundError:
            return report([{
                "name": "setup", "status": "skip",
                "message": f"No data to test on ({SAMPLE_DATA_PATH} not found).",
            }])

    try:
        frame = serve_snapshot_frame(df_raw, n_rows=n_rows)
    except KeyError as e:
        return report([{
            "name": "setup", "status": "skip",
            "message": f"Could not build a serve frame — missing raw column {e}.",
        }])

    if len(frame) == 0:
        return report([{
            "name": "setup", "status": "skip",
            "message": "Empty frame — nothing to test.",
        }])

    # --- 1. Determinism (and the timing measurement, from the first call) ---
    try:
        t0 = time.perf_counter()
        p1 = np.asarray(model.predict(frame), dtype=float)
        elapsed = time.perf_counter() - t0
        p2 = np.asarray(model.predict(frame), dtype=float)
    except Exception as e:
        return report([{
            "name": "predict() runs", "status": "fail",
            "message": f"predict() raised on a serve-shaped frame: {e}",
        }])

    drift = float(np.max(np.abs(p1 - p2))) if len(p1) else 0.0
    is_deterministic = drift <= DETERMINISM_TOLERANCE
    if not is_deterministic:
        checks = [{
            "name": "determinism", "status": "fail",
            "message": (
                f"two runs on the same input differ by {drift:.2e} "
                f"(max allowed {DETERMINISM_TOLERANCE:.0e}). Seed every "
                "source of randomness: random_state=..., np.random.seed(...)."
            ),
        }]
    else:
        checks = [{
            "name": "determinism", "status": "pass",
            "message": "two runs on the same input are identical.",
        }]

    # --- 2. Counterfactual probe — does the model read the answer? ---
    #
    # Only meaningful on a deterministic model: the probe reads "the output
    # moved when only the answer changed", and an unseeded model moves on its
    # own. Probing it would accuse it of the wrong crime.
    if not is_deterministic:
        checks.append({
            "name": "target leakage", "status": "skip",
            "message": "not probed — a non-deterministic model moves on its own. "
                       "Fix the seed above, then run this again.",
        })
        return report(checks + [_time_check(elapsed, len(frame), time_budget_s)])

    rng = np.random.default_rng(seed)
    cap = (
        pd.to_numeric(frame["capacity"], errors="coerce")
        .fillna(0).clip(lower=0).astype(int).to_numpy()
    )
    bikes_a = rng.integers(0, cap + 1)
    bikes_a[::3] = 0          # a third of the rows pinned to "empty station"
    bikes_b = cap - bikes_a   # the mirror: same rows, opposite answer

    try:
        pa = np.asarray(
            model.predict(_inject_answer_columns(frame, bikes_a, cap)), dtype=float
        )
        pb = np.asarray(
            model.predict(_inject_answer_columns(frame, bikes_b, cap)), dtype=float
        )
        if pa.shape != p1.shape or pb.shape != p1.shape:
            raise ValueError(f"probe returned {pa.shape} / {pb.shape}, expected {p1.shape}")
    except Exception as e:
        # Fail open — exactly like the platform. An exception here is not
        # evidence of anything.
        checks.append({
            "name": "target leakage", "status": "skip",
            "message": (
                f"could not conclude — predict() raised on the probe frame ({e}). "
                "Not counted against you; the platform skips it too."
            ),
        })
    else:
        moved = float(np.mean(np.abs(pa - pb) > 1e-6))
        if moved >= 0.05:
            checks.append({
                "name": "target leakage", "status": "fail",
                "message": (
                    f"predictions move on {moved:.1%} of rows (limit 5%) when only "
                    f"the answer columns change. Your model reads one of "
                    f"{', '.join(ANSWER_COLUMNS)} — none of them exist at scoring. "
                    "This is target leakage: the feature looks informative because "
                    "it IS the target, so the score collapses on real data."
                ),
            })
        else:
            checks.append({
                "name": "target leakage", "status": "pass",
                "message": f"predictions are stable ({moved:.1%} of rows moved, limit 5%).",
            })

    # --- 3. Time budget ---
    checks.append(_time_check(elapsed, len(frame), time_budget_s))

    return report(checks)


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
        Raw rows to test on. Defaults to SAMPLE_DATA_PATH (first 50 rows).
        Whatever you pass is put through as_serve_frame() first, so predict()
        sees exactly the twelve columns the sandbox serves — no more.
    extra_df : dict[str, pd.DataFrame], optional
        Auxiliary data sources keyed by source key (contract v2).
        If provided AND the loaded model's predict() declares an explicit
        ``extra_df`` parameter, predict is called with it — exactly like
        the scoring sandbox. Otherwise predict(sample_df) is called v1-style.

    Returns
    -------
    dict
        {"valid": bool, "message": str, "warnings": list[str],
         "sample_output": list}

        "warnings" holds non-blocking remarks — things that are legal under
        the contract but usually indicate a bug. Always present, often empty.
    """
    if sample_df is None:
        try:
            sample_df = pd.read_csv(SAMPLE_DATA_PATH, nrows=50)
        except FileNotFoundError:
            return {
                "valid": False,
                "message": f"Sample file not found: {SAMPLE_DATA_PATH}",
                "warnings": [],
                "sample_output": [],
            }

    # File-level checks, in the platform's own order: an empty or oversized
    # file is refused before anything is unpickled.
    try:
        size_bytes = os.path.getsize(pkl_path)
    except OSError as e:
        return {"valid": False, "message": f"Could not read {pkl_path}: {e}",
                "warnings": [], "sample_output": []}

    if size_bytes == 0:
        return {"valid": False,
                "message": "Empty file — the export did not produce anything.",
                "warnings": [], "sample_output": []}

    if size_bytes > MAX_SUBMISSION_MB * 1024 * 1024:
        return {
            "valid": False,
            "message": (
                f"File is {size_bytes / 1024 / 1024:.1f} MB, over the "
                f"{MAX_SUBMISSION_MB} MB limit. Most often a model that "
                "stored the training data inside itself — keep statistics "
                "(means, counts), not frames."
            ),
            "warnings": [], "sample_output": [],
        }

    try:
        with open(pkl_path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        return {"valid": False, "message": f"Could not load .pkl: {e}",
                "warnings": [], "sample_output": []}

    if not hasattr(model, "predict"):
        return {"valid": False, "message": "Object has no predict() method.",
                "warnings": [], "sample_output": []}

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

    # Predict on the serve shape, never on the raw loader frame. The sandbox
    # whitelists the snapshot down to AVAILABLE_FEATURES before calling your
    # model; verifying against anything wider would bless a model that reads
    # columns it will never receive.
    try:
        serve_df = as_serve_frame(sample_df)
    except KeyError as e:
        return {
            "valid": False,
            "message": (
                f"Sample data is missing a raw column needed to build the serve "
                f"frame: {e}. It must carry at least stationcode and snapshot_at."
            ),
            "warnings": [], "sample_output": [],
        }

    try:
        if use_extra:
            output = model.predict(serve_df, extra_df=extra_df)
        else:
            output = model.predict(serve_df)
    except Exception as e:
        return {
            "valid": False,
            "message": (
                f"predict() raised an error: {e}\n"
                "    Note: predict() is called on the twelve columns the sandbox "
                "serves. A KeyError here means the feature you are asking for "
                "does not exist at scoring time (see AVAILABLE_FEATURES)."
            ),
            "warnings": [], "sample_output": [],
        }

    if not isinstance(output, np.ndarray):
        return {
            "valid": False,
            "message": f"predict() must return np.ndarray, got {type(output).__name__}.",
            "warnings": [],
            "sample_output": [],
        }

    if output.shape != (len(serve_df),):
        return {
            "valid": False,
            "message": f"Shape mismatch: expected ({len(serve_df)},), got {output.shape}.",
            "warnings": [],
            "sample_output": [],
        }

    if not np.all((output >= 0.0) & (output <= 1.0)):
        bad = output[(output < 0.0) | (output > 1.0)]
        return {
            "valid": False,
            "message": f"Values out of [0, 1]: {bad[:5]}",
            "warnings": [],
            "sample_output": output.tolist()[:5],
        }

    # Dispersion check — the one the other three miss.
    # "No crash", "no NaN" and "inside [0, 1]" are all satisfied by a model
    # that returns the same number for every row. That is a legitimate
    # model (a constant baseline is exactly that), but it is also what a
    # per-station lookup looks like once its keys stop matching: every
    # lookup misses, every fillna(default) fires, and the output collapses
    # to a single plausible value. Warn, never fail — only the author knows
    # which of the two they meant.
    #
    # max == min rather than std == 0: np.std over thousands of identical
    # float64 values returns ~1e-17, not an exact zero, so a std test would
    # miss the very constant it is looking for on any realistic frame.
    warnings = []
    if len(output) > 1 and float(np.max(output)) == float(np.min(output)):
        warnings.append(
            f"All {len(output)} predictions are identical ({output[0]:.4f}). "
            "Intended for a constant baseline — otherwise your lookup keys "
            "are not matching (see station_key)."
        )

    return {
        "valid": True,
        "message": "Submission is valid. Ready to upload to pastis.ai.",
        "warnings": warnings,
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

    THIS IS NOT YOUR LEADERBOARD SCORE. It is computed on whatever rows you
    pass in — a historical holdout, in practice. The platform scores a *live*
    snapshot, refreshed every 30 minutes, on all of Paris. Two different
    windows, two different quantities: they do not compare, and subtracting
    one from the other is meaningless. Expect several points of gap in either
    direction. Use this number to compare your own models **to each other on
    the same rows**, which is the one thing it is good for.

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
    score = (1 - bs) * 100
    prefix = f"{label} — " if label else ""
    print(f"{prefix}Brier Score : {bs:.4f}  (lower is better)")
    # Deliberately NOT called "leaderboard score": it is the leaderboard's
    # scale computed on your rows, and students read the label, not the docs.
    print(f"{prefix}Score       : {score:.2f} (higher is better — on THIS data, not the leaderboard)")
    return bs
