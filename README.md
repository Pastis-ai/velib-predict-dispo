# Pastis.ai — Vélib Fill Rate Prediction Challenge

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pastis-ai/velib-predict-dispo/blob/main/starter.ipynb)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)

**Predict the fill rate of Paris Vélib stations at a given time.**

---

## The Challenge

Paris has over 1,400 Vélib stations. At any given moment, some are full, some are empty, and many are in between. Your task: given a station and the time of day, predict **taux_remplissage** — the fraction of capacity occupied by available bikes.

$$\text{taux\_remplissage} = \frac{\text{numbikesavailable}}{\text{capacity}} \in [0.0,\; 1.0]$$

This is a **regression** problem. Your model must output a value between 0 and 1.

---

## Your learning path

The kit is a journey — you do **not** need everything on day 1:

| Stage | Where | What you do |
|-------|-------|-------------|
| **1. First submission** | [starter.ipynb](starter.ipynb), Sections 0–7 | Understand the data, train the starter model, submit — day 1 goal |
| **2. Iterate** | Going Further, Levels 1–3 | Feature engineering, better model, tuning — where the points are won |
| **3. Compare & choose** | Bonus section (end of notebook) | Audit your candidate models side by side, pick a champion |
| **4. Scale up** | Level 4 | More months, more arrondissements from the API |
| **5. Expert** | Level 5 | External / auxiliary data (contract v2) |

Feature engineering, model comparison and external data all come **later** in
the journey — the notebook tells you when. Day 1 is about getting a valid
`submission.pkl` on the leaderboard.

---

## Quickstart — Google Colab (no setup required)

1. Click the **Open in Colab** badge above
2. Run the first cell to clone the repo and install dependencies
3. Run all remaining cells (`Runtime > Run all`)
4. Download `submission.pkl` and upload it on the [submission page](https://pastis.ai/scenarios/velib-predict-dispo?tab=submit)

---

## Quickstart — Local Setup

### Prerequisites
**Python 3.12 is required** — the same version used by the Pastis.ai scoring sandbox.  
Do NOT use 3.11 or 3.13.

### macOS / Linux

```bash
git clone https://github.com/pastis-ai/velib-predict-dispo.git
cd velib-predict-dispo

/usr/local/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Open the notebook
code .                          # VS Code: open starter.ipynb in Explorer
jupyter notebook starter.ipynb  # Terminal / browser
```

### Windows (PowerShell)

```powershell
git clone https://github.com/pastis-ai/velib-predict-dispo.git
cd velib-predict-dispo

py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

code .
```

### Verify your setup

```bash
python --version        # must show Python 3.12.x
python -c "import sklearn, pandas, numpy; print('All good!')"
```

> See [SETUP.md](SETUP.md) for a detailed guide including troubleshooting.

---

## Repository Structure

| File | Description |
|------|-------------|
| [starter.ipynb](starter.ipynb) | Main notebook — start here |
| [my_model.py](my_model.py) | Your submission file — improve this |
| [pastis_velib.py](pastis_velib.py) | Base library — do not modify |
| [requirements.txt](requirements.txt) | Python dependencies (frozen versions) |
| [velib_dataset_dev.csv](velib_dataset_dev.csv) | Dev dataset — April 2026, arrondissements 18e/19e/20e |
| [SETUP.md](SETUP.md) | Detailed environment setup guide |

---

## About the dev dataset

The included CSV (`velib_dataset_dev.csv`) covers **April 14–30, 2026** in the **18th, 19th, and 20th arrondissements** of Paris.

| Fact | Value |
|------|-------|
| Rows | ~42,000 |
| Stations | 67 |
| Mean fill rate | ~21% |
| Target variance | Real — well-spread across [0, 1] |

Unlike a sample from the 1st arrondissement (where stations are almost never empty), these outer arrondissements have **genuine variance**: stations that drain completely during rush hours, different weekday vs weekend patterns, and real prediction challenges.

### Three data surfaces, and they are not meant to match

| | What it is | Should it look like the scoring snapshot? |
|---|---|---|
| **1. `velib_dataset_dev.csv`** | An **exploration** fixture: small, fixed, fully known, so the notebook's commentary and charts stay true | **No** |
| **2. `/download` API** | The real **training** set: complete, current, 24 h embargo | Yes, as far as possible |
| **3. Live snapshot** | The 12 served columns, at scoring time | It *is* the reference |

The contract that has to hold is **2 ↔ 3**. The CSV is there to be *looked at*.

> **The CSV is for exploring — it contains everything, answers included. Your
> model will only ever receive the 12 columns of `as_serve_frame()`. Explore on
> the CSV, train on the API.**

Both surfaces carry the same twelve columns: seven trainable features plus the
five answers. Five *other* columns
(`nom_arrondissement_communes_raw`, `code_insee_commune`,
`station_opening_hours`, `is_installed`, `is_returning`) are served at scoring
and exist in neither — a model trained on the seven receives twelve and ignores
the extra five, with no error. Read them with `df.get(...)` if you want them.

---

## The Contract

Your class `MyModel` must inherit from `PastisBaseModel` and implement:

```python
def fit(self, df_raw: pd.DataFrame) -> None:
    # Train your pipeline on the raw DataFrame
    ...

def predict(self, df_raw: pd.DataFrame) -> np.ndarray:
    # Return taux_remplissage for each row
    # REQUIRED: np.ndarray of float in [0.0, 1.0]
    ...
```

The Pastis sandbox calls exactly these two methods — do not change their signatures.

> **Contract v2 (optional):** if the scenario declares auxiliary data sources
> (weather, geography, ...), your `predict()` can opt in to receive them by
> declaring an explicit `extra_df` parameter — see *Going Further → Level 5*
> in [starter.ipynb](starter.ipynb). Contract v1 above stays fully valid;
> no source is declared for this scenario today.

### What `predict()` actually receives

`fit()` and `predict()` do **not** see the same columns, and this is the single
most important thing to understand before you engineer a feature.

Your training CSV has everything. The live snapshot your model is scored on is
passed through a **whitelist** first — these twelve columns, and nothing else:

| Column | Also in the training data? |
|--------|----------------------------|
| `stationcode` | ✅ |
| `name` | ✅ |
| `capacity` | ✅ |
| `snapshot_at` | ✅ |
| `coordonnees_geo` | ✅ |
| `nom_arrondissement_communes` | ✅ |
| `is_renting` | ✅ |
| `nom_arrondissement_communes_raw` | ❌ scoring only |
| `code_insee_commune` | ❌ scoring only |
| `station_opening_hours` | ❌ scoring only |
| `is_installed` | ❌ scoring only |
| `is_returning` | ❌ scoring only |

The five marked *scoring only* are served to your model but absent from **both**
the dev CSV and the `/download` API, so you cannot train on them at all today —
read them defensively (`df.get("is_returning")`) if you use them. A model
trained on the seven available features simply ignores the extra five it is
handed at scoring: no error, nothing to fix.

### Target leakage — the five columns, and why they have a name

```
numbikesavailable · numdocksavailable · mechanical · ebike · has_bike
```

These are in your training data — you need `numbikesavailable` to build the
target, and there is no supervised learning without the answer — and they are
**stripped from the snapshot before `predict()` runs**. Each one is the answer
in a different disguise:

- `numbikesavailable` — the target, literally
- `mechanical + ebike` — sums to `numbikesavailable` exactly
- `numdocksavailable` — `capacity − numdocksavailable` is the target, near-exactly
- `has_bike` — `target > 0`

Putting one of these in your `X` has a name: **target leakage**. And
`numdocksavailable` is the case worth remembering, because it does not look
like cheating — it looks like a perfectly reasonable feature. It is
`capacity − answer`. Train on it, get an R² of 0.99 on your laptop, and watch
the leaderboard score collapse without understanding why.

So these columns are not forbidden in the abstract. They are forbidden **at
serve time** — because at the moment the prediction actually matters, nobody
knows how many bikes are at the station. That is the entire point of the
scenario.

Every upload is probed: the platform runs your model twice on two frames
identical over the twelve served columns and **opposite** over these five. If
your predictions move on more than 5% of rows, the submission is refused. The
probe is simply the question *"does your model still say the same thing when I
lie to it about the answer?"*

### Four more ways a submission gets refused

| Check | Rule |
|-------|------|
| **Determinism** | Two runs on the same input must agree to `1e-9`. Set `random_state=` / `np.random.seed()` everywhere it exists. |
| **Time** | `predict()` must finish in **60 s on a full snapshot** (~1500 rows). That is the budget of *every* scoring cycle, not a one-off allowance. |
| **File size** | 50 MB maximum. Empty or truncated files are refused. Store statistics in your model, not the training frame. |
| **Identical resubmission** | Uploading a file **byte-identical** to one you already submitted is refused (409), naming your previous submission and its date. Since every export produces a new file, this means you picked up an old `.pkl` instead of the one you just wrote. |

> **On models that resemble each other.** Two submissions with identical
> predictions have several possible explanations — a shared tutorial, a public
> model, a pair working together, a copy. Because that fact is *interpretable*
> in more than one way, it is never grounds for an automatic refusal: it is
> flagged to a teacher, and your model is never blocked or penalised for
> resembling someone else's. The four checks above are refusals precisely
> because they admit only one reading.

Two different fingerprints exist, and they answer two different questions:

| Fingerprint | What it covers | What it actually catches |
|---|---|---|
| **`sha256` of the file** | the **bytes** | provenance — which exact artefact was submitted, when, by whom. And an accidental re-upload of the same file (409) |
| **Prediction fingerprint** | the **behaviour** | similarity between models, re-exported or not. It only ever flags |

The file hash is a record, not a filter: re-exporting a model changes its
`sha256` without changing a line of code, so no `.pkl` is ever blocked on its
hash.

The sandbox also runs with **no network access** (`--network none`). Any
`requests` / `urllib` call inside `predict()` works on your laptop and fails at
scoring. External data must arrive through `extra_df` (contract v2) or not at all.

### Check all of this before you upload

`python my_model.py` ends with a **preflight** that replays the platform's own
checks on your machine:

```
Preflight — replaying the platform's submission checks:
  ✓  determinism: two runs on the same input are identical.
  ✓  reads the answer: predictions are stable (0.0% of rows moved, limit 5%).
  ✓  time budget: 0.04s on 1500 rows (budget 60s).
  → Ready to upload.
```

You can also call it directly on any model, trained or reloaded:

```python
from pastis_velib import preflight_check
preflight_check(model, df)
```

A rejection you find here costs a minute. The same rejection found on the
submission page costs a session.

---

## Metric: Brier Score

$$BS = \frac{1}{n} \sum_{i=1}^{n} (p_i - y_i)^2$$

- $p_i$ = predicted fill rate, $y_i$ = true fill rate
- **Lower Brier Score = better model**
- **Leaderboard score = (1 − BS) × 100** — higher is better

Here the Brier Score is simply the mean squared error on the continuous fill rate — **not** `sklearn.metrics.brier_score_loss`, which is reserved for binary-classification probabilities. Your model outputs a rate in [0, 1], so it is scored directly against the true rate.

| BS | Score | Interpretation |
|----|-------|----------------|
| 0.037 | 96.3 | Baseline (always predict mean fill rate) on the dev dataset |
| 0.023 | 97.7 | This starter, feature-engineered (`HistGradientBoostingRegressor`) |
| 0.010 | 99.0 | Strong model |
| < 0.005 | > 99.5 | Suspiciously good — verify there is no data leakage |

The Brier Score is bounded by the variance of the target (~0.037 here), so on this
small dataset every model is compressed into the 96–100 range. The competition lives
in that 1–4 point gap. Download more data from the API and the variance — and the
competitive range — grows.

> **The table above is measured on the dev dataset; the leaderboard is not.** Scoring
> runs on all of Paris, where the fill rate varies much more (variance ≈ 0.08 rather
> than 0.037) — the same naive baseline lands around **91** there, not 96.3. Neither
> number is wrong; they are measured on different data.
>
> Two habits follow. Compare your model to the baseline **on the same dataset** —
> a local score and a leaderboard score are different quantities, not a before/after.
> And expect your leaderboard score to move between runs even when your model does
> not: each run is scored on a fresh snapshot, so read it as a range, not a value.

To make the two scales concrete — measured on the live leaderboard on **2026-08-14**:

| On the live leaderboard | Score |
|---|---|
| Naive baseline (predict the mean) | ~91 |
| Statistical reference models | ~95.6 |
| Best student model observed | 94.95 |

So the 97.7 in the table above, measured on a historical holdout, is **not** a
prediction of where you will land. If your local number is higher than
everything in this list, that is expected — it is a different measurement, not a
better model. Only the leaderboard counts.

**Your goal: beat the baseline. Aim for a meaningful improvement through feature engineering.**

---

## Data API

Download Vélib data directly from the Pastis API — no authentication required.

```python
import requests, pandas as pd

params = {
    "date_from": "2026-04-14",
    "date_to": "2026-04-30",
    "arrondissements": ["Paris 18e", "Paris 19e", "Paris 20e"],
}
with requests.get(
    "https://pastis.ai/api/scenarios/velib-predict-dispo/download",
    params=params, stream=True, timeout=300
) as r:
    r.raise_for_status()
    with open("velib.csv", "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

df = pd.read_csv("velib.csv", parse_dates=["snapshot_at"])
```

See [SETUP.md](SETUP.md) for all endpoints, filtering options, and volume estimates.

---

## Compare your models before submitting (optional, advanced)

Once you have iterated and hold **several candidate models**, the
**Bonus — Pick Your Champion** section at the end of
[starter.ipynb](starter.ipynb) shows how to audit and compare them side by
side with [skore](https://skore.probabl.ai/) — an open-source model-evaluation
library by Probabl (founded by scikit-learn core developers) — and pick the
one worth submitting.

- **Optional and advanced** — nothing on day 1 requires it; come back after
  Going Further Levels 1–3, when you actually have models to compare.
- **Local only** — `pip install skore==0.23.0 ipywidgets` (pinned; verified to
  leave the frozen `requirements.txt` versions untouched). No account, no
  network calls after install; works locally and on Colab.
- **Zero impact on the submission contract** — the `.pkl` you upload is still
  produced by `export_submission()`; skore never ships inside it.

---

## Submit

1. Train your model: `python my_model.py` — it exports, verifies **and** preflights
2. Re-check an existing file at any time: `python my_model.py --verify`
3. Only upload once the preflight prints `→ Ready to upload`
4. Go to the [submission page](https://pastis.ai/scenarios/velib-predict-dispo?tab=submit) and upload `submission.pkl`
5. Check the leaderboard!

> On the leaderboard you appear under a stable botanical pseudonym by default —
> never your real name unless you choose to reveal it.
