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

> Use the dev CSV to develop your pipeline. Download more data from the API to improve your leaderboard score.

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

---

## Metric: Brier Score

$$BS = \frac{1}{n} \sum_{i=1}^{n} (p_i - y_i)^2$$

- $p_i$ = predicted fill rate, $y_i$ = true fill rate
- **Lower Brier Score = better model**
- **Leaderboard score = (1 − BS) × 100** — higher is better

The Brier Score works identically for regression on [0, 1] and for binary classification probabilities.

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

## Submit

1. Train your model: `python my_model.py`
2. Verify the output: `python my_model.py --verify`
3. Go to the [submission page](https://pastis.ai/scenarios/velib-predict-dispo?tab=submit) and upload `submission.pkl`
4. Check the leaderboard!

> On the leaderboard you appear under a stable botanical pseudonym by default —
> never your real name unless you choose to reveal it.
