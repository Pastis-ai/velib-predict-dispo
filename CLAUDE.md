# CLAUDE.md — velib-predict-dispo (v3)

## Project
Public ML starter kit — pastis-ai/velib-predict-dispo
Platform: Pastis.ai (https://pastis.ai)
Audience: Master / Grande École level students (mixed ML experience)

## Use case
Predict `taux_remplissage = numbikesavailable / capacity` for a given station
at a given time. Float in [0.0, 1.0]. Regression problem framed as
probabilistic output.

## Target
- `taux_remplissage = numbikesavailable / capacity` — float in [0.0, 1.0]
- `has_bike` is deprecated as target — trivially imbalanced in the 1st arrondissement
- Ground truth for scoring: taux_remplissage from fresh Vélib data

## Metric
- Brier Score = MSE(y_true, y_pred) where both are in [0.0, 1.0]
- Leaderboard score = (1 − BS) × 100 — higher is better
- Baseline (always predict mean taux_remplissage from train set) → ~75–80 pts

## predict() contract — IMMUTABLE
- Input: raw DataFrame with AVAILABLE_FEATURES columns
- Output: np.ndarray of float in [0.0, 1.0], shape (n_rows,)
- Represents: predicted fill rate (taux_remplissage)
- Sandbox enforces this contract strictly

## Environment
- Python 3.12 mandatory — matches Pastis.ai backend and scoring sandbox exactly
- macOS: `/usr/local/bin/python3.12 -m venv .venv`
- Windows: `py -3.12 -m venv .venv` (requires Python 3.12 from python.org)
- Activation macOS/Linux: `source .venv/bin/activate`
- Activation Windows: `.venv\Scripts\activate`

## Language rules — ABSOLUTE
- Everything in English. No exceptions.
- Notebook markdown cells: English
- Code comments: English
- Docstrings: English
- README.md: English
- SETUP.md: English
- CLAUDE.md: English
- Do NOT use French anywhere in committed files
- Variable names: English or standard math notation

## Git rules — STRICT

### Committed files (exhaustive list — nothing else)
```
CLAUDE.md
README.md
SETUP.md
requirements.txt
pastis_velib.py
my_model.py
starter.ipynb
velib_dataset_dev.csv
```

### Never commit
- `.gitignore` — stays LOCAL ONLY, never commit it
- `velib_dataset_04_2026_1st_dist.csv` — replaced by velib_dataset_dev.csv, never commit
- `download_dev_dataset.py` — local dev helper, not part of the student kit
- `.venv/` — Python virtual environment
- `__pycache__/` — Python bytecode cache
- `*.pkl` — trained model files (the submission deliverable stays local)
- `.DS_Store` — macOS metadata
- `.env`, `.env.local` — environment variables
- `*.ipynb_checkpoints/` — Jupyter notebook checkpoints
- Any file not in the "Committed files" list above

### CRITICAL: CLAUDE.md must always be committed
- CLAUDE.md must ALWAYS be in the committed files list
- CLAUDE.md must NEVER appear in .gitignore under any circumstances
- When in doubt about whether to commit a file, DO NOT commit it

### .gitignore rules
- .gitignore is a LOCAL-ONLY development aid — never commit it to git
- When updating .gitignore, never add patterns that would match committed files
- Patterns to NEVER put in .gitignore:
  `CLAUDE.md`, `README.md`, `SETUP.md`, `requirements.txt`,
  `pastis_velib.py`, `my_model.py`, `starter.ipynb`, `velib_dataset_dev.csv`
- Adding `.gitignore` itself to .gitignore prevents accidental commits via `git add .`

## Coding rules
- `predict()` contract is IMMUTABLE: must return `np.array` of float in [0.0, 1.0]
- `fit()` always receives a raw DataFrame with official columns
- `requirements.txt` versions are frozen — never change them (sandbox compatibility)
- Temporal train/test split ONLY — never random split on time series data
- sklearn ONLY for models in the baseline and my_model.py default
- xgboost and lightgbm are in requirements.txt but only suggested in "Going Further"
  sections — NOT used in default MyModel

## Architecture
- `pastis_velib.py`: base class + helpers — imported by sandbox AND students
- `my_model.py`: student-facing file — clean, structured, well-commented
- `starter.ipynb`: pedagogical notebook — tells the data story, step-by-step
- `velib_dataset_dev.csv`: real data, 18e/19e/20e, April 14–30 2026
- `submission.pkl`: real deliverable — serialized trained pipeline (local only)

## Data sources

### Dev dataset (committed to repo)
`velib_dataset_dev.csv` — April 2026 (14th–30th), arrondissements 18e/19e/20e
~42k rows, 67 stations. Mean fill rate ~21%. Variance is real.
Use this for local development and testing.

### API — download endpoint
```
GET https://pastis.ai/api/scenarios/velib-predict-dispo/download
Params: date_from, date_to, arrondissements (repeatable), stations (repeatable)
No auth required. Returns streamed CSV. Timeout: 300s.
```

### API — utility endpoints
```
GET /api/scenarios/velib-predict-dispo/arrondissements → list of arrondissements
GET /api/scenarios/velib-predict-dispo/stations?search=bastille&limit=8 → station search
GET /api/scenarios/velib-predict-dispo/stats → dataset stats
```

### Volume estimation
- 1 station × 30 days ≈ 1,440 rows ≈ 154 KB
- 10 stations × 30 days ≈ 14,400 rows ≈ 1.5 MB
- All stations × 1 month ≈ 2.2M rows ≈ 233 MB → use `stream=True` + `timeout=300`

## Features contract v1 (immutable — matches velib_dataset_dev.csv)
```
stationcode, name, snapshot_at, capacity, numbikesavailable,
mechanical, ebike, numdocksavailable, is_renting,
nom_arrondissement_communes, coordonnees_geo, has_bike
```

## Column notes
- `nom_arrondissement_communes`: PLURAL (not `nom_arrondissement_commune`)
- `coordonnees_geo`: dict-like string `{"lon": ..., "lat": ...}`
- `has_bike`: int (0/1) — present in CSV but NOT the prediction target
- `is_installed`: NOT present in real data — do not reference it
- `taux_remplissage`: NOT in raw data — computed as `numbikesavailable / capacity`

## Extra features architecture (future-proof)
`make_features(df_raw, extra_df=None)`
`extra_df` is reserved for external data (weather, events...).
Currently ignored. Do not implement now — just keep the signature.
