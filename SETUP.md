# Environment Setup Guide

This guide walks you through setting up Python 3.12 and the project dependencies  
on macOS, Windows, and Linux. If you have never used a terminal before, read every step carefully.

---

## Why Python 3.12?

The Pastis.ai scoring sandbox runs **Python 3.12** exactly.  
If you use a different version (3.11, 3.13, etc.), your model may work locally but fail when submitted.  
Always use Python 3.12 — no exceptions.

---

## Step 1 — Install Python 3.12

### macOS

**Option A: Homebrew (recommended if Homebrew is already installed)**
```bash
brew install python@3.12
which python3.12    # should show /usr/local/bin/python3.12
python3.12 --version  # should show Python 3.12.x
```

**Option B: Direct download from python.org**
1. Go to [python.org/downloads/release/python-3120/](https://www.python.org/downloads/release/python-3120/)
2. Download the macOS installer (`.pkg` file)
3. Run the installer and follow the prompts
4. Open a new terminal and verify: `python3.12 --version`

### Windows

1. Go to [python.org/downloads/release/python-3120/](https://www.python.org/downloads/release/python-3120/)
2. Download **Windows installer (64-bit)**
3. Run the installer
4. **CHECK "Add Python to PATH"** at the bottom of the first installer screen — this is critical
5. Click "Install Now"
6. Open PowerShell and verify:
   ```powershell
   py -3.12 --version   # should show Python 3.12.x
   ```

### Linux (Ubuntu / Debian)

```bash
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev
python3.12 --version
```

If Python 3.12 is not available in your package manager:
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.12 python3.12-venv
```

---

## Step 2 — Clone the Repository

### macOS / Linux
```bash
git clone https://github.com/pastis-ai/velib-predict-dispo.git
cd velib-predict-dispo
```

### Windows (PowerShell)
```powershell
git clone https://github.com/pastis-ai/velib-predict-dispo.git
cd velib-predict-dispo
```

> If `git` is not installed: download Git from [git-scm.com](https://git-scm.com) (Windows)  
> or run `brew install git` (macOS).

---

## Step 3 — Create a Virtual Environment

A virtual environment isolates your project's packages from the rest of your system.

### macOS / Linux
```bash
/usr/local/bin/python3.12 -m venv .venv
```
After this command, a `.venv/` folder appears in your project directory.

**Activate the virtual environment:**
```bash
source .venv/bin/activate
```
You should see `(.venv)` at the start of your terminal prompt.

### Windows (PowerShell)
```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
```

> **Windows execution policy error?**  
> If you see `cannot be loaded because running scripts is disabled`, run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> Then try `.venv\Scripts\activate` again.

---

## Step 4 — Install Dependencies

With the virtual environment **activated** (you see `(.venv)` in your prompt):

```bash
pip install -r requirements.txt
```

This installs the project dependencies (scikit-learn, xgboost, lightgbm, pandas,
numpy, requests, matplotlib, seaborn, tqdm, cloudpickle).  
It takes 1–3 minutes depending on your connection.

---

## Step 5 — Verify Everything Works

```bash
python --version
# Expected: Python 3.12.x

python -c "import sklearn, pandas, numpy, requests, xgboost, lightgbm; print('All packages OK')"
# Expected: All packages OK
```

---

## Step 6 — Open the Notebook

### Option A: VS Code (recommended)
1. Open VS Code in the project folder: `code .`
2. Install the **Jupyter** extension if prompted
3. Open `starter.ipynb`
4. In the top-right corner, click **Select Kernel** → choose **.venv (Python 3.12)**

### Option B: JupyterLab
```bash
pip install jupyterlab
jupyter lab
```
A browser window will open. Click on `starter.ipynb`.

### Option C: Google Colab (no local setup needed)
Click the **Open in Colab** badge in the README.  
Run the first cell to clone the repo and install dependencies.

---

## Step 7 — Train and Export

Once you have edited `my_model.py`:

```bash
# Train and export submission.pkl
python my_model.py

# Verify submission.pkl is valid before uploading
python my_model.py --verify
```

---

## Troubleshooting

### "python3.12: command not found" (macOS)
```bash
/usr/local/bin/python3.12 --version
brew install python@3.12
```

### "py -3.12 is not recognized" (Windows)
Make sure Python 3.12 was installed from [python.org](https://python.org) with "Add to PATH" checked.  
Open a new PowerShell window after installation.

### "ModuleNotFoundError" after pip install
Your virtual environment may not be activated.  
You should see `(.venv)` in your terminal prompt.  
If not, run `source .venv/bin/activate` (macOS/Linux) or `.venv\Scripts\activate` (Windows).

### Conda / Anaconda users
```bash
conda deactivate
source .venv/bin/activate
```

---

## Downloading Data from the API

### Use the included dataset (no internet needed)
The file `velib_dataset_dev.csv` contains real Vélib data for April 14–30, 2026,  
arrondissements 18e, 19e, and 20e — ~42,000 rows, 67 stations.

This dataset has **real variance**: stations that empty during rush hours, genuine weekday vs  
weekend patterns, and a mean fill rate of ~21%. Use this for all local development.

### Download more data via API

**Discover what is available:**
```python
import requests

# List all arrondissements
r = requests.get("https://pastis.ai/api/scenarios/velib-predict-dispo/arrondissements")
print(r.json())

# Search stations by name
r = requests.get(
    "https://pastis.ai/api/scenarios/velib-predict-dispo/stations",
    params={"search": "bastille", "limit": 8}
)
print(r.json()["stations"])

# Check available date range and total rows
r = requests.get("https://pastis.ai/api/scenarios/velib-predict-dispo/stats")
stats = r.json()
print(f"From {stats['first_snapshot']} to {stats['last_snapshot']}")
```

**Download a filtered dataset:**
```python
import requests, pandas as pd

params = {
    "date_from": "2026-01-01",
    "date_to": "2026-03-31",
    "arrondissements": [
        "Paris 18e", "Paris 19e", "Paris 20e",
        "Paris 13e Arrondissement", "Paris 15e Arrondissement",
    ],
}
with requests.get(
    "https://pastis.ai/api/scenarios/velib-predict-dispo/download",
    params=params,
    stream=True,
    timeout=300,
) as r:
    r.raise_for_status()
    with open("velib_large.csv", "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

df = pd.read_csv("velib_large.csv", parse_dates=["snapshot_at"])
print(f"Downloaded: {len(df):,} rows")
```

**Volume reference:**

| Scope | Approx. rows | Approx. size |
|-------|-------------|-------------|
| 1 station × 30 days | 1,440 | 154 KB |
| 10 stations × 30 days | 14,400 | 1.5 MB |
| 1 arrondissement × 30 days | ~50,000 | ~5 MB |
| All stations × 1 month | ~2.2M | ~233 MB |

Use `stream=True` and `timeout=300` for any download larger than a few MB.
