# Trustworthy Stock Forecaster

**A Probabilistic Forecasting & Calibration Toolkit**

AAI/CPE/EE 551 — Engineering Python | Course Project (Summer 2026) — Team 2

---

## Team members

| Name | Email | Stevens ID (CWID) |
|---|---|---|
| Olin Dsouza | odsouza1@stevens.edu | 20033076 |
| Yogesh Patil | yp36032@stevens.edu | 20032007 |
| Kevin Gwinn | kgwinn@stevens.edu | 20017424 |

---

## Project description

### Overview

Most stock-prediction tools output a single predicted value and say nothing about how much
that value can be trusted. A forecast without a measure of uncertainty is unsafe to act on,
and models that look reliable in calm markets frequently become **over-confident during
volatile periods**. This project treats the *trustworthiness* of a forecast as the core
engineering problem.

The toolkit produces **probabilistic forecasts** of daily stock returns — predictions
accompanied by uncertainty intervals rather than point estimates — and then rigorously
measures how **well-calibrated** those intervals are, separately for **calm** and
**volatile** market regimes. The result lets a user see when a model's stated confidence
can actually be believed.

**Headline result** (from the committed dataset): the random-walk model's "90%" prediction
interval covers the realized return on only **77.9%** of volatile days (over-confidence),
while the adaptive EWMA model shrinks that calm-vs-volatile trust gap from 15.4 to 6.1
percentage points and achieves a lower CRPS — its confidence statements are measurably
more trustworthy.

### Functionalities

1. **Load and clean** historical daily price data from CSV files, with full validation
   (`data_loader.py`).
2. **Update data** (optional utility): fetch fresh daily data for any symbol from the
   Alpha Vantage API (`data_updater.py`) — the main program never needs the network.
3. **Compute log-returns** and classify each trading day as *calm* or *volatile* using
   rolling 20-day volatility against a 70th-percentile threshold (`data_loader.py`).
4. **Organize assets** into a `Portfolio` composed of `StockAssets` objects with
   user-friendly printing, sizing, merging, and comparison (`portfolio.py`).
5. **Generate probabilistic forecasts** with two classical models that output full
   predictive distributions: a random-walk baseline and a RiskMetrics EWMA volatility
   model (`forecasters.py`).
6. **Run a rolling-window backtest** (train 1 year, test 1 month, slide monthly) that
   re-fits every model exactly as it would be used in real time (`backtest.py`).
7. **Evaluate calibration** — CRPS and empirical interval coverage — **per regime**, and
   draw **reliability diagrams** with matplotlib (`calibration.py`).
8. **Write results to disk**: a tidy metrics CSV and PNG figures (`output/`).

### Data source

Daily OHLCV price data retrieved from the **Alpha Vantage** free daily time-series API
(<https://www.alphavantage.co/documentation/#daily>) using our own `data_updater.py`
utility, and committed to the repository as three sample CSVs
(`data/aapl_sample.csv`, `data/msft_sample.csv`, `data/jpm_sample.csv`, ~1,000 trading
days each). The data is committed so the whole analysis is **fully reproducible offline**
— the grader never needs an API key or a network connection.

---

## Project file / module structure

```
trustworthy-stock-forecasting/
├── main.ipynb          # MAIN PROGRAM (Jupyter Notebook) — imports the modules
│                       #   and runs the full pipeline end to end
├── data_loader.py      # Module: CSV loading/cleaning, log-returns, rolling
│                       #   volatility, calm/volatile regime labeling
├── data_updater.py     # Module (optional utility): fetch/update symbol CSVs
│                       #   from the Alpha Vantage API; cleanup/remove helpers
├── portfolio.py        # Module: StockAssets + Portfolio (composition) with the
│                       #   __str__ / __len__ / __add__ / __eq__ operator overloads
├── forecasters.py      # Module: Forecaster base class + RandomWalkForecaster and
│                       #   EWMAForecaster subclasses (inheritance), @require_fitted decorator
├── calibration.py      # Module: CRPS, empirical coverage, reliability diagrams
├── backtest.py         # Module: rolling-window split generator, backtest driver,
│                       #   per-regime summaries, CSV export helper
├── test_toolkit.py     # Pytest suite — 13 test cases
├── data/               # Committed dataset (see Data source above)
│   ├── aapl_sample.csv  ├── msft_sample.csv  └── jpm_sample.csv
├── output/             # Generated at run time: metrics CSV + PNG figures
├── requirements.txt    # Python dependencies
├── .gitignore
└── README.md           # This file
```

Every `.py` module was authored in VS Code, has a module header + docstrings, and carries
an `if __name__ == "__main__":` guard so it can be smoke-tested standalone.

---

## Dependencies, environment & installation

* **Python 3.12, 3.13, or 3.14** (developed and tested on 3.13).
* Libraries (see `requirements.txt`): **NumPy**, **Pandas**, **matplotlib**, **pytest**,
  **notebook**, **ipykernel**.
* **Jupyter kernel:** the notebook runs on the standard **`Python 3 (ipykernel)`** kernel,
  registered from the virtual environment created below.

### Install on macOS (Terminal)

```bash
# 1. Check the Python version (must print 3.12.x, 3.13.x, or 3.14.x)
python3 --version

# 2. Clone the repository and enter it
git clone https://github.com/yogesh117/trustworthy-stock-forecasting.git
cd trustworthy-stock-forecasting

# 3. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 4. Install all dependencies
pip install -r requirements.txt
```

### Install on Windows (PowerShell)

```powershell
# 1. Check the Python version (must print 3.12.x, 3.13.x, or 3.14.x)
py --version

# 2. Clone the repository and enter it
git clone https://github.com/yogesh117/trustworthy-stock-forecasting.git
cd trustworthy-stock-forecasting

# 3. Create and activate a virtual environment
py -m venv .venv
.venv\Scripts\Activate.ps1

# 4. Install all dependencies
pip install -r requirements.txt
```

> If PowerShell blocks activation, run
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once, then retry
> step 3.

---

## How to run the program — step by step

All commands are executed **from the repository root** (the folder containing
`main.ipynb`) with the virtual environment **activated** (you should see `(.venv)` in the
prompt).

### Step 1 — Run the test suite (optional but recommended)

```bash
pytest -v
```

Expected result: **13 passed**.

### Step 2 — Run the main program (Jupyter Notebook)

**Option A — Jupyter in the browser (macOS and Windows):**

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```
2. A browser tab opens showing the repository folder. Click **`main.ipynb`**.
3. Confirm the kernel in the top-right corner says **`Python 3 (ipykernel)`**. If it does
   not, choose *Kernel → Change Kernel → Python 3 (ipykernel)*.
4. Run every cell in order: menu *Run → Run All Cells*.
5. The notebook prints the demonstration output cell by cell and displays all figures
   inline. Total runtime is well under one minute.

**Option B — VS Code:**

1. Open the repository folder in VS Code (*File → Open Folder*).
2. Install the Microsoft **Python** and **Jupyter** extensions if prompted.
3. Open `main.ipynb`, click **Select Kernel** (top right) and pick the interpreter at
   `.venv/bin/python` (macOS) or `.venv\Scripts\python.exe` (Windows).
4. Click **Run All**.

### Step 3 — Inspect the generated results

After the run, the `output/` folder contains:

| File | Contents |
|---|---|
| `metrics_summary.csv` | Per-model, per-regime CRPS and coverage at 6 confidence levels |
| `prices_with_regimes.png` | Close prices of all 3 symbols with volatile regimes shaded |
| `reliability_RandomWalk.png` | Reliability diagram, random-walk model |
| `reliability_EWMA_lambda0.94.png` | Reliability diagram, EWMA model |

### (Optional) Smoke-test any module standalone

Each module runs a small self-demonstration under its `__main__` guard, e.g.:

```bash
python data_loader.py
```

(`data_updater.py`'s standalone demo calls the live Alpha Vantage API and needs an
`ALPHAVANTAGE_API_KEY` environment variable — it is an optional utility and **not** part
of the main program flow.)

---

## How the requirements are met

### Part 1 — all 10 components

| # | Requirement | Where / how it is met |
|---|---|---|
| 1 | ≥2 meaningful classes with a relationship | **Inheritance:** `Forecaster` → `RandomWalkForecaster`, `EWMAForecaster` (`forecasters.py`). **Composition:** `Portfolio` is composed of `StockAssets` objects (`portfolio.py`). All have constructors, attributes, methods, and instantiated objects in `main.ipynb`. |
| 2 | ≥2 meaningful functions | `label_regime()` (`data_loader.py`), `compute_calibration()` (`calibration.py`) — plus `crps_normal()`, `empirical_coverage()`, `run_backtest()`, and more. |
| 3 | ≥2 advanced libraries, used critically | **NumPy** (vectorized returns/volatility/CRPS math), **Pandas** (CSV loading, time-series alignment, rolling windows, result tables), **matplotlib** (regime plots, reliability diagrams). |
| 4 | ≥2 exception scenarios + ≥2 pytest cases | `FileNotFoundError` (missing CSV), `ValueError` (insufficient/invalid data), `TypeError` (wrong asset type), `RuntimeError` (unfitted model), `KeyError` (unknown symbol) — raised in the modules and caught/demonstrated in `main.ipynb` §1 and §3. **13 pytest cases** in `test_toolkit.py`. |
| 5 | Meaningful data I/O | Reads 3 CSV price files from `data/` (optionally fetches fresh data via the Alpha Vantage API in `data_updater.py`); writes `output/metrics_summary.csv` and 3 PNG figures. |
| 6 | ≥2 loops and ≥2 if statements | Rolling-window backtest loop + per-model inner loop (`backtest.py`); per-symbol loops in `main.ipynb`; conditionals in regime labeling, interval containment, and input validation throughout. |
| 7 | ≥2 mutable + ≥2 immutable types | Mutable: `list` (records), `dict` (reports, `Portfolio._assets`), `set` (`REGIME_LABELS`, regime sets in `summarize_by_regime`). Immutable: `float`/`int` (prices, counts), `str` (symbols, labels), `tuple` (intervals, `DEFAULT_LEVELS`, `config()`). |
| 8 | `__str__()` + ≥1 more operator overload | `__str__` on `Forecaster`, `StockAssets`, `Portfolio`; plus `__len__` (assets/observations), `__add__` (portfolio merge), `__eq__` (forecaster-config and portfolio comparison). |
| 9 | Docstrings, headers, comments | Every module has a header block; every class and function has a docstring immediately below its definition; meaningful inline comments throughout. |
| 10 | `__name__` | Every module ends with an `if __name__ == "__main__":` standalone demo; `main.ipynb` imports the modules. |

### Part 2 — components included (4 required, 6 provided)

The **four** committed components:

1. **Comprehension** — list/dict comprehensions aggregate per-regime metrics
   (`compute_calibration()`, `summarize_by_regime()`, `Portfolio.summary()`, `main.ipynb` §2).
2. **Built-in library/module** — `statistics` (median CRPS), `math` (erf/sqrt/pi),
   `pathlib`, `urllib.request`, `os`, `functools`.
3. **Generator function** — `rolling_window_splits()` in `backtest.py` *yields* successive
   train/test windows that drive the whole backtest.
4. **Special function** — `map` + `lambda` in `_standard_normal_cdf()` (`calibration.py`)
   and in `label_regime()` (`data_loader.py`); `zip` in `empirical_coverage()` and the
   notebook's plotting loop; `min(..., key=lambda ...)` in `main.ipynb` §5.

Bonus components beyond the required four:

5. **Set operations** — regime-label validation via intersection (`&`) and difference (`-`)
   in `summarize_by_regime()` (`backtest.py`).
6. **Decorator + closure** — `@require_fitted` in `forecasters.py` wraps prediction methods
   and closes over the decorated method; `@classmethod` alternative constructors in
   `portfolio.py`.

### Grading rubric mapping

* **Program runs correctly without errors** — `main.ipynb` executes top to bottom (the
  committed notebook contains the outputs of a full clean run); `pytest -v` → 13 passed.
* **Code structure** — logic is separated into six single-purpose modules plus a thin
  orchestration notebook.
* **Naming & style** — snake_case functions/variables, CapWords classes, UPPER_CASE
  constants, descriptive names throughout.
* **Docstrings / headers / comments** — on every module, class, and function.
* **README** — this file: problem, solution approach, structure, installation, and exact
  run instructions for macOS and Windows.

---

## Main contributions of each team member

| Member | Main contributions |
|---|---|
| **Kevin Gwinn** | Data pipeline (`data_loader.py`): CSV loading/cleaning, log-returns, rolling volatility, regime labeling; data acquisition utility (`data_updater.py`); `StockAssets`/`Portfolio` core (`portfolio.py`); sample dataset preparation. |
| **Yogesh Patil** | Forecasting models (`forecasters.py`): `Forecaster` base class, random-walk and EWMA subclasses, `@require_fitted` decorator; calibration metrics (`calibration.py`): CRPS, coverage, reliability diagrams; backtesting engine (`backtest.py`); repository setup. |
| **Olin Dsouza** | Main program (`main.ipynb`): full pipeline integration and analysis; test suite (`test_toolkit.py`, 13 cases); portfolio operator overloads (`__add__`, `__eq__`); documentation (`README.md`), dependency management (`requirements.txt`, `.gitignore`). |

Each member made ≥5 meaningful commits to the shared repository (program logic, design,
debugging, testing, data handling, and documentation).
