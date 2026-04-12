# CLAUDE.md — Developer Guide for Claude Code

## Project overview

This repository implements a bias-aware daily OHLCV stock prediction pipeline,
following the experiment blueprint from `deep-research-report.md`.  It is
intentionally self-contained: all experiments run on synthetic data by default,
so no API keys or internet access are required.

## Repository layout

```
stock_prediction/
├── deep-research-report.md   # Research context and design rationale
├── run_pipeline.py           # End-to-end experiment entry point
├── requirements.txt          # Pip dependencies
├── src/
│   ├── data.py               # OHLCV loading + synthetic data generator
│   ├── features.py           # Leakage-safe rolling feature engineering
│   ├── labels.py             # Label creation (direction / return / triple-barrier)
│   ├── evaluation.py         # Walk-forward splits with embargo + purging
│   ├── models.py             # LightGBM walk-forward evaluation
│   └── backtest.py           # Transaction-cost-aware backtest
└── tests/
    ├── test_data.py
    ├── test_features.py
    ├── test_labels.py
    ├── test_evaluation.py
    ├── test_models.py
    └── test_backtest.py
```

## Running the pipeline

```bash
# Synthetic data (no internet needed, reproducible)
python run_pipeline.py

# Live data from Yahoo Finance
python run_pipeline.py --symbol SPY --start 2010-01-01

# Key options
python run_pipeline.py --train-years 5 --test-months 6 --cost-bps 2.0
```

Results are written to `results/` as CSV files:
- `binary_metrics.csv`   — fold-by-fold classification metrics
- `regression_metrics.csv` — fold-by-fold regression metrics
- `ternary_metrics.csv`  — triple-barrier fold metrics
- `equity_curve.csv`     — OOS equity curve vs buy-and-hold

## Running tests

```bash
pip install pytest
python -m pytest tests/ -v
```

All 53 tests must pass before committing.

## Development conventions

### No look-ahead bias
Every rolling feature must use only data available *before* bar `t`.
`pandas.Series.rolling(w)` with default `min_periods=w` is the safe pattern —
it produces NaN until `w` bars of history are available, and the window ends at
bar `t-1` when features are shifted correctly.

### Label vs feature discipline
- **Labels** are forward-looking by design (they describe the future).
- **Features** must never be forward-looking.
- Walk-forward splits enforce temporal ordering: train indices always precede
  test indices, with an embargo gap to guard against autocorrelation leakage.

### Purging for triple-barrier labels
Triple-barrier labels have a variable exit time (`t_end`).  When a label
started in the training period exits *during* the test period, it leaks test
information into training.  Call `evaluation.apply_purge()` to remove such rows
(enabled by `purge=True` in `models.walk_forward_evaluate`).

### Adding a new model
1. Add the model logic in `src/models.py` (or a new `src/models_<name>.py`).
2. Follow the same `walk_forward_evaluate` contract: accept `df`, `feature_cols`,
   `label_col`, `spec`, and return a `pd.DataFrame` of per-fold metrics.
3. Add tests in `tests/test_models.py`.

### Adding a new label type
1. Add a function to `src/labels.py` following the signature pattern.
2. Add corresponding tests to `tests/test_labels.py`.

## Key design decisions

| Decision | Rationale |
|---|---|
| Synthetic data as default | Reproducible, no network dependency, fast CI |
| Walk-forward (not k-fold) | Financial time series must not shuffle time |
| Embargo between folds | Auto-correlation can leak test signal into training |
| Purging for triple-barrier | Variable label horizon can extend into test period |
| LightGBM as primary baseline | Hard to beat on small data; honest yardstick |
| Log returns (not price levels) | Stationary; required for most ML models |

## Environment

- Python 3.11+
- Core dependencies: `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `matplotlib`
- Optional (live data): `yfinance` (requires additional transitive deps; see
  `COLAB_GUIDE.md` for a working install sequence)

---

## Current progress

_Last updated: 2026-04-12_

### Completed

| # | Item | Notes |
|---|---|---|
| 1 | Research survey | `deep-research-report.md` — prioritised shortlist of 11 papers/platforms (2024–2026), compute requirements, and reproducible experiment blueprint |
| 2 | Project scaffold | `src/`, `tests/`, `requirements.txt`, `.gitignore` |
| 3 | Data module (`src/data.py`) | `generate_synthetic_ohlcv()` (GBM, reproducible, offline) + `load_ohlcv_from_yfinance()` with graceful fallback |
| 4 | Feature engineering (`src/features.py`) | 30 leakage-safe rolling features across windows (5, 10, 21, 63): return mean/std/skew, momentum, HL range, OC return, volume change, vol ratio, calendar |
| 5 | Labels (`src/labels.py`) | Next-day direction (binary), k-day forward return (regression), triple-barrier (ternary ±1/0) with exit-time output for purging |
| 6 | Walk-forward evaluation (`src/evaluation.py`) | Rolling and expanding windows, configurable embargo, `apply_purge()` for triple-barrier leakage removal |
| 7 | LightGBM baseline (`src/models.py`) | `walk_forward_evaluate()` for all three task types; fold-by-fold accuracy, MCC, AUC, MAE, RMSE, directional accuracy |
| 8 | Backtest (`src/backtest.py`) | Transaction-cost-aware daily backtest; equity curve, annualised Sharpe, max drawdown, win rate |
| 9 | Pipeline script (`run_pipeline.py`) | End-to-end CLI; all 5 steps; results saved to `results/*.csv` |
| 10 | Test suite (`tests/`) | 53 unit tests, all passing; covers correctness, no-mutation guarantees, OHLC validity, no-future-leak, purge mechanics, metric ranges |
| 11 | Documentation | `CLAUDE.md` (developer guide), `COLAB_GUIDE.md` (Google Colab step-by-step) |
| 12 | Merged to `main` | Feature branch `claude/continue-progress-verification-ipW03` merged and pushed |

### Known limitations / tech debt

- `yfinance` install is fragile on some environments (missing `multitasking` wheel); the pipeline falls back to synthetic data automatically, but live-data tests are not part of CI.
- Triple-barrier label generation is a pure-Python loop — slow on large datasets (>10k bars). A vectorised NumPy/Numba version would speed it up significantly.
- The `results/` directory is git-ignored; there is no automated artefact upload or experiment tracking (e.g. MLflow, W&B).
- `__pycache__/` and compiled `.pyc` files are committed — `.gitignore` covers new ones but the initial merge included them.

---

## Next steps

Priority order follows the experiment sequence recommended in `deep-research-report.md`.

### High priority

| # | Task | Why |
|---|---|---|
| N1 | **Add PatchTST baseline** via NeuralForecast (`neuralforecast.models.PatchTST`) | Provides a deep-learning comparator against LightGBM; directly comparable walk-forward metrics |
| N2 | **Add TimesFM / Chronos-Bolt zero-shot baseline** | "No training" comparator; quantifies how much value the trained models add |
| N3 | **Vectorise triple-barrier label loop** | Current pure-Python loop is the pipeline bottleneck; replace with NumPy roll or Numba JIT |
| N4 | **Fix `__pycache__` in git history** | Remove compiled bytecode from the repo (rewrite history or add to `.gitignore` and delete) |

### Medium priority

| # | Task | Why |
|---|---|---|
| N5 | **CI workflow** (GitHub Actions) | Auto-run `pytest tests/` on every push; enforce the "53 tests must pass" gate without manual checks |
| N6 | **Experiment tracking** | Log fold metrics + hyperparameters to CSV/MLflow so multiple runs can be compared |
| N7 | **Hyperparameter search for LightGBM** | Current params are fixed defaults; even a small grid search could meaningfully improve OOS metrics |
| N8 | **Multi-asset extension** | Run the pipeline on S&P 500 constituents (cross-sectional framing) to get more signal — see SSPT/FinWorld in the research report |

### Low priority / future

| # | Task | Why |
|---|---|---|
| N9 | **EigenCluster tokenisation layer** | Research prototype (ICLR 2026); interesting as a representation learning experiment |
| N10 | **LLM-agent strategy (FINSABER-style)** | Requires API keys; lower ROI than the model-stack improvements above |
| N11 | **Regime labelling** | Annotate bull/bear/sideways regimes and condition model training/evaluation on them |
| N12 | **Portfolio-level backtest** | Extend single-asset backtest to a multi-asset universe with position sizing and turnover constraints |
