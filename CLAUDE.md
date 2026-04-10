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
