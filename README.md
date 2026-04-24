# Stock Prediction — Colab-Ready Baseline

A bias-aware daily OHLCV stock prediction pipeline with LightGBM walk-forward evaluation, transaction-cost-aware backtesting, and offline-first design (synthetic data by default).

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/waholulu/stock_prediction/blob/main/notebooks/colab_quickstart.ipynb)

## Quickstart (local)

```bash
pip install -r requirements.txt
python run_pipeline.py --seed 42
```

## Quickstart (Colab)

Click the badge above or open `notebooks/colab_quickstart.ipynb`.

## Optional: live Yahoo Finance data

```bash
pip install -r requirements-data.txt
python run_pipeline.py --symbol SPY --start 2010-01-01 --seed 42
```

## Running tests

```bash
pip install pytest
python -m pytest tests/ -v
```

## Repository layout

```
stock_prediction/
├── run_pipeline.py           # End-to-end CLI
├── requirements.txt          # Core dependencies
├── requirements-data.txt     # Optional: yfinance for live data
├── notebooks/
│   └── colab_quickstart.ipynb
├── src/
│   ├── data.py
│   ├── features.py
│   ├── labels.py
│   ├── evaluation.py
│   ├── models.py
│   └── backtest.py
└── tests/
```

See `COLAB_GUIDE.md` for a manual step-by-step Colab walkthrough and interactive usage examples.
