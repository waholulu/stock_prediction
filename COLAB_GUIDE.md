# Running the Stock Prediction Pipeline in Google Colab

> **One-click version:** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/waholulu/stock_prediction/blob/main/notebooks/colab_quickstart.ipynb)  
> Open `notebooks/colab_quickstart.ipynb` for an automated version of this guide.

This guide walks you through cloning the repository and running the full
experiment pipeline in a free Google Colab notebook — no local Python install
required.

---

## Step 1 — Open a new Colab notebook

Go to [colab.research.google.com](https://colab.research.google.com) and create
a new notebook (File → New notebook).

---

## Step 2 — Clone the repository

Paste the following into the first cell and run it (`Shift+Enter`):

```python
# Clone the repo into the Colab filesystem
!git clone https://github.com/waholulu/stock_prediction.git
%cd stock_prediction
```

---

## Step 3 — Install dependencies

Run this cell to install the core pipeline dependencies:

```python
!pip install -q -r requirements.txt
```

**Optional** — only needed for live Yahoo Finance data (Step 6):

```python
!pip install -q -r requirements-data.txt
```

> The pipeline automatically falls back to synthetic data if `yfinance`
> cannot be imported, so installing `requirements-data.txt` is never a blocker.

---

## Step 4 — Run the tests (optional but recommended)

Verify that everything is working before running experiments:

```python
!pip install -q pytest
!python -m pytest tests/ -v
```

You should see **53 passed** at the end.

---

## Step 5 — Run the full pipeline on synthetic data

This works entirely offline — no API keys or Yahoo Finance access needed:

```python
!python run_pipeline.py
```

Expected output (truncated):

```
[1/5] Generating synthetic OHLCV (3000 trading days) …
  Loaded 3000 bars  (2016-01-01 – 2027-10-22)
[2/5] Computing features and labels …
  30 features, 2936 clean rows
[3/5] Running walk-forward LightGBM evaluation …
  · Binary classification (next-day direction) …
  · Regression (5-day forward return) …
  · Ternary classification with purging (triple-barrier) …
[4/5] Building OOS signal and running backtest …
[5/5] Results summary
...
  Results saved to ./results/
```

---

## Step 6 — Run on live SPY data from Yahoo Finance

```python
!python run_pipeline.py --symbol SPY --start 2010-01-01
```

If `yfinance` is not importable the pipeline prints a warning and falls back to
synthetic data automatically.

---

## Step 7 — Use the modules interactively in a notebook

You can also import the pipeline modules directly and explore results cell by
cell:

```python
import sys
sys.path.insert(0, "/content/stock_prediction")  # adjust if needed

import pandas as pd
import matplotlib.pyplot as plt

from src.data import generate_synthetic_ohlcv, add_basic_returns
from src.features import make_features, get_feature_cols
from src.labels import label_next_day_direction, label_k_day_return, triple_barrier_labels
from src.evaluation import WalkForwardSpec
from src.models import walk_forward_evaluate
from src.backtest import backtest_daily_direction, summarise_backtest
```

### Load and inspect data

```python
df = generate_synthetic_ohlcv(n_days=2000)
df = add_basic_returns(df)
df.head()
```

### Engineer features and create labels

```python
df = make_features(df)
df["y_dir"] = label_next_day_direction(df)
df["y_5d"]  = label_k_day_return(df, k=5)
y_tb, t_end = triple_barrier_labels(df, horizon=10)
df["y_tb"]  = y_tb
df["t_end"] = t_end

feat_cols = get_feature_cols(df)
print(f"{len(feat_cols)} features:", feat_cols[:6], "...")
```

### Walk-forward evaluation

```python
spec = WalkForwardSpec(train_years=3, test_months=6, embargo_days=5)

metrics = walk_forward_evaluate(df, feat_cols, "y_dir",
                                task="classify_binary", spec=spec)
print(metrics)
print("\nMean across folds:")
print(metrics.mean(numeric_only=True).round(4))
```

### Plot fold-by-fold accuracy

```python
plt.figure(figsize=(10, 4))
plt.bar(metrics["fold"], metrics["accuracy"], label="Accuracy")
plt.axhline(0.5, color="red", linestyle="--", label="Random baseline")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.title("Walk-Forward Accuracy — LightGBM Binary Classifier")
plt.legend()
plt.tight_layout()
plt.show()
```

### Backtest the OOS signal

```python
# Collect out-of-sample predictions across all folds
import numpy as np
import lightgbm as lgb
from src.evaluation import walk_forward_splits

df_clean = df.dropna(subset=feat_cols + ["y_dir"]).reset_index(drop=True)
X = df_clean[feat_cols].values.astype("float32")
y = df_clean["y_dir"].values.astype(int)
oos_signal = pd.Series(np.nan, index=df_clean.index)

for train_idx, test_idx, _ in walk_forward_splits(df_clean["date"], spec):
    clf = lgb.LGBMClassifier(n_estimators=200, verbosity=-1)
    clf.fit(pd.DataFrame(X[train_idx], columns=feat_cols), y[train_idx])
    prob = clf.predict_proba(pd.DataFrame(X[test_idx], columns=feat_cols))[:, 1]
    oos_signal.iloc[test_idx] = np.where(prob >= 0.5, 1, -1)

df_clean["signal"] = oos_signal
df_bt = df_clean.dropna(subset=["signal", "ret_1d"])
bt = backtest_daily_direction(df_bt, signal_col="signal", cost_bps=2.0)
summary = summarise_backtest(bt)
print(summary)
```

### Plot the equity curve

```python
plt.figure(figsize=(12, 5))
plt.plot(bt["date"], bt["equity"],    label="Strategy (net of costs)")
plt.plot(bt["date"], bt["bh_equity"], label="Buy & Hold", linestyle="--")
plt.xlabel("Date")
plt.ylabel("Equity (starting at 1.0)")
plt.title("OOS Equity Curve vs Buy-and-Hold")
plt.legend()
plt.tight_layout()
plt.show()
```

---

## Customisation tips

| Goal | What to change |
|---|---|
| Use real market data | `--symbol QQQ --start 2005-01-01` or replace `generate_synthetic_ohlcv()` with `load_ohlcv_from_yfinance()` |
| Change forecast horizon | Pass `k=10` to `label_k_day_return()` |
| Wider / tighter barriers | Adjust `pt_multiplier` and `sl_multiplier` in `triple_barrier_labels()` |
| Longer training window | `WalkForwardSpec(train_years=7, ...)` |
| Expanding window | `WalkForwardSpec(expanding=True, ...)` |
| Different transaction cost | `backtest_daily_direction(..., cost_bps=5.0)` |

---

## Saving results to Google Drive

```python
from google.colab import drive
drive.mount("/content/drive")

import shutil
shutil.copytree(
    "/content/stock_prediction/results",
    "/content/drive/MyDrive/stock_prediction_results",
    dirs_exist_ok=True,
)
print("Results saved to Google Drive.")
```
