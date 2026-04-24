"""
Main pipeline script: S&P 500 daily OHLCV stock prediction.

Runs the full experiment sequence from the deep-research-report:
  1. Load OHLCV data (real from yfinance, or synthetic fallback for offline use).
  2. Feature engineering (rolling OHLCV-derived features).
  3. Label creation: next-day direction, 5-day return, triple-barrier.
  4. Walk-forward evaluation with embargo using LightGBM:
     - Binary classification (next-day direction)
     - Regression (5-day forward return)
     - Ternary classification with purging (triple-barrier)
  5. Simple transaction-cost-aware backtest on out-of-sample predictions.
  6. Print summary table and save results to CSV.

Usage:
    python run_pipeline.py                  # uses synthetic data
    python run_pipeline.py --symbol SPY     # fetches live data from yfinance
    python run_pipeline.py --symbol SPY --start 2010-01-01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Ensure src is importable when run from project root ──────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.data import add_basic_returns, generate_synthetic_ohlcv, load_ohlcv_from_yfinance
from src.features import get_feature_cols, make_features
from src.labels import label_k_day_return, label_next_day_direction, triple_barrier_labels
from src.evaluation import WalkForwardSpec
from src.models import walk_forward_evaluate
from src.backtest import backtest_daily_direction, summarise_backtest


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stock prediction pipeline")
    p.add_argument("--symbol", default=None, help="Ticker symbol (e.g. SPY). Omit for synthetic data.")
    p.add_argument("--start", default="2005-01-01", help="Start date for yfinance download.")
    p.add_argument("--synthetic-days", type=int, default=3000, help="Days for synthetic data.")
    p.add_argument("--train-years", type=float, default=5.0)
    p.add_argument("--test-months", type=float, default=6.0)
    p.add_argument("--embargo-days", type=int, default=5)
    p.add_argument("--cost-bps", type=float, default=2.0, help="Round-trip transaction cost (bps).")
    p.add_argument("--out-dir", default="results", help="Directory to write CSV results.")
    p.add_argument(
        "--seed", type=int, default=42,
        help="Global random seed for NumPy and LightGBM reproducibility.",
    )
    return p.parse_args()


# ── Steps ─────────────────────────────────────────────────────────────────────

def step_load(args: argparse.Namespace) -> pd.DataFrame:
    if args.symbol:
        print(f"[1/5] Downloading OHLCV for {args.symbol} from {args.start} …")
        try:
            df = load_ohlcv_from_yfinance(args.symbol, start=args.start)
        except Exception as exc:
            print(f"  WARNING: yfinance failed ({exc}). Falling back to synthetic data.")
            df = generate_synthetic_ohlcv(n_days=args.synthetic_days)
    else:
        print(f"[1/5] Generating synthetic OHLCV ({args.synthetic_days} trading days) …")
        df = generate_synthetic_ohlcv(n_days=args.synthetic_days)

    print(f"  Loaded {len(df)} bars  ({df['date'].iloc[0].date()} – {df['date'].iloc[-1].date()})")
    return df


def step_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    print("[2/5] Computing features and labels …")
    df = add_basic_returns(df)
    df = make_features(df)
    df["y_dir"] = label_next_day_direction(df)
    df["y_5d"] = label_k_day_return(df, k=5)
    y_tb, t_end = triple_barrier_labels(df, horizon=10)
    df["y_tb"] = y_tb
    df["t_end"] = t_end
    df = df.dropna(subset=["y_dir"]).reset_index(drop=True)
    feat_cols = get_feature_cols(df)
    print(f"  {len(feat_cols)} features, {len(df)} clean rows")
    return df, feat_cols


def step_evaluate(
    df: pd.DataFrame,
    feat_cols: list[str],
    spec: WalkForwardSpec,
    seed: int = 42,
) -> "tuple[dict[str, pd.DataFrame], pd.Series]":
    print("[3/5] Running walk-forward LightGBM evaluation …")
    results: dict[str, pd.DataFrame] = {}

    print("  · Binary classification (next-day direction) …")
    binary_metrics, oos_signal = walk_forward_evaluate(
        df, feat_cols, "y_dir", task="classify_binary", spec=spec,
        seed=seed, return_oos_predictions=True,
    )
    results["binary"] = binary_metrics

    print("  · Regression (5-day forward return) …")
    results["regression"] = walk_forward_evaluate(
        df, feat_cols, "y_5d", task="regress", spec=spec, seed=seed,
    )

    print("  · Ternary classification with purging (triple-barrier) …")
    results["ternary"] = walk_forward_evaluate(
        df, feat_cols, "y_tb", task="classify_ternary", spec=spec,
        purge=True, seed=seed,
    )

    return results, oos_signal


def step_backtest(
    df: pd.DataFrame,
    feat_cols: list[str],
    oos_signal: pd.Series,
    cost_bps: float,
) -> dict[str, dict]:
    """Use pre-computed OOS signal from step_evaluate to run backtest.

    oos_signal must be indexed like df.dropna(subset=feat_cols + ["y_dir"])
    .reset_index(drop=True) — identical subset to walk_forward_evaluate binary.
    """
    print("[4/5] Building OOS signal and running backtest …")

    df_clean = df.dropna(subset=feat_cols + ["y_dir"]).reset_index(drop=True)

    df_bt = df_clean.copy()
    df_bt["signal"] = oos_signal.values
    df_bt = df_bt.dropna(subset=["signal", "ret_1d"])

    bt = backtest_daily_direction(df_bt, signal_col="signal", cost_bps=cost_bps)
    summary = summarise_backtest(bt)

    cov = len(df_bt) / len(df_clean)
    summary["oos_coverage"] = round(cov, 3)
    return {"summary": summary, "equity_curve": bt[["date", "equity", "bh_equity"]]}


def step_report(
    results: dict[str, pd.DataFrame],
    backtest_out: dict,
    out_dir: str,
    cost_bps: float = 2.0,
) -> None:
    print("[5/5] Results summary\n" + "=" * 60)

    print("\n— Binary classification (next-day direction) —")
    b = results["binary"]
    print(b.to_string(index=False))
    print(f"\n  Mean: acc={b['accuracy'].mean():.4f}  MCC={b['mcc'].mean():.4f}  AUC={b['auc'].mean():.4f}")

    print("\n— Regression (5-day return) —")
    r = results["regression"]
    print(r.to_string(index=False))
    print(f"\n  Mean: MAE={r['mae'].mean():.6f}  RMSE={r['rmse'].mean():.6f}  dir_acc={r['dir_acc'].mean():.4f}")

    print("\n— Ternary classification (triple-barrier) —")
    t = results["ternary"]
    print(t.to_string(index=False))
    print(f"\n  Mean: acc={t['accuracy'].mean():.4f}  MCC={t['mcc'].mean():.4f}")

    print(f"\n— Backtest (OOS binary signal, cost={cost_bps:.1f} bps) —")
    s = backtest_out["summary"]
    for k, v in s.items():
        print(f"  {k:25s}: {v:.4f}")

    # Save results
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results["binary"].to_csv(out / "binary_metrics.csv", index=False)
    results["regression"].to_csv(out / "regression_metrics.csv", index=False)
    results["ternary"].to_csv(out / "ternary_metrics.csv", index=False)
    backtest_out["equity_curve"].to_csv(out / "equity_curve.csv", index=False)
    print(f"\n  Results saved to ./{out_dir}/")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    spec = WalkForwardSpec(
        train_years=args.train_years,
        test_months=args.test_months,
        embargo_days=args.embargo_days,
    )

    df = step_load(args)
    df, feat_cols = step_features(df)
    results, oos_signal = step_evaluate(df, feat_cols, spec, seed=args.seed)
    backtest_out = step_backtest(df, feat_cols, oos_signal, args.cost_bps)
    step_report(results, backtest_out, args.out_dir, cost_bps=args.cost_bps)


if __name__ == "__main__":
    main()
