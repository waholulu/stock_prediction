from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.metrics import compute_sharpe


def backtest_daily_direction(
    df: pd.DataFrame,
    signal_col: str,
    ret_col: str = "ret_1d",
    cost_bps: float = 2.0,
) -> Tuple[pd.DataFrame, float]:
    """Run a simple daily directional backtest.

    Args:
        signal_col: column with +1 (long), -1 (short), 0 (flat)
        ret_col: column with daily log returns
        cost_bps: round-trip transaction cost in basis points

    Returns:
        (result_df, sharpe): DataFrame with equity curve columns, and net Sharpe ratio
    """
    out = df.copy()
    signal = out[signal_col].fillna(0).astype(int)

    out["pos"] = signal
    out["strat_ret"] = out["pos"].shift(1) * out[ret_col]

    # Transaction costs when position changes
    turnover = (out["pos"].diff().abs() > 0).astype(float)
    out["cost"] = turnover * (cost_bps / 1e4)
    out["net_ret"] = out["strat_ret"] - out["cost"]

    out["equity"] = np.exp(out["net_ret"].fillna(0).cumsum())
    sharpe = compute_sharpe(out["net_ret"].dropna().values)
    return out, sharpe


def backtest_cost_sensitivity(
    df: pd.DataFrame,
    signal_col: str,
    ret_col: str = "ret_1d",
    cost_bps_list: List[float] = None,
) -> pd.DataFrame:
    """Run backtest at multiple cost levels, returning a summary table."""
    if cost_bps_list is None:
        cost_bps_list = [0, 1, 2, 5, 10]

    results = []
    for bps in cost_bps_list:
        _, sharpe = backtest_daily_direction(df, signal_col, ret_col, cost_bps=bps)
        results.append({"cost_bps": bps, "sharpe": sharpe})
    return pd.DataFrame(results)
