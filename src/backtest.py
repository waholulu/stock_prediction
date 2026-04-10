"""
Simple transaction-cost-aware backtest for a daily directional strategy.

The backtest:
- Takes a signal series (+1 long, -1 short, 0 flat).
- Applies next-bar execution: position entered using the *next* bar's return.
- Deducts a round-trip transaction cost (in basis points) whenever the position
  changes.
- Reports equity curve, cumulative net return, annualised Sharpe, and max
  drawdown.

Functions:
- backtest_daily_direction(): run backtest on a signal + return series
- summarise_backtest()      : compute performance metrics from backtest output
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def backtest_daily_direction(
    df: pd.DataFrame,
    signal_col: str,
    ret_col: str = "ret_1d",
    cost_bps: float = 2.0,
) -> pd.DataFrame:
    """Backtest a daily directional strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``signal_col`` (+1, -1, or 0) and ``ret_col`` (log return).
    signal_col : str
        Column with position signal.  +1 = long, -1 = short, 0 = flat.
    ret_col : str
        Column with daily log returns of the underlying.
    cost_bps : float
        Round-trip transaction cost per trade, in basis points (1 bps = 0.0001).

    Returns
    -------
    pd.DataFrame
        Original DataFrame extended with:
        - ``pos``       : filled signal (NaN → 0)
        - ``strat_ret`` : gross strategy log return (position * next-bar return)
        - ``cost``      : transaction cost applied on position changes
        - ``net_ret``   : net strategy return (strat_ret - cost)
        - ``equity``    : cumulative equity starting at 1.0
        - ``bh_equity`` : buy-and-hold equity for comparison
    """
    if signal_col not in df.columns:
        raise ValueError(f"Column '{signal_col}' not found in DataFrame.")
    if ret_col not in df.columns:
        raise ValueError(f"Column '{ret_col}' not found in DataFrame.")

    out = df.copy()
    signal = out[signal_col].fillna(0).astype(float)

    # Position is entered at the *next* bar (no look-ahead)
    out["pos"] = signal
    out["strat_ret"] = out["pos"].shift(1) * out[ret_col]

    # Cost when position changes (non-zero turnover)
    turnover = (out["pos"].diff().abs() > 0).astype(float)
    # First bar: assume we enter from flat, so mark as turnover if pos != 0
    turnover.iloc[0] = float(out["pos"].iloc[0] != 0)
    out["cost"] = turnover * (cost_bps / 1e4)

    out["net_ret"] = out["strat_ret"] - out["cost"]

    # Cumulative equity (starting at 1.0)
    out["equity"] = np.exp(out["net_ret"].fillna(0).cumsum())
    out["bh_equity"] = np.exp(out[ret_col].fillna(0).cumsum())

    return out


def summarise_backtest(bt: pd.DataFrame, trading_days: int = 252) -> dict[str, float]:
    """Compute performance summary from backtest output.

    Parameters
    ----------
    bt : pd.DataFrame
        Output of ``backtest_daily_direction``.
    trading_days : int
        Number of trading days per year for annualisation.

    Returns
    -------
    dict with keys:
    - ``total_net_return`` : cumulative net log return (not annualised)
    - ``annual_sharpe``    : annualised Sharpe ratio of net returns
    - ``max_drawdown``     : maximum peak-to-trough drawdown of equity curve
    - ``win_rate``         : fraction of non-flat bars with positive net return
    - ``n_trades``         : number of position changes
    - ``bh_total_return``  : buy-and-hold cumulative log return
    """
    net = bt["net_ret"].dropna()

    total_net_return = float(net.sum())
    mean_ret = net.mean()
    std_ret = net.std(ddof=1)
    annual_sharpe = (
        float(mean_ret / (std_ret + 1e-12) * np.sqrt(trading_days))
        if std_ret > 0 else 0.0
    )

    # Max drawdown on equity curve
    equity = bt["equity"].dropna()
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / (roll_max + 1e-12)
    max_drawdown = float(drawdown.min())

    # Win rate: fraction of active bars (pos != 0) with net_ret > 0
    active = bt[bt["pos"].shift(1).fillna(0) != 0]["net_ret"].dropna()
    win_rate = float((active > 0).mean()) if len(active) > 0 else float("nan")

    n_trades = int((bt["pos"].diff().abs() > 0).sum())
    bh_total_return = float(bt["bh_equity"].iloc[-1] - 1) if "bh_equity" in bt.columns else float("nan")

    return {
        "total_net_return": total_net_return,
        "annual_sharpe": annual_sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "n_trades": n_trades,
        "bh_total_return": bh_total_return,
    }
