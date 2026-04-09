"""
Label creation for stock prediction tasks.

Three label families (matching the research report blueprint):

1. ``label_next_day_direction`` — binary UP/DOWN based on next-day log return sign.
2. ``label_k_day_return``       — forward k-day log return (regression target).
3. ``triple_barrier_labels``    — ternary {+1, 0, -1} trading-congruent labels
                                   based on profit-take / stop-loss / time-out.

All functions return Series aligned with the input DataFrame index and use
only *forward-looking* price data (i.e., the label is about the future, which is
correct — it is the *features* that must not be future-leaking).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def label_next_day_direction(
    df: pd.DataFrame,
    ret_col: str = "ret_1d",
) -> pd.Series:
    """Binary label: 1 if next-day log return > 0, else 0.

    The last row will be NaN (no next-day return available).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``ret_col``.
    ret_col : str
        Column with daily log returns.

    Returns
    -------
    pd.Series of int (0 or 1), with NaN at the last position.
    """
    if ret_col not in df.columns:
        raise ValueError(f"Column '{ret_col}' not found in DataFrame.")
    y = (df[ret_col].shift(-1) > 0).astype(float)
    y.iloc[-1] = np.nan  # last row has no future return
    return y.rename("y_dir")


def label_k_day_return(
    df: pd.DataFrame,
    k: int = 5,
    price_col: str = "close",
) -> pd.Series:
    """Forward k-day log return (regression target).

    log(close[t+k] / close[t])

    The last k rows will be NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``price_col``.
    k : int
        Forecast horizon in trading days.
    price_col : str
        Column to compute returns from.

    Returns
    -------
    pd.Series of float.
    """
    if price_col not in df.columns:
        raise ValueError(f"Column '{price_col}' not found in DataFrame.")
    fwd = np.log(df[price_col].shift(-k)) - np.log(df[price_col])
    return fwd.rename(f"y_{k}d")


def triple_barrier_labels(
    df: pd.DataFrame,
    price_col: str = "close",
    horizon: int = 10,
    pt_multiplier: float = 1.0,
    sl_multiplier: float = 1.0,
    vol_lookback: int = 20,
) -> tuple[pd.Series, pd.Series]:
    """Simplified triple-barrier labeling.

    For each bar *t*:
    - Compute a volatility estimate ``σ[t]`` as the rolling std of log returns
      over the previous ``vol_lookback`` days.
    - Set an upper barrier at ``log(P[t]) + pt_multiplier * σ[t]``
      and a lower barrier at ``log(P[t]) - sl_multiplier * σ[t]``.
    - Walk forward up to ``horizon`` bars:
        - If log price hits the upper barrier first → label = +1
        - If log price hits the lower barrier first → label = -1
        - If neither is hit within the horizon   → label =  0
    - The *exit index* (bar at which the label was determined) is also returned
      so callers can apply purging during walk-forward CV.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``price_col``.
    price_col : str
        Close price column.
    horizon : int
        Maximum number of bars to hold the position.
    pt_multiplier : float
        Profit-take barrier in units of σ.
    sl_multiplier : float
        Stop-loss barrier in units of σ.
    vol_lookback : int
        Rolling window for volatility estimation.

    Returns
    -------
    labels : pd.Series of float {-1, 0, +1, NaN}
        NaN when σ is unavailable or near zero.
    t_end : pd.Series of float
        Integer index of the bar at which the barrier was crossed (or horizon
        expiry).  NaN when ``labels`` is NaN.
    """
    if price_col not in df.columns:
        raise ValueError(f"Column '{price_col}' not found in DataFrame.")

    px = df[price_col].values.astype(np.float64)
    log_px = np.log(px)

    # Volatility proxy: rolling std of 1-day log returns
    log_ret = pd.Series(np.diff(log_px, prepend=np.nan))
    vol = log_ret.rolling(vol_lookback).std().values

    n = len(df)
    labels = np.full(n, np.nan)
    t_end_arr = np.full(n, np.nan)

    for t in range(n - 1):
        v = vol[t]
        if np.isnan(v) or v < 1e-10:
            continue

        pt_level = log_px[t] + pt_multiplier * v
        sl_level = log_px[t] - sl_multiplier * v

        hit = 0
        end_t = min(t + horizon, n - 1)
        for tau in range(t + 1, min(t + horizon + 1, n)):
            lp = log_px[tau]
            if lp >= pt_level:
                hit = 1
                end_t = tau
                break
            if lp <= sl_level:
                hit = -1
                end_t = tau
                break

        labels[t] = hit
        t_end_arr[t] = end_t

    return (
        pd.Series(labels, index=df.index, name="y_tb"),
        pd.Series(t_end_arr, index=df.index, name="t_end"),
    )
