from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def label_next_day_direction(df: pd.DataFrame, ret_col: str = "ret_1d") -> pd.Series:
    """Binary label: 1 if next day's return > 0, else 0."""
    return (df[ret_col].shift(-1) > 0).astype(int)


def label_k_day_return(df: pd.DataFrame, k: int = 5, price_col: str = "close") -> pd.Series:
    """Forward log return over k days (regression target)."""
    return np.log(df[price_col].shift(-k)) - np.log(df[price_col])


def triple_barrier_labels(
    df: pd.DataFrame,
    price_col: str = "close",
    horizon: int = 10,
    pt: float = 1.0,
    sl: float = 1.0,
    vol_lookback: int = 20,
) -> Tuple[pd.Series, pd.Series]:
    """Simplified triple-barrier labeling.

    Returns:
        labels: +1 profit-take, -1 stop-loss, 0 timeout
        t_end: index of barrier hit
    """
    px = df[price_col].values.astype(float)
    logret = np.diff(np.log(px), prepend=np.nan)
    vol = pd.Series(logret).rolling(vol_lookback).std().values

    labels = np.full(len(df), np.nan)
    t_end = np.full(len(df), np.nan)

    for t in range(len(df) - horizon):
        if np.isnan(vol[t]) or vol[t] == 0:
            continue
        pt_level = np.log(px[t]) + pt * vol[t]
        sl_level = np.log(px[t]) - sl * vol[t]

        hit = 0
        end_t = t + horizon
        for tau in range(t + 1, t + horizon + 1):
            lr = np.log(px[tau])
            if lr >= pt_level:
                hit = +1
                end_t = tau
                break
            if lr <= sl_level:
                hit = -1
                end_t = tau
                break
        labels[t] = hit
        t_end[t] = end_t

    return pd.Series(labels, index=df.index), pd.Series(t_end, index=df.index)


def add_all_labels(
    df: pd.DataFrame,
    k: int = 5,
    tb_horizon: int = 10,
    tb_pt: float = 1.0,
    tb_sl: float = 1.0,
    tb_vol_lookback: int = 20,
) -> pd.DataFrame:
    """Add all three label types to the DataFrame."""
    df = df.copy()
    df["y_dir"] = label_next_day_direction(df)
    df["y_5d"] = label_k_day_return(df, k=k)
    df["y_tb"], df["t_end"] = triple_barrier_labels(
        df, horizon=tb_horizon, pt=tb_pt, sl=tb_sl, vol_lookback=tb_vol_lookback
    )
    return df
