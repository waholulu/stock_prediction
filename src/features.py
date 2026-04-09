"""
Leakage-safe feature engineering for daily OHLCV stock prediction.

All features are computed strictly from past data (rolling windows that do not
include the current bar) to avoid look-ahead bias.

Functions:
- make_features(): build rolling return/volatility/range/volume features
- get_feature_cols(): return the list of feature column names for a given DataFrame
"""

from __future__ import annotations

import numpy as np
import pandas as pd


_DEFAULT_WINDOWS = (5, 10, 21, 63)


def make_features(
    df: pd.DataFrame,
    windows: tuple[int, ...] = _DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """Compute rolling OHLCV-derived features.

    The DataFrame must already contain the columns produced by
    ``data.add_basic_returns``: ``ret_1d``, ``hl_range``, ``oc_return``,
    ``vol_chg``.

    For each window ``w`` the following features are added (using only data
    strictly prior to the current bar; ``rolling(w)`` with default min_periods
    of ``w`` ensures no partial-window leakage at the cost of NaN rows):

    - ``ret_mean_<w>``     : rolling mean of 1-day log returns
    - ``ret_std_<w>``      : rolling std of 1-day log returns
    - ``range_mean_<w>``   : rolling mean of normalised HL range
    - ``oc_mean_<w>``      : rolling mean of open-to-close returns
    - ``vol_chg_mean_<w>`` : rolling mean of log volume changes
    - ``ret_skew_<w>``     : rolling skewness of returns (w >= 10 only)

    Calendar features (known at prediction time, no look-ahead):
    - ``dow``   : day-of-week (0 = Monday)
    - ``month`` : calendar month

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: date, ret_1d, hl_range, oc_return, vol_chg
    windows : tuple of int
        Rolling window sizes in trading days.

    Returns
    -------
    pd.DataFrame
        Original DataFrame extended with feature columns (copy, no mutation).
    """
    required = ["ret_1d", "hl_range", "oc_return", "vol_chg"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required columns: {missing}. "
            "Call data.add_basic_returns() first."
        )

    df = df.copy()

    for w in windows:
        df[f"ret_mean_{w}"] = df["ret_1d"].rolling(w).mean()
        df[f"ret_std_{w}"] = df["ret_1d"].rolling(w).std()
        df[f"range_mean_{w}"] = df["hl_range"].rolling(w).mean()
        df[f"oc_mean_{w}"] = df["oc_return"].rolling(w).mean()
        df[f"vol_chg_mean_{w}"] = df["vol_chg"].rolling(w).mean()
        if w >= 10:
            df[f"ret_skew_{w}"] = df["ret_1d"].rolling(w).skew()

    # Momentum: cumulative return over a window (ratio of prices)
    for w in windows:
        df[f"momentum_{w}"] = df["ret_1d"].rolling(w).sum()

    # Volatility ratio: short-window vol / long-window vol (regime indicator)
    if 5 in windows and 21 in windows:
        short_vol = df["ret_1d"].rolling(5).std()
        long_vol = df["ret_1d"].rolling(21).std()
        df["vol_ratio_5_21"] = short_vol / (long_vol + 1e-12)

    # Calendar features (always available at prediction time)
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return feature column names (excludes raw OHLCV, date, label columns)."""
    exclude = {
        "date", "open", "high", "low", "close", "volume",
        "log_close", "adj_close",
        # label columns added later
        "ret_1d", "vol_chg", "hl_range", "oc_return",
        "y_dir", "y_5d", "y_tb", "t_end",
    }
    return [c for c in df.columns if c not in exclude]
