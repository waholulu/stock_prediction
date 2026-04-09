from __future__ import annotations

from typing import List, Sequence, Tuple

import pandas as pd


def make_features(
    df: pd.DataFrame,
    windows: Sequence[int] = (5, 10, 21, 63),
) -> pd.DataFrame:
    """Build leakage-safe rolling features from basic returns columns.

    Expects columns: ret_1d, hl_range, oc_return, vol_chg, date.
    """
    df = df.copy()
    for w in windows:
        df[f"ret_mean_{w}"] = df["ret_1d"].rolling(w).mean()
        df[f"ret_std_{w}"] = df["ret_1d"].rolling(w).std()
        df[f"range_mean_{w}"] = df["hl_range"].rolling(w).mean()
        df[f"oc_mean_{w}"] = df["oc_return"].rolling(w).mean()
        df[f"vol_chg_mean_{w}"] = df["vol_chg"].rolling(w).mean()
    # Calendar features (known at prediction time)
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    return df


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Return the list of feature column names (excludes raw OHLCV and target cols)."""
    exclude = {
        "date", "open", "high", "low", "close", "volume",
        "log_close", "ret_1d", "vol_chg", "hl_range", "oc_return",
        "y_dir", "y_5d", "y_tb", "t_end",
    }
    return [c for c in df.columns if c not in exclude]
