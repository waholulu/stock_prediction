from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd


def load_ohlcv(
    symbol: str = "SPY",
    start: str = "2000-01-01",
    end: Optional[str] = None,
    cache_dir: Optional[str] = "data/raw",
) -> pd.DataFrame:
    """Fetch OHLCV from yfinance with optional local CSV cache."""
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{symbol}_{start}_{end or 'latest'}.csv")
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path, parse_dates=["date"])
            return df

    import yfinance as yf

    raw = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
    df = raw.copy()
    # Handle multi-level columns from newer yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={c: c.lower().replace(" ", "_") for c in df.columns})
    df = df.reset_index()
    # Normalize date column name
    date_col = [c for c in df.columns if c.lower() == "date"]
    if date_col:
        df = df.rename(columns={date_col[0]: "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").dropna(subset=["open", "high", "low", "close"])
    df = df[["date", "open", "high", "low", "close", "volume"]].reset_index(drop=True)

    if cache_dir:
        df.to_csv(cache_path, index=False)

    return df


def add_basic_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns and basic derived series from OHLCV."""
    df = df.copy()
    df["log_close"] = np.log(df["close"])
    df["ret_1d"] = df["log_close"].diff()
    df["vol_chg"] = np.log(df["volume"].replace(0, np.nan)).diff()
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]
    df["oc_return"] = (df["close"] - df["open"]) / df["open"]
    return df


def load_and_prepare(
    symbol: str = "SPY",
    start: str = "2000-01-01",
    end: Optional[str] = None,
    cache_dir: Optional[str] = "data/raw",
) -> pd.DataFrame:
    """Load OHLCV and add basic returns, dropping NaN rows."""
    df = load_ohlcv(symbol, start, end, cache_dir)
    df = add_basic_returns(df)
    df = df.dropna().reset_index(drop=True)
    return df
