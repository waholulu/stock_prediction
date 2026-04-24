"""
Data loading utilities for stock prediction pipeline.

Provides:
- load_ohlcv_from_yfinance(): fetch real data from Yahoo Finance
- generate_synthetic_ohlcv(): produce reproducible synthetic OHLCV data for testing
- add_basic_returns(): compute log returns and derived series
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_ohlcv_from_yfinance(
    symbol: str = "SPY",
    start: str = "2000-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance.

    Returns a DataFrame with columns: date, open, high, low, close, volume.
    Raises ImportError if yfinance is not available.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is required to fetch live data. "
            "Install it with: pip install -r requirements-data.txt"
        ) from exc

    df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
    # Flatten MultiIndex columns if present (yfinance >= 0.2 may return them)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    df = df.reset_index()
    # Normalize the date column name
    date_col = [c for c in df.columns if c.lower() in ("date", "datetime", "index")][0]
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"adj_close": "adj_close"})  # keep if present

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Downloaded DataFrame is missing columns: {missing}")

    df = df.sort_values("date").dropna(subset=required).reset_index(drop=True)
    return df[["date", "open", "high", "low", "close", "volume"]]


def generate_synthetic_ohlcv(
    n_days: int = 2000,
    start: str = "2016-01-01",
    seed: int = 42,
    initial_price: float = 300.0,
    annual_vol: float = 0.16,
    annual_drift: float = 0.07,
) -> pd.DataFrame:
    """Generate synthetic daily OHLCV data using geometric Brownian motion.

    Suitable for unit tests and offline development without internet access.

    Parameters
    ----------
    n_days : int
        Number of trading days to simulate.
    start : str
        First date in the series (business days from here).
    seed : int
        Random seed for reproducibility.
    initial_price : float
        Starting close price.
    annual_vol : float
        Annualised volatility (e.g. 0.16 = 16 %).
    annual_drift : float
        Annualised drift / expected return.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)

    daily_vol = annual_vol / np.sqrt(252)
    daily_drift = annual_drift / 252

    # Simulate close prices via GBM
    log_returns = rng.normal(daily_drift - 0.5 * daily_vol**2, daily_vol, size=n_days)
    log_prices = np.log(initial_price) + np.cumsum(log_returns)
    close = np.exp(log_prices)

    # Construct OHLC around close with realistic intraday noise
    intraday_noise = rng.uniform(0.002, 0.012, size=n_days)
    open_price = close * np.exp(rng.normal(0, daily_vol * 0.3, size=n_days))
    high = np.maximum(close, open_price) * (1 + intraday_noise)
    low = np.minimum(close, open_price) * (1 - intraday_noise)

    # Volume: log-normal around a baseline
    base_volume = 8e7
    volume = rng.lognormal(mean=np.log(base_volume), sigma=0.4, size=n_days).astype(int)

    return pd.DataFrame(
        {
            "date": dates,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def add_basic_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add log-return and price-derived columns to an OHLCV DataFrame.

    New columns added (all computable without look-ahead):
    - log_close  : natural log of close
    - ret_1d     : 1-day log return of close
    - vol_chg    : 1-day log change in volume
    - hl_range   : (high - low) / close  (normalised daily range)
    - oc_return  : (close - open) / open (open-to-close return)
    """
    df = df.copy()
    df["log_close"] = np.log(df["close"])
    df["ret_1d"] = df["log_close"].diff()
    df["vol_chg"] = np.log(df["volume"].replace(0, np.nan)).diff()
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]
    df["oc_return"] = (df["close"] - df["open"]) / df["open"]
    return df
