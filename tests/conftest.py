"""Shared test fixtures: synthetic OHLCV data for unit tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_ohlcv():
    """Generate 500 rows of synthetic OHLCV data."""
    np.random.seed(42)
    n = 500
    dates = pd.bdate_range("2010-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
    high = close * (1 + np.abs(np.random.randn(n) * 0.005))
    low = close * (1 - np.abs(np.random.randn(n) * 0.005))
    open_ = close * (1 + np.random.randn(n) * 0.003)
    volume = np.random.randint(1_000_000, 10_000_000, size=n).astype(float)

    return pd.DataFrame({
        "date": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
