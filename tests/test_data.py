import numpy as np
import pandas as pd

from src.data import add_basic_returns


def test_add_basic_returns_columns(synthetic_ohlcv):
    df = add_basic_returns(synthetic_ohlcv)
    for col in ["log_close", "ret_1d", "vol_chg", "hl_range", "oc_return"]:
        assert col in df.columns, f"Missing column: {col}"


def test_add_basic_returns_no_nan_after_drop(synthetic_ohlcv):
    df = add_basic_returns(synthetic_ohlcv).dropna()
    assert len(df) > 0
    for col in ["ret_1d", "hl_range", "oc_return"]:
        assert df[col].isna().sum() == 0


def test_dates_sorted(synthetic_ohlcv):
    df = add_basic_returns(synthetic_ohlcv)
    assert df["date"].is_monotonic_increasing
