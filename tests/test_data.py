"""Tests for src/data.py"""
import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import generate_synthetic_ohlcv, add_basic_returns


def test_synthetic_shape():
    df = generate_synthetic_ohlcv(n_days=100)
    assert df.shape == (100, 6)
    assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]


def test_synthetic_ohlc_validity():
    df = generate_synthetic_ohlcv(n_days=500)
    assert (df["high"] >= df["close"]).all(), "high must be >= close"
    assert (df["high"] >= df["open"]).all(), "high must be >= open"
    assert (df["low"] <= df["close"]).all(), "low must be <= close"
    assert (df["low"] <= df["open"]).all(), "low must be <= open"
    assert (df["close"] > 0).all(), "close must be positive"
    assert (df["volume"] > 0).all(), "volume must be positive"


def test_synthetic_dates_monotonic():
    df = generate_synthetic_ohlcv(n_days=200)
    assert df["date"].is_monotonic_increasing


def test_synthetic_reproducible():
    df1 = generate_synthetic_ohlcv(seed=0)
    df2 = generate_synthetic_ohlcv(seed=0)
    pd.testing.assert_frame_equal(df1, df2)


def test_synthetic_different_seeds():
    df1 = generate_synthetic_ohlcv(seed=1)
    df2 = generate_synthetic_ohlcv(seed=2)
    assert not df1["close"].equals(df2["close"])


def test_add_basic_returns_columns():
    df = generate_synthetic_ohlcv(n_days=50)
    df = add_basic_returns(df)
    for col in ["log_close", "ret_1d", "vol_chg", "hl_range", "oc_return"]:
        assert col in df.columns, f"Missing column: {col}"


def test_add_basic_returns_first_row_nan():
    df = generate_synthetic_ohlcv(n_days=50)
    df = add_basic_returns(df)
    assert np.isnan(df["ret_1d"].iloc[0]), "First ret_1d must be NaN"


def test_add_basic_returns_no_mutation():
    df = generate_synthetic_ohlcv(n_days=50)
    original_cols = list(df.columns)
    _ = add_basic_returns(df)
    assert list(df.columns) == original_cols, "add_basic_returns must not mutate input"


def test_add_basic_returns_hl_range_positive():
    df = generate_synthetic_ohlcv(n_days=200)
    df = add_basic_returns(df)
    assert (df["hl_range"] >= 0).all()
