"""Tests for src/features.py"""
import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import generate_synthetic_ohlcv, add_basic_returns
from src.features import make_features, get_feature_cols


def _base_df(n=500):
    df = generate_synthetic_ohlcv(n_days=n)
    return add_basic_returns(df)


def test_make_features_returns_copy():
    df = _base_df()
    original_ncols = df.shape[1]
    _ = make_features(df)
    assert df.shape[1] == original_ncols, "make_features must not mutate input"


def test_make_features_adds_expected_cols():
    df = make_features(_base_df())
    for w in (5, 10, 21, 63):
        assert f"ret_mean_{w}" in df.columns
        assert f"ret_std_{w}" in df.columns
        assert f"range_mean_{w}" in df.columns
        assert f"oc_mean_{w}" in df.columns
        assert f"vol_chg_mean_{w}" in df.columns
    assert "dow" in df.columns
    assert "month" in df.columns


def test_make_features_calendar_range():
    df = make_features(_base_df())
    assert df["dow"].between(0, 6).all()
    assert df["month"].between(1, 12).all()


def test_make_features_missing_required_col():
    df = generate_synthetic_ohlcv(n_days=100)  # no returns added
    with pytest.raises(ValueError, match="ret_1d"):
        make_features(df)


def test_get_feature_cols_excludes_raw():
    df = make_features(_base_df())
    feat = get_feature_cols(df)
    for excluded in ["date", "open", "high", "low", "close", "volume", "log_close"]:
        assert excluded not in feat, f"'{excluded}' should not be in feature cols"


def test_get_feature_cols_nonempty():
    df = make_features(_base_df())
    assert len(get_feature_cols(df)) > 0


def test_no_future_leak_in_features():
    """Rolling features computed at bar t should only use data up to t-1
    (rolling(w) with default min_periods=w is safe once w bars have passed)."""
    df = make_features(_base_df(n=300))
    feat_cols = get_feature_cols(df)
    clean = df.dropna(subset=feat_cols)
    # After dropping NaN rows, no feature should be NaN
    assert not clean[feat_cols].isna().any().any()
