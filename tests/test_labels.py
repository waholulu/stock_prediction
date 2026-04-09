"""Tests for src/labels.py"""
import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import generate_synthetic_ohlcv, add_basic_returns
from src.labels import label_next_day_direction, label_k_day_return, triple_barrier_labels


def _base_df(n=300):
    df = generate_synthetic_ohlcv(n_days=n)
    return add_basic_returns(df)


def test_next_day_direction_values():
    df = _base_df()
    y = label_next_day_direction(df)
    valid = y.dropna()
    assert set(valid.unique()).issubset({0.0, 1.0}), "Labels must be 0 or 1"


def test_next_day_direction_last_nan():
    df = _base_df()
    y = label_next_day_direction(df)
    assert np.isnan(y.iloc[-1]), "Last label must be NaN (no future return)"


def test_next_day_direction_length():
    df = _base_df(n=100)
    y = label_next_day_direction(df)
    assert len(y) == len(df)


def test_next_day_direction_missing_col():
    df = generate_synthetic_ohlcv(n_days=50)
    with pytest.raises(ValueError):
        label_next_day_direction(df, ret_col="ret_1d")


def test_k_day_return_length():
    df = _base_df()
    y = label_k_day_return(df, k=5)
    assert len(y) == len(df)


def test_k_day_return_last_k_nan():
    df = _base_df(n=100)
    k = 5
    y = label_k_day_return(df, k=k)
    assert y.iloc[-k:].isna().all(), f"Last {k} labels must be NaN"


def test_k_day_return_missing_col():
    df = generate_synthetic_ohlcv(n_days=50)
    with pytest.raises(ValueError):
        label_k_day_return(df, price_col="nonexistent_col")


def test_triple_barrier_labels_values():
    df = _base_df()
    labels, t_end = triple_barrier_labels(df)
    valid = labels.dropna()
    assert set(valid.unique()).issubset({-1.0, 0.0, 1.0})


def test_triple_barrier_labels_length():
    df = _base_df(n=200)
    labels, t_end = triple_barrier_labels(df)
    assert len(labels) == len(df)
    assert len(t_end) == len(df)


def test_triple_barrier_t_end_in_range():
    df = _base_df(n=200)
    labels, t_end = triple_barrier_labels(df, horizon=10)
    valid_end = t_end.dropna()
    assert (valid_end >= 0).all()
    assert (valid_end < len(df)).all()


def test_triple_barrier_symmetric_barriers():
    """With equal pt/sl multipliers on a GBM, +1/-1 counts should be roughly similar."""
    df = generate_synthetic_ohlcv(n_days=2000, annual_drift=0.0)
    df = add_basic_returns(df)
    labels, _ = triple_barrier_labels(df, horizon=10, pt_multiplier=1.0, sl_multiplier=1.0)
    valid = labels.dropna()
    n_up = (valid == 1).sum()
    n_dn = (valid == -1).sum()
    # Ratio should be within 30/70 range (very loose — synthetic data)
    ratio = n_up / (n_up + n_dn + 1e-9)
    assert 0.30 < ratio < 0.70, f"Unexpected label imbalance: {n_up} up vs {n_dn} down"
