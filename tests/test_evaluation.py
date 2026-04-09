"""Tests for src/evaluation.py"""
import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import generate_synthetic_ohlcv, add_basic_returns
from src.evaluation import WalkForwardSpec, walk_forward_splits, apply_purge


def _dates(n=1000):
    df = generate_synthetic_ohlcv(n_days=n)
    return df["date"]


def test_no_train_test_overlap():
    dates = _dates(1000)
    spec = WalkForwardSpec(train_years=2, test_months=6, embargo_days=5)
    for train_idx, test_idx, _ in walk_forward_splits(dates, spec):
        assert len(set(train_idx) & set(test_idx)) == 0, "Train and test must not overlap"


def test_train_before_test():
    dates = _dates(1000)
    spec = WalkForwardSpec(train_years=2, test_months=6, embargo_days=5)
    for train_idx, test_idx, _ in walk_forward_splits(dates, spec):
        assert train_idx[-1] < test_idx[0], "All train indices must precede test indices"


def test_non_overlapping_test_folds():
    dates = _dates(2000)
    spec = WalkForwardSpec(train_years=2, test_months=6, embargo_days=5)
    splits = list(walk_forward_splits(dates, spec))
    for i in range(len(splits) - 1):
        _, test_i, _ = splits[i]
        _, test_j, _ = splits[i + 1]
        assert len(set(test_i) & set(test_j)) == 0, "Test folds must not overlap"


def test_embargo_respected():
    """embargo_end should be test_end + embargo_days (clamped to n)."""
    dates = _dates(1000)
    embargo_days = 10
    spec = WalkForwardSpec(train_years=2, test_months=3, embargo_days=embargo_days)
    splits = list(walk_forward_splits(dates, spec))
    n = len(dates)
    for _, test_idx, embargo_end in splits:
        test_end = test_idx[-1] + 1  # exclusive end
        expected = min(n, test_end + embargo_days)
        assert embargo_end == expected, (
            f"embargo_end={embargo_end} expected {expected}"
        )


def test_yields_tuples():
    dates = _dates(600)
    spec = WalkForwardSpec(train_years=1, test_months=3, embargo_days=2)
    for result in walk_forward_splits(dates, spec):
        assert len(result) == 3


def test_minimum_folds():
    dates = _dates(2000)
    spec = WalkForwardSpec(train_years=2, test_months=6, embargo_days=5)
    splits = list(walk_forward_splits(dates, spec))
    assert len(splits) >= 3, "Should produce multiple folds with 2000 days"


def test_expanding_window():
    dates = _dates(1500)
    spec = WalkForwardSpec(train_years=2, test_months=6, expanding=True)
    splits = list(walk_forward_splits(dates, spec))
    # With expanding window, train_idx[0] should always be 0
    for train_idx, _, _ in splits:
        assert train_idx[0] == 0


def test_apply_purge_removes_leaking_rows():
    dates = generate_synthetic_ohlcv(n_days=500)["date"]
    dates = pd.Series(dates).reset_index(drop=True)
    train_idx = np.arange(0, 200)
    test_idx = np.arange(200, 250)
    # Simulate some label_end_times that extend into the test period
    t_end = pd.Series(np.arange(500, dtype=float))
    t_end[195:200] = 210.0  # these overlap the test window
    purged = apply_purge(train_idx, test_idx, dates, t_end)
    assert len(purged) < len(train_idx), "Purge should remove some rows"
    # None of the purged indices should have t_end >= test_start
    for idx in purged:
        te = t_end.iloc[idx]
        if not np.isnan(te):
            assert int(te) < test_idx[0]
