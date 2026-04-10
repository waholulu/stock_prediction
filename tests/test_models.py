"""Tests for src/models.py"""
import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import generate_synthetic_ohlcv, add_basic_returns
from src.features import make_features, get_feature_cols
from src.labels import label_next_day_direction, label_k_day_return, triple_barrier_labels
from src.evaluation import WalkForwardSpec
from src.models import walk_forward_evaluate


def _prepared_df(n=1500):
    df = generate_synthetic_ohlcv(n_days=n, seed=7)
    df = add_basic_returns(df)
    df = make_features(df)
    df["y_dir"] = label_next_day_direction(df)
    df["y_5d"] = label_k_day_return(df, k=5)
    y_tb, t_end = triple_barrier_labels(df)
    df["y_tb"] = y_tb
    df["t_end"] = t_end
    return df


_SPEC = WalkForwardSpec(train_years=1, test_months=6, embargo_days=5)


def test_binary_returns_dataframe():
    df = _prepared_df()
    feat = get_feature_cols(df)
    result = walk_forward_evaluate(df, feat, "y_dir", task="classify_binary", spec=_SPEC)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_binary_has_expected_columns():
    df = _prepared_df()
    feat = get_feature_cols(df)
    result = walk_forward_evaluate(df, feat, "y_dir", task="classify_binary", spec=_SPEC)
    for col in ("fold", "accuracy", "mcc", "auc"):
        assert col in result.columns


def test_binary_metrics_in_range():
    df = _prepared_df()
    feat = get_feature_cols(df)
    result = walk_forward_evaluate(df, feat, "y_dir", task="classify_binary", spec=_SPEC)
    assert result["accuracy"].between(0, 1).all()
    assert result["auc"].between(0, 1).all()
    assert result["mcc"].between(-1, 1).all()


def test_regression_returns_dataframe():
    df = _prepared_df()
    feat = get_feature_cols(df)
    result = walk_forward_evaluate(df, feat, "y_5d", task="regress", spec=_SPEC)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_regression_has_expected_columns():
    df = _prepared_df()
    feat = get_feature_cols(df)
    result = walk_forward_evaluate(df, feat, "y_5d", task="regress", spec=_SPEC)
    for col in ("fold", "mae", "rmse", "dir_acc"):
        assert col in result.columns


def test_regression_metrics_positive():
    df = _prepared_df()
    feat = get_feature_cols(df)
    result = walk_forward_evaluate(df, feat, "y_5d", task="regress", spec=_SPEC)
    assert (result["mae"] >= 0).all()
    assert (result["rmse"] >= 0).all()
    assert result["dir_acc"].between(0, 1).all()


def test_ternary_returns_dataframe():
    df = _prepared_df()
    feat = get_feature_cols(df)
    result = walk_forward_evaluate(df, feat, "y_tb", task="classify_ternary", spec=_SPEC)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_purge_runs_without_error():
    df = _prepared_df()
    feat = get_feature_cols(df)
    result = walk_forward_evaluate(
        df, feat, "y_tb", task="classify_ternary", spec=_SPEC, purge=True
    )
    assert isinstance(result, pd.DataFrame)
