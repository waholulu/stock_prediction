import numpy as np

from src.data import add_basic_returns
from src.features import make_features, get_feature_cols


def test_make_features_no_nan_after_drop(synthetic_ohlcv):
    df = add_basic_returns(synthetic_ohlcv).dropna().reset_index(drop=True)
    df_feat = make_features(df).dropna().reset_index(drop=True)
    assert len(df_feat) > 0
    feat_cols = get_feature_cols(df_feat)
    for col in feat_cols:
        assert df_feat[col].isna().sum() == 0, f"NaN in {col}"


def test_feature_count(synthetic_ohlcv):
    df = add_basic_returns(synthetic_ohlcv).dropna().reset_index(drop=True)
    windows = (5, 10, 21, 63)
    df_feat = make_features(df, windows=windows).dropna().reset_index(drop=True)
    feat_cols = get_feature_cols(df_feat)
    expected = len(windows) * 5 + 2  # 5 rolling features per window + dow + month
    assert len(feat_cols) == expected


def test_rolling_uses_past_only(synthetic_ohlcv):
    """Rolling features at row i should only depend on rows <= i."""
    df = add_basic_returns(synthetic_ohlcv).dropna().reset_index(drop=True)
    df_feat = make_features(df, windows=(5,))
    # The ret_mean_5 at index 10 should equal mean of ret_1d[6:11]
    expected = df["ret_1d"].iloc[6:11].mean()
    actual = df_feat["ret_mean_5"].iloc[10]
    assert abs(actual - expected) < 1e-10
