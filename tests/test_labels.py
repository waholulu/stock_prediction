import numpy as np

from src.data import add_basic_returns
from src.labels import label_next_day_direction, label_k_day_return, triple_barrier_labels


def test_direction_label_values(synthetic_ohlcv):
    df = add_basic_returns(synthetic_ohlcv).dropna().reset_index(drop=True)
    y = label_next_day_direction(df)
    valid = y.dropna()
    assert set(valid.unique()).issubset({0, 1})


def test_k_day_return_is_float(synthetic_ohlcv):
    df = add_basic_returns(synthetic_ohlcv).dropna().reset_index(drop=True)
    y = label_k_day_return(df, k=5)
    valid = y.dropna()
    assert len(valid) > 0
    assert valid.dtype == np.float64


def test_triple_barrier_values(synthetic_ohlcv):
    df = add_basic_returns(synthetic_ohlcv).dropna().reset_index(drop=True)
    labels, t_end = triple_barrier_labels(df, horizon=10, pt=1.0, sl=1.0)
    valid = labels.dropna()
    assert set(valid.unique()).issubset({-1, 0, 1})


def test_triple_barrier_t_end_not_past(synthetic_ohlcv):
    df = add_basic_returns(synthetic_ohlcv).dropna().reset_index(drop=True)
    labels, t_end = triple_barrier_labels(df, horizon=10)
    valid_mask = ~t_end.isna()
    indices = np.arange(len(df))
    assert (t_end[valid_mask].values >= indices[valid_mask]).all()
