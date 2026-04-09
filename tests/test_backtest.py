import numpy as np
import pandas as pd

from src.backtest import backtest_daily_direction


def test_zero_cost(synthetic_ohlcv):
    df = synthetic_ohlcv.copy()
    df["ret_1d"] = np.random.randn(len(df)) * 0.01
    df["signal"] = 1  # always long
    out, _ = backtest_daily_direction(df, "signal", cost_bps=0)
    np.testing.assert_array_almost_equal(
        out["net_ret"].dropna().values,
        out["strat_ret"].dropna().values,
    )


def test_flat_signal_flat_equity(synthetic_ohlcv):
    df = synthetic_ohlcv.copy()
    df["ret_1d"] = np.random.randn(len(df)) * 0.01
    df["signal"] = 0
    out, sharpe = backtest_daily_direction(df, "signal", cost_bps=2)
    assert abs(out["equity"].iloc[-1] - 1.0) < 1e-10


def test_cost_applied_on_flip(synthetic_ohlcv):
    df = synthetic_ohlcv.head(10).copy()
    df["ret_1d"] = 0.01
    df["signal"] = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
    out, _ = backtest_daily_direction(df, "signal", cost_bps=10)
    # Every row after the first has a position change, so cost should be applied
    cost_rows = out["cost"].iloc[1:]  # first row has NaN diff
    assert (cost_rows.iloc[1:] > 0).all()
