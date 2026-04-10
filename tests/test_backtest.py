"""Tests for src/backtest.py"""
import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import generate_synthetic_ohlcv, add_basic_returns
from src.backtest import backtest_daily_direction, summarise_backtest


def _base_df(n=500):
    df = generate_synthetic_ohlcv(n_days=n)
    return add_basic_returns(df)


def test_backtest_adds_columns():
    df = _base_df()
    df["signal"] = 1  # always long
    bt = backtest_daily_direction(df, signal_col="signal")
    for col in ("pos", "strat_ret", "cost", "net_ret", "equity", "bh_equity"):
        assert col in bt.columns, f"Missing column: {col}"


def test_backtest_equity_starts_near_one():
    df = _base_df()
    df["signal"] = 1
    bt = backtest_daily_direction(df, signal_col="signal")
    # First equity value: position was 0 on bar 0 (no prior position)
    # so first strat_ret is NaN, equity starts at 1.0
    assert abs(bt["equity"].iloc[0] - 1.0) < 1e-9


def test_backtest_zero_cost_equals_gross():
    df = _base_df(n=200)
    df["signal"] = np.where(df["ret_1d"].shift(1).fillna(0) > 0, 1, -1)
    bt = backtest_daily_direction(df, signal_col="signal", cost_bps=0.0)
    # With zero cost, net_ret should equal strat_ret (no NaN rows)
    mask = bt["strat_ret"].notna() & bt["net_ret"].notna()
    np.testing.assert_allclose(
        bt.loc[mask, "net_ret"].values,
        bt.loc[mask, "strat_ret"].values,
        atol=1e-10,
    )


def test_backtest_missing_signal_col():
    df = _base_df()
    with pytest.raises(ValueError):
        backtest_daily_direction(df, signal_col="nonexistent")


def test_backtest_missing_ret_col():
    df = _base_df()
    df["signal"] = 1
    with pytest.raises(ValueError):
        backtest_daily_direction(df, signal_col="signal", ret_col="nonexistent")


def test_backtest_flat_signal():
    df = _base_df(n=200)
    df["signal"] = 0  # flat — no returns, only costs at entry
    bt = backtest_daily_direction(df, signal_col="signal")
    # With a flat signal, strat_ret should be 0 everywhere (pos=0 always)
    assert (bt["strat_ret"].fillna(0) == 0).all()


def test_summarise_keys():
    df = _base_df(n=300)
    df["signal"] = 1
    bt = backtest_daily_direction(df, signal_col="signal")
    summary = summarise_backtest(bt)
    for key in ("total_net_return", "annual_sharpe", "max_drawdown", "win_rate", "n_trades"):
        assert key in summary, f"Missing key: {key}"


def test_summarise_max_drawdown_nonpositive():
    df = _base_df(n=500)
    df["signal"] = 1
    bt = backtest_daily_direction(df, signal_col="signal")
    summary = summarise_backtest(bt)
    assert summary["max_drawdown"] <= 0


def test_summarise_n_trades_positive():
    df = _base_df(n=200)
    # Alternating signal generates many trades
    df["signal"] = np.tile([1, -1], len(df) // 2 + 1)[: len(df)]
    bt = backtest_daily_direction(df, signal_col="signal")
    summary = summarise_backtest(bt)
    assert summary["n_trades"] > 0


def test_buy_and_hold_equity_matches_underlying():
    df = _base_df(n=300)
    df["signal"] = 1
    bt = backtest_daily_direction(df, signal_col="signal")
    # bh_equity = cumulative exp(sum of ret_1d)
    expected_bh = np.exp(df["ret_1d"].fillna(0).cumsum().values)
    np.testing.assert_allclose(bt["bh_equity"].values, expected_bh, rtol=1e-6)
