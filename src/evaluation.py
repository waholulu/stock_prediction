"""
Bias-aware walk-forward evaluation for time-series prediction.

Key concepts:
- Walk-forward (expanding or rolling) train/test splits with strict temporal ordering.
- Embargo: a gap between the end of the test fold and the start of the next
  training fold, preventing leakage via auto-correlation.
- Purging: remove training samples whose *label horizon* overlaps the test
  interval (important for triple-barrier labels whose exit may be during test).

Classes / functions:
- WalkForwardSpec   : configuration dataclass
- walk_forward_splits: generator yielding (train_idx, test_idx, embargo_end)
- apply_purge       : filter training indices by label end-time vs. test start
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class WalkForwardSpec:
    """Configuration for walk-forward CV.

    Attributes
    ----------
    train_years : float
        Length of the training window in calendar years (approx 252 trading
        days per year).
    test_months : float
        Length of each test fold in calendar months (approx 21 trading days).
    embargo_days : int
        Number of trading days to drop between the end of a test fold and the
        start of the *next* training window.  This prevents short-term
        auto-correlation from leaking test information into training.
    expanding : bool
        If True, use an expanding (growing) training window instead of a
        fixed-length rolling window.  Expanding tends to give more stable
        models but may include very stale data.
    """

    train_years: float = 5.0
    test_months: float = 6.0
    embargo_days: int = 5
    expanding: bool = False


def walk_forward_splits(
    dates: pd.Series | list,
    spec: WalkForwardSpec | None = None,
) -> "Generator[tuple[np.ndarray, np.ndarray, int]]":
    """Generate walk-forward (train, test) index pairs.

    Parameters
    ----------
    dates : array-like of datetime-like
        Ordered dates for the full dataset.  Must be monotonically increasing.
    spec : WalkForwardSpec, optional
        Split configuration. Defaults to WalkForwardSpec().

    Yields
    ------
    train_idx : np.ndarray of int
        Positional indices for the training fold.
    test_idx : np.ndarray of int
        Positional indices for the test fold.
    embargo_end : int
        Positional index *after* which the next fold begins (embargo applied).
    """
    if spec is None:
        spec = WalkForwardSpec()

    dates = pd.to_datetime(pd.Series(dates, dtype="object")).reset_index(drop=True)
    n = len(dates)
    train_days = max(1, int(spec.train_years * 252))
    test_days = max(1, int(spec.test_months * 21))
    embargo = max(0, int(spec.embargo_days))

    start = train_days  # first possible test start

    while start + test_days <= n:
        test_start = start
        test_end = start + test_days  # exclusive

        if spec.expanding:
            train_start = 0
        else:
            train_start = max(0, test_start - train_days)

        train_idx = np.arange(train_start, test_start)
        test_idx = np.arange(test_start, test_end)
        embargo_end = min(n, test_end + embargo)

        if len(train_idx) == 0 or len(test_idx) == 0:
            start = test_end
            continue

        yield train_idx, test_idx, embargo_end
        start = test_end  # advance by one test block (non-overlapping)


def apply_purge(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    dates: pd.Series,
    label_end_times: pd.Series,
) -> np.ndarray:
    """Remove training samples whose label window overlaps the test interval.

    This prevents the case where a triple-barrier label starting in the
    training period has an *exit bar* that falls inside the test period,
    which would introduce look-ahead bias.

    Parameters
    ----------
    train_idx : np.ndarray of int
        Training positional indices (before purging).
    test_idx : np.ndarray of int
        Test positional indices.
    dates : pd.Series of datetime
        Full date series (indexed 0..n-1).
    label_end_times : pd.Series
        Per-row integer position (or NaN) of the bar at which the label is
        finalised (e.g. ``t_end`` from ``triple_barrier_labels``).

    Returns
    -------
    np.ndarray of int
        Purged training indices.
    """
    dates = pd.to_datetime(pd.Series(dates, dtype="object"))
    test_start_date = dates.iloc[test_idx[0]]

    # label_end_times may be fractional floats from NaN propagation
    le = label_end_times.iloc[train_idx]
    valid_mask = le.notna()
    end_positions = le[valid_mask].astype(int).clip(0, len(dates) - 1)
    end_dates = dates.iloc[end_positions.values]
    end_dates.index = le[valid_mask].index

    # Build keep mask over train_idx
    keep = np.ones(len(train_idx), dtype=bool)
    valid_positions = np.where(valid_mask.values)[0]
    leak_mask = end_dates.values >= test_start_date.to_datetime64()
    keep[valid_positions[leak_mask]] = False

    return train_idx[keep]
