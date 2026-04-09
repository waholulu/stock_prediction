from __future__ import annotations

from typing import Generator, Tuple

import numpy as np
import pandas as pd

from src.config import SplitSpec


def walk_forward_splits(
    dates: pd.Series,
    spec: SplitSpec,
) -> Generator[Tuple[np.ndarray, np.ndarray, int], None, None]:
    """Generate walk-forward train/test splits with embargo.

    Yields (train_idx, test_idx, embargo_end_idx) tuples.
    """
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    n = len(dates)
    train_days = int(spec.train_years * 252)
    test_days = int(spec.test_months * 21)
    embargo = int(spec.embargo_days)

    start = train_days
    while start + test_days <= n:
        train_start = start - train_days
        test_start = start
        test_end = min(start + test_days, n)
        embargo_end = min(n, test_end + embargo)

        train_idx = np.arange(train_start, test_start)
        test_idx = np.arange(test_start, test_end)

        yield train_idx, test_idx, embargo_end
        start = test_end


def apply_purge_by_label_end(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    dates: pd.Series,
    label_end_times: pd.Series,
) -> np.ndarray:
    """Purge training samples whose label horizon overlaps the test interval.

    label_end_times: per-row end index (numeric) for the label.
    """
    dates = pd.to_datetime(pd.Series(dates))
    test_start_date = dates.iloc[test_idx[0]]

    le = label_end_times.iloc[train_idx]
    # Convert numeric indices to dates
    if np.issubdtype(le.dropna().dtype, np.number):
        valid = le.dropna().astype(int).clip(0, len(dates) - 1)
        le_dates = dates.iloc[valid.values]
        le_dates = le_dates.reindex(train_idx)
    else:
        le_dates = le

    keep = (le_dates < test_start_date) | le_dates.isna()
    return train_idx[keep.values]
