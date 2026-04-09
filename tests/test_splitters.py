import numpy as np
import pandas as pd

from src.config import SplitSpec
from src.splitters import walk_forward_splits


def test_train_before_test(synthetic_ohlcv):
    dates = synthetic_ohlcv["date"]
    spec = SplitSpec(train_years=1.0, test_months=3, embargo_days=3)
    for train_idx, test_idx, _ in walk_forward_splits(dates, spec):
        assert train_idx[-1] < test_idx[0], "Train must precede test"


def test_embargo_gap(synthetic_ohlcv):
    dates = synthetic_ohlcv["date"]
    spec = SplitSpec(train_years=1.0, test_months=3, embargo_days=5)
    splits = list(walk_forward_splits(dates, spec))
    assert len(splits) > 0, "Should produce at least one split"
    for train_idx, test_idx, embargo_end in splits:
        assert embargo_end >= test_idx[-1]


def test_no_overlap_between_folds(synthetic_ohlcv):
    dates = synthetic_ohlcv["date"]
    spec = SplitSpec(train_years=1.0, test_months=3, embargo_days=3)
    splits = list(walk_forward_splits(dates, spec))
    for i in range(1, len(splits)):
        prev_test_end = splits[i - 1][1][-1]
        curr_test_start = splits[i][1][0]
        assert curr_test_start > prev_test_end, "Test folds must not overlap"
