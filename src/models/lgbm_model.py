from __future__ import annotations

from typing import List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import LightGBMConfig, SplitSpec
from src.metrics import compute_classification_metrics
from src.splitters import apply_purge_by_label_end, walk_forward_splits


def eval_lightgbm_walkforward(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = "y_dir",
    spec: SplitSpec = SplitSpec(),
    lgbm_config: LightGBMConfig = LightGBMConfig(),
    purge: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run LightGBM with walk-forward evaluation.

    Returns:
        fold_metrics: DataFrame with per-fold accuracy, MCC, AUC
        oos_predictions: DataFrame with date, y_true, y_pred, y_prob for all OOS samples
    """
    X = df[feature_cols].values
    y = df[label_col].values.astype(int)
    dates = df["date"]

    fold_metrics = []
    oos_rows = []

    for i, (train_idx, test_idx, embargo_end) in enumerate(
        walk_forward_splits(dates, spec), start=1
    ):
        if purge and label_col == "y_tb" and "t_end" in df.columns:
            train_idx = apply_purge_by_label_end(
                train_idx, test_idx, dates, df["t_end"]
            )

        dtrain = lgb.Dataset(X[train_idx], label=y[train_idx])
        params = lgbm_config.to_params()
        model = lgb.train(params, dtrain, num_boost_round=lgbm_config.num_boost_round)

        prob = model.predict(X[test_idx])
        pred = (prob >= 0.5).astype(int)

        metrics = compute_classification_metrics(y[test_idx], pred, prob)
        metrics["fold"] = i
        fold_metrics.append(metrics)

        for j, idx in enumerate(test_idx):
            oos_rows.append({
                "date": dates.iloc[idx],
                "y_true": int(y[idx]),
                "y_pred": int(pred[j]),
                "y_prob": float(prob[j]),
            })

    return pd.DataFrame(fold_metrics), pd.DataFrame(oos_rows)
