"""
LightGBM baseline for walk-forward stock prediction.

Provides:
- LGBMClassifier wrapper: next-day direction / triple-barrier ternary labels
- LGBMRegressor wrapper: k-day forward return
- walk_forward_evaluate(): run full walk-forward evaluation and collect metrics

Metrics collected per fold:
- Classification: accuracy, MCC, ROC-AUC
- Regression: MAE, RMSE, directional accuracy (sign of predicted vs actual return)
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    mean_absolute_error,
    roc_auc_score,
)

from .evaluation import WalkForwardSpec, apply_purge, walk_forward_splits


_LGBM_CLF_PARAMS = dict(
    objective="binary",
    learning_rate=0.03,
    num_leaves=31,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=1,
    min_data_in_leaf=50,
    metric="binary_logloss",
    verbosity=-1,
    n_estimators=200,
)

_LGBM_REG_PARAMS = dict(
    objective="regression",
    learning_rate=0.03,
    num_leaves=31,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=1,
    min_data_in_leaf=50,
    metric="rmse",
    verbosity=-1,
    n_estimators=200,
)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def walk_forward_evaluate(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    task: Literal["classify_binary", "classify_ternary", "regress"] = "classify_binary",
    spec: WalkForwardSpec | None = None,
    purge: bool = False,
) -> pd.DataFrame:
    """Run walk-forward evaluation with LightGBM.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``date``, all ``feature_cols``, ``label_col``, and
        optionally ``t_end`` (needed when ``purge=True``).
    feature_cols : list[str]
        Predictor column names.
    label_col : str
        Target column (e.g. ``"y_dir"``, ``"y_5d"``, ``"y_tb"``).
    task : str
        ``"classify_binary"``  — binary 0/1 classification (next-day direction).
        ``"classify_ternary"`` — ternary -1/0/+1 (triple-barrier).
        ``"regress"``          — continuous target (k-day return).
    spec : WalkForwardSpec, optional
        Walk-forward configuration.
    purge : bool
        If True, apply purging based on ``df["t_end"]`` (for triple-barrier
        labels).  Requires ``t_end`` column in ``df``.

    Returns
    -------
    pd.DataFrame
        One row per fold with fold number and relevant metrics.
    """
    if spec is None:
        spec = WalkForwardSpec()

    import lightgbm as lgb

    # Drop rows where the label or any feature is NaN
    subset = feature_cols + [label_col]
    df_clean = df.dropna(subset=subset).reset_index(drop=True)

    X = df_clean[feature_cols].values.astype(np.float32)
    dates = df_clean["date"]
    t_end = df_clean["t_end"] if "t_end" in df_clean.columns else pd.Series(
        [np.nan] * len(df_clean)
    )

    fold_metrics: list[dict] = []

    for fold_i, (train_idx, test_idx, _) in enumerate(
        walk_forward_splits(dates, spec), start=1
    ):
        if purge and "t_end" in df_clean.columns:
            train_idx = apply_purge(train_idx, test_idx, dates, t_end)

        if len(train_idx) < 10 or len(test_idx) < 1:
            continue

        if task == "classify_binary":
            y = df_clean[label_col].values.astype(int)
            y_train = y[train_idx]
            y_test = y[test_idx]

            # Skip fold if only one class present in training data
            if len(np.unique(y_train)) < 2:
                continue

            model = lgb.LGBMClassifier(**_LGBM_CLF_PARAMS)
            X_train_df = pd.DataFrame(X[train_idx], columns=feature_cols)
            X_test_df = pd.DataFrame(X[test_idx], columns=feature_cols)
            model.fit(X_train_df, y_train)
            prob = model.predict_proba(X_test_df)[:, 1]
            pred = (prob >= 0.5).astype(int)

            metrics = {
                "fold": fold_i,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "accuracy": accuracy_score(y_test, pred),
                "mcc": matthews_corrcoef(y_test, pred),
                "auc": roc_auc_score(y_test, prob),
            }

        elif task == "classify_ternary":
            y_raw = df_clean[label_col].values.astype(float)
            # Map {-1, 0, +1} → {0, 1, 2} for LightGBM multiclass
            label_map = {-1.0: 0, 0.0: 1, 1.0: 2}
            y_mapped = np.array([label_map.get(v, 1) for v in y_raw])
            y_train = y_mapped[train_idx]
            y_test = y_mapped[test_idx]

            if len(np.unique(y_train)) < 2:
                continue

            params = _LGBM_CLF_PARAMS.copy()
            params["objective"] = "multiclass"
            params["num_class"] = 3
            params["metric"] = "multi_logloss"
            model = lgb.LGBMClassifier(**params)
            X_train_df = pd.DataFrame(X[train_idx], columns=feature_cols)
            X_test_df = pd.DataFrame(X[test_idx], columns=feature_cols)
            model.fit(X_train_df, y_train)
            pred = model.predict(X_test_df)

            metrics = {
                "fold": fold_i,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "accuracy": accuracy_score(y_test, pred),
                "mcc": matthews_corrcoef(y_test, pred),
                "auc": float("nan"),  # multiclass AUC is less meaningful here
            }

        else:  # regress
            y = df_clean[label_col].values.astype(np.float32)
            y_train = y[train_idx]
            y_test = y[test_idx]

            model = lgb.LGBMRegressor(**_LGBM_REG_PARAMS)
            X_train_df = pd.DataFrame(X[train_idx], columns=feature_cols)
            X_test_df = pd.DataFrame(X[test_idx], columns=feature_cols)
            model.fit(X_train_df, y_train)
            pred = model.predict(X_test_df)

            dir_acc = accuracy_score(
                (y_test > 0).astype(int), (pred > 0).astype(int)
            )
            metrics = {
                "fold": fold_i,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "mae": mean_absolute_error(y_test, pred),
                "rmse": _rmse(y_test, pred),
                "dir_acc": dir_acc,
            }

        fold_metrics.append(metrics)

    return pd.DataFrame(fold_metrics)
