from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score, mean_absolute_error


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute accuracy, MCC, and optionally AUC."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    return metrics


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute MAE, RMSE, and directional accuracy."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    dir_acc = float(np.mean((y_true > 0) == (y_pred > 0))) if len(y_true) > 0 else 0.0
    return {"mae": mae, "rmse": rmse, "directional_accuracy": dir_acc}


def compute_sharpe(returns: np.ndarray, annualize: int = 252) -> float:
    """Annualized Sharpe ratio."""
    returns = np.asarray(returns)
    returns = returns[~np.isnan(returns)]
    if len(returns) == 0 or np.std(returns) < 1e-12:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(annualize))


def compute_max_drawdown(equity_curve: np.ndarray) -> float:
    """Maximum drawdown from an equity curve."""
    equity_curve = np.asarray(equity_curve)
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / (peak + 1e-12)
    return float(np.min(drawdown))
