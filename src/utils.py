from __future__ import annotations

import json
import logging
import os
import random

import numpy as np
import pandas as pd


def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("stock_prediction")
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dirs(results_dir: str = "results") -> None:
    for subdir in ["metrics", "predictions", "plots", "models"]:
        os.makedirs(os.path.join(results_dir, subdir), exist_ok=True)


def save_metrics(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def save_json(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
