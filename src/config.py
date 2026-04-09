from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import yaml


@dataclass
class DataConfig:
    symbol: str = "SPY"
    start: str = "2000-01-01"
    end: Optional[str] = None


@dataclass
class FeatureConfig:
    windows: List[int] = field(default_factory=lambda: [5, 10, 21, 63])


@dataclass
class TripleBarrierConfig:
    horizon: int = 10
    pt_multiplier: float = 1.0
    sl_multiplier: float = 1.0
    vol_lookback: int = 20


@dataclass
class KDayReturnConfig:
    k: int = 5


@dataclass
class LabelConfig:
    triple_barrier: TripleBarrierConfig = field(default_factory=TripleBarrierConfig)
    k_day_return: KDayReturnConfig = field(default_factory=KDayReturnConfig)


@dataclass
class SplitSpec:
    train_years: float = 5.0
    test_months: float = 6.0
    embargo_days: int = 5


@dataclass
class EvalConfig:
    train_years: float = 5.0
    test_months: float = 6.0
    embargo_days: int = 5
    purge_triple_barrier: bool = True

    def to_split_spec(self) -> SplitSpec:
        return SplitSpec(
            train_years=self.train_years,
            test_months=self.test_months,
            embargo_days=self.embargo_days,
        )


@dataclass
class LightGBMConfig:
    objective: str = "binary"
    learning_rate: float = 0.03
    num_leaves: int = 31
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.9
    bagging_freq: int = 1
    min_data_in_leaf: int = 50
    num_boost_round: int = 200
    metric: str = "binary_logloss"
    verbosity: int = -1

    def to_params(self) -> dict:
        return {
            "objective": self.objective,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "min_data_in_leaf": self.min_data_in_leaf,
            "metric": self.metric,
            "verbosity": self.verbosity,
        }


@dataclass
class PatchTSTConfig:
    h: int = 5
    input_size: int = 64
    max_steps: int = 500
    freq: str = "B"


@dataclass
class TimesFMConfig:
    max_context: int = 512
    max_horizon: int = 64
    forecast_horizon: int = 5
    normalize_inputs: bool = True


@dataclass
class ModelsConfig:
    lightgbm: LightGBMConfig = field(default_factory=LightGBMConfig)
    patchtst: PatchTSTConfig = field(default_factory=PatchTSTConfig)
    timesfm: TimesFMConfig = field(default_factory=TimesFMConfig)


@dataclass
class BacktestConfig:
    cost_bps: float = 2.0
    cost_bps_sensitivity: List[float] = field(default_factory=lambda: [0, 1, 2, 5, 10])


@dataclass
class OutputConfig:
    results_dir: str = "results"

    @property
    def metrics_dir(self) -> str:
        return os.path.join(self.results_dir, "metrics")

    @property
    def predictions_dir(self) -> str:
        return os.path.join(self.results_dir, "predictions")

    @property
    def plots_dir(self) -> str:
        return os.path.join(self.results_dir, "plots")

    @property
    def models_dir(self) -> str:
        return os.path.join(self.results_dir, "models")


@dataclass
class ProjectConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    seed: int = 42


def _build_dataclass(cls, raw: dict):
    """Recursively build a dataclass from a dict, ignoring unknown keys."""
    if raw is None:
        return cls()
    import dataclasses

    fieldnames = {f.name for f in dataclasses.fields(cls)}
    filtered = {}
    for k, v in raw.items():
        if k not in fieldnames:
            continue
        f = next(f for f in dataclasses.fields(cls) if f.name == k)
        if dataclasses.is_dataclass(f.type if isinstance(f.type, type) else None):
            filtered[k] = _build_dataclass(f.type, v)
        else:
            filtered[k] = v
    return cls(**filtered)


def load_config(path: str = "config/default.yaml") -> ProjectConfig:
    """Load YAML config and return a ProjectConfig dataclass."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    data = _build_dataclass(DataConfig, raw.get("data"))
    features = _build_dataclass(FeatureConfig, raw.get("features"))

    labels_raw = raw.get("labels", {})
    labels = LabelConfig(
        triple_barrier=_build_dataclass(TripleBarrierConfig, labels_raw.get("triple_barrier")),
        k_day_return=_build_dataclass(KDayReturnConfig, labels_raw.get("k_day_return")),
    )

    evaluation = _build_dataclass(EvalConfig, raw.get("evaluation"))

    models_raw = raw.get("models", {})
    models = ModelsConfig(
        lightgbm=_build_dataclass(LightGBMConfig, models_raw.get("lightgbm")),
        patchtst=_build_dataclass(PatchTSTConfig, models_raw.get("patchtst")),
        timesfm=_build_dataclass(TimesFMConfig, models_raw.get("timesfm")),
    )

    backtest = _build_dataclass(BacktestConfig, raw.get("backtest"))
    output = _build_dataclass(OutputConfig, raw.get("output"))
    seed = raw.get("seed", 42)

    return ProjectConfig(
        data=data,
        features=features,
        labels=labels,
        evaluation=evaluation,
        models=models,
        backtest=backtest,
        output=output,
        seed=seed,
    )
