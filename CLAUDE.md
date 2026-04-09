# CLAUDE.md

## Project Overview

Stock prediction project using daily OHLCV data. Implements walk-forward evaluation with embargo/purge, LightGBM baseline, and (planned) PatchTST + TimesFM models.

## Quick Reference

```bash
# Install
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run pipeline (when implemented)
python -m src.pipeline --config config/default.yaml --phase all
```

## Project Structure

```
config/default.yaml    — All hyperparameters (centralized, do not hardcode magic numbers)
src/config.py          — Dataclass config loader (YAML -> Python)
src/data.py            — OHLCV download (yfinance) + basic returns
src/features.py        — Rolling feature engineering
src/labels.py          — 3 label types: direction, k-day return, triple-barrier
src/splitters.py       — Walk-forward splits with embargo + purge
src/metrics.py         — Classification/regression metrics, Sharpe, max drawdown
src/backtest.py        — Transaction-cost-aware daily backtest
src/models/lgbm_model.py      — LightGBM walk-forward wrapper
src/models/patchtst_model.py  — (planned) PatchTST via NeuralForecast
src/models/timesfm_model.py   — (planned) TimesFM zero-shot inference
src/pipeline.py        — (planned) Main orchestrator
```

## Key Design Rules

- **No look-ahead bias**: All features use only past data. Labels use shift(-k). Walk-forward splits enforce strict temporal ordering.
- **Config-driven**: All tunable parameters live in `config/default.yaml`. Load via `from src.config import load_config`.
- **Evaluation discipline**: Always use walk-forward with embargo. Use purge for triple-barrier labels. Report accuracy, MCC, AUC per fold.
- **Transaction costs**: Backtest must include cost_bps. Never report Sharpe without costs.

## Testing

Tests use synthetic OHLCV data (fixture in `tests/conftest.py`), no network calls needed.

```bash
pytest tests/ -v          # All tests
pytest tests/test_data.py # Specific module
```

## Current Status

See `implementation-plan.md` for completed items and next steps. Core modules (data, features, labels, splitters, metrics, backtest, LightGBM) are implemented and tested. PatchTST, TimesFM, and pipeline orchestrator are next.
