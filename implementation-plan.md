# 实施计划

## 已完成

### 基础架构 (Phase 0)
- [x] `.gitignore`, `requirements.txt`, `requirements-gpu.txt`
- [x] `config/default.yaml` — 所有超参数集中管理
- [x] `src/config.py` — dataclass 嵌套配置加载器
- [x] `src/utils.py` — 日志、种子、结果持久化工具
- [x] `Makefile` — 便捷命令

### 核心模块 (Phase 1-4, 6)
- [x] `src/data.py` — OHLCV 下载 + 基础收益率 (log_close, ret_1d, vol_chg, hl_range, oc_return)
- [x] `src/features.py` — 滚动特征工程 (5/10/21/63 日窗口 × 5 指标 + 日历特征)
- [x] `src/labels.py` — 三种标签 (次日方向、k日收益、三重屏障)
- [x] `src/splitters.py` — walk-forward 时序分割 + embargo + purge
- [x] `src/metrics.py` — 分类/回归指标 + Sharpe + 最大回撤
- [x] `src/backtest.py` — 交易成本回测 + 成本敏感性分析

### 模型 (Phase 5a)
- [x] `src/models/lgbm_model.py` — LightGBM walk-forward 评估 wrapper

### 测试 (16/16 通过)
- [x] `tests/conftest.py` — 合成 OHLCV 数据 fixture
- [x] `tests/test_data.py` — 数据管道测试
- [x] `tests/test_features.py` — 特征工程测试 (含前瞻性检查)
- [x] `tests/test_labels.py` — 标签值域/类型测试
- [x] `tests/test_splitters.py` — 时序分割正确性测试
- [x] `tests/test_backtest.py` — 回测算术正确性测试

---

## 下一步开发计划

### Step 1: PatchTST 模型 wrapper
**文件**: `src/models/patchtst_model.py`

```python
def eval_patchtst_walkforward(df, spec, patchtst_config):
    # 对每个 walk-forward fold:
    # 1. 将数据转为 NeuralForecast 长格式 (unique_id="SPY", ds=date, y=ret_1d)
    # 2. 训练 PatchTST(h=5, input_size=64, max_steps=500, freq="B")
    # 3. 预测收益率 -> (pred > 0).astype(int) 转方向信号
    # 4. 计算 accuracy, MCC
    # 返回 fold_metrics + oos_predictions
```

**依赖**: `pip install neuralforecast`
**注意**: NeuralForecast 要求 PyTorch，建议 GPU 环境运行

### Step 2: TimesFM 零样本推理 wrapper
**文件**: `src/models/timesfm_model.py`

```python
def eval_timesfm_walkforward(df, spec, timesfm_config):
    # 对每个 walk-forward fold:
    # 1. 取训练窗口末尾 512 个 ret_1d 作为 context
    # 2. timesfm.TimesFM_2p5_200M_torch 零样本推理
    # 3. 预测 horizon=5 的收益率 -> 转方向信号
    # 返回 fold_metrics + oos_predictions
```

**依赖**: `pip install timesfm torch`
**注意**: 用 `try/except ImportError` 保护，未安装时优雅跳过

### Step 3: 管道编排器
**文件**: `src/pipeline.py`

串联所有组件:
1. 加载配置 → 设置种子 → 创建输出目录
2. `load_and_prepare()` → `make_features()` → `add_all_labels()`
3. 依次运行 LightGBM / PatchTST / TimesFM (各自 walk-forward)
4. 拼接每个模型的 OOS predictions → 转信号 → `backtest_daily_direction()`
5. 生成模型对比汇总表 → 保存到 `results/`

命令行接口:
```bash
python -m src.pipeline --config config/default.yaml --phase all
python -m src.pipeline --phase lgbm   # 只跑 LightGBM
```

### Step 4: 可视化 + 分析 notebooks
- `notebooks/01_data_exploration.ipynb` — 价格走势、收益分布、滚动波动率
- `notebooks/02_lgbm_baseline.ipynb` — LightGBM 逐折结果、特征重要性
- `notebooks/03_model_comparison.ipynb` — 三模型净值曲线叠加、成本敏感性表

### Step 5: 端到端集成测试
**文件**: `tests/test_pipeline.py`

用合成小数据 (50 行, train_years=0.05) 跑完整管道，验证:
- `results/metrics/` 下有 JSON/CSV 输出
- `results/predictions/` 下有 OOS 预测文件
- 无报错完成

---

## 技术架构图

```
config/default.yaml
        │
        ▼
   src/config.py ──→ ProjectConfig
        │
        ▼
   src/data.py ──→ OHLCV DataFrame
        │
        ▼
   src/features.py ──→ 特征 DataFrame
        │
        ▼
   src/labels.py ──→ y_dir / y_5d / y_tb
        │
        ├──→ src/splitters.py ──→ walk-forward folds
        │         │
        │         ▼
        ├──→ src/models/lgbm_model.py ──→ OOS predictions
        ├──→ src/models/patchtst_model.py ──→ OOS predictions
        ├──→ src/models/timesfm_model.py ──→ OOS predictions
        │         │
        │         ▼
        └──→ src/backtest.py ──→ 净值曲线 + Sharpe
                  │
                  ▼
             src/metrics.py ──→ 模型对比汇总
```
