# Asset Quality Scorer — 项目规范

> 本文件供 Claude Code 和开发者共同遵守。所有约定必须在改代码前先看这里。

---

## 1. 代码组织规范

### 绝对禁止

- **禁止用版本号命名文件**：`data_v2.py`、`metrics_v2.py`、`train_v2_regression.py` 这类命名是反模式，只会让目录越来越乱。
- **禁止为每个新实验创建新脚本**：不能因为要试一个新架构就新建 `train_dinov2.py`、`train_clip_prompt.py`。
- **禁止在文件名/目录名里写版本号**（`_v2`、`_v3`、`_new`、`_final`）。

### 正确做法：组件化 + 注册表

**模型** 统一放在 `quality_scorer/models/`，每个架构一个文件，通过注册表调用：

```
quality_scorer/
  models/
    __init__.py        ← 注册表，暴露 build_model(cfg)
    convnext.py        ← ConvNeXt + CrossModalFusion（当前主力）
    dinov2.py          ← DINOv2 backbone（未来）
    ordinal.py         ← 序数分类头（未来）
```

**数据** 统一在 `quality_scorer/data.py`，用参数控制行为，不拆文件。

**指标** 统一在 `quality_scorer/metrics.py`，所有通道、所有任务复用同一套函数。

**入口脚本只有两个**，读 config 派发，不写死任何参数：

```
scripts/
  train.py    ← 统一训练入口
  eval.py     ← 统一评估入口（指定 exp_id 即可）
```

---

## 2. 实验命名规范

### Experiment ID 格式

```
{arch}_{channel}_{descriptor}
```

| 字段 | 说明 | 示例 |
|------|------|------|
| `arch` | backbone 简称 | `convnext_base`, `dinov2_base` |
| `channel` | 通道名 或 `all` | `metallic`, `normal_map`, `all` |
| `descriptor` | 描述关键变化，用下划线分词 | `baseline`, `clip_prompt`, `aux_roughness` |

**合法命名示例：**
```
convnext_base_metallic_baseline
convnext_base_metallic_clip_prompt
convnext_base_metallic_aux_roughness
convnext_base_all_ordinal_head
dinov2_base_normal_map_baseline
```

**非法命名（禁止）：**
```
convnext_base_metallic_regression_v2   ✗ 有版本号
convnext_base_metallic_new             ✗ 无意义后缀
metallic_final                         ✗ 缺少 arch
```

---

## 3. Checkpoint 目录结构

所有训练输出统一写到：

```
outputs/runs/{exp_id}/
```

每个 run 目录内容固定为：

```
outputs/runs/convnext_base_metallic_baseline/
  config.yaml        ← 训练时用的完整配置快照（自动复制）
  best.pt            ← 按主指标（MAE）保存的最佳权重
  best_srcc.pt       ← 按 SRCC 保存（可选）
  best_binary_f1.pt  ← 按 F1 保存（可选）
  train_log.json     ← 每 epoch 的 train/val 指标曲线
  eval_test.json     ← 训练完成后写入的 test set 结果（运行 eval.py 生成）
```

**禁止**在 `outputs/` 下直接建散乱目录或把旧结果命名为 `_v2`。

---

## 4. Config 规范

实验配置放在 `config/` 下，文件名 = exp_id：

```
config/
  convnext_base_metallic_baseline.yaml
  convnext_base_metallic_clip_prompt.yaml
```

Config 文件必须能完整复现实验（包含 arch、dataset、sampler、loss、optimizer 所有参数），不依赖脚本里的 hardcode。

---

## 5. 结果追踪

`outputs/runs/` 下每个 exp_id 的 `eval_test.json` 是结果的唯一权威来源，格式：

```json
{
  "exp_id": "convnext_base_metallic_baseline",
  "channel": "metallic",
  "ckpt": "best",
  "mae": 0.6257,
  "srcc": 0.8162,
  "plcc": 0.8149,
  "acc": 0.6355,
  "within_1": 0.8581,
  "kappa_qwk": 0.8030,
  "kappa_lin": 0.8141,
  "binary_f1": 0.8014,
  "per_score_mae": {"0": 0.86, "1": 0.92, "2": 1.62, "3": 0.38, "4": 0.74, "5": 0.30}
}
```

`eval.py --compare` 可一次性汇总所有 exp_id 的结果做横向对比，不需要手动整理。

---

## 6. 当前遗留文件说明

以下文件是历史遗留，**新代码不要引用，逐步迁移**：

| 文件 | 状态 | 迁移目标 |
|------|------|---------|
| `quality_scorer/data.py` | 旧版 Phase 1 数据 | 合并到新 `data.py` |
| `quality_scorer/data_v2.py` | 当前在用 | 重命名为 `data.py`（完成迁移后） |
| `quality_scorer/metrics.py` | 旧版 | 合并到 `metrics.py` |
| `quality_scorer/metrics_v2.py` | 当前在用 | 重命名为 `metrics.py` |
| `scripts/train_v2_regression.py` | 当前在用 | 迁移到 `scripts/train.py` |
| `scripts/eval_v2_regression.py` | 当前在用 | 迁移到 `scripts/eval.py` |
| `outputs/phase2_regression/` | 旧结果 | 保留参考，新实验用 `outputs/runs/` |

---

## 7. 开发流程

新实验的标准流程：

```
1. 在 quality_scorer/models/ 中添加或修改架构（组件化，不建新脚本）
2. 在 config/ 中新建 {exp_id}.yaml
3. python scripts/train.py --config config/{exp_id}.yaml
4. python scripts/eval.py  --exp-id {exp_id}
5. git commit（代码变更 + config + eval_test.json）
```

**不需要**改脚本文件名、不需要新建 train_xxx.py。

---

## 8. GPU 使用约定

- **最多同时训练 1 个通道**（服务器 GPU 资源有限）
- 训练前先 `nvidia-smi` 确认空闲 GPU
- 后台训练用 `nohup ... > logs/{exp_id}.log 2>&1 &`，日志写到 `logs/`
