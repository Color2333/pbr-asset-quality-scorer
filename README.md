# Asset Quality Scorer

独立于 `screening/` 的 3D asset 质量打分与 embedding 实验目录。

## 目标

- Phase 1: 将现有 PBR 四通道二分类升级为连续质量 scorer。
- Phase 1.5: 增加几何维度，形成多维 asset quality profile。
- Phase 2: 从 scorer backbone 抽取中间层 feature，构建 asset embedding，并用 tier / finalScore / 通道分数验证空间结构。

## 环境

当前机器 shell 中没有检测到 `conda` 命令。本目录先提供可复现配置：

```bash
bash asset_quality_scorer/scripts/create_env.sh
conda activate asset-quality-scorer
```

如果脚本提示找不到 conda，需要先安装 Miniconda/Anaconda，或把已有 conda 初始化到当前 shell。

## 目录

```text
asset_quality_scorer/
  environment.yml          # conda 环境
  config/                  # 实验配置
  quality_scorer/          # 新 scorer / embedding 代码
  scripts/                 # 启动脚本
  outputs/                 # 新实验输出，默认不写入 screening/
  notebooks/               # t-SNE / UMAP 分析草稿
```

## 与旧目录的关系

短期内会复用旧数据和 checkpoint：

- 数据图像：`screening/data_v2`
- 原始 CSV 标签：`screening/data_38k`
- 已有 binary/ordinal checkpoint：`screening/models_*`

新训练结果、embedding、可视化和报告默认写到 `asset_quality_scorer/outputs/`。

## Phase 1 训练

先跑单通道：

```bash
bash asset_quality_scorer/scripts/run_phase1_ordinal.sh roughness
```

调试小样本：

```bash
bash asset_quality_scorer/scripts/run_phase1_ordinal.sh roughness \
  --epochs 1 --batch-size 16 --num-workers 0 \
  --max-train-samples 64 --max-val-samples 32
```

跑配置里的四通道：

```bash
bash asset_quality_scorer/scripts/run_phase1_ordinal.sh all
```
