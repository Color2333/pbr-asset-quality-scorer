# PBR 材质质量评估 Demo — 模型卡（当前生产版本）

> 本文档只描述 demo 页当前挂载的**最佳版本** `dinov2_large_multitask_emd_all`，不含历史尝试。
> 数字来源：`outputs/runs/dinov2_large_multitask_emd_all/{demo_predictions.json, summary.json}`、`docs/技术报告_PBR材质质量评估.md`。
> 更新日期：2026-06-05

---

## 1. Demo 页

- **入口**：`scripts/demo.py`（Flask，默认端口 7862，`python scripts/demo.py [--port 7862]`）
- **数据**：加载离线预计算的 `outputs/runs/dinov2_large_multitask_emd_all/demo_predictions.json`（由 `scripts/predict_demo.py` 对 test split 全量推理生成），**页面本身不占 GPU**
- **展示内容**：每个资产的渲染图 + 4 通道贴图（base_color / normal_map / roughness / metallic），模型预测分 vs 人工真分 vs 误差
- **筛选器**：`all` / `accurate`（MAE≤0.4）/ `big_err`（MAE≥1.0）/ `metal_wrong`（|metallic 误差|≥2）；支持按名称、总误差、metallic 误差排序

---

## 2. 任务定义

给定一个 3D 资产的渲染图 + 各 PBR 通道贴图，对 4 个通道（`base_color` / `normal_map` / `roughness` / `metallic`）各预测 0–5 的连续质量分。**北极星指标为 SRCC**（与人工打分的 Spearman 排序相关），辅以 MAE / exact acc / within-1 等。

产品目的：自动分层筛选 3D 训练数据（FinalScore / Tier 预测、高质量资产筛选）。

---

## 3. 模型架构

**Exp ID**：`dinov2_large_multitask_emd_all`
**Config**：`config/dinov2_large_multitask_emd.yaml`
**代码**：`quality_scorer/models/dinov2.py`（`DINOv2MultiTaskScorer`）

```
4 通道贴图 (224², RGB)            CLIP 特征 (离线预计算)
        │                                │
DINOv2 ViT-L/14 reg4 (共享主干)    ViT-L/14 CLS: render 768 + base_color 768 = 1536
        │                                │
取第 8/16/24 层 patch tokens             │
各 mean-pool → 3 × 1024 = 3072           │
        │                                │
        ├──── CrossModalFusion (cross-attn, 多尺度特征 attend 到 CLIP) ──┐
        │                                                              │
        └──── clip_direct bypass（CLIP 直通） ──────────────────────────┤
                                                                       │
                  融合 MLP: LN → Linear(→512) → GELU → Dropout(0.3)
                                       │
              ┌────────────────────────┼────────────────────────┐
        4 × EMD score 头          4 × binary 头            （辅助头：tier /
        Linear(512→6)             Linear(512→1)             pbrType / 缺陷）
        6-bin 分布 → E[score]     valid/invalid
```

### 关键设计

| 组件 | 选择 | 理由 |
|---|---|---|
| **Backbone** | DINOv2 ViT-L/14 reg4（timm，304M） | dense 自监督特征对材质纹理更友好，比 ConvNeXt 稳定 +0.01–0.02 SRCC |
| **多任务共享主干** | 4 通道共享 backbone，各自独立头 | **最有效的单一改动**：metallic 0.565→0.605（强通道梯度逼主干学通用材质表示，弱通道受益），全通道不掉 |
| **EMD 分布头**（NIMA 风格） | Linear(512→6) 输出 6-bin 分布，squared EMD loss，推理取期望 | SRCC 持平回归（0.792）但 exact acc +0.02、asym acc 最高；天然"差一档轻罚、差多档重罚"，对齐"5→4 可接受"的产品容忍度 |
| **晚融合**（通道独立出分） | 不做跨通道交互 | FinalScore 实测就是 5 维加权和（R²=0.998），标签不含交互项；所有跨通道尝试均被证伪 |
| **CLIP 语义路径** | render + base_color CLS 拼 1536-d，cross-attn + 直通 | 提供"该不该是金属"等物体级语义上下文 |
| **分辨率 224** | 不用 448 | 质量信号是低频/全局的，448 无红利甚至有害 |

### 训练配方

- **损失**：squared EMD（主）+ 0.2·BCE(valid/invalid) + 0.05·成对 margin 排序损失（直接优化排序，与 SRCC 同向）
- **metallic 特殊处理**：loss 权重 ×1.5，回流主干梯度 ×0.5
- **解冻调度**：epoch 5 解冻 stage4（最后 6 层）→ epoch 10 解冻 stage3-4 → epoch 15 全解冻；**best.pt 实际取自 ep9 val 峰值（浅解冻阶段）**。后续实验证实：39k 噪声数据撑不起全解冻 304M ViT-L（ep16 起 val 崩），浅解冻 + LLRD(0.65) 是正确姿势
- **正则**：Dropout 0.3（融合 MLP）+ drop_path 0.2（stochastic depth）+ RandomErasing
- **优化器**：AdamW lr=1e-4, wd=1e-4，cosine 退火，bf16 AMP，batch 12
- **Checkpoint**：按 val srcc_mean 存 `best.pt`，另按 metallic val SRCC 存 `best_metallic.pt`

---

## 4. 数据集

**文件**：`dataset/sampled_all.csv`（49,165 个资产）
**划分**（按 pbrType 分层）：train 39,329 / val 4,918 / test 4,918（有效 test 评估 n=4,917）
**图像**：每资产渲染图 + 4 通道贴图，预处理为 224² tensor cache（`cache/224`）；CLIP 特征离线预计算（`features/clip_vitl14_openai_render_base_color.pt`）

### 标签结构（人工标注，单标注者）

| 字段 | 含义 |
|---|---|
| baseColor / normal / roughness / metallic / rendered | 各维度 0–5 整数分 |
| finalScore | 综合分，实测 = 5 维加权和（R²=0.998）：`0.189·bc + 0.351·normal + 0.152·rough + 0.150·metal + 0.147·rendered` |
| tier | Tier1（电影级，0.4%）~ Tier5（勉强可用） |
| pbrType | physical 88.0% / stylized 8.4% / uncertain 3.6% |
| 缺陷标记 ×4 | hasTextOrPattern / FakeAOOrGlow / normalAbnormalTint / normalIsFlipped |

### 分布特点

- **来源**：Sketchfab 59% / 3D66 14% / Objaverse 9% / Games 7.6% / Unreal 5.5%
- **finalScore** 均值 2.30；4–5 分高质量样本仅 2.9%（高分尾部稀缺）
- **47% 的 metallic 贴图近乎全黑**，且标签从 0 散到 5（"正确的全黑非金属" vs "漏标的全黑"像素相同、分数相反）——这是 metallic 通道的根本难点
- **单标注、无重复标注** → 无法直接测人类一致性天花板；NR-IQA 文献参照下 0.8 量级 SRCC 已是合理好成绩

---

## 5. 评估指标（test split, n=4,917）

### 5.1 核心指标

| 通道 | test SRCC | val MAE (ep9) | 备注 |
|---|:---:|:---:|---|
| roughness | **0.896** | 0.416 | 最强通道，近天花板 |
| base_color | **0.842** | 0.394 | 近天花板 |
| normal_map | **0.807** | 0.527 | 近天花板；尾部（满分）已用过采样修复 |
| metallic | **0.624** | 0.882 | 数据噪声硬天花板（见 5.4） |
| **均值** | **0.792** | 0.555 | |

分类视角（test，多任务 EMD vs 其他头）：**exact acc 0.607**、**asymmetric acc（5→4 免罚）0.636** —— 均为三种头（回归/序数/EMD）中最高。

### 5.2 产品级指标（分层筛选目标，已达标）

| 指标 | 数值 |
|---|:---:|
| FinalScore 预测 SRCC（4 通道预测分 → finalScore） | **0.831** |
| Tier 5 档分类 exact / **within-1** | 63% / **98.6%** |
| 高质量筛选（Tier1-2）二分 AUC | **0.924** |
| 保留 top5% 的精度 | 0.66（6.4× 随机） |
| 保留 top30% 的召回 | 0.92 |
| normal 满分召回（尾部过采样后） | 10% → **61.6%**（normal SRCC 仅 -0.002） |

### 5.3 train–test 泛化诊断

| 通道 | train SRCC | test SRCC | gap |
|---|:---:|:---:|:---:|
| roughness | 0.933 | 0.895 | 0.037 |
| base_color | 0.890 | 0.838 | 0.052 |
| normal_map | 0.868 | 0.808 | 0.060 |
| **metallic** | 0.788 | 0.627 | **0.161**（3–4×） |

正则栈可把 metallic gap 砍到 0.067，但 test 纹丝不动 → **metallic 是 noise-limited，不是 gap-limited**。

### 5.4 metallic 近黑/非黑拆分（test）

| 子集 | n | SRCC | MAE |
|---|---:|:---:|---:|
| 近黑（非黑像素<2%） | 2,289 | 0.555 | 1.11 |
| 非黑（有金属可看） | 2,628 | 0.686 | 0.75 |
| 非黑（最严阈值 ≥30%） | 598 | 0.709 | 0.71 |

两个关键事实：① 非黑子集封顶 ~0.71，够不着其他通道一档 → metallic 难是**全局性**的；② 近黑 0.555 仍高于 map-only 贝叶斯地板（0.24–0.44）→ 模型已在用 base_color/render 上下文"捞回"。metallic ~0.62 的硬天花板有 ~6 条独立证据（正则、synthneg 反事实、跨通道三种实现、backbone 三重确认、语义≠质量探针、双流对照），架构层已穷尽。

### 5.5 附属能力：缺陷检测器（独立交付物，DINOv3 版为生产）

| 缺陷 | AUC | AP | @精度90% 的召回 |
|---|:---:|:---:|:---:|
| 有文字/印花 | 0.948 | 0.815 | 0.41 |
| 法线异色 | 0.931 | 0.855 | 0.69 |

---

## 6. 进行中工作

### 6.1 VLM 视觉先验注入 metallic（攻坚天花板）

**动机**：metallic 是"看物体该不该是金属"的世界知识判断，纯视觉编码器（CLIP/DINOv2）区分不开"正确的全黑"vs"漏标的全黑"（视觉天花板 AUC ≈ 0.58）。

**方法**（`scripts/precompute_vlm_prior.py` + `scripts/vlm_metal_prior_probe.py`）：
- 用 Qwen2.5-VL（nf4 量化）对每个资产的 white render + base_color 提问（如 "Does this model likely have large metal parts?"），离线预计算
- 输出与 tensor cache 对齐的 memmap：`p_yes` 标量 [N] + 最后隐层 [N, 3584]（fp16）
- 验证集：近黑子群 n=2,289（其中漏标 1,139），指标 = 区分"漏标 vs 正确全黑"的 AUC

**Probe 结果**（`outputs/vlm_metal_prior/probe_summary*.json`）：

| Prompt 变体 | AUC（全量 n=2289） |
|---|:---:|
| v4_largemetal（最优单体） | **0.632** |
| v2_error | 0.630 |
| v1_hasmetal | 0.628 |
| v5_bbox | 0.544（失败） |
| 多 prompt 集成 | ~0.64 |
| 高置信极端子集 | ~0.68 |
| **视觉信息天花板（参照）** | 0.58 |

**结论**：**首次穿透 0.58 视觉天花板**，证明 VLM 世界知识携带视觉特征之外的增量信息；prompt 工程边际已尽，剩余误差主要是标签噪声。
**下一步**：把 `p_yes` 标量 / 3584-d 隐层注入 `DINOv2MultiTaskScorer` 的 metallic 头重训，验证 test SRCC 能否突破 0.62–0.63 带。

### 6.2 扩充到 100k 数据集重训

**配置**：`config/dinov2_large_multitask_emd_100k.yaml`

| 项 | 50k 版 | 100k 版 |
|---|---|---|
| CSV | sampled_all.csv（49,165） | **sampled_all_0604.csv（104,621）** |
| train 样本 | 39,329 | ~83,700（2.1×） |
| tensor cache | cache/224 | cache/224_0604 |
| CLIP 特征 | …render_base_color.pt | …render_base_color_0604.pt |
| epochs | 30 | 20（样本量翻倍按比例缩） |
| 解冻里程碑 | 5 / 10 / 15 | 4 / 8 / 12 |
| 新增 | — | 尾部过采样（power 0.5, cap 10）默认开启 |

**当前状态**（2026-06-05，训练中，epoch 7/20）：

| val 指标 | 50k best（ep9） | 100k ep6/7 |
|---|:---:|:---:|
| srcc_mean | 0.806 | 0.781 |
| base_color | 0.849 | 0.778 |
| normal_map | 0.828 | 0.806 |
| roughness | 0.908 | 0.862 |
| **metallic** | 0.640 | **0.688**（ep6 峰值） |

早期信号：**metallic val SRCC +0.05**（更多数据稀释标签噪声的预期方向），强通道尚未收敛（解冻刚开始）。注意 100k 的 val 集不同，不能与 50k 直接比，最终以 test 评估为准。

---

## 7. 产物索引

| 产物 | 路径 |
|---|---|
| 生产权重 | `outputs/runs/dinov2_large_multitask_emd_all/best.pt` |
| Demo 页 | `scripts/demo.py`（预测数据 `demo_predictions.json`） |
| 部署管线 | `../screening/pbr_infer/`（含 DEPLOYMENT_SOP） |
| 完整技术报告 | `docs/技术报告_PBR材质质量评估.md` |
| VLM probe 结果 | `outputs/vlm_metal_prior/` |
| 100k 训练日志 | `outputs/runs/dinov2_large_multitask_emd_100k/train_log.json`、`logs/emd_100k.log` |
