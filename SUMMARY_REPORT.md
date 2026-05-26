# Image-only 3D Asset Quality Scorer 总结报告

## 当前定位

当前阶段只使用已有图片信息，不加入几何特征。系统输入来自 `grid_pbr.png` 和 `grid_white.png`，核心能力包括：

- 四通道 PBR 图片质量 scorer：`normal_map`、`roughness`、`metallic`、`base_color`
- 从 scorer backbone 提取的 6144 维 asset embedding
- 基于 embedding 和四通道预测分数的 asset-level fusion scorer
- UMAP / t-SNE 可视化、定量 probe、错误样例分析

当前目标不是做完整 3D 资产理解，而是把 image-only 的 PBR/texture quality 判断先做稳定、可解释、可复用。

## 已完成工作

### Phase 1：四通道连续质量 Scorer

已经把原来的二分类思路升级为 ordinal regression scorer。每个通道使用 ConvNeXt V2 backbone + CORAL ordinal head，输出 0-5 的连续 expected score。

训练结果摘要：

| Channel | Expected MAE | Best Binary F1 | 判断 |
|---|---:|---:|---|
| normal_map | 0.639 | 0.867 | 稳定可用 |
| roughness | 0.690 | 0.800 | 稳定可用 |
| metallic | 0.684 | 0.812 | 可用，但预测更容易走极端 |
| base_color | 0.619 | 0.336 | 连续分数可用，硬分类较弱 |

结论：`normal_map` 和 `roughness` 是当前最可靠的质量信号；`metallic` 和 `base_color` 有信息量，但需要校准和错误样例分析。

### Phase 2：Asset Embedding

已经完成 full raw-grid embedding 抽取：

- asset 数量：37996
- embedding 维度：6144
- 来源：四通道 scorer backbone feature 拼接
- 输出：`asset_quality_scorer/outputs/phase2_embedding_raw/asset_embeddings.npz`
- 元数据：`asset_quality_scorer/outputs/phase2_embedding_raw/asset_scores.csv`

UMAP 和 t-SNE 可视化显示，不同质量 tier 在 embedding 空间里有区域性分布，但不是严格线性分层。这符合预期：image embedding 同时编码质量、材质类型、风格、贴图复杂度等视觉因素。

### 定量验证

embedding-only probe：

- `final_score` Ridge MAE：0.368，基线 0.753
- `final_score` Ridge R2：0.726
- tier linear accuracy：0.632，基线 0.345
- tier macro F1：0.539

fusion scorer：

- 输入：128 维 PCA embedding + 四通道 expected score + 四通道 pred score，共 136 维
- `final_score` GBDT MAE：0.329，基线 0.707
- `final_score` GBDT R2：0.739
- tier GBDT accuracy：0.686，基线 0.378
- tier GBDT macro F1：0.541

结论：当前 embedding 不是单纯记录纹理外观，而是已经编码了可预测的质量信息。fusion scorer 是目前最适合作为 asset-level image-only 质量分的版本。

## Image-only 内容分析

全量数据分布：

- raw-grid asset：37996
- 有效 tier asset：37447
- tier 分布：
  - Tier 1：57
  - Tier 2：1978
  - Tier 3：10523
  - Tier 4：15266
  - Tier 5：9623

通道预测与人工通道分数关系：

| Channel | MAE | Pearson r | 主要现象 |
|---|---:|---:|---|
| normal_map | 0.568 | 0.831 | 与 tier 和 final score 趋势一致，信号强 |
| roughness | 0.603 | 0.871 | 当前最强通道相关性 |
| metallic | 1.100 | 0.630 | 容易偏 0/5，非极端区间拟合弱 |
| base_color | 0.621 | 0.621 | 趋势存在，但分类边界不清晰 |

趋势图和误差图：

- `asset_quality_scorer/outputs/image_only_analysis/score_trends_by_tier.png`
- `asset_quality_scorer/outputs/image_only_analysis/channel_error_summary.png`

## 错误样例分析

已导出当前 fusion scorer 的高误差样例：

- CSV：`asset_quality_scorer/outputs/image_only_error_cases/top_final_score_errors.csv`
- 拼图：`asset_quality_scorer/outputs/image_only_error_cases/top_final_score_errors_contact_sheet.jpg`

观察到的主要失败模式：

- 模型容易高估“PBR 通道看起来丰富完整，但人工 final score 低”的资产。
- 高误差样例里，`normal_map`、`metallic`、`base_color` 经常给出偏高分。
- 这说明模型主要学到了图片上的 PBR 完整性、纹理强度、法线/金属响应等视觉信号。
- 但它还不能完全理解人工 final score 中隐含的“资产是否真正可用、是否重复低质、是否风格不合格”等综合判断。

这个结果说明当前系统边界清楚：它已经是一个 image-based PBR/texture quality scorer，但还不是完整人工审美/可用性替代器。

## 当前判断

当前 image-only 方案是成立的：

1. 四通道 scorer 可以输出连续质量分。
2. scorer backbone embedding 能表达质量相关视觉信息。
3. fusion scorer 能把 embedding 和通道预测分数整合成资产级质量分。
4. 错误样例暴露的问题主要是校准和语义边界问题，而不是方向错误。

当前最值得继续做的是把这个 image-only scorer 做稳，而不是加入新模态。

## 下一步

短期优先级：

1. 四通道分数校准  
   对 expected score 做 calibration，让输出分数更接近真实 0-5 分布，尤其处理 `metallic` 和 `base_color` 的极端预测。

2. 错误样例回看  
   按高误差样例分组，判断错误来自标签噪声、通道 scorer 高估、图片内容不足，还是资产类型本身难判。

3. Image-only inference pipeline  
   输入一个 asset 的 `grid_pbr.png` 和 `grid_white.png`，输出四通道分、fusion 分、embedding 和近邻案例。

4. Embedding retrieval 验证  
   用 embedding 找相似 asset，检查近邻是否在质量、材质风格、贴图复杂度上相近。

5. 针对 base_color / metallic 做专项改进  
   对这两个通道检查标签分布和错误样例，必要时把任务从硬分类调整为更平滑的 texture/style quality regression。

## 主要产物

- 总报告：`asset_quality_scorer/SUMMARY_REPORT.md`
- Phase 1 报告：`asset_quality_scorer/outputs/phase1_ordinal/PHASE1_REPORT.md`
- 全量 embedding：`asset_quality_scorer/outputs/phase2_embedding_raw/asset_embeddings.npz`
- embedding 元数据：`asset_quality_scorer/outputs/phase2_embedding_raw/asset_scores.csv`
- fusion scorer：`asset_quality_scorer/outputs/asset_fusion_scorer/fusion_scorer.pkl`
- image-only 分析：`asset_quality_scorer/outputs/image_only_analysis/image_only_summary.json`
- 高误差样例：`asset_quality_scorer/outputs/image_only_error_cases/top_final_score_errors.csv`


For Feature
 未来工作总结

  ---
  一、定向CLIP特征（最高优先级）

  问题：CLS token是通用语义摘要，对材质属性判别不够精准，尤其无法区分"全黑metallic正确"vs"全黑metallic错误"。

  方案：为每个通道设计针对性的文本prompt，用CLIP文本-图像相似度替代或补充CLS token。

  # 示例：metallic通道
  prompts = [
      "a metallic object with shiny metal surface",
      "a non-metallic object made of wood, fabric or plastic",
      "an object with both metallic and non-metallic parts",
      ...
  ]
  sim_scores = [B, N_prompts]  # 直接编码材质语义

  工作项：  
  - 为4个通道各自设计5-10个prompt，重点是metallic
  - 重新提取特征（concat到现有clip_feat，特征文件更新）
  - 重训metallic，观察全黑样本预测是否改善
  - 验证有效后推广其他通道
  
  ---
  二、CLIP patch token / DINOv2（中期）
 
  问题：CLS token丢失空间信息，局部缺陷（法线图局部偏色、roughness局部异常）会被全局均值掩盖。

  方案A：用CLIP的256个patch token做注意力池化，保留空间信息。

  方案B：换用DINOv2——训练目标是自监督局部特征一致性，材质/纹理表达显著优于CLIP，尤其适合normal_map和roughness。
  
  工作项：
  - 先用patch token做实验，成本低
  - 若效果好再考虑完整换DINOv2 backbone

  ---
  三、多通道图像辅助输入（中期）
 
  问题：每个模型只看单一通道图像，但PBR通道间有强相关性。全黑metallic是否正确，roughness和base_color贴图的视觉内容
  直接告诉答案。
  
  方案：为metallic模型加入roughness/base_color图像的轻量特征，concat到fusion层。
  
  主图: metallic [B, 3, 224, 224]  → ConvNeXt
  辅助: roughness + base_color → 冻结backbone提取 stage4 → pool → Linear(2048→256)
  fusion_in: 2816 + 256 = 3072

  工作项：
  - tensor cache已有所有通道数据，只需改dataset和模型forward
  - 优先在metallic上验证，因为该通道的跨通道依赖最强
  
  ---
  四、标注数据扩充（长期/数据侧）
 
  问题：以下场景纯靠调参无法突破：

  ┌─────────────────────────────┬─────────────────────────────┐
  │            问题             │          根本原因           │
  ├─────────────────────────────┼─────────────────────────────┤
  │ normal_map score=5 MAE=1.73 │ 训练集只有232个score=5样本  │
  ├─────────────────────────────┼─────────────────────────────┤
  │ base_color binary F1=0.60   │ invalid只占4.1%，结构性稀缺 │
  ├─────────────────────────────┼─────────────────────────────┤
  │ metallic score=2 MAE=1.62   │ 双峰分布，中间分数天然稀少  │
  ├─────────────────────────────┼─────────────────────────────┤
  │ base_color score=1 MAE=1.26 │ 只有389个score=1样本        │
  └─────────────────────────────┴─────────────────────────────┘
  
  工作项：
  - 针对性补充稀缺分值的标注（尤其是高分质量样本）
  - 考虑用当前模型的预测结果辅助筛选待标注样本（主动学习）
  
  ---
  五、已完成 / 当前状态

  ┌─────────────────────────────────────────────────────┬──────┐
  │                        模块                         │ 状态 │
  ├─────────────────────────────────────────────────────┼──────┤
  │ ConvNeXt多尺度 + CrossModalFusion + clip_direct架构 │ ✅   │
  ├─────────────────────────────────────────────────────┼──────┤
  │ 4通道全部训练完成                                   │ ✅   │
  ├─────────────────────────────────────────────────────┼──────┤
  │ Huber + binary + defect + ranking多任务损失         │ ✅   │
  ├─────────────────────────────────────────────────────┼──────┤
  │ 解决metallic全黑placeholder问题                     │ ✅   │
  ├─────────────────────────────────────────────────────┼──────┤
  │ per-score tail oversample                           │ ✅   │
  ├─────────────────────────────────────────────────────┼──────┤
  │ 测试集完整评估（MAE/SRCC/acc/QWK/Kappa）            │ ✅   │
  ├─────────────────────────────────────────────────────┼──────┤
  │ 可视化页面（分布图 + 案例图）                       │ ✅   │
  └─────────────────────────────────────────────────────┴──────┘
  
  ---
  优先级排序：定向CLIP prompt → 多通道辅助输入 → patch token/DINOv2 → 数据补充
  
  前两项改动量小、预期收益明确，适合近期推进。