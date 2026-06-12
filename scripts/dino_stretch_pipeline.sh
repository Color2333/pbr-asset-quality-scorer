#!/bin/bash
# 近黑增亮小实验编排: stretch1(baseline, 已在跑) 完 -> 评估 -> 跑 stretch8 -> 评估 -> 对比近黑
# 全程 GPU1 串行(与 VLM 全量共卡, 串行避免显存挤爆)
set -u
cd /storage/datasets/art-data-intern/intern-container/haojiang_code/Pbr_auto
PY=/storage/home/haojiang/miniconda3/envs/asset-quality-scorer/bin/python
L=asset_quality_scorer/logs

# 1) 等 baseline(stretch1) 训练进程结束
while pgrep -f "train_multitask.py.*stretch1" >/dev/null 2>&1; do sleep 60; done
sleep 15
CUDA_VISIBLE_DEVICES=1 $PY asset_quality_scorer/scripts/eval_multitask.py \
  --exp-id dinov2_large_multitask_stretch1 > $L/eval_dino_stretch1.log 2>&1
echo "STRETCH1_EVAL_DONE $(date)"

# 2) 跑 stretch8(实验组)
CUDA_VISIBLE_DEVICES=1 $PY asset_quality_scorer/scripts/train_multitask.py \
  --config asset_quality_scorer/config/dinov2_large_multitask_stretch8.yaml \
  > $L/dino_stretch8.log 2>&1
echo "STRETCH8_TRAIN_DONE $(date)"
sleep 15
CUDA_VISIBLE_DEVICES=1 $PY asset_quality_scorer/scripts/eval_multitask.py \
  --exp-id dinov2_large_multitask_stretch8 > $L/eval_dino_stretch8.log 2>&1
echo "STRETCH8_EVAL_DONE $(date)"

# 3) 对比近黑 metallic
$PY - <<'PYEOF'
import json
from pathlib import Path
PKG=Path('asset_quality_scorer')
def rd(exp):
    p=PKG/'outputs/runs'/exp/'eval_test.json'
    return json.loads(p.read_text()) if p.exists() else None
b=rd('dinov2_large_multitask_stretch1'); s=rd('dinov2_large_multitask_stretch8')
print("=== 近黑增亮对比 (DINOv2 冻结) ===")
for name,d in [('baseline x1',b),('stretch x8',s)]:
    if d:
        nb=d.get('metallic_nearblack',{})
        print(f"  {name}: metallic_srcc={d['metallic']['srcc']}  近黑srcc={nb.get('srcc')}  auc_missing={nb.get('auc_missing')}")
print("ALL_DONE")
PYEOF
