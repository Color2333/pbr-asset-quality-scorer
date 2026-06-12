#!/bin/bash
# After the packed retrain (GPU1, pid $1) finishes: eval it, then run the CONTROL
# (same recipe, standard unpacked inputs) on GPU1, then eval it. Isolates "packed
# input" effect from the "unfreeze+LLRD+retrain recipe" effect.
set -u
cd /storage/datasets/art-data-intern/intern-container/haojiang_code/Pbr_auto
PY=/storage/home/haojiang/miniconda3/envs/asset-quality-scorer/bin/python
PACKED_PID=$1; L=asset_quality_scorer/logs
ts() { date +%Y%m%d_%H%M%S; }

while ps -p "$PACKED_PID" >/dev/null 2>&1; do sleep 120; done
echo "PACKED_TRAIN_DONE $(date)"
sleep 20
CUDA_VISIBLE_DEVICES=1 $PY asset_quality_scorer/scripts/eval_multitask.py \
  --exp-id dinov2_large_multitask_emd_packed > "$L/eval_emd_packed_$(ts).log" 2>&1
echo "PACKED_EVAL_DONE $(date)"

# control: same recipe, standard inputs
CUDA_VISIBLE_DEVICES=1 $PY asset_quality_scorer/scripts/train_multitask.py \
  --config asset_quality_scorer/config/dinov2_large_multitask_emd_pe_ctrl.yaml \
  > "$L/train_emd_pe_ctrl_$(ts).log" 2>&1
echo "CTRL_TRAIN_DONE $(date)"
sleep 20
CUDA_VISIBLE_DEVICES=1 $PY asset_quality_scorer/scripts/eval_multitask.py \
  --exp-id dinov2_large_multitask_emd_pe_ctrl > "$L/eval_emd_pe_ctrl_$(ts).log" 2>&1
echo "ALL_DONE $(date)"
