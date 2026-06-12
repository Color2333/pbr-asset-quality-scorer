#!/bin/bash
# Overnight orchestration for the two extremes-loss experiments.
#   Run1 (sigma=0.25)   already running on GPU1  (pid passed as $1)
#   pilot               running on GPU0          (pid passed as $2)
# After pilot frees GPU0 -> launch Run2 (extreme-weight) on GPU0.
# After both trainings finish -> eval each (4-channel, saves probs) -> mark done.
set -u
cd /storage/datasets/art-data-intern/intern-container/haojiang_code/Pbr_auto
PY=/storage/home/haojiang/miniconda3/envs/qwen-vl/bin/python
RUN1_PID=$1   # sharp_sigma025 on GPU1
PILOT_PID=$2  # judge pilot on GPU0
L=asset_quality_scorer/logs

# 1) wait for pilot to free GPU0, then launch Run2 (extreme-weight) there
while ps -p "$PILOT_PID" >/dev/null 2>&1; do sleep 30; done
sleep 20
CUDA_VISIBLE_DEVICES=0 $PY asset_quality_scorer/scripts/vlm_scorer_sft.py \
  --items 40000 --extreme-weight 2.5 --val-assets 1500 --eval-every 10000 \
  --lora-r 16 --lr 1e-4 --out qwen25_extreme_w25 > $L/extreme_w25.log 2>&1 &
RUN2_PID=$!
echo "RUN2_LAUNCHED pid=$RUN2_PID $(date)"

# 2) wait for both trainings to finish
while ps -p "$RUN1_PID" >/dev/null 2>&1 || ps -p "$RUN2_PID" >/dev/null 2>&1; do sleep 120; done
echo "BOTH_TRAININGS_DONE $(date)"
sleep 20

# 3) eval each on test (4-channel default, saves preds+probs). Run1->GPU1, Run2->GPU0, parallel
CUDA_VISIBLE_DEVICES=1 $PY asset_quality_scorer/scripts/vlm_scorer_eval.py \
  --adapter asset_quality_scorer/outputs/runs/vlm_scorer_qwen25_sharp_sigma025/best \
  --out-tag sharp_sigma025_oldtest > $L/eval_sharp_sigma025.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 $PY asset_quality_scorer/scripts/vlm_scorer_eval.py \
  --adapter asset_quality_scorer/outputs/runs/vlm_scorer_qwen25_extreme_w25/best \
  --out-tag extreme_w25_oldtest > $L/eval_extreme_w25.log 2>&1 &
wait
echo "ALL_EVALS_DONE $(date)"
