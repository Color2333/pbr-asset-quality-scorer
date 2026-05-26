"""Metrics for continuous regression scorer (v2).

Primary metrics (aligned with NR-IQA literature):
  - MAE:     mean absolute error on continuous predicted score
  - SRCC:    Spearman rank-order correlation coefficient
  - PLCC:    Pearson linear correlation coefficient
  - within_1: fraction of predictions within 1 of true label (rounded)

Binary metrics for valid/invalid head:
  - F1 at fixed threshold and best-sweep threshold
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from quality_scorer.metrics import binary_metrics, sweep_invalid_threshold


@torch.no_grad()
def eval_regression_epoch(
    model,
    dataloader,
    device: torch.device,
    invalid_max_score: int,
) -> dict:
    model.eval()
    all_scores: list[float] = []
    all_pred_scores: list[float] = []
    all_binary_probs: list[float] = []
    all_true_scores_int: list[int] = []

    for images, clip_feats, scores, _, _defects in dataloader:
        images = images.to(device, non_blocking=True)
        clip_feats = clip_feats.to(device, non_blocking=True)
        scores = scores.to(device, non_blocking=True)

        pred_score, binary_logit, _defect_logits = model(images, clip_feats)
        binary_prob = torch.sigmoid(binary_logit)

        all_scores.extend(scores.cpu().tolist())
        all_pred_scores.extend(pred_score.cpu().tolist())
        all_binary_probs.extend(binary_prob.cpu().tolist())
        all_true_scores_int.extend(scores.long().cpu().tolist())

    labels = np.asarray(all_scores, dtype=np.float32)
    preds = np.asarray(all_pred_scores, dtype=np.float32)
    labels_int = np.asarray(all_true_scores_int, dtype=np.int64)

    mae = float(np.abs(preds - labels).mean()) if labels.size else 0.0
    within_1 = float((np.abs(np.round(preds) - labels_int) <= 1).mean()) if labels.size else 0.0

    srcc = float(spearmanr(preds, labels).statistic) if labels.size > 1 else 0.0
    plcc = float(pearsonr(preds, labels).statistic) if labels.size > 1 else 0.0

    return {
        "mae": round(mae, 4),
        "within_1": round(within_1, 4),
        "srcc": round(srcc, 4),
        "plcc": round(plcc, 4),
        "binary_at_0_5": binary_metrics(all_binary_probs, all_true_scores_int, 0.5, invalid_max_score),
        "binary_best": sweep_invalid_threshold(all_binary_probs, all_true_scores_int, invalid_max_score),
    }
