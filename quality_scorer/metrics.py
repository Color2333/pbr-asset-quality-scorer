"""Metrics for PBR asset quality scoring.

Ordinal / Phase-1:
  compute_pos_weight, eval_ordinal_epoch

Regression / Phase-2 (aligned with NR-IQA literature):
  MAE, SRCC, PLCC, within_1, Cohen's Kappa (linear + quadratic)

Binary (valid/invalid head):
  binary_metrics, sweep_invalid_threshold
  eval_regression_epoch
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from quality_scorer.ordinal import logits_to_class_probs, make_coral_levels


# ── ordinal helpers ───────────────────────────────────────────────────────────

def compute_pos_weight(scores: Iterable[int], num_classes: int, device: torch.device) -> torch.Tensor:
    scores_tensor = torch.tensor(list(scores), dtype=torch.long)
    if scores_tensor.numel() == 0:
        return torch.ones(num_classes - 1, device=device)
    levels = make_coral_levels(scores_tensor, num_classes)
    pos = levels.sum(dim=0)
    neg = levels.shape[0] - pos
    return (neg / pos.clamp_min(1.0)).to(device)


# ── binary (valid/invalid) metrics ───────────────────────────────────────────

def binary_metrics(invalid_probs, scores, threshold: float, invalid_max_score: int) -> dict:
    probs = np.asarray(list(invalid_probs), dtype=np.float32)
    labels = (np.asarray(list(scores), dtype=np.int64) <= invalid_max_score).astype(np.int64)
    preds = (probs >= threshold).astype(np.int64)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall    = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "threshold": round(float(threshold), 4),
        "f1": round(float(f1), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def sweep_invalid_threshold(invalid_probs, scores, invalid_max_score: int,
                             lo=0.05, hi=0.95, steps=45) -> dict:
    best = None
    for threshold in np.linspace(lo, hi, steps + 1):
        m = binary_metrics(invalid_probs, scores, float(threshold), invalid_max_score)
        if best is None or m["f1"] > best["f1"]:
            best = m
    return best or binary_metrics([], [], 0.5, invalid_max_score)


# ── Phase-1 ordinal eval ──────────────────────────────────────────────────────

@torch.no_grad()
def eval_ordinal_epoch(
    model,
    dataloader,
    device: torch.device,
    num_classes: int,
    invalid_max_score: int,
) -> dict:
    model.eval()
    all_scores: list[int] = []
    all_pred_scores: list[int] = []
    all_invalid_probs: list[float] = []
    all_expected_scores: list[float] = []

    class_values = torch.arange(num_classes, device=device, dtype=torch.float32).view(1, -1)
    for images, scores in dataloader:
        images = images.to(device, non_blocking=True)
        scores = scores.to(device, non_blocking=True)
        logits = model(images)
        class_probs = logits_to_class_probs(logits)
        pred_scores = class_probs.argmax(dim=1)
        invalid_probs = class_probs[:, : invalid_max_score + 1].sum(dim=1)
        expected_scores = (class_probs * class_values).sum(dim=1)

        all_scores.extend(scores.cpu().tolist())
        all_pred_scores.extend(pred_scores.cpu().tolist())
        all_invalid_probs.extend(invalid_probs.cpu().tolist())
        all_expected_scores.extend(expected_scores.cpu().tolist())

    labels        = np.asarray(all_scores,         dtype=np.int64)
    argmax_preds  = np.asarray(all_pred_scores,    dtype=np.int64)
    expected      = np.asarray(all_expected_scores, dtype=np.float32)
    expected_preds = np.clip(np.round(expected).astype(np.int64), 0, num_classes - 1)

    acc              = float((argmax_preds == labels).mean())       if labels.size else 0.0
    ordinal_mae      = float(np.abs(argmax_preds - labels).mean())  if labels.size else 0.0
    expected_mae     = float(np.abs(expected - labels).mean())      if labels.size else 0.0
    within_1         = float((np.abs(argmax_preds - labels) <= 1).mean()) if labels.size else 0.0
    expected_within_1 = float((np.abs(expected_preds - labels) <= 1).mean()) if labels.size else 0.0
    return {
        "ordinal_acc": round(acc, 4),
        "ordinal_mae": round(ordinal_mae, 4),
        "expected_mae": round(expected_mae, 4),
        "within_1": round(within_1, 4),
        "expected_within_1": round(expected_within_1, 4),
        "binary_at_0_5": binary_metrics(all_invalid_probs, all_scores, 0.5, invalid_max_score),
        "binary_best": sweep_invalid_threshold(all_invalid_probs, all_scores, invalid_max_score),
    }


# ── Phase-2 regression eval ───────────────────────────────────────────────────

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
        images     = images.to(device, non_blocking=True)
        clip_feats = clip_feats.to(device, non_blocking=True)
        scores     = scores.to(device, non_blocking=True)
        pred_score, binary_logit, _defect_logits = model(images, clip_feats)
        binary_prob = torch.sigmoid(binary_logit)

        all_scores.extend(scores.cpu().tolist())
        all_pred_scores.extend(pred_score.cpu().tolist())
        all_binary_probs.extend(binary_prob.cpu().tolist())
        all_true_scores_int.extend(scores.long().cpu().tolist())

    labels     = np.asarray(all_scores,         dtype=np.float32)
    preds      = np.asarray(all_pred_scores,    dtype=np.float32)
    labels_int = np.asarray(all_true_scores_int, dtype=np.int64)

    mae     = float(np.abs(preds - labels).mean())                         if labels.size else 0.0
    within_1 = float((np.abs(np.round(preds) - labels_int) <= 1).mean())  if labels.size else 0.0
    srcc    = float(spearmanr(preds, labels).statistic)                    if labels.size > 1 else 0.0
    plcc    = float(pearsonr(preds, labels).statistic)                     if labels.size > 1 else 0.0
    return {
        "mae": round(mae, 4),
        "within_1": round(within_1, 4),
        "srcc": round(srcc, 4),
        "plcc": round(plcc, 4),
        "binary_at_0_5": binary_metrics(all_binary_probs, all_true_scores_int, 0.5, invalid_max_score),
        "binary_best": sweep_invalid_threshold(all_binary_probs, all_true_scores_int, invalid_max_score),
    }
