"""Evaluate Phase 2 regression checkpoints on the test split.

Usage:
    python asset_quality_scorer/scripts/eval_v2_regression.py
    python asset_quality_scorer/scripts/eval_v2_regression.py --channel metallic
    python asset_quality_scorer/scripts/eval_v2_regression.py --ckpt best_srcc  # best / best_mae / best_srcc / best_binary_f1
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "asset_quality_scorer"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT))

from quality_scorer.constants import ALL_CHANNELS
from quality_scorer.convnext_regression import ConvNeXtRegressionScorer
from quality_scorer.data_v2 import CHANNEL_DEFECT_COLS, TensorCacheCLIPDataset, build_score_lookup
from quality_scorer.metrics import binary_metrics, sweep_invalid_threshold


def _resolve(p):
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def eval_channel(channel: str, ckpt_name: str, device: torch.device) -> dict:
    output_dir = _resolve(f"asset_quality_scorer/outputs/phase2_regression/convnext_base_{channel}_regression_v2")
    ckpt_path = output_dir / f"{ckpt_name}.pt"
    if not ckpt_path.exists():
        print(f"  [skip] {ckpt_path} not found")
        return {}

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    invalid_max_score = ckpt.get("invalid_max_score", 1)
    defect_cols = ckpt.get("defect_cols", CHANNEL_DEFECT_COLS.get(channel, []))

    # Detect architecture version from checkpoint: old checkpoints lack clip_direct
    state = ckpt["model_state_dict"]
    has_clip_direct = any(k.startswith("clip_direct") for k in state)

    model = ConvNeXtRegressionScorer(
        clip_dim=1536,
        attn_proj_dim=256,
        attn_heads=4,
        hidden_dim=512,
        dropout=0.0,
        n_defect_labels=len(defect_cols),
        freeze_features=False,
        use_clip_direct=has_clip_direct,
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    score_by_model = build_score_lookup(
        _resolve("screening/data_38k"), channel
    )
    test_ds = TensorCacheCLIPDataset(
        tensor_cache_root=_resolve("screening/cache_224_tensors"),
        clip_feature_path=_resolve("screening/features/clip_vitl14_openai_base_color_render.pt"),
        split_image_root=_resolve("screening/data_v2") / channel,
        split="test",
        channel=channel,
        score_by_model=score_by_model,
        invalid_max_score=invalid_max_score,
        is_train=False,
        manifest_path=_resolve("screening/channel_store_38k/manifest.csv"),
        defect_cols=defect_cols,
    )
    loader = DataLoader(
        test_ds, batch_size=128, num_workers=4,
        pin_memory=device.type == "cuda",
    )
    print(f"  test samples: {len(test_ds)}")

    all_scores, all_preds, all_binary_probs, all_scores_int = [], [], [], []

    with torch.no_grad():
        for images, clip_feats, scores, _, _defects in loader:
            images    = images.to(device, non_blocking=True)
            clip_feats = clip_feats.to(device, non_blocking=True)
            scores    = scores.to(device, non_blocking=True)
            pred_score, binary_logit, _ = model(images, clip_feats)
            binary_prob = torch.sigmoid(binary_logit)
            all_scores.extend(scores.cpu().tolist())
            all_preds.extend(pred_score.cpu().tolist())
            all_binary_probs.extend(binary_prob.cpu().tolist())
            all_scores_int.extend(scores.long().cpu().tolist())

    labels   = np.array(all_scores,     dtype=np.float32)
    preds    = np.array(all_preds,       dtype=np.float32)
    labels_i = np.array(all_scores_int, dtype=np.int64)

    preds_i   = np.clip(np.round(preds), 0, 5).astype(np.int64)
    mae       = float(np.abs(preds - labels).mean())
    acc       = float((preds_i == labels_i).mean())
    within1   = float((np.abs(preds_i - labels_i) <= 1).mean())
    srcc      = float(spearmanr(preds, labels).statistic)
    plcc      = float(pearsonr(preds, labels).statistic)
    kappa_lin = float(cohen_kappa_score(labels_i, preds_i, weights="linear"))
    kappa_qwk = float(cohen_kappa_score(labels_i, preds_i, weights="quadratic"))
    b05       = binary_metrics(all_binary_probs, all_scores_int, 0.5, invalid_max_score)
    bbest     = sweep_invalid_threshold(all_binary_probs, all_scores_int, invalid_max_score)

    # Per-score breakdown
    per_score_mae = {}
    per_score_acc = {}
    per_score_n   = {}
    for s in range(6):
        mask = labels_i == s
        n_s = int(mask.sum())
        if n_s > 0:
            per_score_mae[s] = round(float(np.abs(preds[mask] - labels[mask]).mean()), 4)
            per_score_acc[s] = round(float((np.round(preds[mask]).astype(np.int64) == labels_i[mask]).mean()), 4)
            per_score_n[s]   = n_s

    return dict(
        mae=round(mae, 4), srcc=round(srcc, 4), plcc=round(plcc, 4),
        acc=round(acc, 4), within_1=round(within1, 4),
        kappa_lin=round(kappa_lin, 4), kappa_qwk=round(kappa_qwk, 4),
        binary_f1=b05["f1"], binary_best_f1=bbest["f1"],
        binary_best_thr=bbest["threshold"],
        per_score_mae=per_score_mae,
        per_score_acc=per_score_acc,
        per_score_n=per_score_n,
        n=len(labels),
        val_metrics=ckpt.get("val_metrics", {}),
        ckpt_epoch=ckpt.get("epoch", "?"),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--channel", choices=ALL_CHANNELS, default=None)
    p.add_argument("--ckpt", default="best", help="best / best_mae / best_srcc / best_binary_f1")
    p.add_argument("--device", choices=("cuda", "cpu"), default=None)
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    channels = [args.channel] if args.channel else ALL_CHANNELS

    print(f"\n{'='*72}")
    print(f"Phase 2 regression — TEST SET evaluation  (ckpt={args.ckpt})")
    print(f"device: {device}")
    print(f"{'='*72}\n")

    results = {}
    for ch in channels:
        print(f"[{ch}]  (val best at epoch {_get_best_epoch(ch)})")
        r = eval_channel(ch, args.ckpt, device)
        if r:
            results[ch] = r
            print(f"  MAE={r['mae']:.4f}  SRCC={r['srcc']:.4f}  PLCC={r['plcc']:.4f}  "
                  f"acc={r['acc']:.4f}  within_1={r['within_1']:.4f}  "
                  f"kappa_lin={r['kappa_lin']:.4f}  kappa_qwk={r['kappa_qwk']:.4f}  "
                  f"F1@0.5={r['binary_f1']:.4f}  bestF1={r['binary_best_f1']:.4f}@{r['binary_best_thr']:.2f}")
            print(f"  per-score MAE: { {k: v for k,v in r['per_score_mae'].items()} }")
            print(f"  per-score acc: { {k: v for k,v in r['per_score_acc'].items()} }")
        print()

    if len(results) > 1:
        print(f"\n{'─'*100}")
        print(f"{'channel':12}  {'MAE':>6}  {'SRCC':>6}  {'PLCC':>6}  {'acc':>6}  {'w1':>6}  {'QWK':>6}  {'linK':>6}  {'F1@0.5':>7}  {'bestF1':>7}")
        print(f"{'─'*100}")
        for ch, r in results.items():
            print(f"{ch:12}  {r['mae']:>6.4f}  {r['srcc']:>6.4f}  {r['plcc']:>6.4f}  "
                  f"{r['acc']:>6.4f}  {r['within_1']:>6.4f}  {r['kappa_qwk']:>6.4f}  {r['kappa_lin']:>6.4f}  "
                  f"{r['binary_f1']:>7.4f}  {r['binary_best_f1']:>7.4f}")


def _get_best_epoch(channel: str) -> str:
    try:
        import json
        p = PROJECT_ROOT / f"asset_quality_scorer/outputs/phase2_regression/convnext_base_{channel}_regression_v2/summary.json"
        d = json.loads(p.read_text())
        mae_list = d["last_log"]["val_mae"]
        ep = d["last_log"]["epoch"][mae_list.index(min(mae_list))]
        return str(ep)
    except Exception:
        return "?"


if __name__ == "__main__":
    main()
