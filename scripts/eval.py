"""Evaluate regression checkpoints on the test split.

Usage:
    # Evaluate all channels (looks under outputs/runs/)
    python asset_quality_scorer/scripts/eval.py

    # Single channel
    python asset_quality_scorer/scripts/eval.py --channel metallic

    # Specific exp_id (outputs/runs/{exp_id}/)
    python asset_quality_scorer/scripts/eval.py --exp-id convnext_base_metallic_baseline

    # Specific checkpoint file inside exp dir
    python asset_quality_scorer/scripts/eval.py --ckpt best_srcc

    # Compare all exp_ids that match a pattern
    python asset_quality_scorer/scripts/eval.py --compare
"""
from __future__ import annotations

import argparse
import json
import sys
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
from quality_scorer.data import CHANNEL_DEFECT_COLS, TensorCacheCLIPDataset
from quality_scorer.metrics import binary_metrics, sweep_invalid_threshold
from quality_scorer.models import build_model

RUNS_ROOT = PACKAGE_ROOT / "outputs" / "runs"


def _resolve(p):
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _find_run_dir(exp_id: str | None, channel: str | None) -> Path | None:
    """Resolve the checkpoint directory for a given exp_id or channel."""
    if exp_id:
        d = RUNS_ROOT / exp_id
        if d.exists():
            return d
        # fallback: legacy phase2_regression location
        d2 = PACKAGE_ROOT / "outputs" / "phase2_regression" / exp_id
        return d2 if d2.exists() else None
    if channel:
        # Search runs/ for newest dir containing the channel name
        candidates = sorted(RUNS_ROOT.glob(f"*_{channel}_*"))
        if candidates:
            return candidates[-1]
        # fallback legacy
        d = PACKAGE_ROOT / "outputs" / "phase2_regression" / f"convnext_base_{channel}_regression_v2"
        return d if d.exists() else None
    return None


def eval_run(run_dir: Path, channel: str, ckpt_name: str, device: torch.device) -> dict:
    ckpt_path = run_dir / f"{ckpt_name}.pt"
    if not ckpt_path.exists():
        print(f"  [skip] {ckpt_path} not found")
        return {}

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    invalid_max_score = ckpt.get("invalid_max_score", 1)
    defect_cols = ckpt.get("defect_cols", CHANNEL_DEFECT_COLS.get(channel, []))
    state = ckpt["model_state_dict"]
    has_clip_direct = any(k.startswith("clip_direct") for k in state)
    # spatial fusion → aux_fusion.pos_embed; any aux → aux_fusion.* present
    spatial_aux = any(k.startswith("aux_fusion.pos_embed") for k in state)
    use_aux     = any(k.startswith("aux_fusion") for k in state)
    aux_fusion_mode = "spatial" if spatial_aux else "pooled"
    # Auto-detect clip_dim: LayerNorm weight in clip_direct has shape [clip_dim]
    if has_clip_direct and "clip_direct.0.weight" in state:
        clip_dim = int(state["clip_direct.0.weight"].shape[0])
    else:
        # Infer from cross_modal stage proj + fusion input dim
        clip_dim = 1536

    arch = ckpt.get("arch", "convnext_base")
    model = build_model(
        arch,
        clip_dim=clip_dim, attn_proj_dim=256, attn_heads=4, hidden_dim=512, dropout=0.0,
        n_defect_labels=len(defect_cols), freeze_features=False,
        use_clip_direct=has_clip_direct,
        use_aux=use_aux, aux_proj_dim=256, aux_fusion_mode=aux_fusion_mode,
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    # with_prompts models have clip_dim > 1536 (prompt sims appended)
    channel_cls_path = (
        _resolve("asset_quality_scorer/features/clip_vitl14_openai_scoring_channels.pt")
        if clip_dim > 1536 else None
    )
    test_ds = TensorCacheCLIPDataset(
        tensor_cache_root=_resolve("asset_quality_scorer/cache/224"),
        clip_feature_path=_resolve("asset_quality_scorer/features/clip_vitl14_openai_render_base_color.pt"),
        csv_path=_resolve("asset_quality_scorer/dataset/sampled_all.csv"),
        channel_cls_path=channel_cls_path,
        split="test",
        channel=channel,
        invalid_max_score=invalid_max_score,
        is_train=False,
        defect_cols=defect_cols,
    )
    loader = DataLoader(test_ds, batch_size=128, num_workers=4, pin_memory=device.type == "cuda")
    print(f"  test samples: {len(test_ds)}")

    all_scores, all_preds, all_binary_probs, all_scores_int = [], [], [], []
    with torch.no_grad():
        # dataset returns 6-tuple: (img, aux_imgs, clip_feat, score, binary, defects)
        for batch in loader:
            images, _aux, clip_feats, scores, _binary, _defects = batch
            images     = images.to(device, non_blocking=True)
            clip_feats = clip_feats.to(device, non_blocking=True)
            scores     = scores.to(device, non_blocking=True)
            # aux_images=None → AuxFusion uses empty fallback (main_proj only)
            pred_score, binary_logit, _ = model(images, clip_feats, aux_images=None)
            binary_prob = torch.sigmoid(binary_logit)
            all_scores.extend(scores.cpu().tolist())
            all_preds.extend(pred_score.cpu().tolist())
            all_binary_probs.extend(binary_prob.cpu().tolist())
            all_scores_int.extend(scores.long().cpu().tolist())

    labels   = np.array(all_scores,     dtype=np.float32)
    preds    = np.array(all_preds,      dtype=np.float32)
    labels_i = np.array(all_scores_int, dtype=np.int64)
    preds_i  = np.clip(np.round(preds), 0, 5).astype(np.int64)

    mae       = float(np.abs(preds - labels).mean())
    acc       = float((preds_i == labels_i).mean())
    within1   = float((np.abs(preds_i - labels_i) <= 1).mean())
    srcc      = float(spearmanr(preds, labels).statistic)
    plcc      = float(pearsonr(preds, labels).statistic)
    kappa_lin = float(cohen_kappa_score(labels_i, preds_i, weights="linear"))
    kappa_qwk = float(cohen_kappa_score(labels_i, preds_i, weights="quadratic"))
    b05       = binary_metrics(all_binary_probs, all_scores_int, 0.5, invalid_max_score)
    bbest     = sweep_invalid_threshold(all_binary_probs, all_scores_int, invalid_max_score)

    per_score_mae, per_score_acc, per_score_n = {}, {}, {}
    for s in range(6):
        mask = labels_i == s
        n_s = int(mask.sum())
        if n_s > 0:
            per_score_mae[s] = round(float(np.abs(preds[mask] - labels[mask]).mean()), 4)
            per_score_acc[s] = round(float((preds_i[mask] == labels_i[mask]).mean()), 4)
            per_score_n[s]   = n_s

    result = dict(
        mae=round(mae, 4), srcc=round(srcc, 4), plcc=round(plcc, 4),
        acc=round(acc, 4), within_1=round(within1, 4),
        kappa_lin=round(kappa_lin, 4), kappa_qwk=round(kappa_qwk, 4),
        binary_f1=b05["f1"], binary_best_f1=bbest["f1"], binary_best_thr=bbest["threshold"],
        per_score_mae=per_score_mae, per_score_acc=per_score_acc, per_score_n=per_score_n,
        n=len(labels), val_metrics=ckpt.get("val_metrics", {}), ckpt_epoch=ckpt.get("epoch", "?"),
    )
    # Write eval result alongside the checkpoint
    (run_dir / "eval_test.json").write_text(json.dumps(
        {"exp_id": run_dir.name, "channel": channel, "ckpt": ckpt_name, **result}, indent=2
    ))
    return result


def _print_result(ch: str, r: dict) -> None:
    print(f"  MAE={r['mae']:.4f}  SRCC={r['srcc']:.4f}  PLCC={r['plcc']:.4f}  "
          f"acc={r['acc']:.4f}  within_1={r['within_1']:.4f}  "
          f"kappa_lin={r['kappa_lin']:.4f}  kappa_qwk={r['kappa_qwk']:.4f}  "
          f"F1@0.5={r['binary_f1']:.4f}  bestF1={r['binary_best_f1']:.4f}@{r['binary_best_thr']:.2f}")
    print(f"  per-score MAE: { {k: v for k,v in r['per_score_mae'].items()} }")
    print(f"  per-score acc: { {k: v for k,v in r['per_score_acc'].items()} }")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--channel",  choices=ALL_CHANNELS, default=None)
    p.add_argument("--exp-id",   default=None, help="Specific exp_id under outputs/runs/")
    p.add_argument("--ckpt",     default="best", help="best / best_mae / best_srcc / best_binary_f1")
    p.add_argument("--device",   choices=("cuda", "cpu"), default=None)
    p.add_argument("--compare",  action="store_true", help="Print comparison table for all runs")
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.compare:
        _compare_all(device, args.ckpt)
        return

    channels = [args.channel] if args.channel else ALL_CHANNELS

    print(f"\n{'='*72}")
    print(f"TEST SET evaluation  (ckpt={args.ckpt})")
    print(f"device: {device}")
    print(f"{'='*72}\n")

    results = {}
    for ch in channels:
        run_dir = _find_run_dir(args.exp_id, ch)
        if run_dir is None:
            print(f"[{ch}]  no run dir found — skip")
            continue
        print(f"[{ch}]  {run_dir.name}")
        r = eval_run(run_dir, ch, args.ckpt, device)
        if r:
            results[ch] = r
            _print_result(ch, r)
        print()

    if len(results) > 1:
        _print_summary_table(results)


def _compare_all(device: torch.device, ckpt: str) -> None:
    """Print a comparison table for every exp_id under outputs/runs/."""
    if not RUNS_ROOT.exists():
        print(f"No runs found at {RUNS_ROOT}")
        return
    all_results = {}
    for run_dir in sorted(RUNS_ROOT.iterdir()):
        if not run_dir.is_dir():
            continue
        # Infer channel from exp_id (second segment: convnext_base_{channel}_...)
        parts = run_dir.name.split("_")
        ch = next((c for c in ALL_CHANNELS if c in run_dir.name), None)
        if ch is None:
            continue
        print(f"[{run_dir.name}]")
        r = eval_run(run_dir, ch, ckpt, device)
        if r:
            all_results[run_dir.name] = (ch, r)
        print()
    if all_results:
        _print_summary_table({k: v[1] for k, v in all_results.items()}, label_col="exp_id")


def _print_summary_table(results: dict, label_col: str = "channel") -> None:
    print(f"\n{'─'*110}")
    print(f"{label_col:35}  {'MAE':>6}  {'SRCC':>6}  {'PLCC':>6}  {'acc':>6}  {'w1':>6}  {'QWK':>6}  {'linK':>6}  {'F1@0.5':>7}  {'bestF1':>7}")
    print(f"{'─'*110}")
    for key, r in results.items():
        print(f"{key:35}  {r['mae']:>6.4f}  {r['srcc']:>6.4f}  {r['plcc']:>6.4f}  "
              f"{r['acc']:>6.4f}  {r['within_1']:>6.4f}  {r['kappa_qwk']:>6.4f}  {r['kappa_lin']:>6.4f}  "
              f"{r['binary_f1']:>7.4f}  {r['binary_best_f1']:>7.4f}")


if __name__ == "__main__":
    main()
