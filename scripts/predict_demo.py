"""One-shot precompute: run the best multitask model over the test split and
dump per-asset predictions (pred vs ground-truth, all 4 channels) to JSON for
the demo page. No GPU held by the web server — the demo just reads this JSON.

Usage:
    CUDA_VISIBLE_DEVICES=0 python asset_quality_scorer/scripts/predict_demo.py \
        --exp dinov2_large_multitask_emd_all [--limit 0] [--batch 8]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "asset_quality_scorer"
sys.path.insert(0, str(PROJECT_ROOT)); sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT / "scripts"))

from train_multitask import MultiChannelDataset, _collate, _CHANNELS
from quality_scorer.models import DINOv2MultiTaskScorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="dinov2_large_multitask_emd_all")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = PACKAGE_ROOT / "outputs" / "runs" / args.exp
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)

    # infer head config from the experiment name / checkpoint
    emd = "emd" in args.exp
    aux = "aux" in args.exp
    dp  = 0.2 if "dp" in args.exp.split("_") else 0.0
    model = DINOv2MultiTaskScorer(
        metallic_film=False, freeze_features=False,
        emd_channels=("all" if emd else None),
        aux_supervision=aux, drop_path_rate=dp,
    ).to(dev)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    print(f"loaded {args.exp} (emd={emd} aux={aux})")

    ds = MultiChannelDataset(
        PACKAGE_ROOT / "cache/224",
        PACKAGE_ROOT / "features/clip_vitl14_openai_render_base_color.pt",
        PACKAGE_ROOT / "dataset/sampled_all.csv",
        args.split, is_train=False, invalid_max_score=1)
    names = [s[0] for s in ds.samples]
    if args.limit and len(ds.samples) > args.limit:
        import random; random.seed(0)
        keep = sorted(random.sample(range(len(ds.samples)), args.limit))
        ds.samples = [ds.samples[i] for i in keep]
        names = [names[i] for i in keep]
    dl = DataLoader(ds, batch_size=args.batch, num_workers=4, collate_fn=_collate)
    print(f"{len(ds.samples)} {args.split} assets")

    preds = {ch: [] for ch in _CHANNELS}
    gts   = {ch: [] for ch in _CHANNELS}
    with torch.no_grad():
        for imgs, clips, scores, _, _ in dl:
            imgs = {c: v.to(dev) for c, v in imgs.items()}; clips = clips.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                out = model(imgs, clips)
            for ch in _CHANNELS:
                preds[ch].extend(out[ch][0].float().cpu().tolist())
                gts[ch].extend(scores[ch].tolist())

    # per-channel SRCC for the header
    srcc = {ch: round(float(spearmanr(preds[ch], gts[ch]).statistic), 4) for ch in _CHANNELS}
    srcc_mean = round(float(np.mean(list(srcc.values()))), 4)

    assets = []
    for i, name in enumerate(names):
        assets.append({
            "name": name,
            "pred": {ch: round(float(preds[ch][i]), 2) for ch in _CHANNELS},
            "gt":   {ch: int(gts[ch][i]) for ch in _CHANNELS},
        })

    out_json = {
        "exp_id": args.exp, "split": args.split,
        "srcc": srcc, "srcc_mean": srcc_mean,
        "n": len(assets), "assets": assets,
    }
    out_path = run_dir / "demo_predictions.json"
    out_path.write_text(json.dumps(out_json, ensure_ascii=False, indent=1))
    print(f"per-channel SRCC: {srcc}  mean={srcc_mean}")
    print(f"wrote {out_path}  ({len(assets)} assets)")


if __name__ == "__main__":
    main()
