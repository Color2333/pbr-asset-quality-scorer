"""Evaluate a multitask checkpoint on the test split. One protocol for every
run (baseline / vlmprior / 100k) so comparisons are apples-to-apples.

Reports per-channel SRCC/MAE + the metallic near-black subset (the actual
battleground): within-near-black SRCC and missing-vs-correct AUC (pred score
inverted as the detector).

Usage:
    python asset_quality_scorer/scripts/eval_multitask.py --exp-id dinov2_large_multitask_emd_vlmprior
    python asset_quality_scorer/scripts/eval_multitask.py --exp-id dinov2_large_multitask_emd_all
Writes outputs/runs/{exp_id}/eval_test.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "asset_quality_scorer"
sys.path.insert(0, str(PROJECT_ROOT)); sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT / "scripts"))

from train_multitask import MultiChannelDataset, _collate, _CHANNELS, _resolve  # noqa: E402
from quality_scorer.models import DINOv2MultiTaskScorer  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--config", default=None, help="defaults to config/{exp_id}.yaml")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--batch-size", type=int, default=24)
    args = ap.parse_args()

    cfg_path = Path(args.config) if args.config else PACKAGE_ROOT / "config" / f"{args.exp_id}.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    d, m = cfg["data"], cfg["model"]
    run_dir = PACKAGE_ROOT / "outputs/runs" / args.exp_id
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vlm_cfg = d.get("vlm_prior")
    if vlm_cfg:
        vlm_cfg = dict(vlm_cfg)
        vlm_cfg["path_prefix"] = str(_resolve(vlm_cfg["path_prefix"]))
    ds = MultiChannelDataset(_resolve(d["tensor_cache_root"]), _resolve(d["clip_feature_path"]),
                             _resolve(d["csv_path"]), "test", False,
                             int(m.get("invalid_max_score", 1)), d.get("pbr_filter"),
                             False, None, bool(d.get("zero_render_clip", False)), vlm_cfg,
                             float(d.get("metallic_stretch", 1.0)))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=8,
                    collate_fn=_collate, pin_memory=True)
    print(f"exp={args.exp_id}  test={len(ds)}  vlm_prior={'on' if vlm_cfg else 'off'}")

    backbone = m.get("backbone", "dinov2")
    if backbone == "convnext":
        from quality_scorer.models import ConvNeXtMultiTaskScorer
        model = ConvNeXtMultiTaskScorer(
            clip_dim=int(m.get("clip_dim", 1536)), attn_proj_dim=int(m.get("attn_proj_dim", 256)),
            attn_heads=int(m.get("attn_heads", 4)), hidden_dim=int(m.get("hidden_dim", 512)),
            dropout=float(m.get("dropout", 0.3)), freeze_features=True, use_clip_direct=True,
            metallic_grad_scale=float(m.get("metallic_grad_scale", 0.5)),
            ordinal_channels=m.get("ordinal_channels", None),
            emd_channels=m.get("emd_channels", None),
        ).to(device)
    else:
        model = DINOv2MultiTaskScorer(
            clip_dim=int(m.get("clip_dim", 1536)), attn_proj_dim=int(m.get("attn_proj_dim", 256)),
            attn_heads=int(m.get("attn_heads", 4)), hidden_dim=int(m.get("hidden_dim", 512)),
            dropout=float(m.get("dropout", 0.3)), freeze_features=True, use_clip_direct=True,
            metallic_film=bool(m.get("metallic_film", True)),
            metallic_grad_scale=float(m.get("metallic_grad_scale", 0.5)),
            use_cross_channel=bool(m.get("use_cross_channel", False)),
            cc_metallic_only=bool(m.get("cc_metallic_only", False)),
            ordinal_channels=m.get("ordinal_channels", None),
            emd_channels=m.get("emd_channels", None),
            aux_supervision=bool(m.get("aux_supervision", False)),
            use_attn_pool=bool(m.get("use_attn_pool", False)),
            metallic_no_render=bool(m.get("metallic_no_render", False)),
            metallic_spatial_xchannel=bool(m.get("metallic_spatial_xchannel", False)),
            backbone_name=m.get("backbone_name", "vit_large_patch14_reg4_dinov2"),
            vlm_prior_dim=int(m.get("vlm_prior_dim", 0)),
            vlm_proj_dim=int(m.get("vlm_proj_dim", 128)),
        ).to(device)
    state = torch.load(run_dir / args.ckpt, map_location=device, weights_only=False)
    sd = state.get("model_state_dict", state)
    model.load_state_dict(sd)
    model.eval()
    ep = state.get("epoch", "?")
    print(f"loaded {args.ckpt} (epoch {ep})")

    preds = {ch: [] for ch in _CHANNELS}; gts = {ch: [] for ch in _CHANNELS}
    with torch.no_grad():
        for imgs, clips, scores, _, _, vlm, regw in dl:
            imgs = {ch: t.to(device, non_blocking=True) for ch, t in imgs.items()}
            clips = clips.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                out = (model(imgs, clips, vlm_prior=vlm.to(device), vlm_regime_w=regw.to(device)) if vlm_cfg
                       else model(imgs, clips))
            for ch in _CHANNELS:
                preds[ch].extend(out[ch][0].float().cpu().tolist())
                gts[ch].extend(scores[ch].tolist())

    res = {"exp_id": args.exp_id, "ckpt": args.ckpt, "epoch": ep, "n_test": len(ds)}
    srccs = []
    for ch in _CHANNELS:
        p, g = np.array(preds[ch]), np.array(gts[ch])
        s = float(spearmanr(p, g).statistic); srccs.append(s)
        res[ch] = {"srcc": round(s, 4), "mae": round(float(np.abs(p - g).mean()), 4)}
    res["srcc_mean"] = round(float(np.mean(srccs)), 4)

    # metallic near-black battleground
    meta = json.loads((_resolve(d["tensor_cache_root"]) / "meta.json").read_text())
    frac_all = np.load(_resolve(d["csv_path"]).parent / "metallic_nonblack.npy") \
        if (_resolve(d["csv_path"]).parent / "metallic_nonblack.npy").exists() else None
    if frac_all is not None and len(frac_all) == len(meta["model_names"]):
        ci = np.array([s[2] for s in ds.samples])
        nb = frac_all[ci] < 0.02
        p, g = np.array(preds["metallic"]), np.array(gts["metallic"])
        y_missing = (g[nb] <= 2).astype(int)
        res["metallic_nearblack"] = {
            "n": int(nb.sum()),
            "srcc": round(float(spearmanr(p[nb], g[nb]).statistic), 4),
            "auc_missing": round(float(roc_auc_score(y_missing, -p[nb])), 4),
            "srcc_nonblack": round(float(spearmanr(p[~nb], g[~nb]).statistic), 4),
        }
    import numpy as _np
    _np.save(run_dir / "preds.npy", {"preds": {c: _np.array(preds[c]) for c in _CHANNELS},
             "gts": {c: _np.array(gts[c]) for c in _CHANNELS},
             "names": [s[0] for s in ds.samples]}, allow_pickle=True)
    print(json.dumps(res, indent=2))
    (run_dir / "eval_test.json").write_text(json.dumps(res, indent=2))
    print(f"saved -> {run_dir}/eval_test.json")


if __name__ == "__main__":
    main()
