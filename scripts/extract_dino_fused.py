"""Extract DINOv2's per-channel pre-head feature (`fused`, the input to each
score head) for a split, so we can train shared-vs-per-channel heads offline on
the FROZEN trained backbone. Hooks the trained score_heads' input.

Dumps outputs/dino_fused_{split}.npz: feats[ch] [N,H], gt[ch] [N], names.

Usage: CUDA_VISIBLE_DEVICES=1 python asset_quality_scorer/scripts/extract_dino_fused.py --split train
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch, yaml
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "asset_quality_scorer"
sys.path.insert(0, str(PROJECT_ROOT)); sys.path.insert(0, str(PACKAGE_ROOT)); sys.path.insert(0, str(PACKAGE_ROOT / "scripts"))
from train_multitask import MultiChannelDataset, _collate, _CHANNELS, _resolve  # noqa
from quality_scorer.models import DINOv2MultiTaskScorer  # noqa
from torch.utils.data import DataLoader  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", default="dinov2_large_multitask_emd_all")  # ckpt run dir
    ap.add_argument("--config", default="dinov2_large_multitask_emd")       # config name (no .yaml)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=24)
    args = ap.parse_args()
    cfg = yaml.safe_load((PACKAGE_ROOT / "config" / f"{args.config}.yaml").read_text())
    d, m = cfg["data"], cfg["model"]
    dev = "cuda"
    ds = MultiChannelDataset(_resolve(d["tensor_cache_root"]), _resolve(d["clip_feature_path"]),
                             _resolve(d["csv_path"]), args.split, False,
                             int(m.get("invalid_max_score", 1)), d.get("pbr_filter"),
                             False, None, bool(d.get("zero_render_clip", False)), None)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=8, collate_fn=_collate, pin_memory=True)
    model = DINOv2MultiTaskScorer(
        clip_dim=int(m.get("clip_dim", 1536)), attn_proj_dim=int(m.get("attn_proj_dim", 256)),
        attn_heads=int(m.get("attn_heads", 4)), hidden_dim=int(m.get("hidden_dim", 512)),
        dropout=float(m.get("dropout", 0.3)), freeze_features=True, use_clip_direct=True,
        metallic_film=bool(m.get("metallic_film", True)), metallic_grad_scale=float(m.get("metallic_grad_scale", 0.5)),
        use_cross_channel=bool(m.get("use_cross_channel", False)), cc_metallic_only=bool(m.get("cc_metallic_only", False)),
        ordinal_channels=m.get("ordinal_channels", None), emd_channels=m.get("emd_channels", None),
        aux_supervision=bool(m.get("aux_supervision", False)), use_attn_pool=bool(m.get("use_attn_pool", False)),
        metallic_no_render=bool(m.get("metallic_no_render", False)),
        metallic_spatial_xchannel=bool(m.get("metallic_spatial_xchannel", False)),
        backbone_name=m.get("backbone_name", "vit_large_patch14_reg4_dinov2"),
        vlm_prior_dim=int(m.get("vlm_prior_dim", 0)), vlm_proj_dim=int(m.get("vlm_proj_dim", 128)),
    ).to(dev)
    state = torch.load(PACKAGE_ROOT / "outputs/runs" / args.exp_id / "best.pt", map_location=dev, weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state)); model.eval()

    # hook each score head's INPUT (= fused) per channel
    grab = {}
    def mk(ch):
        def hook(mod, inp, out): grab[ch] = inp[0].detach().float().cpu()
        return hook
    for ch in _CHANNELS:
        model.score_heads[ch].register_forward_hook(mk(ch))

    feats = {c: [] for c in _CHANNELS}; gts = {c: [] for c in _CHANNELS}
    with torch.no_grad():
        for imgs, clips, scores, _, _, vlm, regw in dl:
            imgs = {c: t.to(dev, non_blocking=True) for c, t in imgs.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(imgs, clips.to(dev))
            for c in _CHANNELS:
                feats[c].append(grab[c]); gts[c].extend(scores[c].tolist())
    out = {f"feat_{c}": torch.cat(feats[c]).numpy().astype(np.float16) for c in _CHANNELS}
    for c in _CHANNELS: out[f"gt_{c}"] = np.array(gts[c])
    out["names"] = np.array([s[0] for s in ds.samples])
    p = PACKAGE_ROOT / "outputs" / f"dino_fused_{args.split}.npz"
    np.savez(p, **out)
    print(f"saved {p}  feat dim={out['feat_metallic'].shape}  N={len(out['names'])}")


if __name__ == "__main__":
    main()
