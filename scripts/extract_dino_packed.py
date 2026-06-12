"""Channel-specific PACKED inputs through the frozen trained DINOv2, extract fused
features (test split), to test "special per-channel input" cheaply (K-fold OOF
head later). Packing (uint8, from cache/224):
  metallic  : [metallic, adaptive-brightened metallic (per-img p99 stretch), base luma]
  roughness : [roughness, base luma, render luma]   (material context)
  normal_map: original RGB (control)
  base_color: original RGB (control)
Frozen backbone+fusion as feature extractor; heads retrained on the fused output.

Dumps outputs/dino_packed_fused_test.npz. Usage:
  CUDA_VISIBLE_DEVICES=1 python asset_quality_scorer/scripts/extract_dino_packed.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch, yaml
PROJECT_ROOT = Path(__file__).resolve().parents[2]; PKG = PROJECT_ROOT / "asset_quality_scorer"
sys.path.insert(0, str(PROJECT_ROOT)); sys.path.insert(0, str(PKG)); sys.path.insert(0, str(PKG / "scripts"))
from train_multitask import MultiChannelDataset, _CHANNELS, _resolve, IMAGENET_MEAN, IMAGENET_STD  # noqa
from quality_scorer.models import DINOv2MultiTaskScorer  # noqa

cfg = yaml.safe_load((PKG / "config/dinov2_large_multitask_emd.yaml").read_text())
m = cfg["model"]; d = cfg["data"]; dev = "cuda"
CACHE = _resolve(d["tensor_cache_root"])
meta = json.loads((CACHE / "meta.json").read_text())["model_names"]
name2idx = {n: i for i, n in enumerate(meta)}

# test split sample order/names/gt via the dataset (same as eval)
ds = MultiChannelDataset(CACHE, _resolve(d["clip_feature_path"]), _resolve(d["csv_path"]), "test",
                         False, int(m.get("invalid_max_score", 1)), d.get("pbr_filter"), False, None,
                         bool(d.get("zero_render_clip", False)), None)
names = [s[0] for s in ds.samples]; cidx = np.array([name2idx[n] for n in names])
# load raw cache (uint8 N,3,224,224) for needed channels at test indices
raw = {c: np.load(CACHE / f"{c}.npy", mmap_mode="r")[cidx] for c in ["base_color", "normal_map", "roughness", "metallic", "render"]}
luma = lambda x: (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).astype(np.uint8)  # [n,224,224]
base_l, rend_l, me_g, ro_g = luma(raw["base_color"]), luma(raw["render"]), luma(raw["metallic"]), luma(raw["roughness"])


def adaptive_brighten(g):  # per-image p99 stretch -> expose faint near-black grey
    out = np.empty_like(g)
    for i in range(len(g)):
        p = np.percentile(g[i], 99); out[i] = np.clip(g[i] / max(p, 4) * 255, 0, 255)
    return out.astype(np.uint8)


me_bright = adaptive_brighten(me_g)
packed = {
    "metallic":  np.stack([me_g, me_bright, base_l], 1),          # [n,3,224,224]
    "roughness": np.stack([ro_g, base_l, rend_l], 1),
    "normal_map": np.asarray(raw["normal_map"]),                  # control: original RGB
    "base_color": np.asarray(raw["base_color"]),
}
# gt + clip vectors via dataset __getitem__ (authoritative, matches eval order)
gt = {c: [] for c in _CHANNELS}; clip_list = []
for i in range(len(ds)):
    _, clips_i, sc, *_ = ds[i]
    clip_list.append(clips_i)
    for c in _CHANNELS: gt[c].append(int(sc[c]))
gt = {c: np.array(v) for c, v in gt.items()}
clipfeat = torch.stack(clip_list).float()

# build model, load emd_all ckpt
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
st = torch.load(PKG / "outputs/runs/dinov2_large_multitask_emd_all/best.pt", map_location=dev, weights_only=False)
model.load_state_dict(st.get("model_state_dict", st)); model.eval()
grab = {}
for ch in _CHANNELS:
    model.score_heads[ch].register_forward_hook((lambda c: (lambda mod, i, o: grab.__setitem__(c, i[0].detach().float().cpu())))(ch))

mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1); std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
feats = {c: [] for c in _CHANNELS}; B = 24
with torch.no_grad():
    for i in range(0, len(names), B):
        imgs = {c: ((torch.tensor(packed[c][i:i+B]).float() / 255.0 - mean) / std).to(dev) for c in _CHANNELS}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(imgs, clipfeat[i:i+B].to(dev))
        for c in _CHANNELS: feats[c].append(grab[c])
        if (i // B) % 50 == 0: print(f"  {i}/{len(names)}", flush=True)
out = {f"feat_{c}": torch.cat(feats[c]).numpy().astype(np.float16) for c in _CHANNELS}
out.update({f"gt_{c}": gt[c] for c in _CHANNELS}); out["names"] = np.array(names)
np.savez(PKG / "outputs/dino_packed_fused_test.npz", **out)
print("saved dino_packed_fused_test.npz", out["feat_metallic"].shape)
