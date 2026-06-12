"""Build a PACKED tensor cache (cache/224_packed) for retraining DINOv2 to read
channel-specific multi-content inputs:
  metallic  : [metallic, adaptive-brightened metallic (per-img p99 stretch), base luma]
  roughness : [roughness, base luma, render luma]
  normal_map, base_color : original RGB (unchanged)
  render, white_* : copied so the cache is complete.
All uint8 (N,3,224,224), same layout as cache/224, so train_multitask reads it
unchanged via tensor_cache_root.

Usage: python asset_quality_scorer/scripts/build_packed_cache.py
"""
from __future__ import annotations
import shutil
from pathlib import Path
import numpy as np
PKG = Path(__file__).resolve().parents[1]
SRC = PKG / "cache/224"; DST = PKG / "cache/224_packed"
DST.mkdir(parents=True, exist_ok=True)
shutil.copy(SRC / "meta.json", DST / "meta.json")

luma = lambda x: (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).astype(np.uint8)


def adaptive_brighten(g):  # per-image p99 stretch (expose faint near-black grey, no global saturation)
    out = np.empty_like(g)
    for i in range(len(g)):
        p = np.percentile(g[i], 99); out[i] = np.clip(g[i] / max(p, 4) * 255, 0, 255)
    return out.astype(np.uint8)


print("loading source channels...", flush=True)
me = np.load(SRC / "metallic.npy"); ro = np.load(SRC / "roughness.npy")
bc = np.load(SRC / "base_color.npy"); rn = np.load(SRC / "render.npy")
base_l, rend_l = luma(bc), luma(rn)
me_g, ro_g = luma(me), luma(ro)
print("adaptive-brightening metallic...", flush=True)
me_bright = adaptive_brighten(me_g)

np.save(DST / "metallic.npy", np.stack([me_g, me_bright, base_l], 1))
np.save(DST / "roughness.npy", np.stack([ro_g, base_l, rend_l], 1))
print("packed metallic + roughness saved", flush=True)
# copy unchanged channels
for c in ["base_color", "normal_map", "render", "white_model", "white_with_normal"]:
    if (SRC / f"{c}.npy").exists():
        shutil.copy(SRC / f"{c}.npy", DST / f"{c}.npy")
print(f"done -> {DST}")
