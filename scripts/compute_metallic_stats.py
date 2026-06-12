"""One-shot: compute per-asset metallic-map non-black fraction, aligned to the
tensor-cache model_names order. Used to select synthetic missing-metal negatives
(assets that genuinely HAVE metal → zero their metallic map → clean low-score
negative). CPU only. Writes dataset/metallic_nonblack.npy.
"""
from pathlib import Path
import json, numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
cache = PACKAGE_ROOT / "cache/224"
meta = json.loads((cache / "meta.json").read_text())
names = meta["model_names"]
arr = np.load(cache / "metallic.npy", mmap_mode="r")   # [N,3,224,224] uint8
N = len(names)
print(f"{N} assets; computing non-black fraction…")

frac = np.zeros(N, dtype=np.float32)
for i in range(N):
    img = np.asarray(arr[i], dtype=np.float32) / 255.0
    frac[i] = float((img.max(0) > 0.06).mean())
    if i % 5000 == 0:
        print(f"  {i}/{N}")

out = PACKAGE_ROOT / "dataset" / "metallic_nonblack.npy"
np.save(out, frac)
print(f"wrote {out}")
print(f"  非黑占比分布: <2%={float((frac<0.02).mean()):.2f}  "
      f">20%={float((frac>0.2).mean()):.2f}  >30%={float((frac>0.3).mean()):.2f}  "
      f">50%={float((frac>0.5).mean()):.2f}")
