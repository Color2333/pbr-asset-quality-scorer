"""Convert an eval_multitask-style preds.npy ({preds,gts,names}) into the
demo_predictions.json schema. For DINOv2/ConvNeXt runs evaluated via eval_multitask.
Usage: python ... --run archive/convnext_base_multitask_emd --label "ConvNeXt-B EMD"
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
PKG = Path(__file__).resolve().parents[1]
CH = ["base_color", "normal_map", "roughness", "metallic"]
ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--label", default=None)
a = ap.parse_args()
rd = PKG / "outputs/runs" / a.run
d = np.load(rd / "preds.npy", allow_pickle=True).item()
preds, gts, names = d["preds"], d["gts"], d["names"]
srcc = {c: round(float(spearmanr(preds[c], gts[c]).statistic), 4) for c in CH}
assets = [{"name": names[i], "pred": {c: round(float(preds[c][i]), 2) for c in CH},
           "gt": {c: int(gts[c][i]) for c in CH}} for i in range(len(names))]
out = {"exp_id": a.label or a.run, "split": "test", "srcc": srcc,
       "srcc_mean": round(float(np.mean(list(srcc.values()))), 4), "n": len(assets), "assets": assets}
(rd / "demo_predictions.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(f"wrote {rd}/demo_predictions.json  mean={out['srcc_mean']}  ({len(assets)})")
