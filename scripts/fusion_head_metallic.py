"""Fusion head for metallic: learn a small head over [model preds + fullcover 6-bin
probs + SOP map-grey features + near-black frac], leakage-free K-fold OOF on the
test set. Tests whether learned fusion (esp. + the SOP grey signal) beats simple
average / single-best, on BOTH SRCC and the protect-excellence guard score.

The SOP signal (key new ingredient): the gold-set finding showed near-black GT
scores "map cleanliness per SOP" — unreasonable grey in non-metal regions = low.
We operationalize it as map-grey features (fraction/mean/spread of mid-tone pixels
in the metallic map) and see if a head can use it.

Usage: python asset_quality_scorer/scripts/fusion_head_metallic.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "scripts"))
from vlm_scorer_eval import build_items

rows = build_items("test"); names = [r["name"] for r in rows]
gt = np.array([r["scores"]["metallic"] for r in rows], float)
N = len(gt); is5 = gt == 5


def pred(run):
    m = {a["name"]: a["pred"]["metallic"] for a in
         json.loads((PKG / "outputs/runs" / run / "demo_predictions.json").read_text())["assets"]}
    return np.array([m[n] for n in names])


dino = pred("dinov2_large_multitask_emd_all")
conv = pred("archive/convnext_base_multitask_emd")
q60 = pred("vlm_scorer_a_old50k_oldtest")
qfc = pred("vlm_scorer_fullcover_oldtest")
probs = np.load(PKG / "outputs/runs/vlm_scorer_fullcover_oldtest/probs.npy", allow_pickle=True).item()["metallic"]  # (N,6)
nbf = dict(zip(json.loads((PKG / "cache/224/meta.json").read_text())["model_names"],
               np.load(PKG / "dataset/metallic_nonblack.npy")))
nb = np.array([nbf.get(n, 1.0) for n in names])

# SOP map-grey features: load metallic map (downsized), measure the "unreasonable grey"
_cache = PKG / "outputs/sop_grey_feats.npy"
if _cache.exists():
    GREY = np.load(_cache); print(f"loaded cached SOP grey feats {GREY.shape}", flush=True)
else:
    print("computing SOP map-grey features (loading metallic maps)...", flush=True)
    GREY = np.zeros((N, 4))
    for i, n in enumerate(names):
        p = PKG.parent / "datasets0526/metallic" / f"{n}.png"
        try:
            im = Image.open(p).convert("L"); im.thumbnail((256, 256)); v = np.asarray(im, float).ravel() / 255.0
            grey = ((v > 0.04) & (v < 0.96))             # ambiguous mid-tone (not clean-0, not solid metal)
            GREY[i] = [grey.mean(), v[grey].mean() if grey.any() else 0,
                       v[grey].std() if grey.any() else 0, float((v > 0.04).mean())]
        except Exception:
            GREY[i] = [0, 0, 0, 0]
        if (i + 1) % 1500 == 0: print(f"  {i+1}/{N}", flush=True)
    np.save(_cache, GREY)

preds = {"DINOv2": dino, "ConvNeXt": conv, "Qwen60k": q60, "Qwen-full": qfc}
base = np.column_stack([dino, conv, q60, qfc])           # 4 model point preds
FEATS = {
    "仅模型预测(4)": base,
    "+ fullcover概率(6)": np.column_stack([base, probs]),
    "+ SOP灰特征(4)": np.column_stack([base, probs, GREY]),
    "+ 近黑frac": np.column_stack([base, probs, GREY, nb]),
}
HEADS = {"Ridge": lambda: Ridge(alpha=1.0),
         "GBM": lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05),
         "MLP": lambda: MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=800,
                                     early_stopping=True, random_state=0)}


def oof(X, fn, seed=0):
    kf = KFold(5, shuffle=True, random_state=seed); pr = np.zeros(N)
    Xn = (X - X.mean(0)) / (X.std(0) + 1e-6)
    for tr, te in kf.split(Xn):
        m = fn(); m.fit(Xn[tr], gt[tr]); pr[te] = m.predict(Xn[te])
    return pr


def guard(v, K=0.2):
    k = int(N * K); o = np.argsort(v)
    feat = np.zeros(N, bool); feat[o[-k:]] = True; rej = np.zeros(N, bool); rej[o[:k]] = True
    rec5 = (feat & is5).sum() / is5.sum(); tr5 = (rej & is5).sum() / is5.sum()
    return 0.4 * rec5 + 0.4 * (1 - tr5) + 0.2 * roc_auc_score(is5.astype(int), v), rec5, tr5


def line(lbl, v):
    g, r5, t5 = guard(v)
    print(f"  {lbl:<26} SRCC={spearmanr(v, gt).statistic:.4f}  守护分={g:.3f}  真5精选={r5*100:.0f}%  误杀={t5*100:.1f}%")


print(f"\nN={N}  GT std={gt.std():.2f}\n=== 参照 ===")
for s, v in preds.items(): line(f"单模 {s}", v)
line("简单平均(4模型)", base.mean(1))
print("\n=== 融合头(K折OOF, 逐步加特征) ===")
for fname, X in FEATS.items():
    print(f"[{fname}]  dim={X.shape[1]}")
    for hname, fn in HEADS.items():
        line(f"  {hname}", oof(X, fn))
