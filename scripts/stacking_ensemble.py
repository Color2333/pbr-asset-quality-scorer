"""Stacking ensemble over all available scorer models, with leakage-free K-fold
out-of-fold evaluation. Compares: each single model, simple average, and
GBM/RF/Ridge stacking (with disagreement + near-black features).

Auto-discovers every model in REGISTRY that has a demo_predictions.json.
Stacking only earns its keep with MANY diverse base learners — rerun this once
ConvNeXt / Qwen3 / the metallic experts have landed.

Usage:  python asset_quality_scorer/scripts/stacking_ensemble.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

PKG = Path(__file__).resolve().parents[1]
CH = ["base_color", "normal_map", "roughness", "metallic"]
REGISTRY = [
    ("DINOv2", "dinov2_large_multitask_emd_all"),
    ("ConvNeXt", "archive/convnext_base_multitask_emd"),
    ("Qwen2.5", "vlm_scorer_a_old50k_oldtest"),
    ("Qwen3", "vlm_scorer_qwen3_smoke10k_oldtest"),
    ("MeExpert", "vlm_scorer_qwen25_me_expert_old"),       # metallic-only (overlay)
    ("MeExpert0604", "vlm_scorer_qwen25_me_expert_0604"),  # metallic-only
]


def load(run):
    p = PKG / "outputs/runs" / run / "demo_predictions.json"
    return {a["name"]: a for a in json.loads(p.read_text())["assets"]} if p.exists() else None


def oof(X, y, fn, seed=0):
    kf = KFold(5, shuffle=True, random_state=seed); pr = np.zeros(len(y))
    for tr, te in kf.split(X):
        m = fn(); m.fit(X[tr], y[tr]); pr[te] = m.predict(X[te])
    return pr


def main():
    models = [(s, load(r)) for s, r in REGISTRY]
    models = [(s, a) for s, a in models if a is not None]
    print("loaded:", [s for s, _ in models])
    names = sorted(set.intersection(*[set(a) for _, a in models]))
    frac = dict(zip(json.loads((PKG / "cache/224/meta.json").read_text())["model_names"],
                    np.load(PKG / "dataset/metallic_nonblack.npy")))
    nbf = np.array([frac.get(n, 1.0) for n in names])
    print(f"{len(models)} models, {len(names)} common assets\n")

    GBM = lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05)
    RF = lambda: RandomForestRegressor(n_estimators=300, max_depth=6, n_jobs=4)
    RIDGE = lambda: Ridge(alpha=1.0)

    for ch in CH:
        # which models actually predict this channel meaningfully (experts = metallic only)
        usable = [(s, a) for s, a in models if not s.startswith("Me") or ch == "metallic"]
        cols = {s: np.array([a[n]["pred"][ch] for n in names]) for s, a in usable}
        g = np.array([usable[0][1][n]["gt"][ch] for n in names]).astype(float)
        base = np.column_stack([cols[s] for s, _ in usable])
        feat = np.column_stack([base, base.std(1), nbf, (nbf < 0.02).astype(float)])
        singles = {s: spearmanr(cols[s], g).statistic for s, _ in usable}
        avg = spearmanr(base.mean(1), g).statistic
        res = {"avg": avg,
               "ridge": spearmanr(oof(base, g, RIDGE), g).statistic,
               "gbm": spearmanr(oof(feat, g, GBM), g).statistic,
               "rf": spearmanr(oof(feat, g, RF), g).statistic}
        bs = max(singles, key=singles.get)
        print(f"[{ch}] {len(usable)}模型  单模最佳={bs}:{singles[bs]:.4f}  "
              f"平均={res['avg']:.4f}  Ridge={res['ridge']:.4f}  GBM={res['gbm']:.4f}  RF={res['rf']:.4f}")
        best = max(res, key=res.get)
        print(f"      → 最佳集成: {best}={res[best]:.4f}  (vs 单模最佳 {singles[bs]:.4f}, "
              f"{'+' if res[best]>singles[bs] else ''}{res[best]-singles[bs]:.4f})")


if __name__ == "__main__":
    main()
