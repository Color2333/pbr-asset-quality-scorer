"""Oracle ceiling-decomposition for the A/B label-noise split (a SMALL, zero-cost
experiment — pure numpy, no GPU, no training).

Question: the metallic SRCC ceiling (~0.62 here) — how much is dragged by
  A = likely LABEL errors (models say HIGHER than GT), vs
  B = model BLIND SPOT (models say LOWER than GT; e.g. correct empty map on wood)?

Two views, both honest about their bias:

  (1) LEAVE-OUT with RANDOM CONTROL. Drop A (or B) from the test, recompute SRCC.
      Caveat: dropping high-error points ALWAYS raises SRCC. So we also drop the
      SAME COUNT of RANDOM points (avg over seeds). The A/B effect is only the
      part ABOVE the random control. This is the trustworthy number.

  (2) SUBSTITUTION upper bound. A: set GT<-consensus (assume label fixed toward
      models). B: set pred<-GT (assume blind spot fixed). Recompute SRCC. These
      are UPPER bounds (circular for A by construction) — read as "best case", not
      as a result.

Reference scorer = ensemble mean of the loaded models (our current best).

Usage:  python asset_quality_scorer/scripts/ab_oracle.py [--thr 1.5]
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

PKG = Path(__file__).resolve().parents[1]
CH = ["base_color", "normal_map", "roughness", "metallic"]
MODELS = [("DINOv2", "dinov2_large_multitask_emd_all"),
          ("Qwen", "vlm_scorer_a_old50k_oldtest"),
          ("ConvNeXt", "archive/convnext_base_multitask_emd")]
SEEDS = list(range(40))   # random-control repetitions (deterministic, no RNG-in-loop surprises)


def load(run):
    p = PKG / "outputs/runs" / run / "demo_predictions.json"
    return {a["name"]: a for a in json.loads(p.read_text())["assets"]} if p.exists() else None


def srcc(p, g):
    return float(spearmanr(p, g).statistic)


def rand_drop_srcc(pred, gt, k, n):
    """mean SRCC after dropping k random rows, over SEEDS draws."""
    if k == 0:
        return srcc(pred, gt)
    vals = []
    for s in SEEDS:
        rng = np.random.default_rng(s)
        drop = rng.choice(n, size=k, replace=False)
        keep = np.ones(n, bool); keep[drop] = False
        vals.append(srcc(pred[keep], gt[keep]))
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=1.5, help="suspicion threshold for A/B membership")
    args = ap.parse_args()

    loaded = [(s, load(r)) for s, r in MODELS]
    loaded = [(s, a) for s, a in loaded if a is not None]
    names = sorted(set.intersection(*[set(a) for _, a in loaded]))
    n = len(names)
    print(f"{len(loaded)} models {[s for s,_ in loaded]}, {n} assets, thr={args.thr}\n")

    for ch in CH:
        gt = np.array([loaded[0][1][nm]["gt"][ch] for nm in names], float)
        P = np.column_stack([[a[nm]["pred"][ch] for nm in names] for _, a in loaded])
        ens = P.mean(1)                         # reference scorer = ensemble mean
        cons = P.mean(1)                         # consensus for substitution
        errs = P - gt[:, None]
        allpos = (errs > 0).all(1); allneg = (errs < 0).all(1)
        susp = np.where(allpos | allneg, np.abs(errs).min(1), 0.0)
        A = (allpos) & (susp > args.thr)         # models HIGHER than GT  -> label?
        B = (allneg) & (susp > args.thr)         # models LOWER than GT   -> blind spot?
        nA, nB = int(A.sum()), int(B.sum())

        base = srcc(ens, gt)
        # leave-out (real) vs random control (same count)
        def drop(mask):
            keep = ~mask
            return srcc(ens[keep], gt[keep])
        dropA, dropB, dropAB = drop(A), drop(B), drop(A | B)
        rcA, rcB, rcAB = (rand_drop_srcc(ens, gt, nA, n),
                          rand_drop_srcc(ens, gt, nB, n),
                          rand_drop_srcc(ens, gt, nA + nB, n))
        # substitution upper bounds
        gtA = gt.copy(); gtA[A] = cons[A]          # A: fix label toward consensus
        subA = srcc(ens, gtA)
        predB = ens.copy(); predB[B] = gt[B]       # B: fix model toward truth
        subB = srcc(predB, gt)
        predboth = ens.copy(); predboth[B] = gt[B]
        gtboth = gt.copy(); gtboth[A] = cons[A]
        subboth = srcc(predboth, gtboth)

        print(f"=== {ch} ===  base SRCC(ensemble)={base:.4f}   A={nA}  B={nB}")
        print(f"  leave-out 真实 vs 随机对照(剔等量随机点):")
        print(f"    剔A : {dropA:.4f}  (随机对照 {rcA:.4f}  净效应 {dropA-rcA:+.4f})")
        print(f"    剔B : {dropB:.4f}  (随机对照 {rcB:.4f}  净效应 {dropB-rcB:+.4f})")
        print(f"    剔A+B: {dropAB:.4f} (随机对照 {rcAB:.4f}  净效应 {dropAB-rcAB:+.4f})")
        print(f"  substitution 上界(乐观,非结论):  改A标签→{subA:.4f}   修B模型→{subB:.4f}   两者→{subboth:.4f}")
        print()


if __name__ == "__main__":
    main()
