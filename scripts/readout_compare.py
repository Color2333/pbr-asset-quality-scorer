"""比较不同读出方式 (期望 / argmax / 温度缩放) 对 SRCC/acc/within1 的影响。
需 probs.npy (vlm_scorer_eval 重评后产生)。"""
import sys, json, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
sys.path.insert(0, "asset_quality_scorer/scripts")
from vlm_scorer_eval import build_items, CHANNELS
PKG=Path("asset_quality_scorer")
run=sys.argv[1] if len(sys.argv)>1 else "vlm_scorer_a_old50k_oldtest"
probs=np.load(PKG/"outputs/runs"/run/"probs.npy",allow_pickle=True).item()
rows=build_items("test"); g={c:np.array([r['scores'][c] for r in rows]) for c in CHANNELS}
ks=np.arange(6)
def m(p,gt):
    r=np.clip(np.round(p),0,5).astype(int)
    return spearmanr(p,gt).statistic,(r==gt).mean(),(np.abs(r-gt)<=1).mean()
print(f"{run}\n{'通道':<12}{'读出':<10}{'SRCC':>8}{'acc':>7}{'w1':>7}{'std':>7}")
for c in CHANNELS:
    P=probs[c]  # [N,6]
    for name,p in [("期望",(P*ks).sum(1)),("argmax",P.argmax(1).astype(float)),
                   ("T=0.5",((P**2/( (P**2).sum(1,keepdims=True)))*ks).sum(1))]:
        s,a,w=m(p,g[c]); print(f"{c:<12}{name:<10}{s:>8.3f}{a*100:>6.0f}%{w*100:>6.0f}%{p.std():>7.2f}")
