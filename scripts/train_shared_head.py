"""Train heads on the FROZEN DINOv2 `fused` features (cached): compare a single
SHARED EMD head (all 4 channels) vs 4 per-channel EMD heads. Answers "does one
shared head for 4 channels work as well as per-channel heads" on the trained
backbone. Fast (tiny heads on cached 512-d features).

Needs outputs/dino_fused_{train,test}.npz (from extract_dino_fused.py).
Usage: python asset_quality_scorer/scripts/train_shared_head.py
"""
from __future__ import annotations
import numpy as np, torch, torch.nn as nn
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
PKG = Path(__file__).resolve().parents[1]
CH = ["base_color", "normal_map", "roughness", "metallic"]
dev = "cuda" if torch.cuda.is_available() else "cpu"
tr = np.load(PKG / "outputs/dino_fused_train.npz", allow_pickle=True)
te = np.load(PKG / "outputs/dino_fused_test.npz", allow_pickle=True)
Xtr = {c: torch.tensor(tr[f"feat_{c}"], dtype=torch.float32) for c in CH}
Ytr = {c: torch.tensor(tr[f"gt_{c}"], dtype=torch.float32) for c in CH}
Xte = {c: torch.tensor(te[f"feat_{c}"], dtype=torch.float32) for c in CH}
Yte = {c: te[f"gt_{c}"].astype(float) for c in CH}
ks = torch.arange(6.)


def soft(y, sigma=0.75):
    w = torch.exp(-0.5 * ((ks[None] - y[:, None]) / sigma) ** 2); return w / w.sum(1, keepdim=True)


class EMD(nn.Module):
    def __init__(s, shared=False):
        super().__init__()
        s.shared = shared
        if shared:
            s.emb = nn.Embedding(4, 512)               # channel id -> additive bias
            s.h = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 6))
        else:
            s.h = nn.ModuleDict({c: nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 6)) for c in CH})

    def forward(s, x, ci):
        if s.shared:
            return s.h(x + s.emb(torch.full((len(x),), ci, device=x.device)))
        return s.h[CH[ci]](x)


def train_eval(shared):
    torch.manual_seed(0)
    m = EMD(shared).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    Xg = {c: Xtr[c].to(dev) for c in CH}; Tg = {c: soft(Ytr[c]).to(dev) for c in CH}
    for ep in range(60):
        m.train(); perm = torch.randperm(len(Xtr[CH[0]]))
        for i in range(0, len(perm), 1024):
            idx = perm[i:i + 1024]; opt.zero_grad(); loss = 0
            for ci, c in enumerate(CH):
                lp = torch.log_softmax(m(Xg[c][idx], ci), 1)
                loss = loss - (Tg[c][idx] * lp).sum(1).mean()
            loss.backward(); opt.step()
    m.eval(); res = {}
    with torch.no_grad():
        for ci, c in enumerate(CH):
            p = torch.softmax(m(Xte[c].to(dev), ci), 1).cpu()
            score = (p * ks).sum(1).numpy()
            res[c] = (score, spearmanr(score, Yte[c]).statistic)
    return res


def guard(v, g, K=0.2):
    N = len(g); is5 = g == 5; k = int(N * K); o = np.argsort(v)
    feat = np.zeros(N, bool); feat[o[-k:]] = True; rej = np.zeros(N, bool); rej[o[:k]] = True
    r5 = (feat & is5).sum() / is5.sum(); t5 = (rej & is5).sum() / is5.sum()
    return 0.4 * r5 + 0.4 * (1 - t5) + 0.2 * roc_auc_score(is5.astype(int), v)


print(f"{'通道':<12}{'每通道头SRCC':>14}{'共享头SRCC':>12}{'Δ':>8}")
per = train_eval(False); sh = train_eval(True)
for c in CH:
    print(f"{c:<12}{per[c][1]:>14.4f}{sh[c][1]:>12.4f}{sh[c][1]-per[c][1]:>+8.4f}")
gm = Yte["metallic"]
print(f"\nmetallic 守护分: 每通道头={guard(per['metallic'][0],gm):.3f}  共享头={guard(sh['metallic'][0],gm):.3f}")
print("(参照: 原模型 digit/EMD readout test metallic 0.617)")
