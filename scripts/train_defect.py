"""Defect detector (2 viable defects only): hasTextOrPattern (base_color) +
normalHasAbnormalTint (normal_map). Dropped the rare two (fake-AO 2% / flipped
0.4%) — too few positives. Shared ConvNeXt-Base backbone (capacity isn't the
bottleneck; smaller = less overfit on rare positives), each defect routed to its
channel. Class-weighted BCE, shallow unfreeze. Reports AUC + operating points.

Usage:
    CUDA_VISIBLE_DEVICES=1 python asset_quality_scorer/scripts/train_defect.py
"""
from __future__ import annotations
import csv, json, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quality_scorer.models.convnext import AttentionPool2d

PKG = Path(__file__).resolve().parents[1]
CACHE = PKG / "cache/224"; CSV = PKG / "dataset/sampled_all.csv"
OUT = PKG / "outputs/runs/convnext_defect_text_tint_ms"; OUT.mkdir(parents=True, exist_ok=True)
DEFECTS = {"hasTextOrPattern": "base_color", "normalHasAbnormalTint": "normal_map"}
DNAMES = list(DEFECTS)
MEAN=[0.485,0.456,0.406]; STD=[0.229,0.224,0.225]
_AUG = transforms.Compose([transforms.ConvertImageDtype(torch.float32),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.Normalize(MEAN,STD)])
_EVAL = transforms.Compose([transforms.ConvertImageDtype(torch.float32), transforms.Normalize(MEAN,STD)])


class DefectDS(Dataset):
    def __init__(self, split, is_train):
        self.tfm = _AUG if is_train else _EVAL
        meta = json.loads((CACHE/"meta.json").read_text()); cidx={n:i for i,n in enumerate(meta["model_names"])}
        self._arr = {c: None for c in ("base_color","normal_map")}
        self.samples=[]
        with open(CSV,newline="",encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r.get("split")!=split: continue
                name=r["model"].removeprefix("raw_data/").replace("/","__").removesuffix(".glb")
                if name not in cidx: continue
                y=[1.0 if r.get(d)=="True" else 0.0 for d in DNAMES]
                self.samples.append((cidx[name], y))
    def arr(self,c):
        if self._arr[c] is None: self._arr[c]=np.load(CACHE/f"{c}.npy",mmap_mode="r")
        return self._arr[c]
    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        ci,y=self.samples[i]
        bc=self.tfm(torch.from_numpy(np.array(self.arr("base_color")[ci],copy=True)))
        nm=self.tfm(torch.from_numpy(np.array(self.arr("normal_map")[ci],copy=True)))
        return bc, nm, torch.tensor(y)


class DefectNet(nn.Module):
    """Shared ConvNeXt-Base backbone; text head reads base_color, tint head reads normal.
    Multi-scale + AttentionPool2d (NOT GAP): local defects (text/pattern in a corner)
    would be averaged away by GAP — attention pooling keeps 'which patches matter',
    and multi-scale (stage2/3/4) covers high-freq text + low-freq tint."""
    D = 256 + 512 + 1024   # multi-scale concat = 1792
    def __init__(self):
        super().__init__()
        bb=convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
        self.features=bb.features
        self.pool2=AttentionPool2d(256); self.pool3=AttentionPool2d(512); self.pool4=AttentionPool2d(1024)
        def head(): return nn.Sequential(nn.LayerNorm(self.D),nn.Linear(self.D,256),nn.GELU(),nn.Dropout(0.3),nn.Linear(256,1))
        self.text_head=head(); self.tint_head=head()
        for p in self.features.parameters(): p.requires_grad_(False)
    def unfreeze_last(self):
        for i in (6,7):
            for p in self.features[i].parameters(): p.requires_grad_(True)
    def enc(self,x):
        for i in range(4): x=self.features[i](x)
        s2=self.pool2(x)
        for i in range(4,6): x=self.features[i](x)
        s3=self.pool3(x)
        for i in range(6,8): x=self.features[i](x)
        s4=self.pool4(x)
        return torch.cat([s2,s3,s4],dim=1)   # [B,1792]
    def forward(self, bc, nm):
        return self.text_head(self.enc(bc)).squeeze(1), self.tint_head(self.enc(nm)).squeeze(1)


class DefectNetDINOv3(nn.Module):
    """DINOv3 ViT-L backbone (Eva); patch tokens → attention pool → 2 routed heads.
    Tests whether DINOv3's stronger DENSE features help local-defect detection."""
    def __init__(self):
        super().__init__()
        import timm
        from quality_scorer.models.dinov2 import TokenAttentionPool
        self.bb = timm.create_model('vit_large_patch16_dinov3', pretrained=True,
                                    num_classes=0, dynamic_img_size=True)
        self.npref = self.bb.num_prefix_tokens
        D = self.bb.embed_dim
        self.pool = TokenAttentionPool(D)
        def head(): return nn.Sequential(nn.LayerNorm(D),nn.Linear(D,256),nn.GELU(),nn.Dropout(0.3),nn.Linear(256,1))
        self.text_head=head(); self.tint_head=head()
        for p in self.bb.parameters(): p.requires_grad_(False)
    def unfreeze_last(self):
        for blk in self.bb.blocks[-6:]:
            for p in blk.parameters(): p.requires_grad_(True)
    def enc(self, x):
        t = self.bb.forward_features(x)        # [B, npref+N, D]
        return self.pool(t[:, self.npref:, :]) # patch tokens → [B,D]
    def forward(self, bc, nm):
        return self.text_head(self.enc(bc)).squeeze(1), self.tint_head(self.enc(nm)).squeeze(1)


def main():
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--backbone",default="convnext",choices=["convnext","dinov3"])
    args=ap.parse_args()
    global OUT
    if args.backbone=="dinov3": OUT=PKG/"outputs/runs/dinov3_defect_text_tint"; OUT.mkdir(parents=True,exist_ok=True)
    dev="cuda"
    tr=DefectDS("train",True); te=DefectDS("test",False)
    # class weights (pos rate)
    Y=np.array([y for _,y in tr.samples]); posrate=Y.mean(0)
    pw=torch.tensor((1-posrate)/np.clip(posrate,1e-3,1), dtype=torch.float32, device=dev)
    print(f"train {len(tr)} test {len(te)}; pos rate {dict(zip(DNAMES,posrate.round(3)))}; pos_weight {pw.tolist()}")
    bs = 16 if args.backbone=="dinov3" else 48
    trl=DataLoader(tr,batch_size=bs,shuffle=True,num_workers=8,pin_memory=True)
    tel=DataLoader(te,batch_size=bs*2,shuffle=False,num_workers=8)
    m=(DefectNetDINOv3() if args.backbone=="dinov3" else DefectNet()).to(dev)
    print(f"backbone={args.backbone}")
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=2e-4,weight_decay=1e-4)
    amp=torch.bfloat16
    best_auc=0.0
    for ep in range(1,16):
        if ep==4: m.unfreeze_last(); opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4); print("  unfreeze last stage")
        m.train()
        for bc,nm,y in trl:
            bc,nm,y=bc.to(dev),nm.to(dev),y.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=amp):
                lt,li=m(bc,nm)
                loss=F.binary_cross_entropy_with_logits(lt,y[:,0],pos_weight=pw[0])+\
                     F.binary_cross_entropy_with_logits(li,y[:,1],pos_weight=pw[1])
            loss.backward(); opt.step()
        # eval
        m.eval(); P=[[],[]]; L=[[],[]]
        with torch.no_grad():
            for bc,nm,y in tel:
                bc,nm=bc.to(dev),nm.to(dev)
                with torch.autocast("cuda",dtype=amp): lt,li=m(bc,nm)
                P[0].extend(torch.sigmoid(lt).float().cpu().tolist()); P[1].extend(torch.sigmoid(li).float().cpu().tolist())
                L[0].extend(y[:,0].tolist()); L[1].extend(y[:,1].tolist())
        aucs=[roc_auc_score(L[k],P[k]) for k in (0,1)]
        print(f"  ep{ep:2d} loss={loss.item():.3f}  text AUC={aucs[0]:.3f}  tint AUC={aucs[1]:.3f}")
        if np.mean(aucs)>best_auc:
            best_auc=np.mean(aucs)
            torch.save({"model_state_dict":m.state_dict(),"epoch":ep}, OUT/"best.pt")
            bestP,bestL=[list(x) for x in P],[list(x) for x in L]
    # final operating points on best
    print("\n=== 最佳检测器 (test) ===")
    res={}
    for k,d in enumerate(DNAMES):
        auc=roc_auc_score(bestL[k],bestP[k]); ap=average_precision_score(bestL[k],bestP[k])
        prec,rec,_=precision_recall_curve(bestL[k],bestP[k])
        pr80=float(prec[rec>=0.8].max()) if (rec>=0.8).any() else 0
        rc90=float(rec[prec>=0.9].max()) if (prec>=0.9).any() else 0
        res[d]={"auc":round(auc,3),"ap":round(ap,3),"prec@rec80":round(pr80,2),"rec@prec90":round(rc90,2)}
        print(f"  {d:<26}: AUC={auc:.3f} AP={ap:.3f} @召回80%精度={pr80:.2f} @精度90%召回={rc90:.2f}")
    (OUT/"eval_test.json").write_text(json.dumps(res,indent=2,ensure_ascii=False))
    print(f"\nsaved {OUT}/best.pt + eval_test.json")

if __name__=="__main__": main()
