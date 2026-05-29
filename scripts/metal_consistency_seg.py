"""Proof-of-life: weak-supervised metallic-region segmentation + consistency signal.

Hypothesis (破局点2): metallic quality is a cross-channel CONSISTENCY judgement —
"does the metallic map light up where the appearance says metal should be?".
High-score (4-5) samples have CORRECT metallic maps, so their binarized metallic
render IS a free pseudo-GT mask of "where metal should be".

Stage 1 (this script): train a segmentation model
    input  = base_color + white_model   (BOTH independent of the metallic map —
             render is EXCLUDED on purpose: its specular is rendered FROM the
             metallic map, so using it would be circular and blind to the
             dominant "missing metal" bug)
    target = binarized metallic render of high-score samples (pixel-aligned —
             all channels are same-camera renders of the same geometry)

Two validations this script reports:
  (A) Can it learn?  metal-class IoU on HELD-OUT high-score val samples.
  (B) Can it flag bugs?  agreement = IoU(predicted_mask, actual_metallic_mask)
      computed per quality-score bin. If consistency is a real quality signal,
      agreement should be HIGH for good (4-5) and LOWER for bad (0-1) samples —
      because bad samples' actual metallic disagrees with what appearance predicts.

Run (queued after the 38k retrain frees GPU0):
    CUDA_VISIBLE_DEVICES=0 python asset_quality_scorer/scripts/metal_consistency_seg.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _model_path_to_name(model_path: str) -> str:
    return model_path.removeprefix("raw_data/").replace("/", "__").removesuffix(".glb")


class MetalSegDataset(Dataset):
    """Returns (input6[, score, name]); target mask built from metallic on the fly.

    score_filter: keep only samples whose metallic score is in this set (None = all).
    """

    def __init__(self, cache_root: Path, csv_path: Path, split: str,
                 mask_thresh: int = 102, score_filter: set[int] | None = None):
        meta = json.loads((cache_root / "meta.json").read_text())
        self.idx = {n: i for i, n in enumerate(meta["model_names"])}
        for ch in ("base_color", "white_model", "metallic"):
            if ch not in meta["channels"]:
                raise ValueError(f"cache missing channel '{ch}'")
        self.cache_root = cache_root
        self.mask_thresh = mask_thresh
        self._bc = self._wm = self._me = None  # lazy memmaps (per-worker)

        self.samples: list[tuple[str, int]] = []
        with Path(csv_path).open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("split") != split:
                    continue
                try:
                    score = int(row["metallic"])
                except (KeyError, ValueError):
                    continue
                if score_filter is not None and score not in score_filter:
                    continue
                name = _model_path_to_name(row["model"])
                if name in self.idx:
                    self.samples.append((name, score))

    def _arr(self, ch):
        return np.load(self.cache_root / f"{ch}.npy", mmap_mode="r")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        if self._bc is None:
            self._bc, self._wm, self._me = (self._arr("base_color"),
                                            self._arr("white_model"),
                                            self._arr("metallic"))
        name, score = self.samples[i]
        ci = self.idx[name]
        bc = torch.from_numpy(np.array(self._bc[ci], copy=True)).float() / 255.0
        wm = torch.from_numpy(np.array(self._wm[ci], copy=True)).float() / 255.0
        bc = (bc - IMAGENET_MEAN) / IMAGENET_STD
        wm = (wm - IMAGENET_MEAN) / IMAGENET_STD
        inp = torch.cat([bc, wm], dim=0)                       # [6, H, W]
        me = np.array(self._me[ci], copy=True).mean(0)         # grayscale [H, W]
        mask = torch.from_numpy((me > self.mask_thresh).astype(np.float32))[None]  # [1,H,W]
        return inp, mask, score, name


def build_model() -> nn.Module:
    m = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    # 3→6 input channels: duplicate pretrained RGB filters for base_color & white_model
    old = m.backbone.conv1
    new = nn.Conv2d(6, old.out_channels, kernel_size=old.kernel_size,
                    stride=old.stride, padding=old.padding, bias=old.bias is not None)
    with torch.no_grad():
        new.weight[:, :3] = old.weight
        new.weight[:, 3:] = old.weight
    m.backbone.conv1 = new
    # num_classes → 1 (metal logit)
    m.classifier[-1] = nn.Conv2d(256, 1, kernel_size=1)
    m.aux_classifier = None
    return m


def metal_iou(pred_bin: torch.Tensor, gt_bin: torch.Tensor) -> torch.Tensor:
    """Per-sample metal-class IoU. pred_bin/gt_bin: [B,1,H,W] in {0,1}. → [B]."""
    inter = (pred_bin * gt_bin).flatten(1).sum(1)
    union = ((pred_bin + gt_bin) > 0).float().flatten(1).sum(1)
    return inter / union.clamp_min(1.0)


@torch.no_grad()
def evaluate(model, loader, device, mask_thresh):
    model.eval()
    ious, scores = [], []
    for inp, mask, score, _ in loader:
        inp = inp.to(device)
        logit = model(inp)["out"]
        pred = (torch.sigmoid(logit) > 0.5).float().cpu()
        ious.append(metal_iou(pred, mask))
        scores.append(score)
    ious = torch.cat(ious).numpy()
    scores = torch.cat(scores).numpy()
    return ious, scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-root", default="asset_quality_scorer/cache/224")
    p.add_argument("--csv", default="asset_quality_scorer/dataset/sampled_all.csv")
    p.add_argument("--mask-thresh", type=int, default=102)   # 0.4 * 255
    p.add_argument("--hi-score", type=int, default=4)         # pseudo-GT from score >= this
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="asset_quality_scorer/outputs/runs/metal_consistency_seg")
    args = p.parse_args()

    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    root = PROJECT_ROOT / args.cache_root
    csvp = PROJECT_ROOT / args.csv
    out = PROJECT_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    hi = set(range(args.hi_score, 6))
    train_ds = MetalSegDataset(root, csvp, "train", args.mask_thresh, score_filter=hi)
    val_hi_ds = MetalSegDataset(root, csvp, "val",  args.mask_thresh, score_filter=hi)
    val_all_ds = MetalSegDataset(root, csvp, "val", args.mask_thresh, score_filter=None)
    print(f"train(hi>={args.hi_score}): {len(train_ds)}  val(hi): {len(val_hi_ds)}  val(all): {len(val_all_ds)}")

    dl = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh,
                                   num_workers=args.num_workers, pin_memory=True)
    train_loader = dl(train_ds, True)
    val_hi_loader = dl(val_hi_ds, False)
    val_all_loader = dl(val_all_ds, False)

    model = build_model().to(dev)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-4)

    best_iou = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for inp, mask, _, _ in train_loader:
            inp, mask = inp.to(dev), mask.to(dev)
            logit = model(inp)["out"]
            loss = F.binary_cross_entropy_with_logits(logit, mask)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        ious, _ = evaluate(model, val_hi_loader, dev, args.mask_thresh)
        miou = float(ious.mean())
        print(f"  ep{ep:02d} loss={tot/len(train_loader):.4f}  val_hi metal-IoU={miou:.4f}", flush=True)
        if miou > best_iou:
            best_iou = miou
            torch.save({"model_state_dict": model.state_dict(), "epoch": ep,
                        "val_iou": miou, "args": vars(args)}, out / "best.pt")

    # ── Validation (B): consistency = IoU(pred, actual metallic) by score bin ──
    ious_all, scores_all = evaluate(model, val_all_loader, dev, args.mask_thresh)
    by_bin = {}
    for s in range(6):
        m = scores_all == s
        if m.any():
            by_bin[s] = {"n": int(m.sum()), "mean_iou": round(float(ious_all[m].mean()), 4)}
    report = {
        "best_val_hi_iou": round(best_iou, 4),
        "consistency_iou_by_score": by_bin,
        "interpretation": "If consistency is a quality signal, mean_iou should rise with score.",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("\n=== Validation (A) learnability: best val(hi) metal-IoU =", round(best_iou, 4))
    print("=== Validation (B) consistency IoU by quality score (want monotonic ↑):")
    for s, v in by_bin.items():
        print(f"    score {s}: n={v['n']:<5} mean IoU(pred, actual)={v['mean_iou']}")
    print(f"\nreport -> {out/'report.json'}")


if __name__ == "__main__":
    main()
