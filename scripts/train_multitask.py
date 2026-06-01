"""Multi-task training: one DINOv2 backbone, 4 PBR channel heads, FiLM on metallic.

Usage:
    python asset_quality_scorer/scripts/train_multitask.py \
        --config asset_quality_scorer/config/dinov2_large_multitask.yaml
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "asset_quality_scorer"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT))

import csv
import numpy as np
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from quality_scorer.models import DINOv2MultiTaskScorer

_CHANNELS  = DINOv2MultiTaskScorer.CHANNELS   # (bc, nm, ro, me)
_SCORE_COL = {"base_color": "baseColor", "normal_map": "normal",
              "roughness": "roughness", "metallic": "metallic"}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_AUG = transforms.Compose([
    transforms.ConvertImageDtype(torch.float32),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
_EVAL = transforms.Compose([
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class MultiChannelDataset(Dataset):
    """Loads all 4 channel images + shared CLIP features per sample."""

    def __init__(self, cache_root: Path, clip_path: Path, csv_path: Path,
                 split: str, is_train: bool, invalid_max_score: int = 1):
        self.is_train = is_train
        self.tfm = _AUG if is_train else _EVAL
        self.invalid_max_score = invalid_max_score

        # Load cache metadata
        meta = json.loads((cache_root / "meta.json").read_text())
        cache_idx = {n: i for i, n in enumerate(meta["model_names"])}

        # Lazy memmaps (opened per worker)
        self._cache_root = cache_root
        self._arrays: dict[str, np.ndarray | None] = {ch: None for ch in _CHANNELS}

        # Load CLIP features
        clip_data = torch.load(clip_path, map_location="cpu", weights_only=False)
        clip_names = clip_data["model_names"]
        clip_idx = {n: i for i, n in enumerate(clip_names)}
        bc  = clip_data["features"]["base_color"]
        rn  = clip_data["features"]["render"]
        # pre-build full clip matrix: [N, 1536] fp32
        N = len(clip_names)
        self._clip_mat = torch.cat([
            torch.from_numpy(np.array(bc, dtype=np.float32)),
            torch.from_numpy(np.array(rn, dtype=np.float32)),
        ], dim=1)                                          # [N, 1536]
        self._clip_idx = clip_idx

        # Build sample list (intersect cache ∩ clip ∩ CSV ∩ split, all 4 scores valid)
        self.samples: list[tuple[str, dict[str, int]]] = []
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("split") != split:
                    continue
                name = row["model"].removeprefix("raw_data/").replace("/", "__").removesuffix(".glb")
                if name not in cache_idx or name not in clip_idx:
                    continue
                scores = {}
                ok = True
                for ch, col in _SCORE_COL.items():
                    try:
                        scores[ch] = int(row[col])
                        assert 0 <= scores[ch] <= 5
                    except Exception:
                        ok = False; break
                if ok:
                    self.samples.append((name, scores, cache_idx[name], clip_idx[name]))

    def _arr(self, ch: str) -> np.ndarray:
        if self._arrays[ch] is None:
            self._arrays[ch] = np.load(self._cache_root / f"{ch}.npy", mmap_mode="r")
        return self._arrays[ch]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        name, scores, ci, ki = self.samples[i]
        imgs = {}
        for ch in _CHANNELS:
            raw = torch.from_numpy(np.array(self._arr(ch)[ci], copy=True))  # [3,H,W] uint8
            imgs[ch] = self.tfm(raw)
        clip = self._clip_mat[ki]                            # [1536]
        score_t  = {ch: torch.tensor(s, dtype=torch.float32) for ch, s in scores.items()}
        binary_t = {ch: torch.tensor(float(s <= self.invalid_max_score)) for ch, s in scores.items()}
        return imgs, clip, score_t, binary_t


def _collate(batch):
    imgs_list, clips, scores_list, bin_list = zip(*batch)
    imgs = {ch: torch.stack([b[ch] for b in imgs_list]) for ch in _CHANNELS}
    clips = torch.stack(clips)
    scores  = {ch: torch.stack([b[ch] for b in scores_list]) for ch in _CHANNELS}
    binaries = {ch: torch.stack([b[ch] for b in bin_list])  for ch in _CHANNELS}
    return imgs, clips, scores, binaries


def _ranking_loss(pred, target, margin=0.5):
    dt = target.unsqueeze(0) - target.unsqueeze(1)
    dp = pred.unsqueeze(0)   - pred.unsqueeze(1)
    mask = dt.abs() > 0.5
    if not mask.any():
        return pred.sum() * 0.0
    return F.relu(margin - dt[mask].sign() * dp[mask]).mean()


def _resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def train(cfg: dict) -> None:
    d, m, t, u, o = cfg["data"], cfg["model"], cfg["train"], cfg["unfreeze"], cfg["output"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_id  = f"dinov2_large_multitask_{o.get('exp_suffix','v1')}"
    out_dir = _resolve(o.get("root", "asset_quality_scorer/outputs/runs")) / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*72}\nTraining: {exp_id}\n{'='*72}")

    # ── dataset ────────────────────────────────────────────────────────────
    cache_root = _resolve(d["tensor_cache_root"])
    clip_path  = _resolve(d["clip_feature_path"])
    csv_path   = _resolve(d["csv_path"])
    inv_max    = int(m.get("invalid_max_score", 1))

    train_ds = MultiChannelDataset(cache_root, clip_path, csv_path, "train", True,  inv_max)
    val_ds   = MultiChannelDataset(cache_root, clip_path, csv_path, "val",   False, inv_max)
    print(f"  train: {len(train_ds)}  val: {len(val_ds)}")

    nw = int(t.get("num_workers", 8))
    bs = int(t.get("batch_size", 16))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, collate_fn=_collate, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs*2, shuffle=False,
                              num_workers=nw, collate_fn=_collate, pin_memory=True)

    # ── model ──────────────────────────────────────────────────────────────
    model = DINOv2MultiTaskScorer(
        clip_dim            = int(m.get("clip_dim", 1536)),
        attn_proj_dim       = int(m.get("attn_proj_dim", 256)),
        attn_heads          = int(m.get("attn_heads", 4)),
        hidden_dim          = int(m.get("hidden_dim", 512)),
        dropout             = float(m.get("dropout", 0.3)),
        freeze_features     = True,
        use_clip_direct     = True,
        metallic_film       = bool(m.get("metallic_film", True)),
        metallic_grad_scale = float(m.get("metallic_grad_scale", 0.5)),
        use_cross_channel   = bool(m.get("use_cross_channel", False)),
        cc_n_heads          = int(m.get("cc_n_heads", 4)),
        cc_me_bc_bias       = float(m.get("cc_me_bc_bias", 1.0)),
        cc_me_ro_bias       = float(m.get("cc_me_ro_bias", 1.0)),
    ).to(device)
    cc = model.cross_channel
    print(f"  metallic_film={model.film_gen is not None}  "
          f"metallic_grad_scale={model.metallic_grad_scale}  "
          f"cross_channel={cc is not None}")

    # per-channel loss weights
    ch_weights = {ch: float(t.get("channel_weights", {}).get(ch, 1.0)) for ch in _CHANNELS}
    print(f"  channel weights: {ch_weights}")

    # ── AMP ────────────────────────────────────────────────────────────────
    use_amp  = device.type == "cuda"
    amp_dt   = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler   = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dt == torch.float16)
    if use_amp:
        print(f"  amp: {'bf16' if amp_dt==torch.bfloat16 else 'fp16'}")

    # ── optimizer ──────────────────────────────────────────────────────────
    lr  = float(t.get("lr", 1e-4))
    wd  = float(t.get("weight_decay", 1e-4))
    bin_w  = float(t.get("binary_loss_weight",  0.2))
    rank_w = float(t.get("ranking_loss_weight", 0.05))
    huber_d = float(t.get("huber_delta", 1.0))

    def make_opt():
        return optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)
    optimizer = make_opt()
    epochs    = int(t.get("epochs", 30))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── training loop ──────────────────────────────────────────────────────
    best = {"mae_mean": float("inf"), "srcc_mean": -2.0}
    for ei in range(epochs):
        epoch = ei + 1

        # progressive unfreeze
        if ei == int(u.get("stage4_epoch", 5)):
            model.unfreeze_stage4(); optimizer = make_opt()
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-ei)
            print("  [unfreeze] stage4")
        elif ei == int(u.get("stage34_epoch", 10)):
            model.unfreeze_stage34(); optimizer = make_opt()
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-ei, eta_min=1e-6)
            print("  [unfreeze] stage3+4")
        elif ei == int(u.get("stage234_epoch", 15)):
            model.unfreeze_stage234(); optimizer = make_opt()
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-ei, eta_min=1e-6)
            print("  [unfreeze] stage2+3+4")

        model.train()
        running = 0.0
        for imgs, clips, scores, binaries in train_loader:
            imgs   = {ch: v.to(device, non_blocking=True) for ch, v in imgs.items()}
            clips  = clips.to(device, non_blocking=True)
            scores   = {ch: v.to(device, non_blocking=True) for ch, v in scores.items()}
            binaries = {ch: v.to(device, non_blocking=True) for ch, v in binaries.items()}

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dt, enabled=use_amp):
                preds = model(imgs, clips)   # {ch: (score, binary_logit)}
                loss = torch.tensor(0.0, device=device)
                for ch in _CHANNELS:
                    ps, pb = preds[ch]
                    w = ch_weights[ch]
                    loss = loss + w * F.huber_loss(ps, scores[ch], delta=huber_d)
                    if bin_w > 0:
                        loss = loss + w * bin_w * F.binary_cross_entropy_with_logits(pb, binaries[ch])
                    if rank_w > 0:
                        loss = loss + w * rank_w * _ranking_loss(ps, scores[ch])

            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            running += float(loss.item())

        scheduler.step()
        train_loss = running / max(len(train_loader), 1)

        # ── validation ────────────────────────────────────────────────────
        model.eval()
        preds_all  = {ch: [] for ch in _CHANNELS}
        scores_all = {ch: [] for ch in _CHANNELS}
        with torch.no_grad():
            for imgs, clips, scores, _ in val_loader:
                imgs  = {ch: v.to(device, non_blocking=True) for ch, v in imgs.items()}
                clips = clips.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=amp_dt, enabled=use_amp):
                    preds = model(imgs, clips)
                for ch in _CHANNELS:
                    preds_all[ch].extend(preds[ch][0].float().cpu().tolist())
                    scores_all[ch].extend(scores[ch].tolist())

        metrics = {}
        for ch in _CHANNELS:
            p = np.array(preds_all[ch]); s = np.array(scores_all[ch])
            metrics[ch] = {
                "mae":  round(float(np.abs(p-s).mean()), 4),
                "srcc": round(float(spearmanr(p, s).statistic), 4),
            }
        mae_mean  = round(np.mean([metrics[ch]["mae"]  for ch in _CHANNELS]), 4)
        srcc_mean = round(np.mean([metrics[ch]["srcc"] for ch in _CHANNELS]), 4)

        ch_str = "  ".join(f"{ch[:2]}={metrics[ch]['srcc']:.3f}" for ch in _CHANNELS)
        print(f"  epoch={epoch:2d} loss={train_loss:.4f}  srcc_mean={srcc_mean:.4f}  "
              f"mae_mean={mae_mean:.4f}  [{ch_str}]", flush=True)

        if srcc_mean > best["srcc_mean"]:
            best = {"mae_mean": mae_mean, "srcc_mean": srcc_mean,
                    "epoch": epoch, "per_channel": metrics}
            torch.save({"model_state_dict": model.state_dict(), "arch": "dinov2_large_multitask",
                        "epoch": epoch, "metrics": best}, out_dir / "best.pt")

    print(f"\nDone. Best: srcc_mean={best['srcc_mean']}  mae_mean={best['mae_mean']}")
    for ch in _CHANNELS:
        mc = best["per_channel"][ch]
        print(f"  {ch:12s}: mae={mc['mae']}  srcc={mc['srcc']}")

    # Diagnostic: what cross-channel attention did the model actually learn?
    if model.cross_channel is not None:
        bias_summary = model.cross_channel.get_attn_bias_summary()
        print("\n  Learned cross-channel attention biases (non-zero):")
        for pair, val in sorted(bias_summary.items(), key=lambda x: -abs(x[1])):
            bar = "█" * int(abs(val) * 5)
            print(f"    {pair}: {val:+.3f}  {bar}")
        best["attn_bias"] = bias_summary

    (out_dir / "summary.json").write_text(json.dumps({"exp_id": exp_id, **best}, indent=2))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = yaml.safe_load(open(_resolve(args.config)))
    train(cfg)
