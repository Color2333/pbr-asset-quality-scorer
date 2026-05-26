"""Train Phase 2 regression scorer: ConvNeXt multi-scale + cross-modal CLIP + defect heads.

Usage:
    python asset_quality_scorer/scripts/train.py --channel metallic
    python asset_quality_scorer/scripts/train.py --all   # sequential (one channel at a time)

Output:
    outputs/runs/{exp_id}/   where exp_id = convnext_base_{channel}_{exp_suffix}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "asset_quality_scorer"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT))

from quality_scorer.constants import ALL_CHANNELS
from quality_scorer.data import CHANNEL_DEFECT_COLS, TensorCacheCLIPDataset, build_score_lookup
from quality_scorer.metrics import eval_regression_epoch
from quality_scorer.models import ConvNeXtRegressionScorer


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _make_loader(dataset, batch_size, num_workers, sampler=None, shuffle=False):
    kw = dict(batch_size=batch_size, num_workers=num_workers,
              pin_memory=torch.cuda.is_available())
    if sampler is not None:
        kw["sampler"] = sampler
    else:
        kw["shuffle"] = shuffle
    if num_workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = 4
    return DataLoader(dataset, **kw)


def _save_checkpoint(out_dir, fname, model, optimizer, epoch, channel, args, metrics):
    state = dict(
        epoch=epoch,
        model_state_dict=model.state_dict(),
        channel=channel,
        arch="convnext_base",
        invalid_max_score=args.invalid_max_score,
        defect_cols=CHANNEL_DEFECT_COLS.get(channel, []),
        val_metrics=metrics,
    )
    if args.save_optimizer:
        state["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(state, out_dir / fname)


def _hardlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def _ranking_loss(pred: torch.Tensor, target: torch.Tensor, margin: float) -> torch.Tensor:
    diff_target = target.unsqueeze(0) - target.unsqueeze(1)
    diff_pred   = pred.unsqueeze(0)   - pred.unsqueeze(1)
    mask = diff_target.abs() > 0.5
    if not mask.any():
        return pred.sum() * 0.0
    loss = F.relu(margin - diff_target[mask].sign() * diff_pred[mask])
    return loss.mean()


def train_channel(args: argparse.Namespace, channel: str) -> dict:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    exp_id  = f"convnext_base_{channel}_{args.exp_suffix}"
    out_dir = _resolve(args.output_root) / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    defect_cols = CHANNEL_DEFECT_COLS.get(channel, [])

    print(f"\n{'='*72}")
    print(f"Training: {exp_id}")
    print(f"{'='*72}")
    print(f"  device          : {device}")
    print(f"  defect labels   : {defect_cols or '(none)'}")
    print(f"  output_dir      : {out_dir}")

    score_by_model = build_score_lookup(_resolve(args.labels_root), channel)

    def _ds(split, is_train):
        return TensorCacheCLIPDataset(
            tensor_cache_root=_resolve(args.tensor_cache_root),
            clip_feature_path=_resolve(args.clip_feature_path),
            split_image_root=_resolve(args.split_image_root) / channel,
            split=split,
            channel=channel,
            score_by_model=score_by_model,
            invalid_max_score=args.invalid_max_score,
            is_train=is_train,
            manifest_path=_resolve(args.manifest_path) if args.manifest_path else None,
            defect_cols=defect_cols,
        )

    train_ds = _ds("train", True)
    val_ds   = _ds("val",   False)
    print(f"  train           : {len(train_ds)}  scores={train_ds.score_counts()}")
    print(f"  val             : {len(val_ds)}")

    sampler = WeightedRandomSampler(
        train_ds.get_sample_weights(
            args.mid_oversample_factor, args.oversample_lo_score, args.oversample_hi_score,
            tail_factor=args.tail_oversample_factor, tail_lo_score=args.tail_lo_score,
        ),
        num_samples=len(train_ds), replacement=True,
    )
    train_loader = _make_loader(train_ds, args.batch_size, args.num_workers, sampler=sampler)
    val_loader   = _make_loader(val_ds,   args.batch_size * 2, args.num_workers)

    model = ConvNeXtRegressionScorer(
        clip_dim=args.clip_dim,
        attn_proj_dim=args.attn_proj_dim,
        attn_heads=args.attn_heads,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        n_defect_labels=len(defect_cols),
        freeze_features=True,
    ).to(device)

    print(f"  binary_loss_w   : {args.binary_loss_weight}")
    print(f"  defect_loss_w   : {args.defect_loss_weight}  (labels: {defect_cols})")
    print(f"  ranking_loss_w  : {args.ranking_loss_weight}")
    print(f"  mid_oversample  : {args.mid_oversample_factor}x  lo={args.oversample_lo_score} hi={args.oversample_hi_score}")
    if args.tail_oversample_factor:
        print(f"  tail_oversample : {args.tail_oversample_factor}x  lo={args.tail_lo_score}")

    def make_opt():
        return optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=args.weight_decay,
        )

    optimizer = make_opt()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    log = dict(channel=channel, arch="convnext_base", exp_id=exp_id,
               epoch=[], train_loss=[], val_mae=[], val_srcc=[], val_plcc=[],
               val_within_1=[], val_binary_f1=[], val_binary_best_f1=[])
    best = dict(mae=float("inf"), srcc=-2.0, binary_f1=-1.0)
    patience = 0

    for ei in range(args.epochs):
        epoch = ei + 1

        if ei == args.unfreeze_stage4_epoch:
            model.unfreeze_stage4(); optimizer = make_opt()
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - ei)
            print("  [unfreeze] stage4")
        elif ei == args.unfreeze_stage34_epoch:
            model.unfreeze_stage34(); optimizer = make_opt()
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - ei, eta_min=1e-6)
            print("  [unfreeze] stage3+4")
        elif ei == args.unfreeze_stage234_epoch:
            model.unfreeze_stage234(); optimizer = make_opt()
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - ei, eta_min=1e-6)
            print("  [unfreeze] stage2+3+4")

        model.train()
        running = 0.0
        for imgs, clips, scores, binaries, defects in tqdm(
            train_loader, desc=f"  epoch {epoch:>2}/{args.epochs}", leave=False
        ):
            imgs     = imgs.to(device, non_blocking=True)
            clips    = clips.to(device, non_blocking=True)
            scores   = scores.to(device, non_blocking=True)
            binaries = binaries.to(device, non_blocking=True)
            defects  = defects.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred_score, binary_logit, defect_logits = model(imgs, clips)

            loss = F.huber_loss(pred_score, scores, delta=args.huber_delta)
            if args.binary_loss_weight > 0:
                loss = loss + args.binary_loss_weight * F.binary_cross_entropy_with_logits(
                    binary_logit, binaries)
            if args.defect_loss_weight > 0 and defect_logits is not None and defects.shape[1] > 0:
                loss = loss + args.defect_loss_weight * F.binary_cross_entropy_with_logits(
                    defect_logits, defects)
            if args.ranking_loss_weight > 0:
                loss = loss + args.ranking_loss_weight * _ranking_loss(
                    pred_score, scores, args.ranking_margin)

            loss.backward()
            optimizer.step()
            running += float(loss.item())

        scheduler.step()
        train_loss = running / max(len(train_loader), 1)
        metrics = eval_regression_epoch(model, val_loader, device, args.invalid_max_score)
        b05   = metrics["binary_at_0_5"]
        bbest = metrics["binary_best"]

        log["epoch"].append(epoch)
        log["train_loss"].append(round(train_loss, 4))
        log["val_mae"].append(metrics["mae"])
        log["val_srcc"].append(metrics["srcc"])
        log["val_plcc"].append(metrics["plcc"])
        log["val_within_1"].append(metrics["within_1"])
        log["val_binary_f1"].append(b05["f1"])
        log["val_binary_best_f1"].append(bbest["f1"])
        (out_dir / "train_log.json").write_text(json.dumps(log, indent=2))

        print(
            f"  epoch={epoch:>2} loss={train_loss:.4f}  "
            f"MAE={metrics['mae']:.4f} SRCC={metrics['srcc']:.4f} PLCC={metrics['plcc']:.4f}  "
            f"w1={metrics['within_1']:.4f}  "
            f"F1@0.5={b05['f1']:.4f} bestF1={bbest['f1']:.4f}@{bbest['threshold']:.2f}"
        )

        improved = False
        if metrics["mae"] < best["mae"] - 1e-4:
            best["mae"] = metrics["mae"]
            _save_checkpoint(out_dir, "best_mae.pt", model, optimizer, epoch, channel, args, metrics)
            _hardlink(out_dir / "best_mae.pt", out_dir / "best.pt")
            improved = True
        if metrics["srcc"] > best["srcc"] + 1e-4:
            best["srcc"] = metrics["srcc"]
            _save_checkpoint(out_dir, "best_srcc.pt", model, optimizer, epoch, channel, args, metrics)
            improved = True
        if b05["f1"] > best["binary_f1"] + 1e-4:
            best["binary_f1"] = b05["f1"]
            _save_checkpoint(out_dir, "best_binary_f1.pt", model, optimizer, epoch, channel, args, metrics)
            improved = True

        patience = 0 if improved else patience + 1
        if patience >= args.patience:
            print(f"  early stop: patience={args.patience}")
            break

    (out_dir / "summary.json").write_text(
        json.dumps(dict(exp_id=exp_id, channel=channel, best=best, last_log=log), indent=2)
    )
    return dict(exp_id=exp_id, channel=channel, best=best, output_dir=str(out_dir))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=PACKAGE_ROOT / "config" / "phase2_regression.yaml")
    p.add_argument("--channel", choices=ALL_CHANNELS, default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--device", choices=("cuda", "cpu"), default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--save-optimizer", action="store_true")
    args = p.parse_args()

    cfg = _load_config(args.config)
    d, m, t, u, o = cfg["data"], cfg["model"], cfg["train"], cfg["unfreeze"], cfg["output"]

    args.tensor_cache_root       = d["tensor_cache_root"]
    args.clip_feature_path       = d["clip_feature_path"]
    args.split_image_root        = d["split_image_root"]
    args.labels_root             = d["labels_root"]
    args.manifest_path           = d.get("manifest_path")
    args.clip_dim                = int(m.get("clip_dim", 1536))
    args.attn_proj_dim           = int(m.get("attn_proj_dim", 256))
    args.attn_heads              = int(m.get("attn_heads", 4))
    args.hidden_dim              = int(m.get("hidden_dim", 512))
    args.dropout                 = float(m.get("dropout", 0.3))
    args.invalid_max_score       = int(m["invalid_max_score"])
    args.output_root             = o.get("root", "asset_quality_scorer/outputs/runs")
    args.exp_suffix              = o.get("exp_suffix", "baseline")
    args.epochs                  = args.epochs     or int(t["epochs"])
    args.batch_size              = args.batch_size or int(t["batch_size"])
    args.num_workers             = int(t["num_workers"])
    args.lr                      = float(t["lr"])
    args.weight_decay            = float(t["weight_decay"])
    args.patience                = int(t["patience"])
    args.defect_loss_weight      = float(t.get("defect_loss_weight", 0.1))
    args.ranking_loss_weight     = float(t.get("ranking_loss_weight", 0.05))
    args.ranking_margin          = float(t.get("ranking_margin", 0.5))
    args.huber_delta             = float(t.get("huber_delta", 1.0))

    channel_for_override = args.channel or None
    _ovr = t.get("channel_oversample", {}).get(channel_for_override, {}) if channel_for_override else {}
    args.mid_oversample_factor  = float(_ovr.get("mid_oversample_factor", t.get("mid_oversample_factor", 4.0)))
    args.oversample_lo_score    = int(_ovr.get("lo_score", t.get("lo_score", 1)))
    _hi = _ovr.get("hi_score", t.get("hi_score", None))
    args.oversample_hi_score    = int(_hi) if _hi is not None else None
    _tail = _ovr.get("tail_oversample_factor", t.get("tail_oversample_factor", None))
    args.tail_oversample_factor = float(_tail) if _tail is not None else None
    args.tail_lo_score          = int(_ovr.get("tail_lo_score", t.get("tail_lo_score", 4)))
    args.binary_loss_weight     = float(_ovr.get("binary_loss_weight", t.get("binary_loss_weight", 0.2)))

    args.unfreeze_stage4_epoch   = int(u.get("stage4_epoch", 5))
    args.unfreeze_stage34_epoch  = int(u.get("stage34_epoch", 10))
    args.unfreeze_stage234_epoch = int(u.get("stage234_epoch", 15))
    args.config_channels         = tuple(d["channels"])

    if not args.all and args.channel is None:
        p.error("pass --channel <name> or --all")
    return args


def main() -> int:
    args = parse_args()
    channels = args.config_channels if args.all else (args.channel,)
    summaries = [train_channel(args, ch) for ch in channels]
    print("\nDone.")
    for s in summaries:
        print(f"  {s['exp_id']}: {s['best']}  → {s['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
