from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "asset_quality_scorer"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT))

from quality_scorer.constants import ALL_CHANNELS
from quality_scorer.models import ConvNeXtOrdinalScorer
from quality_scorer.data import PBRScoreDataset, build_score_lookup
from quality_scorer.metrics import compute_pos_weight, eval_ordinal_epoch
from quality_scorer.ordinal import CoralEntropyLoss
from quality_scorer.transforms import get_transforms


UNFREEZE_STAGE3_EPOCH = 5
UNFREEZE_STAGE23_EPOCH = 10


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_config(path: Path) -> dict:
    with path.open("r") as file:
        return yaml.safe_load(file)


def _make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    sampler=None,
) -> DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if sampler is not None:
        kwargs["sampler"] = sampler
    else:
        kwargs["shuffle"] = shuffle
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return DataLoader(dataset, **kwargs)


def _save_checkpoint(
    output_dir: Path,
    filename: str,
    model,
    optimizer,
    epoch: int,
    channel: str,
    args: argparse.Namespace,
    metrics: dict,
):
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "channel": channel,
        "arch": f"convnext_ordinal_{args.backbone}",
        "backbone": args.backbone,
        "num_classes": args.num_classes,
        "invalid_max_score": args.invalid_max_score,
        "val_metrics": metrics,
    }
    if args.save_optimizer:
        state["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(state, output_dir / filename)


def _replace_hardlink(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        import shutil

        shutil.copy2(source, target)


def train_channel(args: argparse.Namespace, channel: str) -> dict:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    image_root = _resolve_path(args.image_root) / channel
    labels_root = _resolve_path(args.labels_root)
    output_dir = _resolve_path(args.output_root) / f"convnext_{args.backbone}_{channel}_coral"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 72}")
    print(f"Phase 1 ordinal scorer: {channel}")
    print(f"{'=' * 72}")
    lmdb_path = _resolve_path(args.lmdb_path) if args.lmdb_path else None
    print(f"  device      : {device}")
    print(f"  image_root  : {image_root}")
    print(f"  labels_root : {labels_root}")
    print(f"  output_dir  : {output_dir}")
    print(f"  image backend: {'lmdb → ' + str(lmdb_path) if lmdb_path else 'path'}")

    score_by_model = build_score_lookup(labels_root, channel)
    backend = "lmdb" if lmdb_path else "path"
    train_ds = PBRScoreDataset(
        image_root,
        "train",
        score_by_model,
        transform=get_transforms(True, channel),
        num_classes=args.num_classes,
        image_backend=backend,
        lmdb_path=lmdb_path,
        channel=channel,
    )
    val_ds = PBRScoreDataset(
        image_root,
        "val",
        score_by_model,
        transform=get_transforms(False, channel),
        num_classes=args.num_classes,
        image_backend=backend,
        lmdb_path=lmdb_path,
        channel=channel,
    )
    if args.max_train_samples:
        train_ds.samples = train_ds.samples[: args.max_train_samples]
    if args.max_val_samples:
        val_ds.samples = val_ds.samples[: args.max_val_samples]

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(f"empty train/val dataset for {channel}")

    print(f"  train       : {len(train_ds)}  scores={train_ds.score_counts()}")
    print(f"  val         : {len(val_ds)}  scores={val_ds.score_counts()}")

    # Weighted sampler: oversample intermediate scores to counter bimodal collapse
    sample_weights = train_ds.get_sample_weights(mid_factor=args.mid_oversample_factor)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_ds),
        replacement=True,
    )
    train_loader = _make_loader(train_ds, args.batch_size, shuffle=False, num_workers=args.num_workers, sampler=sampler)
    val_loader = _make_loader(val_ds, args.batch_size * 2, False, args.num_workers)

    model = ConvNeXtOrdinalScorer(
        backbone=args.backbone,
        num_classes=args.num_classes,
        freeze_features=True,
    ).to(device)
    pos_weight = compute_pos_weight(
        (score for _, score in train_ds.samples),
        args.num_classes,
        device,
    )
    criterion = CoralEntropyLoss(
        args.num_classes,
        pos_weight=pos_weight,
        entropy_lambda=args.entropy_lambda,
        label_smoothing=args.label_smoothing,
    )
    print(f"  pos_weight      : {[round(float(x), 3) for x in pos_weight.detach().cpu()]}")
    print(f"  entropy_lambda  : {args.entropy_lambda}")
    print(f"  label_smoothing : {args.label_smoothing}")
    print(f"  mid_oversample  : {args.mid_oversample_factor}x")

    def make_optimizer():
        return optim.AdamW(
            filter(lambda param: param.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    optimizer = make_optimizer()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    log = {
        "channel": channel,
        "arch": f"convnext_ordinal_{args.backbone}",
        "epoch": [],
        "train_loss": [],
        "val_ordinal_mae": [],
        "val_expected_mae": [],
        "val_within_1": [],
        "val_expected_within_1": [],
        "val_binary_f1": [],
        "val_binary_best_f1": [],
    }
    best = {
        "expected_mae": float("inf"),
        "ordinal_mae": float("inf"),
        "within_1": -1.0,
        "binary_f1": -1.0,
    }
    patience = 0

    for epoch_idx in range(args.epochs):
        epoch = epoch_idx + 1
        if epoch_idx == UNFREEZE_STAGE3_EPOCH:
            model.unfreeze_stage3()
            optimizer = make_optimizer()
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - epoch_idx)
            print("  [unfreeze] stage3")
        elif epoch_idx == UNFREEZE_STAGE23_EPOCH:
            model.unfreeze_stage23()
            optimizer = make_optimizer()
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.epochs - epoch_idx,
                eta_min=1e-6,
            )
            print("  [unfreeze] stage2+stage3")

        model.train()
        running_loss = 0.0
        for images, scores in tqdm(train_loader, desc=f"  epoch {epoch:>2}/{args.epochs}", leave=False):
            images = images.to(device, non_blocking=True)
            scores = scores.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), scores)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())

        scheduler.step()
        train_loss = running_loss / max(len(train_loader), 1)
        metrics = eval_ordinal_epoch(
            model,
            val_loader,
            device,
            num_classes=args.num_classes,
            invalid_max_score=args.invalid_max_score,
        )
        binary_at_0_5 = metrics["binary_at_0_5"]
        binary_best = metrics["binary_best"]

        log["epoch"].append(epoch)
        log["train_loss"].append(round(train_loss, 4))
        log["val_ordinal_mae"].append(metrics["ordinal_mae"])
        log["val_expected_mae"].append(metrics["expected_mae"])
        log["val_within_1"].append(metrics["within_1"])
        log["val_expected_within_1"].append(metrics.get("expected_within_1", 0.0))
        log["val_binary_f1"].append(binary_at_0_5["f1"])
        log["val_binary_best_f1"].append(binary_best["f1"])
        with (output_dir / "train_log.json").open("w") as file:
            json.dump(log, file, indent=2)

        print(
            f"  epoch={epoch:>2} loss={train_loss:.4f} "
            f"argMAE={metrics['ordinal_mae']:.4f} expMAE={metrics['expected_mae']:.4f} "
            f"w1={metrics['within_1']:.4f} expW1={metrics.get('expected_within_1', 0):.4f} "
            f"F1@0.5={binary_at_0_5['f1']:.4f} bestF1={binary_best['f1']:.4f}@{binary_best['threshold']:.2f}"
        )

        improved = False
        # Primary criterion: expected_mae (continuous, smoother than argmax MAE)
        if metrics["expected_mae"] < best["expected_mae"] - 1e-4:
            best["expected_mae"] = metrics["expected_mae"]
            _save_checkpoint(output_dir, "best_expected_mae.pt", model, optimizer, epoch, channel, args, metrics)
            _replace_hardlink(output_dir / "best_expected_mae.pt", output_dir / "best.pt")
            improved = True
        if metrics["ordinal_mae"] < best["ordinal_mae"] - 1e-4:
            best["ordinal_mae"] = metrics["ordinal_mae"]
            _save_checkpoint(output_dir, "best_ordinal_mae.pt", model, optimizer, epoch, channel, args, metrics)
            improved = True
        if metrics["within_1"] > best["within_1"] + 1e-4:
            best["within_1"] = metrics["within_1"]
            _save_checkpoint(output_dir, "best_within_1.pt", model, optimizer, epoch, channel, args, metrics)
            improved = True
        if binary_at_0_5["f1"] > best["binary_f1"] + 1e-4:
            best["binary_f1"] = binary_at_0_5["f1"]
            _save_checkpoint(output_dir, "best_binary_f1.pt", model, optimizer, epoch, channel, args, metrics)
            improved = True

        if improved:
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                print(f"  early stop: patience={args.patience}")
                break

    with (output_dir / "summary.json").open("w") as file:
        json.dump({"channel": channel, "best": best, "last_log": log}, file, indent=2)
    return {"channel": channel, "best": best, "output_dir": str(output_dir)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Phase 1 ordinal PBR quality scorers")
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "config" / "phase1_ordinal.yaml")
    parser.add_argument("--channel", choices=ALL_CHANNELS, default=None)
    parser.add_argument("--all", action="store_true", help="train all channels from config")
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--labels-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--backbone", choices=("base", "large"), default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--invalid-max-score", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--entropy-lambda", type=float, default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--mid-oversample-factor", type=float, default=None)
    parser.add_argument("--lmdb-path", default=None, help="LMDB store path (model_name/channel keys)")
    parser.add_argument("--max-train-samples", type=int, default=0, help="debug only")
    parser.add_argument("--max-val-samples", type=int, default=0, help="debug only")
    parser.add_argument("--save-optimizer", action="store_true", help="include optimizer state in checkpoints")
    args = parser.parse_args()

    config = _load_config(args.config)
    args.image_root = args.image_root or config["data"]["image_root"]
    args.labels_root = args.labels_root or config["data"]["labels_root"]
    args.output_root = args.output_root or config["output"]["root"]
    args.backbone = args.backbone or config["model"]["backbone"]
    args.num_classes = args.num_classes or int(config["model"]["num_classes"])
    args.invalid_max_score = args.invalid_max_score or int(config["model"]["invalid_max_score"])
    args.epochs = args.epochs or int(config["train"]["epochs"])
    args.batch_size = args.batch_size or int(config["train"]["batch_size"])
    args.num_workers = args.num_workers if args.num_workers is not None else int(config["train"]["num_workers"])
    args.lr = args.lr or float(config["train"]["lr"])
    args.weight_decay = args.weight_decay or float(config["train"]["weight_decay"])
    args.patience = args.patience or int(config["train"]["patience"])
    args.entropy_lambda = args.entropy_lambda if args.entropy_lambda is not None else float(config["train"].get("entropy_lambda", 0.15))
    args.label_smoothing = args.label_smoothing if args.label_smoothing is not None else float(config["train"].get("label_smoothing", 0.05))
    args.mid_oversample_factor = args.mid_oversample_factor if args.mid_oversample_factor is not None else float(config["train"].get("mid_oversample_factor", 4.0))
    args.lmdb_path = args.lmdb_path or config["data"].get("lmdb_path", None)
    args.config_channels = tuple(config["data"]["channels"])
    if args.invalid_max_score >= args.num_classes:
        parser.error("--invalid-max-score must be < --num-classes")
    if not args.all and args.channel is None:
        parser.error("pass --channel <name> or --all")
    return args


def main() -> int:
    args = parse_args()
    channels = args.config_channels if args.all else (args.channel,)
    summaries = [train_channel(args, channel) for channel in channels]
    print("\nDone.")
    for summary in summaries:
        print(f"  {summary['channel']}: {summary['best']} -> {summary['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
