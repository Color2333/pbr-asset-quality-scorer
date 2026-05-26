from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "asset_quality_scorer"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT))

from quality_scorer.constants import ALL_CHANNELS
from quality_scorer.models import ConvNeXtOrdinalScorer
from quality_scorer.ordinal import logits_to_class_probs
from quality_scorer.transforms import get_transforms
from screening.labels import load_csv


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _crop_channel(model_dir: Path, channel: str) -> Image.Image:
    if channel in ("base_color", "roughness", "metallic"):
        image = Image.open(model_dir / "grid_pbr.png").convert("RGB")
        width, height = image.size
        half_w, half_h = width // 2, height // 2
        boxes = {
            "base_color": (half_w, 0, width, half_h),
            "roughness": (0, half_h, half_w, height),
            "metallic": (half_w, half_h, width, height),
        }
        return image.crop(boxes[channel])
    if channel == "normal_map":
        image = Image.open(model_dir / "grid_white.png").convert("RGB")
        width, height = image.size
        half_w, half_h = width // 2, height // 2
        return image.crop((half_w, 0, width, half_h))
    raise ValueError(f"unsupported channel: {channel}")


class RawChannelDataset(Dataset):
    def __init__(self, records: list[dict], channel: str, transform):
        self.records = records
        self.channel = channel
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        image = _crop_channel(Path(record["model_dir"]), self.channel)
        return record["model_name"], self.transform(image)


def _load_records(labels_root: Path, max_assets: int = 0) -> list[dict]:
    records: list[dict] = []
    for batch_dir in sorted(labels_root.iterdir()):
        if not batch_dir.is_dir() or not batch_dir.name.startswith("batch_"):
            continue
        csv_files = sorted(labels_root.glob(f"{batch_dir.name}_*.csv"))
        if not csv_files:
            continue
        labels = load_csv(str(csv_files[0]))
        for model_name, label in labels.items():
            model_dir = batch_dir / model_name
            if not (model_dir / "grid_pbr.png").exists() or not (model_dir / "grid_white.png").exists():
                continue
            records.append(
                {
                    "model_name": model_name,
                    "model_dir": str(model_dir),
                    "batch": batch_dir.name,
                    "tier": label.get("tier", ""),
                    "final_score": label.get("final_score", -1.0),
                    "pbr_type": label.get("pbr_type", ""),
                    "base_color_score": label.get("base_color_score", -1),
                    "normal_score": label.get("normal_score", -1),
                    "roughness_score": label.get("roughness_score", -1),
                    "metallic_score": label.get("metallic_score", -1),
                }
            )
            if max_assets and len(records) >= max_assets:
                return records
    return records


def _checkpoint_for(output_root: Path, backbone: str, channel: str, checkpoint_name: str) -> Path:
    return output_root / f"convnext_{backbone}_{channel}_coral" / checkpoint_name


def _load_model(ckpt_path: Path, device: torch.device) -> ConvNeXtOrdinalScorer:
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = ConvNeXtOrdinalScorer(
        backbone=checkpoint.get("backbone", "base"),
        num_classes=int(checkpoint.get("num_classes", 6)),
        freeze_features=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model


@torch.no_grad()
def _extract_channel(
    model: ConvNeXtOrdinalScorer,
    records: list[dict],
    channel: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict[str, dict[str, np.ndarray | float | int]]:
    dataset = RawChannelDataset(records, channel, get_transforms(False, channel))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    outputs: dict[str, dict[str, np.ndarray | float | int]] = {}
    class_values = None
    for batch_idx, (model_names, images) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        logits, features = model(images, return_features=True)
        probs = logits_to_class_probs(logits)
        if class_values is None:
            class_values = torch.arange(probs.shape[1], device=device, dtype=probs.dtype).view(1, -1)
        expected = (probs * class_values).sum(dim=1)
        pred = probs.argmax(dim=1)
        for idx, model_name in enumerate(model_names):
            outputs[str(model_name)] = {
                "feature": features[idx].detach().cpu().numpy().astype("float32"),
                "expected_score": float(expected[idx].detach().cpu().item()),
                "pred_score": int(pred[idx].detach().cpu().item()),
            }
        if batch_idx % 20 == 0:
            print(f"  {channel}: batch {batch_idx}/{len(loader)}")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract raw-grid asset embeddings from Phase 1 scorers")
    parser.add_argument("--labels-root", default="screening/data_38k")
    parser.add_argument("--checkpoint-root", default="asset_quality_scorer/outputs/phase1_ordinal")
    parser.add_argument("--output-root", default="asset_quality_scorer/outputs/phase2_embedding_raw")
    parser.add_argument("--backbone", default="base")
    parser.add_argument("--checkpoint-name", default="best.pt")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-assets", type=int, default=0, help="debug only")
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labels_root = _resolve_path(args.labels_root)
    checkpoint_root = _resolve_path(args.checkpoint_root)
    output_root = _resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    records = _load_records(labels_root, max_assets=args.max_assets)
    if not records:
        raise RuntimeError(f"no raw grid records found under {labels_root}")
    print(f"raw_assets={len(records)}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    channel_outputs = {}
    for channel in ALL_CHANNELS:
        ckpt_path = _checkpoint_for(checkpoint_root, args.backbone, channel, args.checkpoint_name)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint missing for {channel}: {ckpt_path}")
        print(f"extracting {channel}: {ckpt_path}")
        model = _load_model(ckpt_path, device)
        channel_outputs[channel] = _extract_channel(
            model,
            records,
            channel,
            args.batch_size,
            args.num_workers,
            device,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    embeddings = []
    rows = []
    for record in records:
        model_name = record["model_name"]
        features = [channel_outputs[channel][model_name]["feature"] for channel in ALL_CHANNELS]
        embeddings.append(np.concatenate(features).astype("float32"))
        row = {key: value for key, value in record.items() if key != "model_dir"}
        for channel in ALL_CHANNELS:
            out = channel_outputs[channel][model_name]
            row[f"{channel}_expected_score"] = out["expected_score"]
            row[f"{channel}_pred_score"] = out["pred_score"]
        rows.append(row)

    embedding_array = np.stack(embeddings).astype("float32")
    np.savez_compressed(
        output_root / "asset_embeddings.npz",
        model_names=np.asarray([record["model_name"] for record in records]),
        embeddings=embedding_array,
    )
    with (output_root / "asset_scores.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "num_assets": len(records),
        "embedding_dim": int(embedding_array.shape[1]),
        "channels": list(ALL_CHANNELS),
        "checkpoint_name": args.checkpoint_name,
        "outputs": {
            "embeddings": str(output_root / "asset_embeddings.npz"),
            "scores": str(output_root / "asset_scores.csv"),
        },
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
