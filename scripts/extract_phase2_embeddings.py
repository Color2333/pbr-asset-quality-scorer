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
from quality_scorer.convnext_backbone import ConvNeXtOrdinalScorer
from quality_scorer.ordinal import logits_to_class_probs
from quality_scorer.transforms import get_transforms
from screening.labels import load_csv


class ChannelImageDataset(Dataset):
    def __init__(self, paths: list[Path], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        return path.stem, self.transform(image)


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _collect_channel_paths(image_root: Path, split: str, channel: str) -> dict[str, Path]:
    split_root = image_root / channel / split
    paths: dict[str, Path] = {}
    for label_dir in ("valid", "invalid"):
        folder = split_root / label_dir
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.png")):
            paths[path.stem] = path
    return paths


def _load_metadata(labels_root: Path) -> dict[str, dict]:
    metadata: dict[str, dict] = {}
    for batch_dir in sorted(labels_root.iterdir()):
        if not batch_dir.is_dir() or not batch_dir.name.startswith("batch_"):
            continue
        csv_files = sorted(labels_root.glob(f"{batch_dir.name}_*.csv"))
        if not csv_files:
            continue
        labels = load_csv(str(csv_files[0]))
        for model_name, label in labels.items():
            metadata[model_name] = {
                "batch": batch_dir.name,
                "tier": label.get("tier", ""),
                "final_score": label.get("final_score", -1.0),
                "pbr_type": label.get("pbr_type", ""),
                "base_color_score": label.get("base_color_score", -1),
                "normal_score": label.get("normal_score", -1),
                "roughness_score": label.get("roughness_score", -1),
                "metallic_score": label.get("metallic_score", -1),
            }
    return metadata


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
    paths_by_stem: dict[str, Path],
    channel: str,
    stems: list[str],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict[str, dict[str, np.ndarray | float | int]]:
    paths = [paths_by_stem[stem] for stem in stems]
    dataset = ChannelImageDataset(paths, get_transforms(False, channel))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    outputs: dict[str, dict[str, np.ndarray | float | int]] = {}
    class_values = None
    for batch_stems, images in loader:
        images = images.to(device, non_blocking=True)
        logits, features = model(images, return_features=True)
        probs = logits_to_class_probs(logits)
        if class_values is None:
            class_values = torch.arange(probs.shape[1], device=device, dtype=probs.dtype).view(1, -1)
        expected = (probs * class_values).sum(dim=1)
        pred = probs.argmax(dim=1)
        for idx, stem in enumerate(batch_stems):
            outputs[str(stem)] = {
                "feature": features[idx].detach().cpu().numpy().astype("float32"),
                "expected_score": float(expected[idx].detach().cpu().item()),
                "pred_score": int(pred[idx].detach().cpu().item()),
                "score_probs": probs[idx].detach().cpu().numpy().astype("float32"),
            }
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Phase 2 asset embeddings from ordinal scorers")
    parser.add_argument("--image-root", default="screening/data_v2")
    parser.add_argument("--labels-root", default="screening/data_38k")
    parser.add_argument("--checkpoint-root", default="asset_quality_scorer/outputs/phase1_ordinal")
    parser.add_argument("--output-root", default="asset_quality_scorer/outputs/phase2_embedding")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--backbone", default="base")
    parser.add_argument("--checkpoint-name", default="best.pt")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-assets", type=int, default=0, help="debug only")
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_root = _resolve_path(args.image_root)
    labels_root = _resolve_path(args.labels_root)
    checkpoint_root = _resolve_path(args.checkpoint_root)
    output_root = _resolve_path(args.output_root) / args.split
    output_root.mkdir(parents=True, exist_ok=True)

    channel_paths = {
        channel: _collect_channel_paths(image_root, args.split, channel)
        for channel in ALL_CHANNELS
    }
    common_stems = set.intersection(*(set(paths) for paths in channel_paths.values()))
    stems = sorted(common_stems)
    if args.max_assets:
        stems = stems[: args.max_assets]
    if not stems:
        raise RuntimeError(f"no common assets found for split={args.split}")

    print(f"split={args.split} common_assets={len(stems)}")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    metadata = _load_metadata(labels_root)
    channel_outputs = {}
    for channel in ALL_CHANNELS:
        ckpt_path = _checkpoint_for(checkpoint_root, args.backbone, channel, args.checkpoint_name)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint missing for {channel}: {ckpt_path}")
        print(f"extracting {channel}: {ckpt_path}")
        model = _load_model(ckpt_path, device)
        channel_outputs[channel] = _extract_channel(
            model,
            channel_paths[channel],
            channel,
            stems,
            args.batch_size,
            args.num_workers,
            device,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    embeddings = []
    score_rows = []
    for stem in stems:
        features = [channel_outputs[channel][stem]["feature"] for channel in ALL_CHANNELS]
        embeddings.append(np.concatenate(features).astype("float32"))
        meta = metadata.get(stem, {})
        row = {
            "model_name": stem,
            **meta,
        }
        for channel in ALL_CHANNELS:
            out = channel_outputs[channel][stem]
            row[f"{channel}_expected_score"] = out["expected_score"]
            row[f"{channel}_pred_score"] = out["pred_score"]
        score_rows.append(row)

    embedding_array = np.stack(embeddings).astype("float32")
    np.savez_compressed(
        output_root / "asset_embeddings.npz",
        model_names=np.asarray(stems),
        embeddings=embedding_array,
    )
    with (output_root / "asset_scores.csv").open("w", newline="") as file:
        fieldnames = list(score_rows[0].keys())
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(score_rows)

    manifest = {
        "split": args.split,
        "num_assets": len(stems),
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
