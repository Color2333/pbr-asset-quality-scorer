"""Dataset classes for PBR asset quality scoring.

PBRScoreDataset       — Phase 1: reads PNG files or LMDB, ordinal scores
TensorCacheCLIPDataset — Phase 2: 224×224 memmap tensors + CLIP features + defect labels
build_score_lookup    — shared helper: model_name → ordinal score from batch CSVs
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from quality_scorer.constants import SCORE_KEYS
from screening.labels import load_csv

# ── image normalisation ───────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_TO_FLOAT  = transforms.ConvertImageDtype(torch.float32)
_NORMALIZE = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

_TRAIN_SPATIAL = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(15),
])
_TRAIN_ERASE = transforms.RandomErasing(p=0.25, scale=(0.02, 0.15))

# ── defect label columns available per channel ────────────────────────────────

CHANNEL_DEFECT_COLS: dict[str, list[str]] = {
    "base_color": ["has_text_or_pattern", "base_color_fake_ao_or_glow"],
    "normal_map":  ["normal_abnormal_tint"],
    "metallic":    ["metallic_invalid"],
    "roughness":   ["roughness_invalid"],
}


# ── Phase 1: file/LMDB-backed dataset ────────────────────────────────────────

class PBRScoreDataset(Dataset):
    """Prepared channel images paired with raw ordinal channel scores.

    Supports two image backends:
      - "path" (default): reads PNG files from image_root/split/invalid|valid/
      - "lmdb": reads from a pre-built LMDB store (key = "model_name/channel").
    """

    def __init__(
        self,
        image_root: Path,
        split: str,
        score_by_model: dict[str, int],
        transform=None,
        num_classes: int = 6,
        image_backend: str = "path",
        lmdb_path: Optional[Path] = None,
        channel: Optional[str] = None,
    ):
        self.samples: list[tuple[str, int]] = []
        self.transform = transform
        self.num_classes = num_classes
        self.image_backend = image_backend
        self.lmdb_path = Path(lmdb_path) if lmdb_path else None
        self.channel = channel
        self._lmdb_env = None

        if image_backend == "lmdb" and (lmdb_path is None or channel is None):
            raise ValueError("lmdb_path and channel are required for image_backend='lmdb'")

        split_root = image_root / split
        for label_dir in ("invalid", "valid"):
            channel_dir = split_root / label_dir
            if not channel_dir.exists():
                continue
            for path in sorted(channel_dir.glob("*.png")):
                score = score_by_model.get(path.stem)
                if score is None or score < 0 or score >= num_classes:
                    continue
                key = path.stem if image_backend == "lmdb" else str(path)
                self.samples.append((key, int(score)))

    def _env(self):
        if self._lmdb_env is None:
            import lmdb
            self._lmdb_env = lmdb.open(
                str(self.lmdb_path), readonly=True, lock=False,
                readahead=False, max_readers=128,
            )
        return self._lmdb_env

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        key, score = self.samples[idx]
        if self.image_backend == "lmdb":
            lmdb_key = f"{key}/{self.channel}".encode("utf-8")
            with self._env().begin(write=False) as txn:
                value = txn.get(lmdb_key)
            if value is None:
                raise KeyError(f"LMDB key not found: {lmdb_key!r}")
            image = Image.open(BytesIO(value)).convert("RGB")
        else:
            image = Image.open(key).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, score

    def score_counts(self) -> dict[int, int]:
        return dict(sorted(Counter(score for _, score in self.samples).items()))

    def get_sample_weights(self, mid_factor: float = 4.0) -> list[float]:
        lo, hi = 1, self.num_classes - 2
        return [mid_factor if lo <= score <= hi else 1.0 for _, score in self.samples]


# ── Phase 2: tensor cache + CLIP features ────────────────────────────────────

def _get_image_transform(is_train: bool, channel: str):
    if not is_train:
        return transforms.Compose([_TO_FLOAT, _NORMALIZE])
    color_jitter = {
        "normal_map": None,
        "roughness":   transforms.ColorJitter(brightness=0.2, contrast=0.2),
        "metallic":    transforms.ColorJitter(brightness=0.2, contrast=0.2),
        "base_color":  transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                               saturation=0.3, hue=0.1),
    }.get(channel)
    steps = [_TRAIN_SPATIAL, _TO_FLOAT, _NORMALIZE]
    if color_jitter is not None:
        steps.insert(1, color_jitter)
    steps.append(_TRAIN_ERASE)
    return transforms.Compose(steps)


def _load_manifest_defect_labels(
    manifest_path: Path, defect_cols: list[str]
) -> dict[str, list[float]]:
    lookup: dict[str, list[float]] = {}
    with manifest_path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            vals = []
            for col in defect_cols:
                try:
                    vals.append(float(row.get(col, 0) or 0))
                except (ValueError, TypeError):
                    vals.append(0.0)
            lookup[row["model_name"]] = vals
    return lookup


class TensorCacheCLIPDataset(Dataset):
    """Loads images from 224×224 uint8 memmap + CLIP features + optional defect labels.

    Args:
        tensor_cache_root : directory with meta.json and {channel}.npy files
        clip_feature_path : path to clip_vitl14_openai_base_color_render.pt
        split_image_root  : screening/data_v2/{channel} — enumerates train/val/test split
        split             : "train" | "val" | "test"
        channel           : which channel (metallic / roughness / normal_map / base_color)
        score_by_model    : dict model_name → ordinal score (0..5)
        invalid_max_score : scores ≤ this are binary-label=1
        is_train          : apply training augmentations
        num_classes       : number of ordinal classes (default 6)
        manifest_path     : path to manifest CSV for defect labels (optional)
        defect_cols       : list of column names to use as defect auxiliary labels
    """

    def __init__(
        self,
        tensor_cache_root: Path,
        clip_feature_path: Path,
        split_image_root: Path,
        split: str,
        channel: str,
        score_by_model: dict[str, int],
        invalid_max_score: int = 1,
        is_train: bool = False,
        num_classes: int = 6,
        manifest_path: Optional[Path] = None,
        defect_cols: Optional[list[str]] = None,
    ):
        self.channel = channel
        self.invalid_max_score = invalid_max_score
        self.num_classes = num_classes
        self.defect_cols = defect_cols or []
        self.transform = _get_image_transform(is_train, channel)

        cache_root = Path(tensor_cache_root)
        meta = json.loads((cache_root / "meta.json").read_text(encoding="utf-8"))
        self._cache_idx: dict[str, int] = {n: i for i, n in enumerate(meta["model_names"])}
        if channel not in meta["channels"]:
            raise ValueError(f"Channel '{channel}' not in tensor cache {meta['channels']}")
        self._array: Optional[np.ndarray] = None
        self._array_path = cache_root / f"{channel}.npy"

        clip_data = torch.load(clip_feature_path, map_location="cpu", weights_only=False)
        clip_names: list[str] = clip_data["model_names"]
        self._clip_idx: dict[str, int] = {n: i for i, n in enumerate(clip_names)}
        bc = clip_data["features"]["base_color"].float()
        rn = clip_data["features"]["render"].float()
        self._clip_tensor: torch.Tensor = torch.cat([bc, rn], dim=1)

        self._defect_lookup: dict[str, list[float]] = {}
        if manifest_path and self.defect_cols:
            self._defect_lookup = _load_manifest_defect_labels(
                Path(manifest_path), self.defect_cols
            )

        self.samples: list[tuple[str, int]] = []
        split_root = Path(split_image_root) / split
        for label_dir in ("invalid", "valid"):
            d = split_root / label_dir
            if not d.exists():
                continue
            for path in sorted(d.glob("*.png")):
                name = path.stem
                score = score_by_model.get(name)
                if score is None or score < 0 or score >= num_classes:
                    continue
                if name not in self._cache_idx or name not in self._clip_idx:
                    continue
                self.samples.append((name, int(score)))

    def _get_array(self) -> np.ndarray:
        if self._array is None:
            self._array = np.load(self._array_path, mmap_mode="r")
        return self._array

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        name, score = self.samples[idx]
        img = torch.from_numpy(np.array(self._get_array()[self._cache_idx[name]], copy=True))
        img = self.transform(img)
        clip_feat = self._clip_tensor[self._clip_idx[name]]
        binary = float(score <= self.invalid_max_score)
        if self.defect_cols:
            defect = torch.tensor(
                self._defect_lookup.get(name, [0.0] * len(self.defect_cols)),
                dtype=torch.float32,
            )
        else:
            defect = torch.zeros(0)
        return (
            img,
            clip_feat,
            torch.tensor(score, dtype=torch.float32),
            torch.tensor(binary, dtype=torch.float32),
            defect,
        )

    def score_counts(self) -> dict[int, int]:
        return dict(sorted(Counter(s for _, s in self.samples).items()))

    def get_sample_weights(
        self,
        mid_factor: float = 4.0,
        lo_score: int = 1,
        hi_score: int | None = None,
        tail_factor: float | None = None,
        tail_lo_score: int = 4,
    ) -> list[float]:
        if hi_score is None:
            hi_score = self.num_classes - 1
        weights = []
        for _, s in self.samples:
            if tail_factor is not None and s >= tail_lo_score:
                weights.append(tail_factor)
            elif lo_score <= s <= hi_score:
                weights.append(mid_factor)
            else:
                weights.append(1.0)
        return weights


# ── shared helper ─────────────────────────────────────────────────────────────

def build_score_lookup(labels_root: Path, channel: str) -> dict[str, int]:
    score_key = SCORE_KEYS[channel]
    lookup: dict[str, int] = {}
    for batch_dir in sorted(labels_root.iterdir()):
        if not batch_dir.is_dir() or not batch_dir.name.startswith("batch_"):
            continue
        csv_files = sorted(labels_root.glob(f"{batch_dir.name}_*.csv"))
        if not csv_files:
            continue
        labels = load_csv(str(csv_files[0]))
        for model_name, label in labels.items():
            score = label.get(score_key, -1)
            if isinstance(score, int) and score >= 0:
                lookup[model_name] = score
    return lookup
