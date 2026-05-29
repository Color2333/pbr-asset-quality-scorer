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
    "base_color": ["hasTextOrPattern", "baseColorHasFakeAOOrGlow"],
    "normal_map":  ["normalHasAbnormalTint", "normalIsFlipped"],
    "metallic":    [],
    "roughness":   [],
}

# ── directional CLIP prompts per channel ──────────────────────────────────────
# These are computed on-the-fly at dataset init from pre-extracted per-channel
# CLS embeddings.  To adjust prompts: just edit the strings here and re-train —
# no re-extraction of image features needed.

CHANNEL_PROMPTS: dict[str, list[str]] = {
    "metallic": [
        "a highly reflective metallic surface made of steel or aluminum",
        "a non-metallic surface made of painted wood, fabric, or plastic",
        "a partially metallic object with mixed metal and non-metal materials",
        "an all-black non-metallic material correctly assigned zero metallic value",
        "an incorrect all-black metallic map where metallic data is missing",
        "a uniform grey metallic map indicating partially metallic material",
        "a metallic texture showing surface oxidation and rust",
        "a chrome or mirror-like metallic surface with strong reflections",
    ],
    "normal_map": [
        "a correct normal map with blue-purple surface relief details",
        "a high-quality normal map showing fine surface bumps and crevices",
        "a flat or blank normal map with uniform blue color and no surface detail",
        "a normal map with abnormal green or orange tint indicating an error",
        "a flipped or inverted normal map causing incorrect lighting",
        "a normal map with sharp and well-defined surface features",
    ],
    "roughness": [
        "a smooth polished surface with very low roughness and high gloss",
        "a rough matte surface with high roughness and no specular highlights",
        "a roughness map with realistic variation between smooth and rough areas",
        "a uniform mid-grey roughness map with no variation",
        "a roughness map showing natural wear patterns and surface texture",
        "a fabric or cloth surface with uniformly high roughness",
    ],
    "base_color": [
        "a clean PBR base color texture with accurate natural colors",
        "a base color texture with baked-in ambient occlusion shadows",
        "a base color texture containing text, logo, or graphic overlay",
        "a stylized or cartoon-like texture with flat unrealistic colors",
        "a photorealistic base color texture for a 3D model",
        "a base color with incorrectly oversaturated or wrong colors",
        "a white or blank base color map with missing texture information",
    ],
}

# CSV score column name for each channel (sampled_all.csv)
_CHANNEL_SCORE_COL: dict[str, str] = {
    "base_color": "baseColor",
    "normal_map": "normal",
    "roughness":  "roughness",
    "metallic":   "metallic",
}


def _model_path_to_name(model_path: str) -> str:
    """raw_data/sketchfab/8b/abc.glb → sketchfab__8b__abc"""
    return model_path.removeprefix("raw_data/").replace("/", "__").removesuffix(".glb")


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


def _compute_prompt_sims(
    channel_cls_path: Path, channel: str, prompts: list[str]
) -> torch.Tensor:
    """Compute image–text cosine similarities from pre-extracted CLS embeddings.

    Loads the CLIP text encoder once, encodes prompts, then does a single matmul.
    Total overhead: ~2s on first call (model load) + negligible matmul.
    Returns: Tensor (N, K) float32.
    """
    import open_clip
    import torch.nn.functional as F

    ch_data = torch.load(channel_cls_path, map_location="cpu", weights_only=False)
    img_embs = ch_data["features"][channel].float()             # (N, D)
    img_embs = F.normalize(img_embs, dim=-1)

    clip_model_name = ch_data.get("clip_model", "ViT-L-14")
    clip_pretrained = ch_data.get("clip_pretrained", "openai")
    print(f"  [prompts] encoding {len(prompts)} prompts for '{channel}' ...")
    model = open_clip.create_model(clip_model_name, pretrained=clip_pretrained).eval()
    tokenizer = open_clip.get_tokenizer(clip_model_name)
    with torch.no_grad():
        tokens = tokenizer(prompts)
        text_embs = F.normalize(model.encode_text(tokens).float(), dim=-1)  # (K, D)
    del model

    sims = img_embs @ text_embs.T   # (N, K)
    print(f"  [prompts] sims shape={sims.shape}  range=[{sims.min():.3f}, {sims.max():.3f}]")
    return sims


class TensorCacheCLIPDataset(Dataset):
    """Loads images from uint8 memmap + CLIP features + optional defect labels.

    Preferred (CSV mode): pass csv_path pointing to sampled_all.csv.
    Legacy mode: pass split_image_root + score_by_model + manifest_path.

    Args:
        tensor_cache_root : directory with meta.json and {channel}.npy files
        clip_feature_path : path to .pt file with CLIP features
        split             : "train" | "val" | "test"
        channel           : which channel (metallic / roughness / normal_map / base_color)
        csv_path          : sampled_all.csv — provides split, scores, defect labels
        channel_cls_path  : per-channel CLS .pt for prompt sim computation
        aux_channels      : additional channel images to load (e.g. ["roughness", "render"])
        split_image_root  : (legacy) screening/data_v2/{channel}
        score_by_model    : (legacy) dict model_name → ordinal score
        invalid_max_score : scores ≤ this are binary-label=1
        is_train          : apply training augmentations
        num_classes       : number of ordinal classes (default 6)
        manifest_path     : (legacy) CSV for defect labels
        defect_cols       : list of column names to use as defect auxiliary labels
    """

    def __init__(
        self,
        tensor_cache_root: Path,
        clip_feature_path: Path,
        split: str,
        channel: str,
        csv_path: Optional[Path] = None,
        channel_cls_path: Optional[Path] = None,
        aux_channels: Optional[list[str]] = None,
        split_image_root: Optional[Path] = None,
        score_by_model: Optional[dict[str, int]] = None,
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

        # Auxiliary channel setup — each aux channel gets its own lazy-loaded memmap
        self.aux_channels: list[str] = list(aux_channels) if aux_channels else []
        self._aux_array_paths: dict[str, Path] = {}
        self._aux_arrays: dict[str, Optional[np.ndarray]] = {}
        self._aux_transforms: dict[str, object] = {}
        for aux_ch in self.aux_channels:
            if aux_ch not in meta["channels"]:
                raise ValueError(f"Aux channel '{aux_ch}' not in tensor cache {meta['channels']}")
            self._aux_array_paths[aux_ch] = cache_root / f"{aux_ch}.npy"
            self._aux_arrays[aux_ch] = None
            self._aux_transforms[aux_ch] = _get_image_transform(is_train, aux_ch)

        clip_data = torch.load(clip_feature_path, map_location="cpu", weights_only=False)
        clip_names: list[str] = clip_data["model_names"]
        self._clip_idx: dict[str, int] = {n: i for i, n in enumerate(clip_names)}
        bc = clip_data["features"]["base_color"].float()
        rn = clip_data["features"]["render"].float()
        cls_tensor: torch.Tensor = torch.cat([bc, rn], dim=1)   # (N, 1536)

        # Optionally compute directional prompt similarities on-the-fly.
        # Requires pre-extracted per-channel CLS embeddings (channel_cls_path).
        # Prompts are defined in CHANNEL_PROMPTS — edit freely; no re-extraction needed.
        self._prompt_idx: dict[str, int] = {}
        self._prompt_tensor: Optional[torch.Tensor] = None
        prompts = CHANNEL_PROMPTS.get(channel)
        if channel_cls_path is not None and prompts:
            self._prompt_tensor = _compute_prompt_sims(
                Path(channel_cls_path), channel, prompts
            )
            # Build index from channel CLS file's model_names
            ch_data = torch.load(Path(channel_cls_path), map_location="cpu", weights_only=False)
            self._prompt_idx = {n: i for i, n in enumerate(ch_data["model_names"])}

        self._clip_tensor: torch.Tensor = cls_tensor
        self.clip_dim: int = cls_tensor.shape[1] + (
            self._prompt_tensor.shape[1] if self._prompt_tensor is not None else 0
        )

        self._defect_lookup: dict[str, list[float]] = {}
        self.samples: list[tuple[str, int]] = []

        if csv_path is not None:
            # ── CSV mode ──────────────────────────────────────────────────────
            score_col = _CHANNEL_SCORE_COL.get(channel)
            if score_col is None:
                raise ValueError(f"No score column defined for channel '{channel}'")
            with Path(csv_path).open(newline="", encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
            for row in rows:
                if row.get("split") != split:
                    continue
                name = _model_path_to_name(row["model"])
                try:
                    score = int(row[score_col])
                except (ValueError, KeyError):
                    continue
                if score < 0 or score >= num_classes:
                    continue
                if name not in self._cache_idx or name not in self._clip_idx:
                    continue
                self.samples.append((name, score))
                if self.defect_cols:
                    vals = []
                    for col in self.defect_cols:
                        try:
                            vals.append(float(row.get(col, 0) or 0))
                        except (ValueError, TypeError):
                            vals.append(0.0)
                    self._defect_lookup[name] = vals
        else:
            # ── legacy mode ───────────────────────────────────────────────────
            if manifest_path and self.defect_cols:
                self._defect_lookup = _load_manifest_defect_labels(
                    Path(manifest_path), self.defect_cols
                )
            split_root = Path(split_image_root) / split
            for label_dir in ("invalid", "valid"):
                d = split_root / label_dir
                if not d.exists():
                    continue
                for path in sorted(d.glob("*.png")):
                    name = path.stem
                    score = (score_by_model or {}).get(name)
                    if score is None or score < 0 or score >= num_classes:
                        continue
                    if name not in self._cache_idx or name not in self._clip_idx:
                        continue
                    self.samples.append((name, int(score)))

    def _get_array(self) -> np.ndarray:
        if self._array is None:
            self._array = np.load(self._array_path, mmap_mode="r")
        return self._array

    def _get_aux_array(self, ch: str) -> np.ndarray:
        if self._aux_arrays[ch] is None:
            self._aux_arrays[ch] = np.load(self._aux_array_paths[ch], mmap_mode="r")
        return self._aux_arrays[ch]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        name, score = self.samples[idx]
        img = torch.from_numpy(np.array(self._get_array()[self._cache_idx[name]], copy=True))
        img = self.transform(img)

        # Auxiliary channel images: [N_aux, 3, H, W]
        if self.aux_channels:
            aux_list = []
            ci = self._cache_idx[name]
            for aux_ch in self.aux_channels:
                aux_img = torch.from_numpy(np.array(self._get_aux_array(aux_ch)[ci], copy=True))
                aux_img = self._aux_transforms[aux_ch](aux_img)
                aux_list.append(aux_img)
            aux_imgs = torch.stack(aux_list, dim=0)
        else:
            aux_imgs = torch.zeros(0, img.shape[0], img.shape[1], img.shape[2])

        clip_feat = self._clip_tensor[self._clip_idx[name]]
        if self._prompt_tensor is not None and name in self._prompt_idx:
            clip_feat = torch.cat([clip_feat, self._prompt_tensor[self._prompt_idx[name]]], dim=0)
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
            aux_imgs,
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
