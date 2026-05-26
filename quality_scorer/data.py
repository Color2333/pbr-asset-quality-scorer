from __future__ import annotations

from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset

from quality_scorer.constants import SCORE_KEYS
from screening.labels import load_csv


class PBRScoreDataset(Dataset):
    """Prepared channel images paired with raw ordinal channel scores.

    Supports two image backends:
      - "path" (default): reads PNG files from image_root/split/invalid|valid/
      - "lmdb": reads from a pre-built LMDB store (key = "model_name/channel").
                Requires lmdb_path and channel to be provided.

    LMDB backend is faster for training because it avoids individual file seeks.
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
        self.samples: list[tuple[str, int]] = []  # (model_name_or_path, score)
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
                # Store model_name for LMDB lookup, or full path for file I/O
                key = path.stem if image_backend == "lmdb" else str(path)
                self.samples.append((key, int(score)))

    def _env(self):
        if self._lmdb_env is None:
            import lmdb
            self._lmdb_env = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                readahead=False,
                max_readers=128,
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
        """Per-sample weights for WeightedRandomSampler.

        Intermediate scores (1 .. K-2) are upweighted by mid_factor to
        counteract class imbalance and prevent extreme-class collapse.
        """
        lo, hi = 1, self.num_classes - 2
        return [mid_factor if lo <= score <= hi else 1.0 for _, score in self.samples]


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
