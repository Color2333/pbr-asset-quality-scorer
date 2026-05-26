from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_SCREENING_ROOT = PROJECT_ROOT / "screening"

DEFAULT_IMAGE_DATA_ROOT = LEGACY_SCREENING_ROOT / "data_v2"
DEFAULT_LABELS_ROOT = LEGACY_SCREENING_ROOT / "data_38k"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "asset_quality_scorer" / "outputs"

ALL_CHANNELS = ("normal_map", "roughness", "metallic", "base_color")

SCORE_KEYS = {
    "base_color": "base_color_score",
    "normal_map": "normal_score",
    "roughness": "roughness_score",
    "metallic": "metallic_score",
}

