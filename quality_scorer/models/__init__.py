"""Model registry — add new architectures here, not new files.

Usage:
    from quality_scorer.models import build_model
    model = build_model("convnext_base", clip_dim=1536, ...)
"""
from __future__ import annotations

from quality_scorer.models.convnext import ConvNeXtRegressionScorer
from quality_scorer.models.convnext_ordinal import ConvNeXtOrdinalScorer
from quality_scorer.models.dinov2 import DINOv2RegressionScorer

_REGISTRY: dict[str, type] = {
    "convnext_base":         ConvNeXtRegressionScorer,
    "convnext_base_ordinal": ConvNeXtOrdinalScorer,
    "dinov2_large":          DINOv2RegressionScorer,
}


def build_model(arch: str, **kwargs):
    if arch not in _REGISTRY:
        raise ValueError(f"Unknown arch '{arch}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[arch](**kwargs)


__all__ = [
    "build_model",
    "ConvNeXtRegressionScorer",
    "ConvNeXtOrdinalScorer",
    "DINOv2RegressionScorer",
]
