from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import (
    ConvNeXt_Base_Weights,
    ConvNeXt_Large_Weights,
    convnext_base,
    convnext_large,
)

from quality_scorer.ordinal import MonotonicCoralHead


_STAGE_DIMS = {
    "base": (512, 1024),
    "large": (768, 1536),
}


def _get_backbone(backbone: str):
    if backbone == "base":
        return convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT), _STAGE_DIMS[backbone]
    if backbone == "large":
        return convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT), _STAGE_DIMS[backbone]
    raise ValueError(f"Unsupported backbone: {backbone!r}")


class ConvNeXtOrdinalScorer(nn.Module):
    """ConvNeXt stage2+stage3 feature extractor with an ordinal quality head."""

    def __init__(self, backbone: str = "base", num_classes: int = 6, freeze_features: bool = True):
        super().__init__()
        bb, (stage2_dim, stage3_dim) = _get_backbone(backbone)
        self.backbone = backbone
        self.features = bb.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = stage2_dim + stage3_dim
        self.head = MonotonicCoralHead(self.feat_dim, num_classes=num_classes)
        if freeze_features:
            for param in self.features.parameters():
                param.requires_grad_(False)

    def unfreeze_stage3(self) -> None:
        for param in self.features[7].parameters():
            param.requires_grad_(True)

    def unfreeze_stage23(self) -> None:
        for idx in (5, 6, 7):
            for param in self.features[idx].parameters():
                param.requires_grad_(True)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(6):
            x = self.features[i](x)
        feat2 = self.pool(x).flatten(1)
        for i in range(6, 8):
            x = self.features[i](x)
        feat3 = self.pool(x).flatten(1)
        return torch.cat([feat2, feat3], dim=1)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        embedding = self.extract_features(x)
        logits = self.head(embedding)
        if return_features:
            return logits, embedding
        return logits

