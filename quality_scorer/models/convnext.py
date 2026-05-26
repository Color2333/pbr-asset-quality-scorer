"""ConvNeXt multi-scale regression scorer with cross-modal CLIP attention + defect heads.

Architecture:
  1. ConvNeXt-Base backbone → stage2 / stage3 / stage4 features via attention pooling
  2. CrossModalFusion: stage features (queries) cross-attend to CLIP context (keys/values)
     — each scale asks "given what CLIP knows about this object, what should I focus on?"
  3. Fusion MLP over cat(multi-scale img feat, cross-attended feat)
  4. Heads:
       score_head   → continuous quality score [0, 5]
       binary_head  → valid/invalid logit (auxiliary)
       defect_head  → per-channel defect logits (auxiliary, optional)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ConvNeXt_Base_Weights, convnext_base

_STAGE2_DIM = 256
_STAGE3_DIM = 512
_STAGE4_DIM = 1024
_MULTISCALE_DIM = _STAGE2_DIM + _STAGE3_DIM + _STAGE4_DIM  # 1792
_CLIP_HALF_DIM = 768  # each of base_color and render


class AttentionPool2d(nn.Module):
    """Soft spatial attention pooling over H×W feature map."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.attn = nn.Conv2d(in_dim, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        w = F.softmax(self.attn(x).view(B, 1, -1), dim=2)  # [B, 1, H*W]
        return (x.view(B, C, -1) * w).sum(dim=2)            # [B, C]


class CrossModalFusion(nn.Module):
    """Multi-scale image features (sequence of 3) cross-attend to CLIP context (2 vectors).

    Each scale asks "given what CLIP knows about this object type,
    which of my spatial features matter most for quality?"
    """

    def __init__(
        self,
        stage_dims: tuple[int, ...] = (_STAGE2_DIM, _STAGE3_DIM, _STAGE4_DIM),
        clip_half_dim: int = _CLIP_HALF_DIM,
        proj_dim: int = 256,
        num_heads: int = 4,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.n_stages = len(stage_dims)

        # Project each stage feature to proj_dim → queries
        self.stage_projs = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d), nn.Linear(d, proj_dim))
            for d in stage_dims
        ])

        # Project each CLIP half (base_color / render) to proj_dim → keys & values
        self.clip_proj = nn.Sequential(
            nn.LayerNorm(clip_half_dim),
            nn.Linear(clip_half_dim, proj_dim),
        )

        self.cross_attn = nn.MultiheadAttention(
            proj_dim, num_heads, dropout=attn_dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(proj_dim)

    @property
    def out_dim(self) -> int:
        return self.n_stages * self.proj_dim

    def forward(self, stage_feats: list[torch.Tensor], clip_feat: torch.Tensor) -> torch.Tensor:
        """
        stage_feats: [s2 [B,256], s3 [B,512], s4 [B,1024]]
        clip_feat:   [B, 1536]  (cat of base_color + render CLIP)
        returns:     [B, 3 * proj_dim]
        """
        # Build query sequence: [B, 3, proj_dim]
        queries = torch.stack(
            [proj(f) for proj, f in zip(self.stage_projs, stage_feats)], dim=1
        )

        # Build key/value sequence from CLIP: [B, 2, proj_dim]
        bc = clip_feat[:, :_CLIP_HALF_DIM].float()
        rn = clip_feat[:, _CLIP_HALF_DIM:].float()
        kv = torch.stack([self.clip_proj(bc), self.clip_proj(rn)], dim=1)

        # Cross-attention with residual + layer norm
        attn_out, _ = self.cross_attn(queries, kv, kv)
        attended = self.norm(queries + attn_out)  # [B, 3, proj_dim]

        return attended.flatten(1)  # [B, 3 * proj_dim]


class ConvNeXtRegressionScorer(nn.Module):

    def __init__(
        self,
        clip_dim: int = 1536,
        attn_proj_dim: int = 256,
        attn_heads: int = 4,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        n_defect_labels: int = 0,
        freeze_features: bool = True,
        use_clip_direct: bool = True,
    ):
        super().__init__()
        bb = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
        self.features = bb.features

        # Attention pooling per stage
        self.pool2 = AttentionPool2d(_STAGE2_DIM)
        self.pool3 = AttentionPool2d(_STAGE3_DIM)
        self.pool4 = AttentionPool2d(_STAGE4_DIM)

        # Cross-modal fusion (image stages ↔ CLIP)
        self.cross_modal = CrossModalFusion(
            stage_dims=(_STAGE2_DIM, _STAGE3_DIM, _STAGE4_DIM),
            clip_half_dim=clip_dim // 2,
            proj_dim=attn_proj_dim,
            num_heads=attn_heads,
        )

        # Direct CLIP bypass: full 1536-dim → attn_proj_dim, no image gating.
        # For degenerate images (e.g. all-black metallic maps) cross-modal attention
        # collapses to a uniform average of its 2 CLIP tokens; this path gives the
        # model an unobstructed route to the full CLIP representation so it can still
        # use semantic context (object material type) as the primary signal.
        self.clip_direct: nn.Module | None
        if use_clip_direct:
            self.clip_direct = nn.Sequential(
                nn.LayerNorm(clip_dim),
                nn.Linear(clip_dim, attn_proj_dim),
                nn.GELU(),
            )
            clip_direct_dim = attn_proj_dim
        else:
            self.clip_direct = None
            clip_direct_dim = 0

        # Fusion MLP: cat(img_multiscale, cross_modal_out[, clip_direct]) → hidden
        fusion_in = _MULTISCALE_DIM + self.cross_modal.out_dim + clip_direct_dim  # 2816 (new) or 2560 (old)
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_in),
            nn.Linear(fusion_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Output heads
        self.score_head = nn.Linear(hidden_dim, 1)   # → sigmoid × 5
        self.binary_head = nn.Linear(hidden_dim, 1)  # → logit (valid/invalid)
        self.defect_head = (
            nn.Linear(hidden_dim, n_defect_labels) if n_defect_labels > 0 else None
        )

        if freeze_features:
            for p in self.features.parameters():
                p.requires_grad_(False)

    # ------------------------------------------------------------------
    def unfreeze_stage4(self) -> None:
        for i in (6, 7):
            for p in self.features[i].parameters():
                p.requires_grad_(True)

    def unfreeze_stage34(self) -> None:
        for i in (4, 5, 6, 7):
            for p in self.features[i].parameters():
                p.requires_grad_(True)

    def unfreeze_stage234(self) -> None:
        for i in (2, 3, 4, 5, 6, 7):
            for p in self.features[i].parameters():
                p.requires_grad_(True)

    # ------------------------------------------------------------------
    def extract_stages(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        for i in range(4):
            x = self.features[i](x)
        s2 = self.pool2(x)
        for i in range(4, 6):
            x = self.features[i](x)
        s3 = self.pool3(x)
        for i in range(6, 8):
            x = self.features[i](x)
        s4 = self.pool4(x)
        return s2, s3, s4

    def forward(
        self,
        image: torch.Tensor,
        clip_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Returns:
            score:         [B]  continuous quality score in [0, 5]
            binary_logit:  [B]  logit for valid/invalid
            defect_logits: [B, n_defect] or None
        """
        s2, s3, s4 = self.extract_stages(image)
        img_feat   = torch.cat([s2, s3, s4], dim=1)           # [B, 1792]
        cross_feat = self.cross_modal([s2, s3, s4], clip_feat) # [B, 768]
        parts = [img_feat, cross_feat]
        if self.clip_direct is not None:
            parts.append(self.clip_direct(clip_feat.float()))  # [B, 256]
        fused = self.fusion(torch.cat(parts, dim=1))           # [B, 512]

        score = torch.sigmoid(self.score_head(fused)).squeeze(1) * 5.0    # [B]
        binary = self.binary_head(fused).squeeze(1)                        # [B]
        defect = self.defect_head(fused) if self.defect_head is not None else None

        return score, binary, defect
