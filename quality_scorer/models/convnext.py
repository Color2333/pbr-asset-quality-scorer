"""ConvNeXt multi-scale regression scorer with cross-modal CLIP attention + defect heads.

Architecture:
  1. ConvNeXt-Base backbone → stage2 / stage3 / stage4 features via attention pooling
  2. CrossModalFusion: stage features (queries) cross-attend to CLIP context (keys/values)
     — each scale asks "given what CLIP knows about this object, what should I focus on?"
  3. [Optional] AuxFusion: main stage4 cross-attends to N auxiliary channel stage4 features
     — auxiliary channels share the same backbone (frozen for aux, no gradient)
  4. Fusion MLP over cat(multi-scale img feat, cross-attended feat[, aux feat])
  5. Heads:
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
        rn = clip_feat[:, _CLIP_HALF_DIM:2 * _CLIP_HALF_DIM].float()
        kv = torch.stack([self.clip_proj(bc), self.clip_proj(rn)], dim=1)

        # Cross-attention with residual + layer norm
        attn_out, _ = self.cross_attn(queries, kv, kv)
        attended = self.norm(queries + attn_out)  # [B, 3, proj_dim]

        return attended.flatten(1)  # [B, 3 * proj_dim]


class AuxFusion(nn.Module):
    """Aggregate N auxiliary channel features into a single context vector.

    Cross-attention: main stage4 pooled feature as query,
    N auxiliary stage4 pooled features as keys/values.
    Handles N=0 gracefully (returns projected main feature).
    """

    def __init__(self, stage4_dim: int = 1024, proj_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.proj_dim = proj_dim
        self.main_proj = nn.Sequential(nn.LayerNorm(stage4_dim), nn.Linear(stage4_dim, proj_dim))
        self.aux_proj  = nn.Sequential(nn.LayerNorm(stage4_dim), nn.Linear(stage4_dim, proj_dim))
        self.cross_attn = nn.MultiheadAttention(proj_dim, num_heads, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(proj_dim)

    def forward(self, main_s4: torch.Tensor, aux_feats: list[torch.Tensor]) -> torch.Tensor:
        """
        main_s4  : [B, 1024]
        aux_feats: list of N [B, 1024]
        returns  : [B, proj_dim]
        """
        q = self.main_proj(main_s4).unsqueeze(1)          # [B, 1, proj_dim]
        if not aux_feats:
            return q.squeeze(1)
        kv = torch.stack([self.aux_proj(f) for f in aux_feats], dim=1)  # [B, N, proj_dim]
        out, _ = self.cross_attn(q, kv, kv)
        return self.norm(q + out).squeeze(1)               # [B, proj_dim]


class SpatialAuxFusion(nn.Module):
    """Patch-level cross-channel consistency fusion.

    The main channel's stage4 feature MAP (queries) cross-attends to the
    auxiliary channels' stage4 feature MAPS (keys/values), keeping spatial
    correspondence intact. Where the pooled AuxFusion can only ask "does the
    object overall look metallic?", this asks per region "does *this patch*
    of the reference channel (e.g. render) agree with *this patch* of the
    main channel?" — the core signal for judging whether a metallic map is
    correct given how the object actually renders.

    Handles N=0 gracefully (returns pooled projected main map). Shares a
    learnable spatial positional embedding between main and aux tokens so the
    attention can align corresponding patches; sliced to the actual token
    count, so it tolerates variable input resolution (7×7 @224, 14×14 @448…).
    """

    def __init__(
        self,
        stage4_dim: int = _STAGE4_DIM,
        proj_dim: int = 256,
        num_heads: int = 4,
        max_tokens: int = 256,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.main_proj = nn.Sequential(nn.LayerNorm(stage4_dim), nn.Linear(stage4_dim, proj_dim))
        self.aux_proj  = nn.Sequential(nn.LayerNorm(stage4_dim), nn.Linear(stage4_dim, proj_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, proj_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.cross_attn = nn.MultiheadAttention(
            proj_dim, num_heads, dropout=attn_dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(proj_dim)

    @staticmethod
    def _tokens(feat_map: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] → [B, H*W, C]
        return feat_map.flatten(2).transpose(1, 2)

    def forward(self, main_map: torch.Tensor, aux_maps: list[torch.Tensor]) -> torch.Tensor:
        """
        main_map: [B, C, H, W]
        aux_maps: list of N [B, C, H, W] (same spatial size as main_map)
        returns : [B, proj_dim]
        """
        q = self.main_proj(self._tokens(main_map))      # [B, N_tok, proj]
        n_tok = q.shape[1]
        q = q + self.pos_embed[:, :n_tok]
        if not aux_maps:
            return self.norm(q).mean(dim=1)
        kv = torch.cat(
            [self.aux_proj(self._tokens(m)) + self.pos_embed[:, :n_tok] for m in aux_maps],
            dim=1,
        )                                                # [B, N_aux*N_tok, proj]
        attn_out, _ = self.cross_attn(q, kv, kv)         # [B, N_tok, proj]
        fused = self.norm(q + attn_out)                  # [B, N_tok, proj]
        return fused.mean(dim=1)                          # [B, proj]


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
        use_aux: bool = False,
        aux_proj_dim: int = 256,
        aux_fusion_mode: str = "pooled",
        aux_trainable: bool = False,
    ):
        super().__init__()
        bb = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
        self.features = bb.features

        # Attention pooling per stage
        self.pool2 = AttentionPool2d(_STAGE2_DIM)
        self.pool3 = AttentionPool2d(_STAGE3_DIM)
        self.pool4 = AttentionPool2d(_STAGE4_DIM)

        # Cross-modal fusion (image stages ↔ CLIP CLS only, never prompt sims)
        self.cross_modal = CrossModalFusion(
            stage_dims=(_STAGE2_DIM, _STAGE3_DIM, _STAGE4_DIM),
            clip_half_dim=_CLIP_HALF_DIM,   # always 768 — base_color and render CLS
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

        # Auxiliary channel fusion: shared backbone extracts stage4 for each aux channel,
        # then cross-attend main stage4 to all aux features.
        # Aux backbone runs with torch.no_grad() — no gradient through aux path.
        self.aux_fusion_mode = aux_fusion_mode
        self.aux_trainable   = aux_trainable
        if use_aux:
            if aux_fusion_mode == "spatial":
                # Patch-level consistency: no aux pooling — keep the full map.
                self.aux_pool4  = None
                self.aux_fusion = SpatialAuxFusion(_STAGE4_DIM, aux_proj_dim, attn_heads)
            else:
                self.aux_pool4  = AttentionPool2d(_STAGE4_DIM)
                self.aux_fusion = AuxFusion(_STAGE4_DIM, aux_proj_dim, attn_heads)
            aux_dim = aux_proj_dim
        else:
            self.aux_pool4  = None
            self.aux_fusion = None
            aux_dim = 0

        # Fusion MLP: cat(img_multiscale, cross_modal_out[, clip_direct][, aux_fusion]) → hidden
        fusion_in = _MULTISCALE_DIM + self.cross_modal.out_dim + clip_direct_dim + aux_dim
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
    def extract_stages(
        self, x: torch.Tensor, return_s4_map: bool = False
    ) -> tuple[torch.Tensor, ...]:
        for i in range(4):
            x = self.features[i](x)
        s2 = self.pool2(x)
        for i in range(4, 6):
            x = self.features[i](x)
        s3 = self.pool3(x)
        for i in range(6, 8):
            x = self.features[i](x)
        s4_map = x
        s4 = self.pool4(x)
        if return_s4_map:
            return s2, s3, s4, s4_map
        return s2, s3, s4

    def _extract_stage4_map(self, x: torch.Tensor) -> torch.Tensor:
        """Run backbone up to stage4, return raw feature map [B, 1024, H, W]."""
        for i in range(8):
            x = self.features[i](x)
        return x

    def forward(
        self,
        image: torch.Tensor,
        clip_feat: torch.Tensor,
        aux_images: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Args:
            image     : [B, 3, H, W]  main channel image
            clip_feat : [B, D]         CLIP CLS features
            aux_images: [B, N, 3, H, W] auxiliary channel images (optional)
        Returns:
            score:         [B]  continuous quality score in [0, 5]
            binary_logit:  [B]  logit for valid/invalid
            defect_logits: [B, n_defect] or None
        """
        spatial_aux = self.aux_fusion is not None and self.aux_fusion_mode == "spatial"
        if spatial_aux:
            s2, s3, s4, s4_map = self.extract_stages(image, return_s4_map=True)
        else:
            s2, s3, s4 = self.extract_stages(image)
        img_feat   = torch.cat([s2, s3, s4], dim=1)           # [B, 1792]
        cross_feat = self.cross_modal([s2, s3, s4], clip_feat) # [B, 768]
        parts = [img_feat, cross_feat]
        if self.clip_direct is not None:
            parts.append(self.clip_direct(clip_feat.float()))  # [B, 256]

        if self.aux_fusion is not None:
            has_aux = aux_images is not None and aux_images.shape[1] > 0
            # Extract aux stage4 maps. When aux_trainable, gradients flow through
            # the shared backbone so it learns cross-channel consistency features;
            # otherwise aux is a frozen reference (cheaper, original behaviour).
            aux_maps: list[torch.Tensor] = []
            if has_aux:
                if self.aux_trainable:
                    for i in range(aux_images.shape[1]):
                        aux_maps.append(self._extract_stage4_map(aux_images[:, i]))
                else:
                    with torch.no_grad():
                        for i in range(aux_images.shape[1]):
                            aux_maps.append(self._extract_stage4_map(aux_images[:, i]))

            if spatial_aux:
                # empty list → SpatialAuxFusion pools main map, preserving dim
                parts.append(self.aux_fusion(s4_map, aux_maps))           # [B, aux_proj_dim]
            else:
                aux_s4_feats = [self.aux_pool4(m) for m in aux_maps]      # [B, 1024] each
                parts.append(self.aux_fusion(s4, aux_s4_feats))           # [B, aux_proj_dim]

        fused = self.fusion(torch.cat(parts, dim=1))               # [B, hidden_dim]

        score = torch.sigmoid(self.score_head(fused)).squeeze(1) * 5.0    # [B]
        binary = self.binary_head(fused).squeeze(1)                        # [B]
        defect = self.defect_head(fused) if self.defect_head is not None else None

        return score, binary, defect
