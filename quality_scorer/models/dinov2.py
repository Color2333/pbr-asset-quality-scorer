"""DINOv2 (ViT-L/14 reg4) regression scorer — drop-in alternative to ConvNeXt.

Mirrors ConvNeXtRegressionScorer's __init__ / forward / unfreeze interface so
train.py, eval.py and metrics.py work unchanged via the model registry.

Architecture:
  1. DINOv2 ViT-L/14 backbone (timm, dynamic_img_size) — strong dense features.
     ViT is isotropic (no conv stages), so we tap 3 transformer depths
     (layers 8/16/24) as a pseudo-multi-scale, mean-pooling patch tokens at each.
  2. CrossModalFusion (reused): the 3 pooled-depth vectors cross-attend to CLIP.
  3. [Optional] clip_direct bypass + AuxFusion / SpatialAuxFusion (破局点1).
  4. Fusion MLP → score / binary / defect heads.

Resolution-agnostic: dynamic_img_size interpolates positional embeddings, so the
same weights run at 224 (16×16 patches) or 448 (32×32 patches).
"""
from __future__ import annotations

import math

import timm
import torch
import torch.nn as nn

from quality_scorer.models.convnext import (
    AuxFusion,
    CrossModalFusion,
    SpatialAuxFusion,
    _CLIP_HALF_DIM,
)

_DINOV2_MODEL  = "vit_large_patch14_reg4_dinov2"
_DINOV2_DIM    = 1024
_TAP_LAYERS    = (7, 15, 23)   # 0-indexed blocks → pseudo s2/s3/s4 (ViT-L has 24)
_MT_CHANNELS   = ("base_color", "normal_map", "roughness", "metallic")
_CLIP_RENDER_DIM = 768  # first half of the 1536-d CLIP feat = render CLS

# ─────────────────────────────────────────────────────────────────────────────
# Cross-Channel Interaction Block
# ─────────────────────────────────────────────────────────────────────────────

class CrossChannelBlock(nn.Module):
    """4-token transformer block for cross-channel quality reasoning.

    Each PBR channel's fused representation becomes one token. A single round
    of multi-head self-attention lets all channels reason about each other
    before their individual score heads — encoding the annotator's logic:

        "metallic quality depends on what base_color (material appearance) and
         roughness (surface smoothness) say about the same region."

    Design choices:
      • Manual MHA: needed so we can inject the SOP-motivated additive bias
        directly into the pre-softmax attention logits.
      • Learnable SOP bias: initialized with physical prior (metallic attends
        more to base_color and roughness), but freely adapted during training.
        If the prior is wrong the model learns to ignore it (→ 0); if it is
        right the model amplifies it. Never hurts, potentially helps a lot.
      • Channel identity embedding: each token carries a learned tag so the
        model knows "this 512-d vector represents the roughness channel". Without
        this all tokens look identical to the attention — the model can't tell
        which channel it's reading from.
      • Residual + LayerNorm: standard transformer stability. Cross-channel
        signal is add-on; if useless, gradients drive it to zero.
      • Small FFN: allows non-linear reasoning after attention (e.g. "metallic
        is white AND roughness is low → this is consistent → boost score").
      • Graceful degradation: missing channels are filled with zero vectors +
        channel embedding, so single-channel inference still works.
    """

    CHANNELS = _MT_CHANNELS  # fixed ordering: bc, nm, ro, me
    _CH_IDX  = {ch: i for i, ch in enumerate(_MT_CHANNELS)}

    def __init__(
        self,
        dim: int = 512,
        n_heads: int = 4,
        dropout: float = 0.1,
        me_bc_init_bias: float = 1.0,
        me_ro_init_bias: float = 1.0,
        # If True, mask out bc/nm/ro → metallic attention (prevent metallic's
        # noisy representation from polluting the cleaner channels).
        # The asymmetry implements: metallic CAN read others, others CANNOT
        # read metallic — matching the intended data flow from the SOP.
        asymmetric_mask: bool = True,
    ):
        super().__init__()
        assert dim % n_heads == 0
        self.dim      = dim
        self.n_heads  = n_heads
        self.head_dim = dim // n_heads

        self.channel_embed = nn.Embedding(len(self.CHANNELS), dim)

        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)

        # SOP-motivated additive attention bias [n_heads, n_ch, n_ch].
        n = len(self.CHANNELS)
        init_bias = torch.zeros(n_heads, n, n)
        me = self._CH_IDX["metallic"]
        bc = self._CH_IDX["base_color"]
        ro = self._CH_IDX["roughness"]
        init_bias[:, me, bc] = me_bc_init_bias
        init_bias[:, me, ro] = me_ro_init_bias
        self.attn_bias = nn.Parameter(init_bias)

        # Asymmetric mask: block bc/nm/ro from attending to metallic.
        # Registered as a buffer (not a parameter) — never trained, never saved
        # to checkpoint in a way that causes shape mismatches.
        if asymmetric_mask:
            # -inf in positions [q, me] for q != me cuts off those paths entirely
            mask = torch.zeros(n, n)
            for q in range(n):
                if q != me:
                    mask[q, me] = float('-inf')   # bc/nm/ro cannot read metallic
            self.register_buffer('hard_mask', mask)   # [n, n]
        else:
            self.register_buffer('hard_mask', torch.zeros(n, n))

        self.norm1 = nn.LayerNorm(dim)
        self.ffn   = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

        self._init_weights()

    def _init_weights(self):
        # Small init for stability: cross-channel starts near identity
        for m in [self.Wq, self.Wk, self.Wv, self.out_proj]:
            nn.init.xavier_uniform_(m.weight, gain=0.5)
        nn.init.zeros_(self.out_proj.weight)   # out_proj → 0 at init
        # channel_embed: normal init so channels are distinguishable from day 1
        nn.init.normal_(self.channel_embed.weight, std=0.02)

    def forward(
        self,
        channel_feats: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            channel_feats: {ch_name: [B, dim]} — may be a subset of CHANNELS.
                           Missing channels are padded with zeros.
        Returns:
            refined: {ch_name: [B, dim]} for every key in channel_feats.
        """
        B = next(iter(channel_feats.values())).shape[0]
        device = next(iter(channel_feats.values())).device
        n = len(self.CHANNELS)

        # Build token sequence [B, n_ch, dim] — zero for absent channels
        tokens = torch.zeros(B, n, self.dim, device=device,
                             dtype=next(iter(channel_feats.values())).dtype)
        present = []
        for ch, feat in channel_feats.items():
            i = self._CH_IDX[ch]
            tokens[:, i] = feat
            present.append(i)

        # Add channel identity embeddings (all positions, including absent ones)
        ids = torch.arange(n, device=device)
        tokens = tokens + self.channel_embed(ids).unsqueeze(0)  # [B, n, dim]

        # ── Multi-head self-attention with SOP bias ────────────────────────
        def split_heads(x):
            # x: [B, n, dim] → [B, n_heads, n, head_dim]
            return x.view(B, n, self.n_heads, self.head_dim).transpose(1, 2)

        Q = split_heads(self.Wq(tokens))   # [B, n_heads, n, head_dim]
        K = split_heads(self.Wk(tokens))
        V = split_heads(self.Wv(tokens))

        scale = self.head_dim ** -0.5
        # Raw attention logits [B, n_heads, n, n]
        attn_logits = (Q @ K.transpose(-2, -1)) * scale
        # Add SOP prior bias (broadcast over batch)
        attn_logits = attn_logits + self.attn_bias.unsqueeze(0)
        # Hard asymmetric mask: sets bc/nm/ro→metallic to -inf before softmax
        # so those paths contribute exactly 0 — metallic's noise cannot
        # flow back into the cleaner channels' representations.
        attn_logits = attn_logits + self.hard_mask.unsqueeze(0).unsqueeze(0)

        attn_w = torch.softmax(attn_logits, dim=-1)
        attn_w = self.attn_drop(attn_w)
        # [B, n_heads, n, head_dim] → [B, n, dim]
        attn_out = (attn_w @ V).transpose(1, 2).reshape(B, n, self.dim)
        attn_out = self.out_proj(attn_out)

        # Residual 1: add cross-channel signal to original tokens
        tokens = self.norm1(tokens + attn_out)

        # FFN + Residual 2
        tokens = self.norm2(tokens + self.ffn(tokens))

        # Return only the channels that were passed in
        return {ch: tokens[:, self._CH_IDX[ch]] for ch in channel_feats}

    def get_attn_bias_summary(self) -> dict:
        """Diagnostic: return mean attention bias per channel pair (for logging)."""
        bias = self.attn_bias.mean(0).detach().cpu()  # [n_ch, n_ch]
        return {
            f"{self.CHANNELS[q]}→{self.CHANNELS[k]}": round(float(bias[q, k]), 3)
            for q in range(len(self.CHANNELS))
            for k in range(len(self.CHANNELS))
            if q != k and abs(float(bias[q, k])) > 0.05
        }


class DINOv2RegressionScorer(nn.Module):

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
        backbone_name: str = _DINOV2_MODEL,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, num_classes=0, dynamic_img_size=True
        )
        self.embed_dim = self.backbone.embed_dim
        self.tap_layers = _TAP_LAYERS
        D = self.embed_dim
        n_scales = len(self.tap_layers)

        # Cross-modal fusion: 3 equal-dim ViT depths ↔ CLIP CLS (base_color + render)
        self.cross_modal = CrossModalFusion(
            stage_dims=(D,) * n_scales,
            clip_half_dim=_CLIP_HALF_DIM,
            proj_dim=attn_proj_dim,
            num_heads=attn_heads,
        )

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

        self.aux_fusion_mode = aux_fusion_mode
        self.aux_trainable   = aux_trainable
        if use_aux:
            if aux_fusion_mode == "spatial":
                self.aux_fusion = SpatialAuxFusion(D, aux_proj_dim, attn_heads)
            else:
                self.aux_fusion = AuxFusion(D, aux_proj_dim, attn_heads)
            aux_dim = aux_proj_dim
        else:
            self.aux_fusion = None
            aux_dim = 0

        fusion_in = n_scales * D + self.cross_modal.out_dim + clip_direct_dim + aux_dim
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_in),
            nn.Linear(fusion_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.score_head  = nn.Linear(hidden_dim, 1)
        self.binary_head = nn.Linear(hidden_dim, 1)
        self.defect_head = (
            nn.Linear(hidden_dim, n_defect_labels) if n_defect_labels > 0 else None
        )

        if freeze_features:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Progressive unfreeze — map ConvNeXt's stage4/34/234 onto ViT block ranges.
    def _unfreeze_blocks(self, start: int) -> None:
        for blk in self.backbone.blocks[start:]:
            for p in blk.parameters():
                p.requires_grad_(True)
        if hasattr(self.backbone, "norm"):
            for p in self.backbone.norm.parameters():
                p.requires_grad_(True)

    def unfreeze_stage4(self) -> None:
        self._unfreeze_blocks(18)   # last 6 of 24

    def unfreeze_stage34(self) -> None:
        self._unfreeze_blocks(12)   # last 12

    def unfreeze_stage234(self) -> None:
        self._unfreeze_blocks(0)    # all blocks

    # ------------------------------------------------------------------
    def _multiscale_tokens(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return patch-token sequences [B, N, D] at the tapped depths."""
        return list(
            self.backbone.get_intermediate_layers(
                x, n=self.tap_layers, return_prefix_tokens=False, norm=True
            )
        )

    @staticmethod
    def _tokens_to_map(tokens: torch.Tensor) -> torch.Tensor:
        """[B, N, D] → [B, D, h, w] (N must be a perfect square)."""
        B, N, D = tokens.shape
        h = int(round(math.sqrt(N)))
        return tokens.transpose(1, 2).reshape(B, D, h, h)

    def _aux_patch_map(self, x: torch.Tensor) -> torch.Tensor:
        """Last-layer patch tokens of an aux image, as a [B, D, h, w] map."""
        tokens = self.backbone.get_intermediate_layers(
            x, n=[self.tap_layers[-1]], return_prefix_tokens=False, norm=True
        )[0]
        return self._tokens_to_map(tokens)

    def forward(
        self,
        image: torch.Tensor,
        clip_feat: torch.Tensor,
        aux_images: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        scale_tokens = self._multiscale_tokens(image)         # 3× [B, N, D]
        pooled = [t.mean(dim=1) for t in scale_tokens]        # 3× [B, D]
        img_feat = torch.cat(pooled, dim=1)                   # [B, 3D]
        cross_feat = self.cross_modal(pooled, clip_feat)
        parts = [img_feat, cross_feat]
        if self.clip_direct is not None:
            parts.append(self.clip_direct(clip_feat.float()))

        if self.aux_fusion is not None:
            has_aux = aux_images is not None and aux_images.shape[1] > 0
            if self.aux_fusion_mode == "spatial":
                main_map = self._tokens_to_map(scale_tokens[-1])
                aux_maps: list[torch.Tensor] = []
                if has_aux:
                    if self.aux_trainable:
                        aux_maps = [self._aux_patch_map(aux_images[:, i])
                                    for i in range(aux_images.shape[1])]
                    else:
                        with torch.no_grad():
                            aux_maps = [self._aux_patch_map(aux_images[:, i])
                                        for i in range(aux_images.shape[1])]
                parts.append(self.aux_fusion(main_map, aux_maps))
            else:
                aux_feats: list[torch.Tensor] = []
                if has_aux:
                    if self.aux_trainable:
                        aux_feats = [self._aux_patch_map(aux_images[:, i]).flatten(2).mean(-1)
                                     for i in range(aux_images.shape[1])]
                    else:
                        with torch.no_grad():
                            aux_feats = [self._aux_patch_map(aux_images[:, i]).flatten(2).mean(-1)
                                         for i in range(aux_images.shape[1])]
                parts.append(self.aux_fusion(pooled[-1], aux_feats))

        fused = self.fusion(torch.cat(parts, dim=1))
        score  = torch.sigmoid(self.score_head(fused)).squeeze(1) * 5.0
        binary = self.binary_head(fused).squeeze(1)
        defect = self.defect_head(fused) if self.defect_head is not None else None
        return score, binary, defect


# ─────────────────────────────────────────────────────────────────────────────
# Multi-task scorer: one shared DINOv2 backbone, 4 channel heads, FiLM for
# metallic.  Training: forward receives all 4 channel images simultaneously.
# Inference: same forward, subset of channels works too.
#
# Design rationale:
#   • Weight-sharing: 4 channel images run through the SAME backbone. Strong
#     channels (roughness 0.895, base_color 0.841) inject quality-relevant
#     gradients that regularise the backbone — "strong channels carry weak".
#   • FiLM on metallic: CLIP render CLS (768-d) → (γ, β) re-scales the shared
#     fused representation just before the metallic head, injecting the
#     "render-based material expectation" as a lightweight per-sample
#     conditioning signal.  γ⊙h+β costs ~2×768×512 ≈ 786k params.
#   • Per-channel binary head: valid/invalid decision per channel (not shared).
#   • Gradient scaling: metallic's noisy labels contribute less to backbone
#     updates via a configurable metallic_grad_scale < 1.
# ─────────────────────────────────────────────────────────────────────────────

class DINOv2MultiTaskScorer(nn.Module):
    """DINOv2 backbone shared across 4 PBR channels with FiLM for metallic."""

    CHANNELS = _MT_CHANNELS

    def __init__(
        self,
        clip_dim: int = 1536,
        attn_proj_dim: int = 256,
        attn_heads: int = 4,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        freeze_features: bool = True,
        use_clip_direct: bool = True,
        metallic_film: bool = True,          # FiLM conditioning on metallic head
        metallic_grad_scale: float = 0.5,   # down-weight metallic → backbone grad
        use_cross_channel: bool = False,     # CrossChannelBlock after fusion MLP
        cc_n_heads: int = 4,
        cc_me_bc_bias: float = 1.0,         # SOP prior: metallic→base_color
        cc_me_ro_bias: float = 1.0,         # SOP prior: metallic→roughness
        backbone_name: str = _DINOV2_MODEL,
    ):
        super().__init__()
        self.metallic_grad_scale = metallic_grad_scale

        # ── shared backbone ────────────────────────────────────────────────
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, num_classes=0, dynamic_img_size=True
        )
        D = self.backbone.embed_dim      # 1024 for ViT-L
        n_scales = len(_TAP_LAYERS)

        # ── shared cross-modal fusion ──────────────────────────────────────
        self.cross_modal = CrossModalFusion(
            stage_dims=(D,) * n_scales,
            clip_half_dim=_CLIP_HALF_DIM,
            proj_dim=attn_proj_dim,
            num_heads=attn_heads,
        )
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

        # ── shared fusion MLP (same weights for all channels) ─────────────
        fusion_in = n_scales * D + self.cross_modal.out_dim + clip_direct_dim
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_in),
            nn.Linear(fusion_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── FiLM generator for metallic ────────────────────────────────────
        # Input: render CLS token (first 768-d of clip_feat).
        # Output: (γ, β) each of size hidden_dim — applied as affine re-scale
        # of the shared fused representation before the metallic head.
        self.film_gen: nn.Module | None
        if metallic_film:
            self.film_gen = nn.Sequential(
                nn.LayerNorm(_CLIP_RENDER_DIM),
                nn.Linear(_CLIP_RENDER_DIM, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
            )
        else:
            self.film_gen = None

        # ── cross-channel interaction (optional) ──────────────────────────
        self.cross_channel: nn.Module | None
        if use_cross_channel:
            self.cross_channel = CrossChannelBlock(
                dim=hidden_dim, n_heads=cc_n_heads,
                me_bc_init_bias=cc_me_bc_bias,
                me_ro_init_bias=cc_me_ro_bias,
            )
        else:
            self.cross_channel = None

        # ── per-channel heads (score + binary, no defect in multi-task) ───
        self.score_heads  = nn.ModuleDict({ch: nn.Linear(hidden_dim, 1) for ch in self.CHANNELS})
        self.binary_heads = nn.ModuleDict({ch: nn.Linear(hidden_dim, 1) for ch in self.CHANNELS})

        if freeze_features:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    # ── progressive unfreeze (same interface as DINOv2RegressionScorer) ───
    def _unfreeze_blocks(self, start: int) -> None:
        for blk in self.backbone.blocks[start:]:
            for p in blk.parameters():
                p.requires_grad_(True)
        if hasattr(self.backbone, "norm"):
            for p in self.backbone.norm.parameters():
                p.requires_grad_(True)

    def unfreeze_stage4(self)   -> None: self._unfreeze_blocks(18)
    def unfreeze_stage34(self)  -> None: self._unfreeze_blocks(12)
    def unfreeze_stage234(self) -> None: self._unfreeze_blocks(0)

    # ── helpers ────────────────────────────────────────────────────────────
    def _encode(self, img: torch.Tensor) -> torch.Tensor:
        """img [B,3,H,W] → fused representation [B, hidden_dim]."""
        tokens = list(self.backbone.get_intermediate_layers(
            img, n=_TAP_LAYERS, return_prefix_tokens=False, norm=True
        ))
        pooled = [t.mean(dim=1) for t in tokens]          # 3× [B, D]
        img_feat  = torch.cat(pooled, dim=1)              # [B, 3D]
        # NOTE: cross_modal and clip_direct need clip_feat — passed in forward.
        # _encode only extracts image features; fusion happens in forward.
        return img_feat, pooled

    def _film(self, fused: torch.Tensor, clip_feat: torch.Tensor) -> torch.Tensor:
        """FiLM: render CLS (first 768-d of clip_feat) → affine transform on fused."""
        if self.film_gen is None:
            return fused
        render_cls = clip_feat[:, :_CLIP_RENDER_DIM].float()
        gamma_beta = self.film_gen(render_cls)             # [B, 2*hidden]
        gamma, beta = gamma_beta.chunk(2, dim=-1)          # each [B, hidden]
        return (1.0 + gamma) * fused + beta                # residual form → stable init

    def forward(
        self,
        channel_imgs: dict[str, torch.Tensor],
        clip_feat: torch.Tensor,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            channel_imgs: dict channel_name → [B, 3, H, W].
                          Pass only the channels you need (train: all 4,
                          eval single-channel: just that one).
            clip_feat:    [B, 1536]  render+base_color CLIP (shared for all channels)
        Returns:
            dict channel_name → (score [B], binary_logit [B])
        """
        out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        clip_f = clip_feat.float()

        # ── Step 1: encode each channel independently ─────────────────────
        fused_per_ch: dict[str, torch.Tensor] = {}
        for ch, img in channel_imgs.items():
            img_feat, pooled = self._encode(img)
            cross_feat = self.cross_modal(pooled, clip_feat)
            parts = [img_feat, cross_feat]
            if self.clip_direct is not None:
                parts.append(self.clip_direct(clip_f))

            fused = self.fusion(torch.cat(parts, dim=1))  # [B, hidden]

            # FiLM: condition metallic fused on render CLS
            if ch == "metallic":
                if self.metallic_grad_scale < 1.0 and self.training:
                    fused = fused * self.metallic_grad_scale + fused.detach() * (1.0 - self.metallic_grad_scale)
                fused = self._film(fused, clip_feat)

            fused_per_ch[ch] = fused

        # ── Step 2: cross-channel interaction (all channels reason together) ─
        if self.cross_channel is not None and len(fused_per_ch) > 1:
            fused_per_ch = self.cross_channel(fused_per_ch)

        # ── Step 3: per-channel score heads ───────────────────────────────
        for ch, fused in fused_per_ch.items():
            score  = torch.sigmoid(self.score_heads[ch](fused)).squeeze(1) * 5.0
            binary = self.binary_heads[ch](fused).squeeze(1)
            out[ch] = (score, binary)

        return out
