"""Multi-task training: one DINOv2 backbone, 4 PBR channel heads, FiLM on metallic.

Usage:
    python asset_quality_scorer/scripts/train_multitask.py \
        --config asset_quality_scorer/config/dinov2_large_multitask.yaml
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml

# Rich for beautiful terminal output
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.text import Text
from rich import box

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "asset_quality_scorer"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT))

import csv
import numpy as np
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from quality_scorer.models import DINOv2MultiTaskScorer

_CHANNELS  = DINOv2MultiTaskScorer.CHANNELS   # (bc, nm, ro, me)
_CLIP_HALF = 768   # CLIP feat = [base_color(768), render(768)]; render half = [768:]
_SCORE_COL = {"base_color": "baseColor", "normal_map": "normal",
              "roughness": "roughness", "metallic": "metallic"}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def _build_aug(strong: bool) -> transforms.Compose:
    """Train-time augmentation. strong=True adds RandomErasing (cutout) — a mild
    geometric regularizer that is safe for all 4 channels (no color jitter, which
    would corrupt normal-map directions). Targets the train-test gap."""
    ops = [
        transforms.ConvertImageDtype(torch.float32),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    if strong:
        ops.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)))
    return transforms.Compose(ops)

_AUG = _build_aug(False)
_EVAL = transforms.Compose([
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# aux-label parsing ───────────────────────────────────────────────────────────
_PBRTYPE_MAP = {"physical": 0, "stylized": 1, "uncertain": 2}
_DEFECT_COLS = ["hasTextOrPattern", "baseColorHasFakeAOOrGlow",
                "normalHasAbnormalTint", "normalIsFlipped"]

def _parse_aux(row: dict) -> dict:
    """Extract auxiliary labels from a CSV row. -1 = missing (CE ignore_index)."""
    tier_s = row.get("tier", "")
    tier = -1
    for k in range(1, 6):
        if f"Tier {k}" in tier_s:
            tier = k - 1
            break
    pbrtype = _PBRTYPE_MAP.get(row.get("pbrType", ""), -1)
    defect = [1.0 if row.get(c) == "True" else 0.0 for c in _DEFECT_COLS]
    return {"tier": tier, "pbrtype": pbrtype, "defect": defect}


class MultiChannelDataset(Dataset):
    """Loads all 4 channel images + shared CLIP features per sample."""

    def __init__(self, cache_root: Path, clip_path: Path, csv_path: Path,
                 split: str, is_train: bool, invalid_max_score: int = 1,
                 pbr_filter: str | None = None, strong_aug: bool = False,
                 synth_neg: dict | None = None, zero_render_clip: bool = False,
                 vlm_prior: dict | None = None, metallic_stretch: float = 1.0):
        self.is_train = is_train
        self.n_synth = 0
        # contrast-stretch the metallic map at load: near-black quality (per SOP, the
        # low-amplitude stray gray on non-metal regions) is near-invisible in the raw
        # map; amplifying it may expose the discriminative signal to the backbone.
        self.metallic_stretch = float(metallic_stretch)
        self.tfm = _build_aug(strong_aug) if is_train else _EVAL
        self.invalid_max_score = invalid_max_score

        # Load cache metadata
        meta = json.loads((cache_root / "meta.json").read_text())
        cache_idx = {n: i for i, n in enumerate(meta["model_names"])}

        # Lazy memmaps (opened per worker)
        self._cache_root = cache_root
        self._arrays: dict[str, np.ndarray | None] = {ch: None for ch in _CHANNELS}

        # Load CLIP features
        clip_data = torch.load(clip_path, map_location="cpu", weights_only=False)
        clip_names = clip_data["model_names"]
        clip_idx = {n: i for i, n in enumerate(clip_names)}
        bc  = clip_data["features"]["base_color"]
        rn  = clip_data["features"]["render"]
        # pre-build full clip matrix: [N, 1536] fp32
        N = len(clip_names)
        self._clip_mat = torch.cat([
            torch.from_numpy(np.array(bc, dtype=np.float32)),
            torch.from_numpy(np.array(rn, dtype=np.float32)),
        ], dim=1)                                          # [N, 1536] = [bc(768), render(768)]
        # Rigorous control for render-circularity: zero the render half so the
        # model can ONLY use base_color context (the clean, non-circular signal).
        if zero_render_clip:
            self._clip_mat[:, _CLIP_HALF:] = 0.0
        self._clip_idx = clip_idx

        # Build sample list (intersect cache ∩ clip ∩ CSV ∩ split, all 4 scores valid)
        self.samples: list[tuple[str, dict[str, int]]] = []
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("split") != split:
                    continue
                # physical-only filter: SOP says physical vs stylized use
                # different judgment rules; train on the clean physical subset.
                if pbr_filter is not None and row.get("pbrType", "") != pbr_filter:
                    continue
                name = row["model"].removeprefix("raw_data/").replace("/", "__").removesuffix(".glb")
                if name not in cache_idx or name not in clip_idx:
                    continue
                scores = {}
                ok = True
                for ch, col in _SCORE_COL.items():
                    try:
                        scores[ch] = int(row[col])
                        assert 0 <= scores[ch] <= 5
                    except Exception:
                        ok = False; break
                if ok:
                    self.samples.append((name, scores, cache_idx[name],
                                         clip_idx[name], _parse_aux(row), False))

        # ── synthetic missing-metal negatives (train only) ───────────────────
        # Take assets that genuinely HAVE metal (high non-black metallic fraction
        # + decent metallic score) → zero their metallic map → label metallic = 0.
        # This injects CLEAN supervision into the ambiguous near-black region:
        # "base_color shows metal material + metallic map is black = missing metal
        #  = bad". A minimal pair vs the original (same object, real map = good).
        # Caveat: render (in CLIP) is NOT re-rendered (image folder read-only), so
        # its render half still shows metal — a possible shortcut. The clean,
        # consistent context signal is base_color (metal albedo, unchanged).
        if synth_neg and is_train:
            frac_path = csv_path.parent / "metallic_nonblack.npy"
            frac = np.load(frac_path)   # aligned to cache meta order (= cache_idx)
            min_score = int(synth_neg.get("min_metallic_score", 3))
            min_frac  = float(synth_neg.get("min_nonblack_frac", 0.2))
            neg_label = int(synth_neg.get("neg_label", 0))
            eligible = [s for s in self.samples
                        if s[1]["metallic"] >= min_score and frac[s[2]] >= min_frac]
            n_target = synth_neg.get("count")
            if n_target is None:
                n_target = int(float(synth_neg.get("ratio", 0.1)) * len(self.samples))
            import random as _r; _r.seed(int(synth_neg.get("seed", 0)))
            picks = eligible if len(eligible) <= n_target else _r.sample(eligible, n_target)
            for (name, scores, ci, ki, aux, _) in picks:
                neg_scores = dict(scores); neg_scores["metallic"] = neg_label
                self.samples.append((name, neg_scores, ci, ki, aux, True))
            self.n_synth = len(picks)

        # ── VLM world-knowledge prior (precompute_vlm_prior.py output) ───────
        # Files are aligned to cache meta order → index by ci. shuffle=True is
        # the spurious-gain control (roughness-dual-stream lesson): permute the
        # prior across assets; if the gain survives, it was never real signal.
        self._vlm = None
        self._vlm_regime = None
        if vlm_prior:
            self._vlm_dim = int(vlm_prior.get("dim", 1))
            prefix = str(vlm_prior["path_prefix"])
            if self._vlm_dim == 1:
                self._vlm = np.load(prefix + "_pyes.npy").astype(np.float32)
                self._vlm = np.clip(self._vlm, 0.0, 1.0)   # 未填充(-1)→0 中性
            else:
                self._vlm = np.load(prefix + "_hidden.npy", mmap_mode="r")
            self._vlm_perm = None
            if vlm_prior.get("shuffle"):
                rng = np.random.RandomState(int(vlm_prior.get("seed", 0)))
                self._vlm_perm = rng.permutation(len(self._vlm))
            # regime gate: soft near-black routing w = sigmoid((thresh - nonblack)/tau).
            # Computed from the metallic MAP (image stat, no label leakage); the prior
            # only flows where visual info is absent. NOTE: the shuffle control permutes
            # the PRIOR but keeps w aligned to the sample — controls prior content only.
            if vlm_prior.get("regime_gate"):
                frac = np.load(csv_path.parent / str(vlm_prior.get(
                    "nonblack_file", "metallic_nonblack.npy")))   # aligned to cache order
                thresh = float(vlm_prior.get("regime_thresh", 0.02))
                tau = float(vlm_prior.get("regime_tau", 0.01))
                z = np.clip((thresh - frac) / tau, -60.0, 60.0)
                self._vlm_regime = (1.0 / (1.0 + np.exp(-z))).astype(np.float32)

    def tail_sample_weights(self, power: float = 0.5, cap: float = 10.0) -> np.ndarray:
        """Per-sample weights for WeightedRandomSampler to oversample rare extreme
        scores. For each sample, weight = max over 4 channels of (inverse score
        frequency)^power, capped. So a sample with a rare extreme in ANY channel
        (e.g. normal_map=5, n≈73, ~10% recall) gets strongly upsampled. Targets
        the high-score tail collapse seen in the best model."""
        from collections import Counter
        counts = {c: Counter() for c in _CHANNELS}
        for s in self.samples:
            for c in _CHANNELS:
                counts[c][int(s[1][c])] += 1
        n = len(self.samples)
        w = np.ones(n, dtype=np.float64)
        for i, s in enumerate(self.samples):
            wi = 1.0
            for c in _CHANNELS:
                freq = counts[c][int(s[1][c])] / n
                wi = max(wi, (1.0 / max(freq, 1e-6)) ** power)
            w[i] = min(wi, cap)
        return w

    def _arr(self, ch: str) -> np.ndarray:
        if self._arrays[ch] is None:
            self._arrays[ch] = np.load(self._cache_root / f"{ch}.npy", mmap_mode="r")
        return self._arrays[ch]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        name, scores, ci, ki, aux, is_synth = self.samples[i]
        imgs = {}
        for ch in _CHANNELS:
            raw = torch.from_numpy(np.array(self._arr(ch)[ci], copy=True))  # [3,H,W] uint8
            if is_synth and ch == "metallic":
                raw = torch.zeros_like(raw)   # zero the metallic map = missing metal
            elif ch == "metallic" and self.metallic_stretch != 1.0:
                raw = (raw.float() * self.metallic_stretch).clamp_(0, 255).to(torch.uint8)
            imgs[ch] = self.tfm(raw)
        clip = self._clip_mat[ki]                            # [1536]
        score_t  = {ch: torch.tensor(s, dtype=torch.float32) for ch, s in scores.items()}
        binary_t = {ch: torch.tensor(float(s <= self.invalid_max_score)) for ch, s in scores.items()}
        aux_t = {
            "tier":    torch.tensor(aux["tier"],    dtype=torch.long),
            "pbrtype": torch.tensor(aux["pbrtype"], dtype=torch.long),
            "defect":  torch.tensor(aux["defect"],  dtype=torch.float32),
        }
        if self._vlm is not None:
            vi = self._vlm_perm[ci] if self._vlm_perm is not None else ci
            if self._vlm_dim == 1:
                vlm_t = torch.tensor([float(self._vlm[vi])], dtype=torch.float32)        # [1]
            else:
                vlm_t = torch.from_numpy(np.array(self._vlm[vi], dtype=np.float32))      # [3584]
        else:
            vlm_t = torch.zeros(1)
        # regime w stays aligned to THIS sample (ci, not vi) even under shuffle
        w_t = torch.tensor(float(self._vlm_regime[ci]) if self._vlm_regime is not None else 1.0)
        return imgs, clip, score_t, binary_t, aux_t, vlm_t, w_t


def _collate(batch):
    imgs_list, clips, scores_list, bin_list, aux_list, vlm_list, w_list = zip(*batch)
    imgs = {ch: torch.stack([b[ch] for b in imgs_list]) for ch in _CHANNELS}
    clips = torch.stack(clips)
    scores  = {ch: torch.stack([b[ch] for b in scores_list]) for ch in _CHANNELS}
    binaries = {ch: torch.stack([b[ch] for b in bin_list])  for ch in _CHANNELS}
    aux = {
        "tier":    torch.stack([a["tier"]    for a in aux_list]),
        "pbrtype": torch.stack([a["pbrtype"] for a in aux_list]),
        "defect":  torch.stack([a["defect"]  for a in aux_list]),
    }
    vlm = torch.stack(vlm_list)
    regime_w = torch.stack(w_list)
    return imgs, clips, scores, binaries, aux, vlm, regime_w


from quality_scorer.ordinal import CoralRankLoss
_coral = CoralRankLoss(num_classes=6)
def _coral_loss(logits, scores):
    return _coral(logits, scores.long())

def _emd_loss(logits, scores):
    """NIMA-style squared Earth Mover's Distance over the 6-bin score distribution.
    Ordinal-aware: penalizes far-off predictions more than near ones (5→0 ≫ 5→4)."""
    p = torch.softmax(logits, dim=1)                       # [B,6]
    cdf_p = torch.cumsum(p, dim=1)
    onehot = torch.zeros_like(p)
    onehot[torch.arange(len(scores), device=scores.device), scores.long()] = 1.0
    cdf_t = torch.cumsum(onehot, dim=1)
    return ((cdf_p - cdf_t) ** 2).sum(1).mean()


def _ranking_loss(pred, target, margin=0.5):
    dt = target.unsqueeze(0) - target.unsqueeze(1)
    dp = pred.unsqueeze(0)   - pred.unsqueeze(1)
    mask = dt.abs() > 0.5
    if not mask.any():
        return pred.sum() * 0.0
    return F.relu(margin - dt[mask].sign() * dp[mask]).mean()


def _resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def train(cfg: dict) -> None:
    d, m, t, u, o = cfg["data"], cfg["model"], cfg["train"], cfg["unfreeze"], cfg["output"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = m.get("backbone", "dinov2")   # "dinov2" | "convnext" | "convnext_early/mid"
    backbone_name = m.get("backbone_name", None)   # override DINOv2 backbone (e.g. DINOv3)
    if backbone.startswith("convnext"):      arch_prefix = "convnext_base"
    elif backbone_name and "dinov3" in backbone_name: arch_prefix = "dinov3_large"
    else:                                    arch_prefix = "dinov2_large"
    exp_id  = f"{arch_prefix}_multitask_{o.get('exp_suffix','v1')}"
    out_dir = _resolve(o.get("root", "asset_quality_scorer/outputs/runs")) / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*72}\nTraining: {exp_id}\n{'='*72}")

    # ── dataset ────────────────────────────────────────────────────────────
    cache_root = _resolve(d["tensor_cache_root"])
    clip_path  = _resolve(d["clip_feature_path"])
    csv_path   = _resolve(d["csv_path"])
    inv_max    = int(m.get("invalid_max_score", 1))

    pbr_filter = d.get("pbr_filter")  # None | "physical" | "stylized"
    strong_aug = bool(t.get("strong_aug", False))
    synth_neg  = d.get("synth_neg")   # None | dict (synthetic missing-metal negatives)
    zero_rn    = bool(d.get("zero_render_clip", False))
    vlm_cfg    = d.get("vlm_prior")   # None | {path_prefix, dim, shuffle, seed}
    if vlm_cfg:
        vlm_cfg = dict(vlm_cfg)
        vlm_cfg["path_prefix"] = str(_resolve(vlm_cfg["path_prefix"]))
        print(f"  vlm_prior: dim={vlm_cfg.get('dim',1)} shuffle={bool(vlm_cfg.get('shuffle'))} ({vlm_cfg['path_prefix']})")
    mstretch = float(d.get("metallic_stretch", 1.0))
    if mstretch != 1.0:
        print(f"  metallic_stretch={mstretch} (对比拉伸 metallic 输入, 暴露近黑低幅信号)")
    train_ds = MultiChannelDataset(cache_root, clip_path, csv_path, "train", True,  inv_max, pbr_filter, strong_aug, synth_neg, zero_rn, vlm_cfg, mstretch)
    val_ds   = MultiChannelDataset(cache_root, clip_path, csv_path, "val",   False, inv_max, pbr_filter, False, None, zero_rn, vlm_cfg, mstretch)
    print(f"  train: {len(train_ds)} (含 {train_ds.n_synth} 合成漏标负样本)  val: {len(val_ds)}  "
          f"pbr_filter={pbr_filter}  strong_aug={strong_aug}")

    nw = int(t.get("num_workers", 8))
    bs = int(t.get("batch_size", 16))
    # Tail oversampling: upsample rare extreme scores (targets normal_map=5 /
    # metallic=0 collapse). power>0 enables a WeightedRandomSampler.
    tail_power = float(t.get("tail_sample_power", 0.0))
    if tail_power > 0:
        from torch.utils.data import WeightedRandomSampler
        wts = train_ds.tail_sample_weights(power=tail_power,
                                           cap=float(t.get("tail_sample_cap", 10.0)))
        sampler = WeightedRandomSampler(torch.as_tensor(wts, dtype=torch.double),
                                        num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler,
                                  num_workers=nw, collate_fn=_collate, pin_memory=True)
        print(f"  tail oversampling: power={tail_power} cap={t.get('tail_sample_cap',10.0)} "
              f"(weight range [{wts.min():.1f},{wts.max():.1f}])")
    else:
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                  num_workers=nw, collate_fn=_collate, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs*2, shuffle=False,
                              num_workers=nw, collate_fn=_collate, pin_memory=True)

    # ── model ──────────────────────────────────────────────────────────────
    if backbone in ("convnext_early", "convnext_mid"):
        from quality_scorer.models import ConvNeXtEarlyFusionScorer, ConvNeXtMidFusionScorer
        Cls = ConvNeXtMidFusionScorer if backbone == "convnext_mid" else ConvNeXtEarlyFusionScorer
        model = Cls(
            clip_dim            = int(m.get("clip_dim", 1536)),
            attn_proj_dim       = int(m.get("attn_proj_dim", 256)),
            attn_heads          = int(m.get("attn_heads", 4)),
            hidden_dim          = int(m.get("hidden_dim", 512)),
            dropout             = float(m.get("dropout", 0.3)),
            freeze_features     = True,
            use_clip_direct     = True,
            metallic_grad_scale = float(m.get("metallic_grad_scale", 1.0)),
            ordinal_channels    = m.get("ordinal_channels", None),
            emd_channels        = m.get("emd_channels", None),
        ).to(device)
    elif backbone == "convnext":
        from quality_scorer.models import ConvNeXtMultiTaskScorer
        model = ConvNeXtMultiTaskScorer(
            clip_dim            = int(m.get("clip_dim", 1536)),
            attn_proj_dim       = int(m.get("attn_proj_dim", 256)),
            attn_heads          = int(m.get("attn_heads", 4)),
            hidden_dim          = int(m.get("hidden_dim", 512)),
            dropout             = float(m.get("dropout", 0.3)),
            freeze_features     = True,
            use_clip_direct     = True,
            metallic_grad_scale = float(m.get("metallic_grad_scale", 0.5)),
            ordinal_channels    = m.get("ordinal_channels", None),
            emd_channels        = m.get("emd_channels", None),
        ).to(device)
    else:
        model = DINOv2MultiTaskScorer(
            clip_dim            = int(m.get("clip_dim", 1536)),
            attn_proj_dim       = int(m.get("attn_proj_dim", 256)),
            attn_heads          = int(m.get("attn_heads", 4)),
            hidden_dim          = int(m.get("hidden_dim", 512)),
            dropout             = float(m.get("dropout", 0.3)),
            freeze_features     = True,
            use_clip_direct     = True,
            metallic_film       = bool(m.get("metallic_film", True)),
            metallic_grad_scale = float(m.get("metallic_grad_scale", 0.5)),
            use_cross_channel   = bool(m.get("use_cross_channel", False)),
            cc_n_heads          = int(m.get("cc_n_heads", 4)),
            cc_me_bc_bias       = float(m.get("cc_me_bc_bias", 1.0)),
            cc_me_ro_bias       = float(m.get("cc_me_ro_bias", 1.0)),
            cc_metallic_only    = bool(m.get("cc_metallic_only", False)),
            metallic_ordinal    = bool(m.get("metallic_ordinal", False)),
            ordinal_channels    = m.get("ordinal_channels", None),
            emd_channels        = m.get("emd_channels", None),
            drop_path_rate      = float(m.get("drop_path_rate", 0.0)),
            aux_supervision     = bool(m.get("aux_supervision", False)),
            use_attn_pool       = bool(m.get("use_attn_pool", False)),
            metallic_no_render  = bool(m.get("metallic_no_render", False)),
            metallic_spatial_xchannel = bool(m.get("metallic_spatial_xchannel", False)),
            msx_heads           = int(m.get("msx_heads", 8)),
            backbone_name       = (backbone_name or "vit_large_patch14_reg4_dinov2"),
            vlm_prior_dim       = int(m.get("vlm_prior_dim", 0)),
            vlm_proj_dim        = int(m.get("vlm_proj_dim", 128)),
        ).to(device)
    cc = model.cross_channel
    print(f"  metallic_film={model.film_gen is not None}  "
          f"metallic_grad_scale={model.metallic_grad_scale}  "
          f"cross_channel={cc is not None}  ordinal={sorted(model.ordinal_channels)}  "
          f"emd={sorted(model.emd_channels)}  "
          f"drop_path={float(m.get('drop_path_rate',0.0))}  aux_sup={model.aux_heads is not None}")

    # per-channel loss weights
    ch_weights = {ch: float(t.get("channel_weights", {}).get(ch, 1.0)) for ch in _CHANNELS}
    print(f"  channel weights: {ch_weights}")

    # ── AMP ────────────────────────────────────────────────────────────────
    use_amp  = device.type == "cuda"
    amp_dt   = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler   = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dt == torch.float16)
    if use_amp:
        print(f"  amp: {'bf16' if amp_dt==torch.bfloat16 else 'fp16'}")

    # ── optimizer ──────────────────────────────────────────────────────────
    lr  = float(t.get("lr", 1e-4))
    wd  = float(t.get("weight_decay", 1e-4))
    bin_w  = float(t.get("binary_loss_weight",  0.2))
    rank_w = float(t.get("ranking_loss_weight", 0.05))
    huber_d = float(t.get("huber_delta", 1.0))
    aux_w  = float(t.get("aux_loss_weight", 0.0))   # auxiliary supervision weight
    if aux_w > 0:
        print(f"  aux supervision: weight={aux_w} (tier/pbrType/defects)")

    # packed-input experiment: unfreeze the patch-embed conv from epoch 0 so the
    # backbone re-learns to read channel-specific multi-content packed inputs.
    if u.get("patch_embed", False) and hasattr(model, "unfreeze_patch_embed"):
        model.unfreeze_patch_embed()
        print("  🔓 patch_embed 解冻 (packed-input 重训)")

    # Layer-wise LR decay (LLRD): backbone blocks get exponentially smaller LR
    # the deeper into the frozen base. Stabilizes full-unfreeze — low layers
    # (generic features) barely move, high layers + heads adapt. Standard ViT
    # fine-tuning practice; directly targets the post-unfreeze overfit we saw.
    llrd = float(t.get("llrd_decay", 0.0))  # 0 = off (uniform LR), 0.75 typical

    def make_opt():
        trainable = lambda module: [p for p in module.parameters() if p.requires_grad]
        if llrd <= 0:
            return optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=lr, weight_decay=wd)
        groups, seen = [], set()
        blocks = model.backbone.blocks
        n_blk = len(blocks)
        for i, blk in enumerate(blocks):
            ps = trainable(blk)
            if ps:
                groups.append({"params": ps, "lr": lr * (llrd ** (n_blk - 1 - i))})
                seen.update(id(p) for p in ps)
        # everything else (heads, fusion, cross_modal, cross_channel, backbone norm/embed) → full LR
        rest = [p for p in model.parameters() if p.requires_grad and id(p) not in seen]
        if rest:
            groups.append({"params": rest, "lr": lr})
        return optim.AdamW(groups, lr=lr, weight_decay=wd)

    optimizer = make_opt()
    epochs    = int(t.get("epochs", 30))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if llrd > 0:
        print(f"  LLRD: decay={llrd}  (backbone block0 lr={lr*(llrd**(len(model.backbone.blocks)-1)):.2e} → blockN lr={lr:.0e})")

    # ── TensorBoard ────────────────────────────────────────────────────────
    from torch.utils.tensorboard import SummaryWriter
    tb_dir = out_dir / "tensorboard"
    tb_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))

    # structured per-epoch log (for viewer dashboard)
    epoch_log: list[dict] = []

    # ── helpers ────────────────────────────────────────────────────────────
    SRCC_COLORS = {  # rich color thresholds for SRCC
        "base_color": (0.82, 0.76),
        "normal_map": (0.78, 0.72),
        "roughness":  (0.88, 0.82),
        "metallic":   (0.62, 0.57),
    }
    def srcc_color(ch: str, val: float) -> str:
        hi, lo = SRCC_COLORS.get(ch, (0.8, 0.7))
        if val >= hi: return "green"
        if val >= lo: return "yellow"
        return "red"

    # ── training loop ──────────────────────────────────────────────────────
    best = {"mae_mean": float("inf"), "srcc_mean": -2.0}
    epoch_times: list[float] = []

    for ei in range(epochs):
        epoch = ei + 1
        t0 = time.time()

        # progressive unfreeze
        unfreeze_msg = None
        if ei == int(u.get("stage4_epoch", 5)):
            model.unfreeze_stage4(); optimizer = make_opt()
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-ei)
            unfreeze_msg = "stage4 解冻"
        elif ei == int(u.get("stage34_epoch", 10)):
            model.unfreeze_stage34(); optimizer = make_opt()
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-ei, eta_min=1e-6)
            unfreeze_msg = "stage3+4 解冻"
        elif ei == int(u.get("stage234_epoch", 15)):
            model.unfreeze_stage234(); optimizer = make_opt()
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-ei, eta_min=1e-6)
            unfreeze_msg = "stage2+3+4 解冻 (全部)"
        if unfreeze_msg:
            console.print(f"  [bold cyan]🔓 {unfreeze_msg}[/bold cyan]")

        # ── train batches with progress bar ──────────────────────────────
        model.train()
        running = 0.0
        with Progress(
            TextColumn(f"  [cyan]ep {epoch:02d}/{epochs}[/cyan]"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TextColumn("[yellow]loss={task.fields[loss]:.4f}[/yellow]"),
            TimeRemainingColumn(),
            console=console, transient=True,
        ) as prog:
            task = prog.add_task("train", total=len(train_loader), loss=0.0)
            for imgs, clips, scores, binaries, aux, vlm, regw in train_loader:
                imgs     = {ch: v.to(device, non_blocking=True) for ch, v in imgs.items()}
                clips    = clips.to(device, non_blocking=True)
                scores   = {ch: v.to(device, non_blocking=True) for ch, v in scores.items()}
                binaries = {ch: v.to(device, non_blocking=True) for ch, v in binaries.items()}
                aux      = {k: v.to(device, non_blocking=True) for k, v in aux.items()}

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=amp_dt, enabled=use_amp):
                    preds = model(imgs, clips, vlm_prior=vlm.to(device, non_blocking=True), vlm_regime_w=regw.to(device, non_blocking=True)) if getattr(model, 'vlm_prior_dim', 0) else model(imgs, clips)
                    loss = torch.tensor(0.0, device=device)
                    for ch in _CHANNELS:
                        ps, pb, aux_logits = preds[ch]
                        w = ch_weights[ch]
                        # main quality loss: CORAL (ordinal) / EMD (distribution) / Huber (reg)
                        if ch in model.ordinal_channels:
                            loss = loss + w * _coral_loss(aux_logits, scores[ch])
                        elif ch in model.emd_channels:
                            loss = loss + w * _emd_loss(aux_logits, scores[ch])
                        else:
                            loss = loss + w * F.huber_loss(ps, scores[ch], delta=huber_d)
                        if bin_w > 0:
                            loss = loss + w * bin_w * F.binary_cross_entropy_with_logits(pb, binaries[ch])
                        if rank_w > 0:
                            loss = loss + w * rank_w * _ranking_loss(ps, scores[ch])
                    # auxiliary supervision (regularizes shared backbone)
                    if aux_w > 0 and "_aux" in preds:
                        ap = preds["_aux"]
                        loss = loss + aux_w * F.cross_entropy(ap["tier"].float(),    aux["tier"],    ignore_index=-1)
                        loss = loss + aux_w * F.cross_entropy(ap["pbrtype"].float(), aux["pbrtype"], ignore_index=-1)
                        loss = loss + aux_w * F.binary_cross_entropy_with_logits(ap["defect"].float(), aux["defect"])

                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
                running += float(loss.item())
                prog.update(task, advance=1, loss=running / (prog.tasks[0].completed + 1))

        scheduler.step()
        train_loss = running / max(len(train_loader), 1)

        # ── validation ────────────────────────────────────────────────────
        model.eval()
        preds_all  = {ch: [] for ch in _CHANNELS}
        scores_all = {ch: [] for ch in _CHANNELS}
        with torch.no_grad():
            for imgs, clips, scores, _, _, vlm, regw in val_loader:
                imgs  = {ch: v.to(device, non_blocking=True) for ch, v in imgs.items()}
                clips = clips.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=amp_dt, enabled=use_amp):
                    preds = model(imgs, clips, vlm_prior=vlm.to(device, non_blocking=True), vlm_regime_w=regw.to(device, non_blocking=True)) if getattr(model, 'vlm_prior_dim', 0) else model(imgs, clips)
                for ch in _CHANNELS:
                    preds_all[ch].extend(preds[ch][0].float().cpu().tolist())
                    scores_all[ch].extend(scores[ch].tolist())

        metrics = {}
        for ch in _CHANNELS:
            p = np.array(preds_all[ch]); s = np.array(scores_all[ch])
            metrics[ch] = {
                "mae":  round(float(np.abs(p-s).mean()), 4),
                "srcc": round(float(spearmanr(p, s).statistic), 4),
            }
        mae_mean  = round(np.mean([metrics[ch]["mae"]  for ch in _CHANNELS]), 4)
        srcc_mean = round(np.mean([metrics[ch]["srcc"] for ch in _CHANNELS]), 4)
        epoch_t   = time.time() - t0
        epoch_times.append(epoch_t)
        eta_s     = np.mean(epoch_times[-5:]) * (epochs - epoch)

        # ── rich table output ─────────────────────────────────────────────
        is_best = srcc_mean > best["srcc_mean"]
        tbl = Table(box=box.SIMPLE_HEAD, show_footer=True,
                    title=f"Epoch {epoch}/{epochs}  loss={train_loss:.4f}"
                          f"  {'⭐ new best  ' if is_best else ''}"
                          f"ETA {int(eta_s//60)}m{int(eta_s%60):02d}s",
                    title_style="bold white")
        tbl.add_column("Channel",  footer="MEAN",    style="bold")
        tbl.add_column("SRCC ↑",   footer=f"{srcc_mean:.4f}", justify="right")
        tbl.add_column("MAE ↓",    footer=f"{mae_mean:.4f}",  justify="right")
        tbl.add_column("Best SRCC",justify="right", style="dim")

        prev_best_ch = (best.get("per_channel") or {})
        for ch in _CHANNELS:
            s_val = metrics[ch]["srcc"]; m_val = metrics[ch]["mae"]
            prev  = (prev_best_ch.get(ch) or {}).get("srcc", 0)
            best_ch = max(s_val, prev)
            delta = s_val - prev
            arrow = " ↑" if delta > 0.002 else (" ↓" if delta < -0.002 else "")
            color = srcc_color(ch, s_val)
            tbl.add_row(
                ch,
                Text(f"{s_val:.4f}{arrow}", style=color),
                f"{m_val:.4f}",
                f"{best_ch:.4f}",
            )
        console.print(tbl)

        # plain-text log line for grep (kept for backward compat)
        ch_str = "  ".join(f"{ch[:2]}={metrics[ch]['srcc']:.3f}" for ch in _CHANNELS)
        print(f"  epoch={epoch:2d} loss={train_loss:.4f}  srcc_mean={srcc_mean:.4f}  "
              f"mae_mean={mae_mean:.4f}  [{ch_str}]", flush=True)

        # ── TensorBoard ───────────────────────────────────────────────────
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("srcc/mean",  srcc_mean,  epoch)
        writer.add_scalar("mae/mean",   mae_mean,   epoch)
        for ch in _CHANNELS:
            writer.add_scalar(f"srcc/{ch}", metrics[ch]["srcc"], epoch)
            writer.add_scalar(f"mae/{ch}",  metrics[ch]["mae"],  epoch)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)
        if device.type == "cuda":
            writer.add_scalar("gpu_mem_gb",
                torch.cuda.memory_allocated(device) / 1e9, epoch)

        # ── structured per-epoch JSON ─────────────────────────────────────
        epoch_entry = {
            "epoch": epoch, "train_loss": train_loss,
            "srcc_mean": srcc_mean, "mae_mean": mae_mean,
            "per_channel": metrics, "epoch_sec": round(epoch_t, 1),
        }
        epoch_log.append(epoch_entry)
        (out_dir / "train_log.json").write_text(
            json.dumps({"exp_id": exp_id, "epochs": epoch_log}, indent=2)
        )

        if is_best:
            best = {"mae_mean": mae_mean, "srcc_mean": srcc_mean,
                    "epoch": epoch, "per_channel": metrics}
            torch.save({"model_state_dict": model.state_dict(), "arch": f"{arch_prefix}_multitask",
                        "epoch": epoch, "metrics": best}, out_dir / "best.pt")

        # also track best-by-metallic checkpoint: srcc_mean peaks early (frozen,
        # strong channels) which under-tests metallic-targeted changes (synthneg).
        # best_metallic.pt = checkpoint at the epoch where metallic val SRCC peaks.
        if metrics["metallic"]["srcc"] > best.get("metallic_srcc_peak", -2.0):
            best["metallic_srcc_peak"] = metrics["metallic"]["srcc"]
            best["metallic_epoch"] = epoch
            torch.save({"model_state_dict": model.state_dict(), "arch": f"{arch_prefix}_multitask",
                        "epoch": epoch, "metallic_srcc": metrics["metallic"]["srcc"]},
                       out_dir / "best_metallic.pt")

    writer.close()

    print(f"\nDone. Best: srcc_mean={best['srcc_mean']}  mae_mean={best['mae_mean']}")
    for ch in _CHANNELS:
        mc = best["per_channel"][ch]
        print(f"  {ch:12s}: mae={mc['mae']}  srcc={mc['srcc']}")

    # Diagnostic: what cross-channel attention did the model actually learn?
    if model.cross_channel is not None:
        bias_summary = model.cross_channel.get_attn_bias_summary()
        print("\n  Learned cross-channel attention biases (non-zero):")
        for pair, val in sorted(bias_summary.items(), key=lambda x: -abs(x[1])):
            bar = "█" * int(abs(val) * 5)
            print(f"    {pair}: {val:+.3f}  {bar}")
        best["attn_bias"] = bias_summary

    (out_dir / "summary.json").write_text(json.dumps({"exp_id": exp_id, **best}, indent=2))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = yaml.safe_load(open(_resolve(args.config)))
    train(cfg)
