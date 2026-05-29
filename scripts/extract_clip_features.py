"""Extract frozen CLIP features for the sampled dataset.

Two modes:
  cls     (default) Extract CLS image embeddings for specified channels.
          Output: features/clip_vitl14_openai_render_base_color.pt
            { model_names, channels, features:{ch: (N,D)}, clip_model, feature_dim }

  prompts  Extract per-channel image–text cosine similarities using directional prompts.
          Output: features/clip_prompt_sims_{channel}.pt  (one file per channel)
            { model_names, channel, prompts:[str], sims:(N,K fp16), clip_model }

Usage:
    # CLS features (render + base_color)
    python asset_quality_scorer/scripts/extract_clip_features.py

    # Prompt similarities for all four scoring channels
    python asset_quality_scorer/scripts/extract_clip_features.py --mode prompts
    python asset_quality_scorer/scripts/extract_clip_features.py --mode prompts --channels metallic
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import sys

import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

_IN_TTY = sys.stderr.isatty()


def _tqdm(it, **kw):
    return tqdm(it, disable=not _IN_TTY, **kw)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CSV        = PROJECT_ROOT / "asset_quality_scorer/dataset/sampled_all.csv"
DEFAULT_IMAGE_ROOT = PROJECT_ROOT / "datasets0526"
DEFAULT_OUTPUT     = PROJECT_ROOT / "asset_quality_scorer/features/clip_vitl14_openai_render_base_color.pt"
FEATURES_DIR       = PROJECT_ROOT / "asset_quality_scorer/features"

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

# ── directional prompts per channel ──────────────────────────────────────────
# Goal: help the model distinguish subtle material-specific failure modes that
# the generic CLS token cannot capture (e.g. all-black metallic: correct vs wrong).

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

PROMPT_CHANNELS = list(CHANNEL_PROMPTS.keys())


def _model_to_name(model_path: str) -> str:
    """raw_data/sketchfab/8b/abc.glb → sketchfab__8b__abc"""
    return model_path.removeprefix("raw_data/").replace("/", "__").removesuffix(".glb")


# ── Dataset ───────────────────────────────────────────────────────────────────

class ChannelDataset(Dataset):
    def __init__(self, csv_path: Path, image_root: Path, channels: list[str]):
        with csv_path.open(newline="", encoding="utf-8-sig") as f:
            all_rows = list(csv.DictReader(f))
        self.image_root = image_root
        self.channels = channels
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ])
        self.rows = [
            r for r in all_rows
            if all((image_root / ch / f"{_model_to_name(r['model'])}.png").exists()
                   for ch in channels)
        ]
        skipped = len(all_rows) - len(self.rows)
        if skipped:
            print(f"  [warn] skipped {skipped} rows with missing images")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        model_name = _model_to_name(self.rows[idx]["model"])
        images = {
            ch: self.transform(Image.open(self.image_root / ch / f"{model_name}.png").convert("RGB"))
            for ch in self.channels
        }
        return model_name, images


def _collate(batch):
    names = [item[0] for item in batch]
    channels = list(batch[0][1].keys())
    return names, {ch: torch.stack([item[1][ch] for item in batch]) for ch in channels}


# ── CLS mode ──────────────────────────────────────────────────────────────────

def extract_cls(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    channels = list(dict.fromkeys(args.channels))
    device, clip_model = _load_clip(args)
    feature_dim = int(getattr(clip_model.visual, "output_dim", 768))
    print(f"Feat dim : {feature_dim}")

    ds = ChannelDataset(Path(args.csv), Path(args.image_root), channels)
    loader = _make_loader(ds, args)

    names_all: list[str] = []
    feats: dict[str, list[torch.Tensor]] = {ch: [] for ch in channels}

    with torch.no_grad():
        for batch_names, images in _tqdm(loader, desc="CLS", unit="batch", ncols=80):
            names_all.extend(batch_names)
            sizes = [images[ch].shape[0] for ch in channels]
            x = torch.cat([images[ch] for ch in channels], dim=0).to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=(args.amp and device.type == "cuda")):
                encoded = clip_model.encode_image(x)
            for ch, feat in zip(channels, torch.split(encoded, sizes, dim=0)):
                feat = feat.detach().cpu()
                if args.feature_dtype == "fp16":
                    feat = feat.half()
                feats[ch].append(feat)

    feature_tensors = {ch: torch.cat(parts, dim=0) for ch, parts in feats.items()}
    payload = {
        "model_names": names_all, "channels": channels,
        "features": feature_tensors,
        "clip_model": args.clip_model, "clip_pretrained": args.clip_pretrained,
        "feature_dim": feature_dim, "feature_dtype": args.feature_dtype,
    }
    torch.save(payload, output)
    meta = {k: v for k, v in payload.items() if k != "features"}
    meta["rows"] = len(names_all)
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved CLS features -> {output}")
    print(f"  shape: {feature_tensors[channels[0]].shape}  dtype: {feature_tensors[channels[0]].dtype}")


# ── prompts mode ──────────────────────────────────────────────────────────────

def extract_prompts(args: argparse.Namespace) -> None:
    channels = list(dict.fromkeys(args.channels))
    device, clip_model = _load_clip(args)
    tokenizer = open_clip.get_tokenizer(args.clip_model)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    for channel in channels:
        prompts = CHANNEL_PROMPTS.get(channel)
        if prompts is None:
            print(f"[{channel}] no prompts defined — skip")
            continue

        print(f"\n[{channel}]  {len(prompts)} prompts")

        # Encode text prompts once
        with torch.no_grad():
            tokens = tokenizer(prompts).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=(args.amp and device.type == "cuda")):
                text_feats = clip_model.encode_text(tokens)           # (K, D)
            text_feats = F.normalize(text_feats.float(), dim=-1)      # (K, D)

        ds = ChannelDataset(Path(args.csv), Path(args.image_root), [channel])
        loader = _make_loader(ds, args)

        names_all: list[str] = []
        sims_all: list[torch.Tensor] = []

        with torch.no_grad():
            for batch_names, images in _tqdm(loader, desc=f"  sims", unit="batch", ncols=80):
                names_all.extend(batch_names)
                x = images[channel].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16,
                                    enabled=(args.amp and device.type == "cuda")):
                    img_feats = clip_model.encode_image(x)            # (B, D)
                img_feats = F.normalize(img_feats.float(), dim=-1)
                sim = (img_feats @ text_feats.T).cpu().half()         # (B, K)
                sims_all.append(sim)

        sims_tensor = torch.cat(sims_all, dim=0)                      # (N, K)
        out_path = FEATURES_DIR / f"clip_prompt_sims_{channel}.pt"
        payload = {
            "model_names": names_all,
            "channel": channel,
            "prompts": prompts,
            "sims": sims_tensor,
            "clip_model": args.clip_model,
            "clip_pretrained": args.clip_pretrained,
        }
        torch.save(payload, out_path)
        meta = {k: v for k, v in payload.items() if k != "sims"}
        meta["n"] = len(names_all)
        meta["n_prompts"] = len(prompts)
        out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        print(f"  Saved -> {out_path}  shape={sims_tensor.shape}")


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_clip(args):
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    print(f"Device   : {device}")
    print(f"Model    : {args.clip_model} / {args.clip_pretrained}")
    clip_model = open_clip.create_model(
        args.clip_model, pretrained=args.clip_pretrained
    ).to(device).eval()
    return device, clip_model


def _make_loader(ds, args):
    return DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        collate_fn=_collate,
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",           choices=["cls", "prompts"], default="cls")
    p.add_argument("--csv",            default=str(DEFAULT_CSV))
    p.add_argument("--image-root",     default=str(DEFAULT_IMAGE_ROOT))
    p.add_argument("--output",         default=str(DEFAULT_OUTPUT),
                   help="Output path (cls mode only)")
    p.add_argument("--channels",       nargs="+",
                   default=None,
                   help="Channels to process. cls default: render base_color. "
                        "prompts default: all four scoring channels.")
    p.add_argument("--clip-model",     default="ViT-L-14")
    p.add_argument("--clip-pretrained",default="openai")
    p.add_argument("--batch-size",     type=int, default=128)
    p.add_argument("--workers",        type=int, default=8)
    p.add_argument("--feature-dtype",  choices=["fp32", "fp16"], default="fp16")
    p.add_argument("--amp",            action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cpu",            action="store_true")
    args = p.parse_args()

    if args.channels is None:
        args.channels = ["render", "base_color"] if args.mode == "cls" else PROMPT_CHANNELS

    print(f"Mode     : {args.mode}")
    print(f"Rows CSV : {args.csv}")
    print(f"Channels : {', '.join(args.channels)}")

    if args.mode == "cls":
        extract_cls(args)
    else:
        extract_prompts(args)


if __name__ == "__main__":
    main()
