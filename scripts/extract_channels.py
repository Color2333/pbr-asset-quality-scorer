"""Extract per-channel images from stitched grid PNGs into a flat dataset directory.

Grid layout (each cell = 2048×2048):
  grid_pbr.png   top-left=render   top-right=base_color
                 bottom-left=roughness  bottom-right=metallic
  grid_white.png top-right=normal_map

Usage:
    python asset_quality_scorer/scripts/extract_channels.py
    python asset_quality_scorer/scripts/extract_channels.py --workers 8 --dry-run
"""
from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SOURCE = Path("/storage/datasets/art-data/onsite/jianqiao_code/argos/pbr_492k/unzip")
DEFAULT_TARGET = Path("/storage/datasets/art-data-intern/haojiang_code/PBR_auto/datasets0526")
DEFAULT_CSV    = PROJECT_ROOT / "asset_quality_scorer/dataset/sampled_all.csv"

# (channel_name, source_grid, crop_box=(left,top,right,bottom))
CHANNELS = [
    ("render",           "grid_pbr.png",   (0,    0,    2048, 2048)),
    ("base_color",       "grid_pbr.png",   (2048, 0,    4096, 2048)),
    ("roughness",        "grid_pbr.png",   (0,    2048, 2048, 4096)),
    ("metallic",         "grid_pbr.png",   (2048, 2048, 4096, 4096)),
    ("normal_map",       "grid_white.png", (2048, 0,    4096, 2048)),
    ("white_model",      "grid_white.png", (0,    2048, 2048, 4096)),
    ("white_with_normal","grid_white.png", (2048, 2048, 4096, 4096)),
]


def model_to_folder(model_path: str) -> str:
    """raw_data/sketchfab/8b/abc.glb → sketchfab__8b__abc"""
    return model_path.removeprefix("raw_data/").replace("/", "__").removesuffix(".glb")


def batch_from_source(source_file: str) -> str:
    """batch_1_单丹婵_2026-03-26.csv → batch_1"""
    parts = source_file.split("_")
    return f"{parts[0]}_{parts[1]}"


def process_one(args_tuple) -> tuple[str, str]:
    """Worker: crop and save all channels for one asset."""
    row, source_root, target_root, dry_run = args_tuple
    from PIL import Image

    model_name = model_to_folder(row["model"])
    batch      = batch_from_source(row["source_file"])
    asset_dir  = Path(source_root) / batch / model_name

    if not asset_dir.exists():
        return model_name, "missing"

    out_paths = {ch: Path(target_root) / ch / f"{model_name}.png" for ch, *_ in CHANNELS}

    # Skip if all outputs already exist
    if all(p.exists() for p in out_paths.values()):
        return model_name, "skip"

    if dry_run:
        return model_name, "dry_run"

    try:
        grids: dict[str, Image.Image] = {}
        for ch, src_file, box in CHANNELS:
            if src_file not in grids:
                src_path = asset_dir / src_file
                if not src_path.exists():
                    return model_name, f"missing_{src_file}"
                grids[src_file] = Image.open(src_path)

            crop = grids[src_file].crop(box)
            out_path = out_paths[ch]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(out_path, format="PNG", optimize=False)

        return model_name, "ok"

    except Exception as e:
        return model_name, f"error:{e}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",     default=str(DEFAULT_CSV))
    p.add_argument("--target",  default=str(DEFAULT_TARGET))
    p.add_argument("--source",  default=str(DEFAULT_SOURCE))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    source_root = Path(args.source)
    target_root = Path(args.target)

    # Create output dirs
    for ch, *_ in CHANNELS:
        (target_root / ch).mkdir(parents=True, exist_ok=True)

    # Load CSV
    with open(args.csv, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} rows from {args.csv}")
    if args.dry_run:
        print("  ** DRY RUN — no files will be written **")

    tasks = [(row, str(source_root), str(target_root), args.dry_run) for row in rows]
    counts: dict[str, int] = {}

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, t): t for t in tasks}
        done = 0
        for fut in as_completed(futures):
            model_name, status = fut.result()
            key = ("error" if (status.startswith("error") or status.startswith("missing_"))
                   else status)
            counts[key] = counts.get(key, 0) + 1
            done += 1
            if done % 1000 == 0 or done == len(rows):
                print(f"  {done}/{len(rows)}  " +
                      "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    print("\nSummary:")
    for k, v in sorted(counts.items()):
        print(f"  {k:12}: {v}")

    # Write manifest CSV alongside images
    if not args.dry_run:
        manifest_path = target_root / "manifest.csv"
        score_cols = ["baseColor", "normal", "roughness", "metallic",
                      "finalScore", "tier", "pbrType", "split"]
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["model_name", "source_file"] + score_cols)
            writer.writeheader()
            for row in rows:
                model_name = model_to_folder(row["model"])
                out = {"model_name": model_name, "source_file": row["source_file"]}
                for col in score_cols:
                    out[col] = row.get(col, "")
                writer.writerow(out)
        print(f"\nManifest written → {manifest_path}")


if __name__ == "__main__":
    main()
