"""Pre-resize channel images into uint8 memmap tensors for fast training.

Images are stored as (N, 3, H, W) uint8 memmaps, one file per channel,
ordered by the rows in the source CSV.

Each (model, channel) pair is an independent task so all workers stay busy.
Uses ProcessPoolExecutor to bypass GIL for PNG decode.

Usage:
    python asset_quality_scorer/scripts/cache_tensors.py
    python asset_quality_scorer/scripts/cache_tensors.py --size 448
    python asset_quality_scorer/scripts/cache_tensors.py --channels render base_color metallic roughness --size 224
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

_IN_TTY = sys.stderr.isatty()


def _tqdm(it, **kw):
    return tqdm(it, disable=not _IN_TTY, **kw)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CSV        = PROJECT_ROOT / "asset_quality_scorer/dataset/sampled_all.csv"
DEFAULT_IMAGE_ROOT = PROJECT_ROOT / "datasets0526"
DEFAULT_OUTPUT_TPL = PROJECT_ROOT / "asset_quality_scorer/cache/{size}"

ALL_CHANNELS = ["render", "base_color", "roughness", "metallic",
                "normal_map", "white_model", "white_with_normal"]


def _model_to_name(model_path: str) -> str:
    """raw_data/sketchfab/8b/abc.glb → sketchfab__8b__abc"""
    return model_path.removeprefix("raw_data/").replace("/", "__").removesuffix(".glb")


def _resize_one(args_tuple) -> tuple[int, str, np.ndarray]:
    """Worker: load one (model, channel) PNG, resize, return (idx, channel, CHW uint8)."""
    idx, model_name, image_root, channel, size = args_tuple
    path = Path(image_root) / channel / f"{model_name}.png"
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8)
    return idx, channel, np.transpose(arr, (2, 0, 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",         default=str(DEFAULT_CSV))
    p.add_argument("--image-root",  default=str(DEFAULT_IMAGE_ROOT))
    p.add_argument("--output",      default=None,
                   help="Output dir. Defaults to asset_quality_scorer/cache/{size}")
    p.add_argument("--channels",    nargs="+", default=ALL_CHANNELS)
    p.add_argument("--size",        type=int, default=224)
    p.add_argument("--workers",     type=int, default=64)
    p.add_argument("--max-pending", type=int, default=256,
                   help="cap in-flight tasks to bound main-process memory (backpressure)")
    args = p.parse_args()

    output = Path(args.output) if args.output else \
        Path(str(DEFAULT_OUTPUT_TPL).format(size=args.size))
    output.mkdir(parents=True, exist_ok=True)

    with open(args.csv, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    channels = list(dict.fromkeys(args.channels))
    N = len(rows)
    total_tasks = N * len(channels)

    print(f"Rows     : {N}")
    print(f"Channels : {', '.join(channels)}")
    print(f"Tasks    : {total_tasks}  ({N} models × {len(channels)} channels)")
    print(f"Workers  : {args.workers}")
    print(f"Size     : {args.size}x{args.size}")
    print(f"Output   : {output}")

    arrays = {
        ch: np.lib.format.open_memmap(
            output / f"{ch}.npy", mode="w+", dtype=np.uint8, shape=(N, 3, args.size, args.size)
        )
        for ch in channels
    }

    # One task per (model, channel) — maximises parallelism
    model_names = [_model_to_name(row["model"]) for row in rows]
    tasks = [
        (i, model_names[i], args.image_root, ch, args.size)
        for i in range(N)
        for ch in channels
    ]

    # Backpressure: cap in-flight tasks so completed-but-unwritten result arrays
    # cannot pile up in the main process faster than the (single-threaded, network-FS)
    # writes drain them — submitting all 344k at once OOM-kills the parent. See
    # screening/cache_224_tensors.py --max-pending for the original pattern.
    ok = err = 0
    max_pending = max(args.max_pending, args.workers)
    task_iter = iter(tasks)
    log_every = 20000
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending: dict = {}
        for _ in range(max_pending):
            t = next(task_iter, None)
            if t is None:
                break
            pending[pool.submit(_resize_one, t)] = t
        with _tqdm(None, total=total_tasks, unit="img", ncols=80) as pbar:
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    t = pending.pop(fut)
                    try:
                        idx, channel, arr = fut.result()
                        arrays[channel][idx] = arr
                        ok += 1
                        if not _IN_TTY and ok % log_every == 0:
                            print(f"  ... {ok}/{total_tasks} written ({100*ok//total_tasks}%)", flush=True)
                    except Exception as e:
                        err += 1
                        if err <= 5:
                            print(f"  [error] idx={t[0]} ch={t[3]}: {e}")
                    pbar.update(1)
                    nt = next(task_iter, None)
                    if nt is not None:
                        pending[pool.submit(_resize_one, nt)] = nt

    for arr in arrays.values():
        arr.flush()

    meta = {
        "csv": str(args.csv),
        "image_root": str(args.image_root),
        "rows": N,
        "channels": channels,
        "size": args.size,
        "dtype": "uint8",
        "layout": "NCHW",
        "model_names": model_names,
    }
    (output / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nDone  ok={ok}  err={err}")
    print(f"Saved tensor cache -> {output}")


if __name__ == "__main__":
    main()
