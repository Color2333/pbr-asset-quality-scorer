from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

CHANNELS = ("normal_map", "roughness", "metallic", "base_color")


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return np.nan


def _tier_id(row: dict[str, str]) -> int:
    match = re.search(r"Tier\s*(\d+)", row.get("tier", ""))
    return int(match.group(1)) if match else -1


def _score_features(rows: list[dict[str, str]]) -> np.ndarray:
    features = []
    for row in rows:
        expected = [_float(row, f"{channel}_expected_score") for channel in CHANNELS]
        pred = [_float(row, f"{channel}_pred_score") for channel in CHANNELS]
        features.append(expected + pred)
    return np.asarray(features, dtype=np.float32)


def _make_tile(row: dict[str, str], pred_score: float, error: float, image_root: Path, tile_size: tuple[int, int]) -> Image.Image:
    tile_w, tile_h = tile_size
    model_dir = image_root / row["batch"] / row["model_name"]
    images = []
    for filename in ("grid_pbr.png", "grid_white.png"):
        path = model_dir / filename
        if path.exists():
            image = Image.open(path).convert("RGB")
            image.thumbnail((tile_w // 2, tile_h - 42))
            canvas = Image.new("RGB", (tile_w // 2, tile_h - 42), "white")
            canvas.paste(image, ((canvas.width - image.width) // 2, (canvas.height - image.height) // 2))
            images.append(canvas)
    if len(images) == 2:
        top = Image.new("RGB", (tile_w, tile_h - 42), "white")
        top.paste(images[0], (0, 0))
        top.paste(images[1], (tile_w // 2, 0))
    else:
        top = Image.new("RGB", (tile_w, tile_h - 42), "white")

    tile = Image.new("RGB", (tile_w, tile_h), "white")
    tile.paste(top, (0, 0))
    draw = ImageDraw.Draw(tile)
    final_score = _float(row, "final_score")
    text = f"GT {final_score:.2f} | Pred {pred_score:.2f} | Err {error:.2f}"
    draw.text((6, tile_h - 38), text, fill=(0, 0, 0))
    draw.text((6, tile_h - 20), row["model_name"][:42], fill=(0, 0, 0))
    return tile


def _save_contact_sheet(cases: list[dict], image_root: Path, output_path: Path, max_images: int) -> None:
    selected = cases[:max_images]
    if not selected:
        return
    columns = 4
    tile_size = (360, 250)
    rows = int(np.ceil(len(selected) / columns))
    sheet = Image.new("RGB", (columns * tile_size[0], rows * tile_size[1]), "white")
    for idx, case in enumerate(selected):
        tile = _make_tile(case["row"], case["pred_final_score"], case["abs_final_error"], image_root, tile_size)
        x = (idx % columns) * tile_size[0]
        y = (idx // columns) * tile_size[1]
        sheet.paste(tile, (x, y))
    sheet.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export high-error image-only fusion cases")
    parser.add_argument("--embedding-npz", default="asset_quality_scorer/outputs/phase2_embedding_raw/asset_embeddings.npz")
    parser.add_argument("--scores-csv", default="asset_quality_scorer/outputs/phase2_embedding_raw/asset_scores.csv")
    parser.add_argument("--fusion-model", default="asset_quality_scorer/outputs/asset_fusion_scorer/fusion_scorer.pkl")
    parser.add_argument("--image-root", default="screening/data_38k")
    parser.add_argument("--output-root", default="asset_quality_scorer/outputs/image_only_error_cases")
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--contact-sheet-images", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = _resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(_resolve_path(args.scores_csv))
    embeddings = np.load(_resolve_path(args.embedding_npz))["embeddings"].astype("float32")
    with _resolve_path(args.fusion_model).open("rb") as file:
        bundle = pickle.load(file)

    score_features = _score_features(rows)
    valid = np.isfinite(score_features).all(axis=1)
    valid &= np.asarray([np.isfinite(_float(row, "final_score")) and _float(row, "final_score") >= 0 for row in rows])
    valid &= np.asarray([_tier_id(row) > 0 for row in rows])

    valid_indices = np.where(valid)[0]
    valid_embeddings = embeddings[valid]
    valid_rows = [rows[int(idx)] for idx in valid_indices]
    embedding_features = bundle["pca_pipeline"].transform(valid_embeddings)
    fusion_features = np.concatenate([embedding_features, score_features[valid]], axis=1)
    pred_final = bundle["gbdt_regressor"].predict(fusion_features)
    pred_tier = bundle["gbdt_classifier"].predict(fusion_features)

    cases = []
    for local_idx, row in enumerate(valid_rows):
        final_score = _float(row, "final_score")
        tier = _tier_id(row)
        case = {
            "row": row,
            "model_name": row["model_name"],
            "batch": row["batch"],
            "tier": tier,
            "final_score": final_score,
            "pred_final_score": float(pred_final[local_idx]),
            "abs_final_error": float(abs(pred_final[local_idx] - final_score)),
            "pred_tier": int(pred_tier[local_idx]),
            "tier_error": int(abs(pred_tier[local_idx] - tier)),
        }
        cases.append(case)
    cases.sort(key=lambda item: item["abs_final_error"], reverse=True)

    csv_path = output_root / "top_final_score_errors.csv"
    with csv_path.open("w", newline="") as file:
        fieldnames = [
            "model_name",
            "batch",
            "tier",
            "pred_tier",
            "tier_error",
            "final_score",
            "pred_final_score",
            "abs_final_error",
            "normal_map_expected_score",
            "roughness_expected_score",
            "metallic_expected_score",
            "base_color_expected_score",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases[: args.top_k]:
            row = case["row"]
            writer.writerow({key: case.get(key, row.get(key, "")) for key in fieldnames})

    _save_contact_sheet(
        cases,
        _resolve_path(args.image_root),
        output_root / "top_final_score_errors_contact_sheet.jpg",
        args.contact_sheet_images,
    )

    summary = {
        "valid_assets": int(valid.sum()),
        "top_k": int(args.top_k),
        "max_abs_final_error": float(cases[0]["abs_final_error"]),
        "mean_top_k_abs_final_error": float(np.mean([case["abs_final_error"] for case in cases[: args.top_k]])),
        "outputs": {
            "csv": str(csv_path),
            "contact_sheet": str(output_root / "top_final_score_errors_contact_sheet.jpg"),
        },
    }
    (output_root / "error_case_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
