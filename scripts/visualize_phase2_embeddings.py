from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scores(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _sample_indices(rows: list[dict[str, str]], sample_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(rows)
    if sample_size <= 0 or sample_size >= n:
        return np.arange(n)

    by_tier: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        tier = row.get("tier") or "unknown"
        by_tier.setdefault(tier, []).append(idx)

    per_tier = max(1, sample_size // max(1, len(by_tier)))
    selected: list[int] = []
    leftovers: list[int] = []
    for indices in by_tier.values():
        indices_array = np.asarray(indices)
        if len(indices_array) <= per_tier:
            selected.extend(indices_array.tolist())
        else:
            selected.extend(rng.choice(indices_array, size=per_tier, replace=False).tolist())
            leftovers.extend(np.setdiff1d(indices_array, np.asarray(selected), assume_unique=False).tolist())

    remaining = sample_size - len(selected)
    if remaining > 0 and leftovers:
        selected.extend(rng.choice(np.asarray(leftovers), size=min(remaining, len(leftovers)), replace=False).tolist())

    return np.asarray(sorted(set(selected)), dtype=np.int64)


def _reduce_embeddings(embeddings: np.ndarray, method: str, seed: int, pca_dims: int) -> np.ndarray:
    scaled = StandardScaler().fit_transform(embeddings)
    if pca_dims > 0 and scaled.shape[1] > pca_dims:
        scaled = PCA(n_components=pca_dims, random_state=seed).fit_transform(scaled)

    if method == "umap":
        import umap

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=30,
            min_dist=0.1,
            metric="euclidean",
            random_state=seed,
        )
        return reducer.fit_transform(scaled)

    reducer = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=30,
        random_state=seed,
        verbose=1,
    )
    return reducer.fit_transform(scaled)


def _score_to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _save_plot(coords: np.ndarray, rows: list[dict[str, str]], output_path: Path, title: str) -> None:
    tiers = np.asarray([_score_to_float(row.get("tier", "")) for row in rows])
    final_scores = np.asarray([_score_to_float(row.get("final_score", "")) for row in rows])
    color_values = tiers if np.isfinite(tiers).any() else final_scores
    color_label = "tier" if np.isfinite(tiers).any() else "final_score"

    plt.figure(figsize=(10, 8), dpi=160)
    scatter = plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=color_values,
        s=6,
        alpha=0.72,
        cmap="viridis",
        linewidths=0,
    )
    plt.title(title)
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    cbar = plt.colorbar(scatter)
    cbar.set_label(color_label)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Phase 2 asset embeddings")
    parser.add_argument("--embedding-npz", default="asset_quality_scorer/outputs/phase2_embedding_raw/asset_embeddings.npz")
    parser.add_argument("--scores-csv", default="asset_quality_scorer/outputs/phase2_embedding_raw/asset_scores.csv")
    parser.add_argument("--output-root", default="asset_quality_scorer/outputs/phase2_embedding_viz")
    parser.add_argument("--method", choices=("umap", "tsne"), default="umap")
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument("--pca-dims", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    embedding_npz = _resolve_path(args.embedding_npz)
    scores_csv = _resolve_path(args.scores_csv)
    output_root = _resolve_path(args.output_root) / args.method
    output_root.mkdir(parents=True, exist_ok=True)

    embeddings_data = np.load(embedding_npz)
    embeddings = embeddings_data["embeddings"]
    model_names = embeddings_data["model_names"]
    rows = _read_scores(scores_csv)
    if len(rows) != embeddings.shape[0]:
        raise RuntimeError(f"row count mismatch: scores={len(rows)} embeddings={embeddings.shape[0]}")

    indices = _sample_indices(rows, args.sample_size, args.seed)
    sampled_embeddings = embeddings[indices].astype("float32")
    sampled_rows = [rows[int(idx)] for idx in indices]
    sampled_names = model_names[indices]

    print(f"method={args.method} total={embeddings.shape[0]} sampled={len(indices)} dim={embeddings.shape[1]}")
    coords = _reduce_embeddings(sampled_embeddings, args.method, args.seed, args.pca_dims)

    coord_csv = output_root / f"{args.method}_coords.csv"
    with coord_csv.open("w", newline="") as file:
        fieldnames = ["model_name", "x", "y"] + list(sampled_rows[0].keys())
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(sampled_rows):
            writer.writerow(
                {
                    "model_name": str(sampled_names[idx]),
                    "x": float(coords[idx, 0]),
                    "y": float(coords[idx, 1]),
                    **row,
                }
            )

    plot_path = output_root / f"{args.method}_by_tier.png"
    _save_plot(coords, sampled_rows, plot_path, f"{args.method.upper()} of Phase 2 Asset Embeddings")

    manifest = {
        "method": args.method,
        "total_assets": int(embeddings.shape[0]),
        "sampled_assets": int(len(indices)),
        "input_dim": int(embeddings.shape[1]),
        "pca_dims": int(args.pca_dims),
        "seed": int(args.seed),
        "outputs": {"coords": str(coord_csv), "plot": str(plot_path)},
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
