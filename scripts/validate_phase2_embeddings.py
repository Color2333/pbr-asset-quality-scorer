from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

CHANNEL_TARGETS = {
    "base_color": "base_color_score",
    "normal_map": "normal_score",
    "roughness": "roughness_score",
    "metallic": "metallic_score",
}


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_scores(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _float_column(rows: list[dict[str, str]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        try:
            values.append(float(row.get(key, "")))
        except ValueError:
            values.append(np.nan)
    return np.asarray(values, dtype=np.float32)


def _tier_ids(rows: list[dict[str, str]]) -> np.ndarray:
    ids = []
    for row in rows:
        match = re.search(r"Tier\s*(\d+)", row.get("tier", ""))
        ids.append(int(match.group(1)) if match else -1)
    return np.asarray(ids, dtype=np.int64)


def _valid_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(len(arrays[0]), dtype=bool)
    for array in arrays:
        mask &= np.isfinite(array)
        mask &= array >= 0
    return mask


def _sample_indices(y_tier: np.ndarray, sample_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(y_tier)
    if sample_size <= 0 or sample_size >= n:
        return np.arange(n)

    selected: list[int] = []
    per_tier = max(1, sample_size // max(1, len(np.unique(y_tier))))
    for tier in sorted(np.unique(y_tier)):
        indices = np.where(y_tier == tier)[0]
        take = min(per_tier, len(indices))
        selected.extend(rng.choice(indices, size=take, replace=False).tolist())

    remaining = sample_size - len(selected)
    if remaining > 0:
        pool = np.setdiff1d(np.arange(n), np.asarray(selected), assume_unique=False)
        selected.extend(rng.choice(pool, size=min(remaining, len(pool)), replace=False).tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def _prepare_features(embeddings: np.ndarray, pca_dims: int, seed: int) -> np.ndarray:
    scaled = StandardScaler().fit_transform(embeddings)
    if pca_dims > 0 and scaled.shape[1] > pca_dims:
        return PCA(n_components=pca_dims, random_state=seed).fit_transform(scaled)
    return scaled


def _regression_probe(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    target: str,
) -> dict:
    mean_pred = np.full_like(y_test, fill_value=float(np.mean(y_train)), dtype=np.float32)
    ridge = Ridge(alpha=10.0)
    ridge.fit(x_train, y_train)
    ridge_pred = ridge.predict(x_test)
    knn = KNeighborsRegressor(n_neighbors=15, weights="distance", metric="cosine")
    knn.fit(x_train, y_train)
    knn_pred = knn.predict(x_test)
    return {
        "target": target,
        "baseline_mae": float(mean_absolute_error(y_test, mean_pred)),
        "ridge_mae": float(mean_absolute_error(y_test, ridge_pred)),
        "ridge_r2": float(r2_score(y_test, ridge_pred)),
        "knn_mae": float(mean_absolute_error(y_test, knn_pred)),
        "knn_r2": float(r2_score(y_test, knn_pred)),
    }


def _classification_probe(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(x_train, y_train)
    dummy_pred = dummy.predict(x_test)

    linear = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=4)
    linear.fit(x_train, y_train)
    linear_pred = linear.predict(x_test)

    knn = KNeighborsClassifier(n_neighbors=15, weights="distance", metric="cosine")
    knn.fit(x_train, y_train)
    knn_pred = knn.predict(x_test)

    return {
        "target": "tier_id",
        "baseline_acc": float(accuracy_score(y_test, dummy_pred)),
        "linear_acc": float(accuracy_score(y_test, linear_pred)),
        "linear_macro_f1": float(f1_score(y_test, linear_pred, average="macro")),
        "knn_acc": float(accuracy_score(y_test, knn_pred)),
        "knn_macro_f1": float(f1_score(y_test, knn_pred, average="macro")),
    }


def _neighbor_consistency(x: np.ndarray, final_score: np.ndarray, tier_id: np.ndarray, k: int) -> dict:
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(x)
    indices = nn.kneighbors(x, return_distance=False)[:, 1:]
    final_delta = np.abs(final_score[:, None] - final_score[indices]).mean(axis=1)
    tier_delta = np.abs(tier_id[:, None] - tier_id[indices]).mean(axis=1)
    tier_match = (tier_id[:, None] == tier_id[indices]).mean(axis=1)
    return {
        "k": int(k),
        "mean_neighbor_final_score_delta": float(final_delta.mean()),
        "mean_neighbor_tier_delta": float(tier_delta.mean()),
        "mean_neighbor_tier_match": float(tier_match.mean()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantitatively validate Phase 2 embeddings")
    parser.add_argument("--embedding-npz", default="asset_quality_scorer/outputs/phase2_embedding_raw/asset_embeddings.npz")
    parser.add_argument("--scores-csv", default="asset_quality_scorer/outputs/phase2_embedding_raw/asset_scores.csv")
    parser.add_argument("--output-root", default="asset_quality_scorer/outputs/phase2_embedding_validation")
    parser.add_argument("--sample-size", type=int, default=20000)
    parser.add_argument("--pca-dims", type=int, default=128)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = _resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    embeddings_data = np.load(_resolve_path(args.embedding_npz))
    embeddings = embeddings_data["embeddings"]
    rows = _read_scores(_resolve_path(args.scores_csv))
    if len(rows) != embeddings.shape[0]:
        raise RuntimeError(f"row count mismatch: scores={len(rows)} embeddings={embeddings.shape[0]}")

    y_final = _float_column(rows, "final_score")
    y_tier = _tier_ids(rows)
    mask = _valid_mask(y_final, y_tier)
    embeddings = embeddings[mask].astype("float32")
    y_final = y_final[mask]
    y_tier = y_tier[mask]
    filtered_rows = [row for row, keep in zip(rows, mask) if keep]

    sampled = _sample_indices(y_tier, args.sample_size, args.seed)
    embeddings = embeddings[sampled]
    y_final = y_final[sampled]
    y_tier = y_tier[sampled]
    sampled_rows = [filtered_rows[int(idx)] for idx in sampled]

    x = _prepare_features(embeddings, args.pca_dims, args.seed)
    train_idx, test_idx = train_test_split(
        np.arange(len(x)),
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y_tier,
    )
    x_train, x_test = x[train_idx], x[test_idx]

    regressions = [_regression_probe(x_train, x_test, y_final[train_idx], y_final[test_idx], "final_score")]
    for channel, target_key in CHANNEL_TARGETS.items():
        y_channel = _float_column(sampled_rows, target_key)
        valid = _valid_mask(y_channel)
        channel_train = train_idx[valid[train_idx]]
        channel_test = test_idx[valid[test_idx]]
        if len(channel_train) == 0 or len(channel_test) == 0:
            continue
        regressions.append(
            _regression_probe(
                x[channel_train],
                x[channel_test],
                y_channel[channel_train],
                y_channel[channel_test],
                f"{channel}_score",
            )
        )

    classification = _classification_probe(x_train, x_test, y_tier[train_idx], y_tier[test_idx])
    neighbors = _neighbor_consistency(x, y_final, y_tier, k=15)
    tier_counts = {str(int(tier)): int((y_tier == tier).sum()) for tier in sorted(np.unique(y_tier))}

    summary = {
        "total_assets": int(embeddings_data["embeddings"].shape[0]),
        "valid_assets": int(mask.sum()),
        "sampled_assets": int(len(x)),
        "input_dim": int(embeddings_data["embeddings"].shape[1]),
        "pca_dims": int(args.pca_dims),
        "test_size": float(args.test_size),
        "seed": int(args.seed),
        "tier_counts": tier_counts,
        "regression": regressions,
        "classification": classification,
        "neighbor_consistency": neighbors,
    }

    (output_root / "validation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    with (output_root / "regression_metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(regressions[0].keys()))
        writer.writeheader()
        writer.writerows(regressions)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
