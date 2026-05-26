from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

CHANNELS = ("normal_map", "roughness", "metallic", "base_color")


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _float_value(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, ""))
    except ValueError:
        return np.nan


def _tier_id(row: dict[str, str]) -> int:
    match = re.search(r"Tier\s*(\d+)", row.get("tier", ""))
    return int(match.group(1)) if match else -1


def _score_features(rows: list[dict[str, str]]) -> np.ndarray:
    features = []
    for row in rows:
        channel_expected = [_float_value(row, f"{channel}_expected_score") for channel in CHANNELS]
        channel_pred = [_float_value(row, f"{channel}_pred_score") for channel in CHANNELS]
        features.append(channel_expected + channel_pred)
    return np.asarray(features, dtype=np.float32)


def _valid_mask(y_final: np.ndarray, y_tier: np.ndarray, score_features: np.ndarray) -> np.ndarray:
    mask = np.isfinite(y_final) & (y_final >= 0) & (y_tier > 0)
    mask &= np.isfinite(score_features).all(axis=1)
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
        selected.extend(rng.choice(indices, size=min(per_tier, len(indices)), replace=False).tolist())
    remaining = sample_size - len(selected)
    if remaining > 0:
        pool = np.setdiff1d(np.arange(n), np.asarray(selected), assume_unique=False)
        selected.extend(rng.choice(pool, size=min(remaining, len(pool)), replace=False).tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def _embedding_pipeline(pca_dims: int, seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=pca_dims, random_state=seed)),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight asset-level fusion scorer")
    parser.add_argument("--embedding-npz", default="asset_quality_scorer/outputs/phase2_embedding_raw/asset_embeddings.npz")
    parser.add_argument("--scores-csv", default="asset_quality_scorer/outputs/phase2_embedding_raw/asset_scores.csv")
    parser.add_argument("--output-root", default="asset_quality_scorer/outputs/asset_fusion_scorer")
    parser.add_argument("--sample-size", type=int, default=30000)
    parser.add_argument("--pca-dims", type=int, default=128)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = _resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    embeddings_data = np.load(_resolve_path(args.embedding_npz))
    embeddings = embeddings_data["embeddings"].astype("float32")
    rows = _read_rows(_resolve_path(args.scores_csv))
    if embeddings.shape[0] != len(rows):
        raise RuntimeError(f"row count mismatch: embeddings={embeddings.shape[0]} rows={len(rows)}")

    score_features = _score_features(rows)
    y_final = np.asarray([_float_value(row, "final_score") for row in rows], dtype=np.float32)
    y_tier = np.asarray([_tier_id(row) for row in rows], dtype=np.int64)
    mask = _valid_mask(y_final, y_tier, score_features)
    embeddings = embeddings[mask]
    score_features = score_features[mask]
    y_final = y_final[mask]
    y_tier = y_tier[mask]

    sampled = _sample_indices(y_tier, args.sample_size, args.seed)
    embeddings = embeddings[sampled]
    score_features = score_features[sampled]
    y_final = y_final[sampled]
    y_tier = y_tier[sampled]

    train_idx, test_idx = train_test_split(
        np.arange(len(y_tier)),
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y_tier,
    )

    pca_pipeline = _embedding_pipeline(args.pca_dims, args.seed)
    embedding_train = pca_pipeline.fit_transform(embeddings[train_idx])
    embedding_test = pca_pipeline.transform(embeddings[test_idx])
    fusion_train = np.concatenate([embedding_train, score_features[train_idx]], axis=1)
    fusion_test = np.concatenate([embedding_test, score_features[test_idx]], axis=1)

    reg_baseline = np.full_like(y_final[test_idx], fill_value=float(np.mean(y_final[train_idx])), dtype=np.float32)
    ridge = Ridge(alpha=10.0)
    ridge.fit(fusion_train, y_final[train_idx])
    ridge_pred = ridge.predict(fusion_test)
    gbdt_reg = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, l2_regularization=0.05, random_state=args.seed)
    gbdt_reg.fit(fusion_train, y_final[train_idx])
    gbdt_pred = gbdt_reg.predict(fusion_test)

    cls_baseline = np.full_like(y_tier[test_idx], fill_value=int(np.bincount(y_tier[train_idx]).argmax()))
    linear_cls = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=4)
    linear_cls.fit(fusion_train, y_tier[train_idx])
    linear_pred = linear_cls.predict(fusion_test)
    gbdt_cls = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, l2_regularization=0.05, random_state=args.seed)
    gbdt_cls.fit(fusion_train, y_tier[train_idx])
    gbdt_cls_pred = gbdt_cls.predict(fusion_test)

    summary = {
        "valid_assets": int(mask.sum()),
        "sampled_assets": int(len(y_tier)),
        "embedding_dim": int(embeddings_data["embeddings"].shape[1]),
        "pca_dims": int(args.pca_dims),
        "fusion_dim": int(fusion_train.shape[1]),
        "test_size": float(args.test_size),
        "seed": int(args.seed),
        "final_score_regression": {
            "baseline_mae": float(mean_absolute_error(y_final[test_idx], reg_baseline)),
            "ridge_mae": float(mean_absolute_error(y_final[test_idx], ridge_pred)),
            "ridge_r2": float(r2_score(y_final[test_idx], ridge_pred)),
            "gbdt_mae": float(mean_absolute_error(y_final[test_idx], gbdt_pred)),
            "gbdt_r2": float(r2_score(y_final[test_idx], gbdt_pred)),
        },
        "tier_classification": {
            "baseline_acc": float(accuracy_score(y_tier[test_idx], cls_baseline)),
            "linear_acc": float(accuracy_score(y_tier[test_idx], linear_pred)),
            "linear_macro_f1": float(f1_score(y_tier[test_idx], linear_pred, average="macro")),
            "gbdt_acc": float(accuracy_score(y_tier[test_idx], gbdt_cls_pred)),
            "gbdt_macro_f1": float(f1_score(y_tier[test_idx], gbdt_cls_pred, average="macro")),
        },
        "outputs": {
            "model": str(output_root / "fusion_scorer.pkl"),
            "summary": str(output_root / "fusion_summary.json"),
        },
    }

    bundle = {
        "pca_pipeline": pca_pipeline,
        "ridge_regressor": ridge,
        "gbdt_regressor": gbdt_reg,
        "linear_classifier": linear_cls,
        "gbdt_classifier": gbdt_cls,
        "channels": CHANNELS,
        "score_feature_keys": [f"{channel}_expected_score" for channel in CHANNELS]
        + [f"{channel}_pred_score" for channel in CHANNELS],
        "summary": summary,
    }
    with (output_root / "fusion_scorer.pkl").open("wb") as file:
        pickle.dump(bundle, file)
    (output_root / "fusion_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
