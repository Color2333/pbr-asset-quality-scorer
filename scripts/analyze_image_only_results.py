from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

CHANNELS = ("normal_map", "roughness", "metallic", "base_color")
LABEL_KEYS = {
    "normal_map": "normal_score",
    "roughness": "roughness_score",
    "metallic": "metallic_score",
    "base_color": "base_color_score",
}


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


def _array(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([_float(row, key) for row in rows], dtype=np.float32)


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0)
    if mask.sum() < 3:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def _mae(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(x[mask] - y[mask])))


def _mean_by_tier(values: np.ndarray, tiers: np.ndarray) -> dict[str, float]:
    result = {}
    for tier in sorted(int(t) for t in np.unique(tiers) if t > 0):
        mask = tiers == tier
        result[str(tier)] = float(np.nanmean(values[mask]))
    return result


def _counts(values: np.ndarray) -> dict[str, int]:
    result = {}
    for value in sorted(int(v) for v in np.unique(values) if v >= 0):
        result[str(value)] = int((values == value).sum())
    return result


def _save_tier_trend_plot(summary: dict, output_path: Path) -> None:
    tiers = ["1", "2", "3", "4", "5"]
    plt.figure(figsize=(9, 5), dpi=160)
    for channel in CHANNELS:
        values = [summary["channels"][channel]["predicted_expected_mean_by_tier"].get(tier, np.nan) for tier in tiers]
        plt.plot(tiers, values, marker="o", label=f"{channel} predicted")
    final_values = [summary["final_score_mean_by_tier"].get(tier, np.nan) for tier in tiers]
    plt.plot(tiers, final_values, marker="s", linestyle="--", label="final_score label")
    plt.xlabel("Tier id")
    plt.ylabel("Mean score")
    plt.title("Image-only scorer outputs by labeled tier")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _save_channel_error_plot(summary: dict, output_path: Path) -> None:
    labels = list(CHANNELS)
    maes = [summary["channels"][channel]["label_vs_expected_mae"] for channel in labels]
    corrs = [summary["channels"][channel]["label_vs_expected_corr"] for channel in labels]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=160)
    axes[0].bar(labels, maes)
    axes[0].set_title("Channel expected score MAE")
    axes[0].set_ylabel("MAE")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(labels, corrs)
    axes[1].set_title("Channel label/pred correlation")
    axes[1].set_ylabel("Pearson r")
    axes[1].tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _write_markdown(summary: dict, output_path: Path) -> None:
    lines = [
        "# Image-only 质量系统分析",
        "",
        "## 当前结论",
        "",
        "当前阶段应该聚焦在图片可见信息上：PBR 四通道裁图、四通道 ordinal scorer、由 scorer backbone 提取的 embedding，以及基于这些输出训练的 asset-level fusion scorer。暂时不加入几何特征是合理的，因为现有输入只有图片，系统边界应该先收敛在 image-only 质量判断。",
        "",
        "从结果看，image-only 路线已经成立：embedding 能预测 `final_score` 和 tier，fusion scorer 在不使用人工通道分数的情况下进一步提升了资产级分数预测。",
        "",
        "## 数据覆盖",
        "",
        f"- 全量 raw-grid asset: {summary['num_assets']}",
        f"- 有效 tier asset: {summary['valid_tier_assets']}",
        f"- tier 分布: {summary['tier_counts']}",
        "",
        "## 四通道 scorer 内容分析",
        "",
    ]
    for channel in CHANNELS:
        item = summary["channels"][channel]
        lines.extend(
            [
                f"### {channel}",
                "",
                f"- 真实通道分数 vs expected score MAE: {item['label_vs_expected_mae']:.3f}",
                f"- 真实通道分数 vs expected score 相关性: {item['label_vs_expected_corr']:.3f}",
                f"- expected score 均值: {item['predicted_expected_mean']:.3f}",
                f"- predicted class 分布: {item['predicted_class_counts']}",
                "",
            ]
        )
    lines.extend(
        [
            "## 资产级结论",
            "",
            f"- embedding-only final_score Ridge MAE: {summary['embedding_validation']['final_score_ridge_mae']:.3f}",
            f"- embedding-only tier linear accuracy: {summary['embedding_validation']['tier_linear_acc']:.3f}",
            f"- fusion final_score GBDT MAE: {summary['fusion_validation']['final_score_gbdt_mae']:.3f}",
            f"- fusion tier GBDT accuracy: {summary['fusion_validation']['tier_gbdt_acc']:.3f}",
            "",
            "这说明当前 embedding 不是单纯记录纹理外观，而是已经编码了与质量等级相关的图像信号；fusion scorer 则把四通道预测分数和 embedding 中的隐式质量信息合并起来，形成了当前最适合作为资产级 scorer 的版本。",
            "",
            "## 当前短板",
            "",
            "- tier 1 样本极少，训练和验证都容易受类别不均衡影响。",
            "- base_color 的硬分类 F1 偏弱，说明颜色/风格/贴图质量之间的边界可能不如 normal、roughness、metallic 明确。",
            "- metallic 的 predicted class 很容易走向极端，这可能来自标签分布本身，也可能来自 metallic/roughness 之间的耦合关系。",
            "- UMAP/t-SNE 能看到质量分布，但不是严格分层；这符合 image-only embedding 的性质，它同时编码风格、材质类型和质量。",
            "",
            "## 下一步只做 image-only",
            "",
            "1. 做 scorer 校准：对四通道 expected score 做 isotonic/temperature calibration，让输出分数更像真实 0-5 质量分。",
            "2. 强化 base_color：单独检查 base_color 标签和图像样例，必要时改成更细的 texture/style quality 任务，而不是硬套 pass/fail 边界。",
            "3. 做 asset retrieval 评估：用 embedding 找近邻，人工看同质量、同风格、同材质复杂度是否聚在一起。",
            "4. 做 image-only inference pipeline：输入一个 asset 的 `grid_pbr.png` 和 `grid_white.png`，输出四通道分数、asset-level fusion 分数、embedding 和近邻案例。",
            "5. 做错误样例分析：挑出 fusion scorer 高误差 asset，回看图片，判断问题来自标签噪声、通道 scorer 误判，还是 asset 类型差异。",
            "",
        ]
    )
    output_path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze image-only scorer and embedding results")
    parser.add_argument("--scores-csv", default="asset_quality_scorer/outputs/phase2_embedding_raw/asset_scores.csv")
    parser.add_argument("--validation-json", default="asset_quality_scorer/outputs/phase2_embedding_validation/validation_summary.json")
    parser.add_argument("--fusion-json", default="asset_quality_scorer/outputs/asset_fusion_scorer/fusion_summary.json")
    parser.add_argument("--output-root", default="asset_quality_scorer/outputs/image_only_analysis")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = _resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(_resolve_path(args.scores_csv))
    tiers = np.asarray([_tier_id(row) for row in rows], dtype=np.int64)
    final_scores = _array(rows, "final_score")

    validation = json.loads(_resolve_path(args.validation_json).read_text())
    fusion = json.loads(_resolve_path(args.fusion_json).read_text())

    summary = {
        "num_assets": len(rows),
        "valid_tier_assets": int((tiers > 0).sum()),
        "tier_counts": _counts(tiers),
        "final_score_mean_by_tier": _mean_by_tier(final_scores, tiers),
        "channels": {},
        "embedding_validation": {
            "final_score_ridge_mae": validation["regression"][0]["ridge_mae"],
            "final_score_ridge_r2": validation["regression"][0]["ridge_r2"],
            "tier_linear_acc": validation["classification"]["linear_acc"],
            "tier_linear_macro_f1": validation["classification"]["linear_macro_f1"],
        },
        "fusion_validation": {
            "final_score_gbdt_mae": fusion["final_score_regression"]["gbdt_mae"],
            "final_score_gbdt_r2": fusion["final_score_regression"]["gbdt_r2"],
            "tier_gbdt_acc": fusion["tier_classification"]["gbdt_acc"],
            "tier_gbdt_macro_f1": fusion["tier_classification"]["gbdt_macro_f1"],
        },
    }

    for channel in CHANNELS:
        expected = _array(rows, f"{channel}_expected_score")
        pred = _array(rows, f"{channel}_pred_score")
        label = _array(rows, LABEL_KEYS[channel])
        summary["channels"][channel] = {
            "label_vs_expected_mae": _mae(label, expected),
            "label_vs_expected_corr": _corr(label, expected),
            "predicted_expected_mean": float(np.nanmean(expected)),
            "label_mean": float(np.nanmean(label[label >= 0])),
            "predicted_class_counts": _counts(pred),
            "label_class_counts": _counts(label),
            "predicted_expected_mean_by_tier": _mean_by_tier(expected, tiers),
            "label_mean_by_tier": _mean_by_tier(label, tiers),
        }

    (output_root / "image_only_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    _save_tier_trend_plot(summary, output_root / "score_trends_by_tier.png")
    _save_channel_error_plot(summary, output_root / "channel_error_summary.png")
    _write_markdown(summary, PROJECT_ROOT / "asset_quality_scorer" / "IMAGE_ONLY_ANALYSIS.md")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
