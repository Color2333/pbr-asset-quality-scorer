from __future__ import annotations

import argparse
import csv
import html
import json
import math
import random
import re
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "asset_quality_scorer" / "outputs"

CHANNELS = ("normal_map", "roughness", "metallic", "base_color")
CHANNEL_LABELS = {
    "normal_map": "Normal Map",
    "roughness": "Roughness",
    "metallic": "Metallic",
    "base_color": "Base Color",
}
TIER_COLORS = {
    "1": "#2563eb",
    "2": "#059669",
    "3": "#ca8a04",
    "4": "#ea580c",
    "5": "#dc2626",
}


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _read_rows(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return rows if limit is None else rows[:limit]


def _float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _tier_id(value: str) -> str:
    match = re.search(r"Tier\s*(\d+)", value or "")
    return match.group(1) if match else str(value or "")


def _mae(pairs: list[tuple[float, float]]) -> float:
    return sum(abs(pred - truth) for truth, pred in pairs) / max(len(pairs), 1)


def _corr(pairs: list[tuple[float, float]]) -> float:
    if len(pairs) < 2:
        return 0.0
    truths = [truth for truth, _ in pairs]
    preds = [pred for _, pred in pairs]
    truth_mean = sum(truths) / len(truths)
    pred_mean = sum(preds) / len(preds)
    numerator = sum((truth - truth_mean) * (pred - pred_mean) for truth, pred in pairs)
    truth_den = math.sqrt(sum((truth - truth_mean) ** 2 for truth in truths))
    pred_den = math.sqrt(sum((pred - pred_mean) ** 2 for pred in preds))
    if truth_den == 0 or pred_den == 0:
        return 0.0
    return numerator / truth_den / pred_den


def _channel_average_metrics(scores_csv: Path) -> dict[str, float]:
    rows = _read_rows(scores_csv)
    pairs = []
    for row in rows:
        try:
            final_score = float(row["final_score"])
            channel_scores = [float(row[f"{channel}_expected_score"]) for channel in CHANNELS]
        except (KeyError, ValueError):
            continue
        if final_score < 0 or not all(math.isfinite(score) for score in channel_scores):
            continue
        pairs.append((final_score, sum(channel_scores) / len(channel_scores)))
    return {
        "count": float(len(pairs)),
        "mae": _mae(pairs),
        "corr": _corr(pairs),
    }


def _row_channel_average(row: dict[str, str]) -> float:
    scores = []
    for channel in CHANNELS:
        value = _float(row.get(f"{channel}_expected_score"), default=float("nan"))
        if math.isfinite(value):
            scores.append(value)
    return sum(scores) / len(scores) if scores else float("nan")


def _stat(label: str, value: str, hint: str = "") -> str:
    hint_html = f'<div class="stat-hint">{_esc(hint)}</div>' if hint else ""
    return f"""
    <section class="stat">
      <div class="stat-label">{_esc(label)}</div>
      <div class="stat-value">{_esc(value)}</div>
      {hint_html}
    </section>
    """


def _bars_svg(values: dict[str, float], title: str, y_label: str, width: int = 640, height: int = 260) -> str:
    if not values:
        return ""
    padding_left, padding_right, padding_top, padding_bottom = 52, 20, 36, 42
    chart_w = width - padding_left - padding_right
    chart_h = height - padding_top - padding_bottom
    max_v = max(max(values.values()), 1.0)
    labels = list(values.keys())
    slot = chart_w / len(labels)
    bar_w = slot * 0.62
    bars = []
    for idx, (label, value) in enumerate(values.items()):
        bar_h = chart_h * value / max_v
        x = padding_left + idx * slot + (slot - bar_w) / 2
        y = padding_top + chart_h - bar_h
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" rx="5" class="bar">'
            f"<title>{_esc(label)}: {_fmt(value, 3)}</title></rect>"
        )
        bars.append(f'<text x="{x + bar_w / 2:.1f}" y="{height - 18}" text-anchor="middle" class="axis">{_esc(label)}</text>')
        bars.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 7:.1f}" text-anchor="middle" class="value">{_fmt(value, 2)}</text>')
    return f"""
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="{_esc(title)}">
      <text x="{padding_left}" y="20" class="chart-title">{_esc(title)}</text>
      <text x="16" y="{padding_top + chart_h / 2:.1f}" transform="rotate(-90 16 {padding_top + chart_h / 2:.1f})" class="axis">{_esc(y_label)}</text>
      <line x1="{padding_left}" y1="{padding_top + chart_h}" x2="{width - padding_right}" y2="{padding_top + chart_h}" class="grid"/>
      <line x1="{padding_left}" y1="{padding_top}" x2="{padding_left}" y2="{padding_top + chart_h}" class="grid"/>
      {''.join(bars)}
    </svg>
    """


def _grouped_channel_svg(summary: dict, width: int = 720, height: int = 280) -> str:
    padding_left, padding_right, padding_top, padding_bottom = 56, 24, 38, 48
    chart_w = width - padding_left - padding_right
    chart_h = height - padding_top - padding_bottom
    channels = list(CHANNELS)
    max_v = 1.15
    slot = chart_w / len(channels)
    bar_w = slot * 0.24
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="通道 MAE 与相关性">',
        '<text x="56" y="22" class="chart-title">通道误差与相关性</text>',
        f'<text x="16" y="{padding_top + chart_h / 2:.1f}" transform="rotate(-90 16 {padding_top + chart_h / 2:.1f})" class="axis">MAE / Pearson r</text>',
        f'<line x1="{padding_left}" y1="{padding_top + chart_h}" x2="{width - padding_right}" y2="{padding_top + chart_h}" class="grid"/>',
        f'<line x1="{padding_left}" y1="{padding_top}" x2="{padding_left}" y2="{padding_top + chart_h}" class="grid"/>',
    ]
    for idx, channel in enumerate(channels):
        item = summary["channels"][channel]
        mae = item["label_vs_expected_mae"]
        corr = item["label_vs_expected_corr"]
        center = padding_left + idx * slot + slot / 2
        for offset, value, cls, label in [
            (-bar_w * 0.6, mae, "bar muted", "MAE"),
            (bar_w * 0.6, corr, "bar", "Pearson r"),
        ]:
            bar_h = chart_h * min(value, max_v) / max_v
            x = center + offset - bar_w / 2
            y = padding_top + chart_h - bar_h
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" rx="5" class="{cls}">'
                f"<title>{CHANNEL_LABELS[channel]} {label}: {_fmt(value, 3)}</title></rect>"
            )
            parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle" class="value">{_fmt(value, 2)}</text>')
        parts.append(f'<text x="{center:.1f}" y="{height - 18}" text-anchor="middle" class="axis">{_esc(CHANNEL_LABELS[channel])}</text>')
    parts.append('<circle cx="538" cy="21" r="5" class="legend-dot"/><text x="550" y="25" class="axis">Pearson r</text>')
    parts.append('<circle cx="632" cy="21" r="5" class="legend-dot muted-fill"/><text x="644" y="25" class="axis">MAE</text>')
    parts.append("</svg>")
    return "".join(parts)


def _trend_svg(summary: dict, width: int = 760, height: int = 320) -> str:
    tiers = ["1", "2", "3", "4", "5"]
    colors = {
        "normal_map": "#2563eb",
        "roughness": "#059669",
        "metallic": "#ca8a04",
        "base_color": "#7c3aed",
        "final_score": "#111827",
    }
    padding_left, padding_right, padding_top, padding_bottom = 56, 120, 38, 46
    chart_w = width - padding_left - padding_right
    chart_h = height - padding_top - padding_bottom

    def x_pos(idx: int) -> float:
        return padding_left + idx * chart_w / (len(tiers) - 1)

    def y_pos(value: float) -> float:
        return padding_top + chart_h - chart_h * value / 5.0

    series: dict[str, list[float]] = {}
    for channel in CHANNELS:
        series[channel] = [summary["channels"][channel]["predicted_expected_mean_by_tier"].get(tier, 0.0) for tier in tiers]
    series["final_score"] = [summary["final_score_mean_by_tier"].get(tier, 0.0) for tier in tiers]

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="不同 tier 的平均预测分趋势">',
        '<text x="56" y="22" class="chart-title">不同 Tier 的平均分趋势</text>',
        f'<text x="18" y="{padding_top + chart_h / 2:.1f}" transform="rotate(-90 18 {padding_top + chart_h / 2:.1f})" class="axis">平均分数 0-5</text>',
        f'<line x1="{padding_left}" y1="{padding_top + chart_h}" x2="{width - padding_right}" y2="{padding_top + chart_h}" class="grid"/>',
        f'<line x1="{padding_left}" y1="{padding_top}" x2="{padding_left}" y2="{padding_top + chart_h}" class="grid"/>',
    ]
    for idx, tier in enumerate(tiers):
        x = x_pos(idx)
        parts.append(f'<text x="{x:.1f}" y="{height - 17}" text-anchor="middle" class="axis">Tier {tier}</text>')
    for name, values in series.items():
        points = " ".join(f"{x_pos(idx):.1f},{y_pos(value):.1f}" for idx, value in enumerate(values))
        stroke = colors[name]
        parts.append(f'<polyline points="{points}" fill="none" stroke="{stroke}" stroke-width="2.4"/>')
        for idx, value in enumerate(values):
            parts.append(f'<circle cx="{x_pos(idx):.1f}" cy="{y_pos(value):.1f}" r="3.8" fill="{stroke}"><title>{_esc(name)} Tier {tiers[idx]}: {_fmt(value, 2)}</title></circle>')
    legend_y = 58
    for name in ["final_score", *CHANNELS]:
        label = "Final score" if name == "final_score" else CHANNEL_LABELS[name]
        parts.append(f'<circle cx="{width - 106}" cy="{legend_y}" r="4" fill="{colors[name]}"/><text x="{width - 94}" y="{legend_y + 4}" class="axis">{_esc(label)}</text>')
        legend_y += 22
    parts.append("</svg>")
    return "".join(parts)


def _sample_umap_points(rows: list[dict[str, str]], max_points: int) -> list[dict[str, str]]:
    buckets: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        tier = _tier_id(row.get("tier", ""))
        if tier in TIER_COLORS:
            buckets.setdefault(tier, []).append(row)
    random.seed(42)
    sampled: list[dict[str, str]] = []
    per_tier = max(1, max_points // max(len(buckets), 1))
    for tier_rows in buckets.values():
        if len(tier_rows) > per_tier:
            sampled.extend(random.sample(tier_rows, per_tier))
        else:
            sampled.extend(tier_rows)
    if len(sampled) > max_points:
        sampled = random.sample(sampled, max_points)
    return sampled


def _scatter_svg(rows: list[dict[str, str]], width: int = 760, height: int = 420) -> str:
    if not rows:
        return ""
    xs = [_float(row.get("x")) for row in rows]
    ys = [_float(row.get("y")) for row in rows]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    padding_left, padding_right, padding_top, padding_bottom = 46, 20, 40, 44
    chart_w = width - padding_left - padding_right
    chart_h = height - padding_top - padding_bottom

    def x_pos(value: float) -> float:
        return padding_left + (value - min_x) / max(max_x - min_x, 1e-6) * chart_w

    def y_pos(value: float) -> float:
        return padding_top + chart_h - (value - min_y) / max(max_y - min_y, 1e-6) * chart_h

    points = []
    for row in rows:
        tier = _tier_id(row.get("tier", ""))
        color = TIER_COLORS.get(tier, "#6b7280")
        title = f"{row.get('model_name', '')} | Tier {tier} | final_score {row.get('final_score', '')}"
        points.append(
            f'<circle cx="{x_pos(_float(row.get("x"))):.1f}" cy="{y_pos(_float(row.get("y"))):.1f}" r="2.1" fill="{color}" opacity="0.62">'
            f"<title>{_esc(title)}</title></circle>"
        )
    legend = []
    legend_x = padding_left
    for tier, color in TIER_COLORS.items():
        legend.append(f'<circle cx="{legend_x}" cy="24" r="4" fill="{color}"/><text x="{legend_x + 9}" y="28" class="axis">Tier {tier}</text>')
        legend_x += 74
    return f"""
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="UMAP embedding scatter">
      <text x="{padding_left}" y="18" class="chart-title">UMAP Embedding 空间抽样</text>
      {''.join(legend)}
      <line x1="{padding_left}" y1="{padding_top + chart_h}" x2="{width - padding_right}" y2="{padding_top + chart_h}" class="grid"/>
      <line x1="{padding_left}" y1="{padding_top}" x2="{padding_left}" y2="{padding_top + chart_h}" class="grid"/>
      {''.join(points)}
      <text x="{padding_left + chart_w / 2:.1f}" y="{height - 12}" text-anchor="middle" class="axis">UMAP x</text>
      <text x="16" y="{padding_top + chart_h / 2:.1f}" transform="rotate(-90 16 {padding_top + chart_h / 2:.1f})" class="axis">UMAP y</text>
    </svg>
    """


def _channel_rows(phase1: dict, image_summary: dict) -> str:
    rows = []
    phase_by_channel = {item["channel"]: item for item in phase1["channels"]}
    notes = {
        "normal_map": "最稳定；与 tier / final score 趋势一致。",
        "roughness": "相关性最高；是当前最强质量信号。",
        "metallic": "有信息量，但更容易给出 0/5 极端预测。",
        "base_color": "连续分数可用；硬分类边界较弱。",
    }
    for channel in CHANNELS:
        phase = phase_by_channel[channel]["best_values"]
        item = image_summary["channels"][channel]
        rows.append(
            "<tr>"
            f"<td>{_esc(CHANNEL_LABELS[channel])}</td>"
            f"<td>{_fmt(phase['expected_mae'])}</td>"
            f"<td>{_fmt(phase['binary_best_f1'])}</td>"
            f"<td>{_fmt(item['label_vs_expected_mae'])}</td>"
            f"<td>{_fmt(item['label_vs_expected_corr'])}</td>"
            f"<td>{_esc(notes[channel])}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _error_rows(rows: list[dict[str, str]]) -> str:
    rendered = []
    for row in rows:
        rendered.append(
            "<tr>"
            f"<td class=\"mono\">{_esc(row.get('model_name', ''))}</td>"
            f"<td>{_esc(row.get('batch', ''))}</td>"
            f"<td>Tier {_esc(row.get('tier', ''))}</td>"
            f"<td>Tier {_esc(row.get('pred_tier', ''))}</td>"
            f"<td>{_fmt(_float(row.get('final_score')), 2)}</td>"
            f"<td>{_fmt(_float(row.get('pred_final_score')), 2)}</td>"
            f"<td>{_fmt(_float(row.get('abs_final_error')), 2)}</td>"
            "</tr>"
        )
    return "\n".join(rendered)


def _metric_explain_cards() -> str:
    items = [
        (
            "Final Score MAE",
            "预测的最终质量分和人工分数平均差多少。越低越好；0.33 表示平均差约三分之一分。",
        ),
        (
            "Tier Accuracy",
            "把资产分到 Tier 1-5 的准确率。越高越好；当前比只猜多数类明显更好。",
        ),
        (
            "Pearson r",
            "模型分数和人工分数的趋势一致性。越接近 1 越说明排序趋势可靠。",
        ),
        (
            "Expected Score",
            "通道模型输出的 0-5 连续质量分，不只是硬判好/坏。",
        ),
    ]
    return "".join(
        f"""
        <div class="explain-card">
          <strong>{_esc(title)}</strong>
          <span>{_esc(text)}</span>
        </div>
        """
        for title, text in items
    )


def _channel_cards(phase1: dict, image_summary: dict) -> str:
    phase_by_channel = {item["channel"]: item for item in phase1["channels"]}
    notes = {
        "normal_map": ("稳定可用", "最容易从图片判断，和人工质量趋势一致。"),
        "roughness": ("最强信号", "与人工通道分相关性最高，是当前最可靠的质量线索。"),
        "metallic": ("需要校准", "有信息量，但预测容易极端化，0/5 边界要继续处理。"),
        "base_color": ("只适合参考", "连续趋势存在，但硬分类较弱，不适合单独做最终裁判。"),
    }
    cards = []
    for channel in CHANNELS:
        phase = phase_by_channel[channel]["best_values"]
        item = image_summary["channels"][channel]
        title, note = notes[channel]
        cards.append(
            f"""
            <article class="channel-card">
              <div class="channel-head">
                <span>{_esc(CHANNEL_LABELS[channel])}</span>
                <b>{_esc(title)}</b>
              </div>
              <p>{_esc(note)}</p>
              <div class="mini-metrics">
                <div><strong>{_fmt(item['label_vs_expected_corr'])}</strong><span>Pearson r</span></div>
                <div><strong>{_fmt(item['label_vs_expected_mae'])}</strong><span>全量 MAE</span></div>
                <div><strong>{_fmt(phase['binary_best_f1'])}</strong><span>Best F1</span></div>
              </div>
            </article>
            """
        )
    return "".join(cards)


def _verdict_card(fusion: dict) -> str:
    mae = fusion["final_score_regression"]["gbdt_mae"]
    baseline_mae = fusion["final_score_regression"]["baseline_mae"]
    acc = fusion["tier_classification"]["gbdt_acc"]
    baseline_acc = fusion["tier_classification"]["baseline_acc"]
    mae_gain = (baseline_mae - mae) / baseline_mae
    acc_gain = acc - baseline_acc
    return f"""
    <div class="verdict">
      <div>
        <span class="eyebrow">当前效果</span>
        <h2>已经能做 image-only 质量预筛，但还不是完整人工评审替代</h2>
        <p>它擅长判断图片里可见的 PBR 完整度、纹理强度、法线和材质响应；不擅长判断资产是否重复低质、风格是否合规、整体是否真的可用。</p>
      </div>
      <div class="verdict-metrics">
        <div><strong>{_fmt(mae)}</strong><span>Final score MAE，比 baseline 降低 {_pct(mae_gain)}</span></div>
        <div><strong>{_pct(acc)}</strong><span>Tier accuracy，比 baseline 高 {_pct(acc_gain)}</span></div>
      </div>
    </div>
    """


def _error_case_cards(rows: list[dict[str, str]], limit: int = 6) -> str:
    cards = []
    for row in rows[:limit]:
        channel_average = _row_channel_average(row)
        cards.append(
            f"""
            <article class="error-card">
              <div class="mono">{_esc(row.get('model_name', ''))}</div>
              <div class="error-line">
                <span>均分 {_fmt(channel_average, 2)}</span>
                <span>真实 {_fmt(_float(row.get('final_score')), 2)}</span>
                <span>预测 {_fmt(_float(row.get('pred_final_score')), 2)}</span>
                <strong>误差 {_fmt(_float(row.get('abs_final_error')), 2)}</strong>
              </div>
              <p>真实 Tier {_esc(row.get('tier', ''))}，预测 Tier {_esc(row.get('pred_tier', ''))}。这类样本通常说明模型看到了完整贴图信号，但没有理解人工 final score 中的综合可用性因素。</p>
            </article>
            """
        )
    return "".join(cards)


def _copy_case_images(row: dict[str, str], image_root: Path, output_dir: Path, index: int) -> list[str]:
    model_dir = image_root / row["batch"] / row["model_name"]
    copied = []
    for filename in ("grid_pbr.png", "grid_white.png"):
        path = model_dir / filename
        if not path.exists():
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"case_{index:02d}_{filename}"
        shutil.copy2(path, output_path)
        copied.append(str(output_path.relative_to(OUTPUT_ROOT)))
    return copied


def _boundary_case_cards(rows: list[dict[str, str]], image_root: Path, limit: int = 5) -> str:
    output_dir = OUTPUT_ROOT / "image_only_error_cases" / "boundary_cases"
    cards = []
    for index, row in enumerate(rows[:limit], start=1):
        image_paths = _copy_case_images(row, image_root, output_dir, index)
        if not image_paths:
            continue
        images_html = "".join(f'<img src="{_esc(src)}" alt="boundary case {index} image">' for src in image_paths)
        channel_average = _row_channel_average(row)
        cards.append(
            f"""
            <figure class="case-card">
              <div class="case-images">{images_html}</div>
              <figcaption>
                <strong>案例 {index}</strong>
                通道均分 {_fmt(channel_average, 2)}，真实分 {_fmt(_float(row.get('final_score')), 2)}，预测分 {_fmt(_float(row.get('pred_final_score')), 2)}，误差 {_fmt(_float(row.get('abs_final_error')), 2)}。
              </figcaption>
            </figure>
            """
        )
    return "".join(cards)


def build_dashboard(args: argparse.Namespace) -> str:
    phase1 = _read_json(_resolve_path(args.phase1_summary))
    image_summary = _read_json(_resolve_path(args.image_summary))
    fusion = _read_json(_resolve_path(args.fusion_summary))
    validation = _read_json(_resolve_path(args.validation_summary))
    error_summary = _read_json(_resolve_path(args.error_summary))
    error_rows = _read_rows(_resolve_path(args.error_csv), limit=args.error_rows)
    umap_rows = _sample_umap_points(_read_rows(_resolve_path(args.umap_csv)), args.umap_points)
    image_root = _resolve_path(args.image_root)
    channel_avg = _channel_average_metrics(_resolve_path(args.scores_csv))

    tier_counts = {f"Tier {tier}": count for tier, count in image_summary["tier_counts"].items()}
    stats = "\n".join(
        [
            _stat("资产覆盖", f"{image_summary['valid_tier_assets']:,}", "有有效人工 tier 的资产数"),
            _stat("最终分误差", _fmt(fusion["final_score_regression"]["gbdt_mae"]), "平均差约 0.33 分，越低越好"),
            _stat("通道均分误差", _fmt(channel_avg["mae"]), "四个通道 expected score 直接平均"),
            _stat("Tier 准确率", _pct(fusion["tier_classification"]["gbdt_acc"]), "预测 Tier 1-5 的准确率"),
        ]
    )
    asset_metric_bars = _bars_svg(
        {
            "只猜均值": fusion["final_score_regression"]["baseline_mae"],
            "四通道均分": channel_avg["mae"],
            "只用 embedding": fusion["final_score_regression"]["ridge_mae"],
            "当前 fusion": fusion["final_score_regression"]["gbdt_mae"],
        },
        "最终质量分误差：越低越好",
        "MAE",
    )
    tier_acc_bars = _bars_svg(
        {
            "只猜多数类": fusion["tier_classification"]["baseline_acc"],
            "线性模型": fusion["tier_classification"]["linear_acc"],
            "当前 fusion": fusion["tier_classification"]["gbdt_acc"],
        },
        "Tier 预测准确率：越高越好",
        "Accuracy",
    )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Image-only Asset Quality Scorer Demo</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #111827;
      --muted: #667085;
      --line: #d9dee7;
      --accent: #2563eb;
      --accent-soft: #dbeafe;
      --good: #059669;
      --warn: #ca8a04;
      --bad: #dc2626;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.55;
    }}
    .page {{ max-width: 1180px; margin: 0 auto; padding: 34px 28px 58px; }}
    .hero {{
      display: grid;
      grid-template-columns: minmax(0, 1.45fr) minmax(320px, 0.55fr);
      gap: 24px;
      align-items: stretch;
      margin-bottom: 24px;
    }}
    h1 {{ font-size: 34px; line-height: 1.16; margin: 0 0 14px; letter-spacing: -0.035em; }}
    h2 {{ font-size: 20px; margin: 0 0 14px; letter-spacing: -0.02em; }}
    h3 {{ font-size: 15px; margin: 0 0 10px; }}
    p {{ margin: 0 0 12px; color: var(--muted); }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 24px;
    }}
    .intro {{ padding: 34px; }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 5px 10px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      font-size: 13px;
      font-weight: 650;
      margin-bottom: 18px;
    }}
    .takeaway {{
      margin-top: 16px;
      padding: 14px 16px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fafbfc;
      color: var(--text);
      font-weight: 600;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin: 0 0 26px;
    }}
    .stat {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px;
    }}
    .stat-label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; }}
    .stat-value {{ font-size: 26px; font-weight: 760; margin-top: 4px; letter-spacing: -0.03em; }}
    .stat-hint {{ color: var(--muted); font-size: 12px; margin-top: 4px; }}
    .grid-2 {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 20px; margin-bottom: 24px; }}
    .grid-3 {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 18px; margin-bottom: 24px; }}
    .flow {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-top: 18px;
    }}
    .step {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
      background: #fafbfc;
    }}
    .step strong {{ display: block; font-size: 14px; margin-bottom: 5px; }}
    .step span {{ color: var(--muted); font-size: 13px; }}
    .verdict {{
      display: grid;
      grid-template-columns: minmax(0, 1.25fr) minmax(280px, 0.75fr);
      gap: 22px;
      align-items: center;
      margin-bottom: 24px;
      padding: 28px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
    }}
    .eyebrow {{ color: var(--accent); font-size: 12px; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; }}
    .verdict h2 {{ font-size: 26px; line-height: 1.24; margin: 8px 0 10px; }}
    .verdict-metrics {{ display: grid; gap: 12px; }}
    .verdict-metrics div {{ border: 1px solid var(--line); border-radius: 16px; padding: 16px; background: #fafbfc; }}
    .verdict-metrics strong {{ display: block; font-size: 30px; line-height: 1; margin-bottom: 6px; }}
    .verdict-metrics span {{ color: var(--muted); font-size: 13px; }}
    .use-list {{ display: grid; gap: 12px; margin-top: 18px; }}
    .use-item {{ display: grid; grid-template-columns: 32px 1fr; gap: 12px; align-items: start; }}
    .use-item b {{ width: 32px; height: 32px; border-radius: 50%; display: grid; place-items: center; background: var(--accent-soft); color: var(--accent); }}
    .use-item strong {{ display: block; margin-bottom: 2px; }}
    .use-item span {{ color: var(--muted); font-size: 13px; }}
    .explain-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
    .explain-card {{ border: 1px solid var(--line); border-radius: 14px; padding: 14px; background: #fafbfc; }}
    .explain-card strong {{ display: block; margin-bottom: 6px; }}
    .explain-card span {{ color: var(--muted); font-size: 13px; }}
    .channel-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }}
    .channel-card {{ border: 1px solid var(--line); border-radius: 16px; padding: 16px; background: #ffffff; }}
    .channel-head {{ display: flex; justify-content: space-between; gap: 12px; align-items: baseline; margin-bottom: 8px; }}
    .channel-head span {{ font-weight: 800; }}
    .channel-head b {{ color: var(--accent); font-size: 12px; }}
    .channel-card p {{ font-size: 13px; min-height: 42px; }}
    .mini-metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 12px; }}
    .mini-metrics div {{ border-top: 1px solid var(--line); padding-top: 8px; }}
    .mini-metrics strong {{ display: block; font-size: 16px; }}
    .mini-metrics span {{ color: var(--muted); font-size: 11px; }}
    .error-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }}
    .error-card {{ border: 1px solid var(--line); border-radius: 14px; padding: 14px; background: #fafbfc; }}
    .error-line {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0; font-size: 12px; }}
    .error-line span, .error-line strong {{ background: white; border: 1px solid var(--line); border-radius: 999px; padding: 3px 8px; }}
    .error-card p {{ font-size: 12px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 10px 9px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-size: 12px; font-weight: 700; }}
    tr:last-child td {{ border-bottom: none; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; word-break: break-all; }}
    .chart-title {{ font-size: 15px; font-weight: 700; fill: var(--text); }}
    .axis {{ font-size: 11px; fill: var(--muted); }}
    .value {{ font-size: 10px; fill: var(--muted); }}
    .grid {{ stroke: var(--line); stroke-width: 1; }}
    .bar {{ fill: var(--accent); }}
    .bar.muted {{ fill: #94a3b8; }}
    .legend-dot {{ fill: var(--accent); }}
    .muted-fill {{ fill: #94a3b8; }}
    svg {{ width: 100%; height: auto; display: block; }}
    .image-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
    figure {{ margin: 0; }}
    figure img {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: white;
      display: block;
    }}
    figcaption {{ color: var(--muted); font-size: 12px; margin-top: 8px; }}
    .boundary-grid {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 14px; }}
    .case-card {{ border: 1px solid var(--line); border-radius: 16px; overflow: hidden; background: #ffffff; }}
    .case-images {{ display: grid; grid-template-columns: 1fr; gap: 1px; background: var(--line); }}
    .case-card img {{ width: 100%; border: none; border-radius: 0; display: block; background: white; }}
    .case-card figcaption {{ padding: 12px; margin: 0; }}
    .case-card figcaption strong {{ display: block; color: var(--text); margin-bottom: 4px; }}
    .contact img {{ max-height: 640px; object-fit: contain; }}
    .small-note {{ color: var(--muted); font-size: 12px; margin-top: 10px; }}
    .footer {{ margin-top: 24px; color: var(--muted); font-size: 12px; }}
    @media (max-width: 980px) {{
      .hero, .grid-2, .grid-3, .image-grid, .verdict, .channel-grid, .explain-grid, .error-grid, .boundary-grid {{ grid-template-columns: 1fr; }}
      .stats {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .flow {{ grid-template-columns: 1fr 1fr; }}
    }}
    @media (max-width: 620px) {{
      .page {{ padding: 18px 14px 40px; }}
      .stats, .flow {{ grid-template-columns: 1fr; }}
      table {{ font-size: 12px; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <div class="panel intro">
        <div class="badge">Image-only 3D Asset Quality Scorer</div>
        <h1>用两张渲染网格图，给 3D 资产做自动质量预筛</h1>
        <p>这个页面只回答三个问题：它用来做什么、现在效果怎么样、哪里还不可靠。底层输入是每个资产的 `grid_pbr.png` 和 `grid_white.png`，系统从中裁出 PBR 通道，预测通道质量和资产级质量。</p>
        <p class="takeaway">一句话结论：已经适合作为大规模资产库的 image-only 预筛和排序信号；不适合作为最终人工验收的唯一裁判。</p>
      </div>
      <div class="panel">
        <h2>它的用途</h2>
        <div class="use-list">
          <div class="use-item"><b>1</b><div><strong>批量预筛</strong><span>在人工看之前，先把明显 PBR/贴图质量差的资产排出来。</span></div></div>
          <div class="use-item"><b>2</b><div><strong>质量排序</strong><span>给大量资产一个 image-only 分数，方便抽样、分层和优先级排序。</span></div></div>
          <div class="use-item"><b>3</b><div><strong>错误审计</strong><span>找出模型高估/低估样本，反过来检查标签噪声和模型盲区。</span></div></div>
          <div class="use-item"><b>4</b><div><strong>相似资产检索</strong><span>embedding 可以支持后续按质量、风格、材质复杂度找近邻。</span></div></div>
        </div>
      </div>
    </section>

    {_verdict_card(fusion)}

    <section class="stats">{stats}</section>

    <section class="grid-2">
      <div class="panel">
        <h2>效果 1：最终质量分更接近人工分</h2>
        {asset_metric_bars}
        <p class="small-note">这里的 MAE 是“预测 final score 与人工 final score 的平均绝对误差”。四通道直接平均已经能降到 {_fmt(channel_avg["mae"])}，说明通道分本身有质量信号；当前 fusion 进一步降到 {_fmt(fusion["final_score_regression"]["gbdt_mae"])}。</p>
      </div>
      <div class="panel">
        <h2>效果 2：Tier 分层明显优于基线</h2>
        {tier_acc_bars}
        <p class="small-note">Tier accuracy 是“预测 Tier 1-5 是否命中人工 tier”。当前 fusion 达到 {_pct(fusion["tier_classification"]["gbdt_acc"])}，说明它不是随机或只学到数据分布。</p>
      </div>
    </section>

    <section class="panel">
      <h2>这些指标到底是什么意思</h2>
      <div class="explain-grid">{_metric_explain_cards()}</div>
      <p class="small-note">四通道均分的 Pearson r 是 {_fmt(channel_avg["corr"])}，说明它和人工 final score 的趋势已经比较一致；fusion 的价值是学习不同通道和视觉 embedding 的非线性组合，而不是简单平均。</p>
    </section>

    <section class="panel">
      <h2>四个通道分别贡献了什么</h2>
      <p>系统不是直接看一张图给总分，而是先理解 PBR 的几个关键通道。这样更容易知道模型为什么判断一个资产好或差。</p>
      <div class="channel-grid">{_channel_cards(phase1, image_summary)}</div>
    </section>

    <section class="grid-2">
      <div class="panel">
        <h2>效果 3：通道分数随人工 Tier 下降</h2>
        {_trend_svg(image_summary)}
        <p class="small-note">Tier 数字越大表示质量越低。如果模型真的学到质量信号，曲线应该从 Tier 1 到 Tier 5 逐步下降。这里整体趋势成立。</p>
      </div>
      <div class="panel">
        <h2>效果 4：Embedding 有质量结构</h2>
        {_scatter_svg(umap_rows)}
        <p class="small-note">这是从 6144 维视觉 embedding 降到 2D 的抽样图。不同 tier 有区域性分布，说明 embedding 不只是记录外观，也包含质量相关信息。</p>
      </div>
    </section>

    <section class="panel">
      <h2>边界样例：直接看 5 个最容易高估的资产</h2>
      <p>下面每个案例都直接展示原始 `grid_pbr.png` 和 `grid_white.png`。这些样本通常看起来贴图完整、PBR 信号丰富，但人工最终分低，因此能暴露 image-only scorer 的边界。</p>
      <div class="boundary-grid">{_boundary_case_cards(error_rows, image_root, limit=5)}</div>
    </section>

    <section class="panel">
      <h2>这些案例说明的问题</h2>
      <p>主要失败模式是“高估”：模型看到 normal / roughness / metallic / base color 都像是完整贴图，就倾向于给高分；但人工 final score 还包含资产是否真正可用、是否重复低质、风格是否合格等综合判断。</p>
      <div class="error-grid">{_error_case_cards(error_rows, limit=3)}</div>
      <p class="small-note">Top-K 平均绝对误差 {_fmt(error_summary["mean_top_k_abs_final_error"])}，最大误差 {_fmt(error_summary["max_abs_final_error"])}。这些案例适合用来做下一轮校准和标签复审。</p>
    </section>

    <section class="grid-2">
      <div class="panel">
        <h2>数据分布背景</h2>
        {_bars_svg(tier_counts, "有效 Tier 样本数", "资产数")}
        <p class="small-note">Tier 4/5 数量最多，Tier 1 极少。所以高质量端的效果更需要继续用人工抽样确认。</p>
      </div>
      <div class="panel">
        <h2>下一步怎么做最有价值</h2>
        <div class="use-list">
          <div class="use-item"><b>1</b><div><strong>分数校准</strong><span>让输出的 0-5 分更贴近人工分布，尤其是 metallic / base_color。</span></div></div>
          <div class="use-item"><b>2</b><div><strong>错误分组</strong><span>把高误差样本分成标签噪声、模型高估、图片信息不足等类别。</span></div></div>
          <div class="use-item"><b>3</b><div><strong>上线预筛 demo</strong><span>输入单个资产目录，输出四通道分、总分、近邻案例和风险解释。</span></div></div>
        </div>
      </div>
    </section>

    <div class="footer">Generated by `asset_quality_scorer/scripts/build_dashboard.py` from current outputs. 图表为页面内 SVG 重绘，contact sheet 来自错误样例导出。</div>
  </main>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a static HTML dashboard for asset_quality_scorer outputs.")
    parser.add_argument("--phase1-summary", default="asset_quality_scorer/outputs/phase1_ordinal/phase1_summary.json")
    parser.add_argument("--image-summary", default="asset_quality_scorer/outputs/image_only_analysis/image_only_summary.json")
    parser.add_argument("--fusion-summary", default="asset_quality_scorer/outputs/asset_fusion_scorer/fusion_summary.json")
    parser.add_argument("--validation-summary", default="asset_quality_scorer/outputs/phase2_embedding_validation/validation_summary.json")
    parser.add_argument("--error-summary", default="asset_quality_scorer/outputs/image_only_error_cases/error_case_summary.json")
    parser.add_argument("--error-csv", default="asset_quality_scorer/outputs/image_only_error_cases/top_final_score_errors.csv")
    parser.add_argument("--umap-csv", default="asset_quality_scorer/outputs/phase2_embedding_viz/umap/umap_coords.csv")
    parser.add_argument("--scores-csv", default="asset_quality_scorer/outputs/phase2_embedding_raw/asset_scores.csv")
    parser.add_argument("--image-root", default="screening/data_38k")
    parser.add_argument("--output", default="asset_quality_scorer/outputs/dashboard.html")
    parser.add_argument("--error-rows", type=int, default=20)
    parser.add_argument("--umap-points", type=int, default=2500)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_dashboard(args), encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
