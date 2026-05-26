from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "asset_quality_scorer"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT))

from quality_scorer.constants import ALL_CHANNELS


METRICS = (
    "val_ordinal_mae",
    "val_expected_mae",
    "val_within_1",
    "val_binary_f1",
    "val_binary_best_f1",
)


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _best_epoch(values: list[float], mode: str) -> tuple[int | None, float | None]:
    if not values:
        return None, None
    if mode == "min":
        best_idx = min(range(len(values)), key=lambda idx: values[idx])
    elif mode == "max":
        best_idx = max(range(len(values)), key=lambda idx: values[idx])
    else:
        raise ValueError(f"unsupported mode: {mode}")
    return best_idx + 1, values[best_idx]


def _load_channel_summary(output_root: Path, backbone: str, channel: str) -> dict:
    output_dir = output_root / f"convnext_{backbone}_{channel}_coral"
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        return {
            "channel": channel,
            "status": "missing",
            "output_dir": str(output_dir),
            "error": f"summary not found: {summary_path}",
        }

    with summary_path.open("r") as file:
        summary = json.load(file)
    log = summary.get("last_log", {})
    epochs = log.get("epoch", [])

    best_ord_epoch, best_ord = _best_epoch(log.get("val_ordinal_mae", []), "min")
    best_exp_epoch, best_exp = _best_epoch(log.get("val_expected_mae", []), "min")
    best_w1_epoch, best_w1 = _best_epoch(log.get("val_within_1", []), "max")
    best_f1_epoch, best_f1 = _best_epoch(log.get("val_binary_f1", []), "max")
    best_sweep_epoch, best_sweep = _best_epoch(log.get("val_binary_best_f1", []), "max")

    checkpoints = {
        name: (output_dir / name).exists()
        for name in (
            "best.pt",
            "best_ordinal_mae.pt",
            "best_within_1.pt",
            "best_binary_f1.pt",
        )
    }
    return {
        "channel": channel,
        "status": "ok",
        "output_dir": str(output_dir),
        "epochs_ran": len(epochs),
        "completed_full_schedule": len(epochs) >= 30,
        "best": summary.get("best", {}),
        "best_epochs": {
            "ordinal_mae": best_ord_epoch,
            "expected_mae": best_exp_epoch,
            "within_1": best_w1_epoch,
            "binary_f1": best_f1_epoch,
            "binary_best_f1": best_sweep_epoch,
        },
        "best_values": {
            "ordinal_mae": best_ord,
            "expected_mae": best_exp,
            "within_1": best_w1,
            "binary_f1": best_f1,
            "binary_best_f1": best_sweep,
        },
        "last_values": {
            metric: (log.get(metric, [None])[-1] if log.get(metric) else None)
            for metric in METRICS
        },
        "checkpoints": checkpoints,
    }


def _quality_flag(row: dict) -> str:
    if row["status"] != "ok":
        return "missing"
    best = row["best_values"]
    if best["expected_mae"] is not None and best["expected_mae"] <= 0.75:
        return "good"
    if best["expected_mae"] is not None and best["expected_mae"] <= 1.0:
        return "usable"
    return "weak"


def _write_markdown(report: dict, path: Path) -> None:
    rows = report["channels"]
    lines: list[str] = []
    lines.append("# Phase 1 Ordinal Scorer Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Channel | Status | Epochs | Ord MAE | Exp MAE | Within-1 | F1@0.5 | Best Sweep F1 | Flag |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        best = row.get("best_values", {})
        lines.append(
            "| {channel} | {status} | {epochs} | {ord_mae} | {exp_mae} | {within1} | {f1} | {sweep} | {flag} |".format(
                channel=row["channel"],
                status=row["status"],
                epochs=row.get("epochs_ran", 0),
                ord_mae=_fmt(best.get("ordinal_mae")),
                exp_mae=_fmt(best.get("expected_mae")),
                within1=_fmt(best.get("within_1")),
                f1=_fmt(best.get("binary_f1")),
                sweep=_fmt(best.get("binary_best_f1")),
                flag=_quality_flag(row),
            )
        )

    lines.append("")
    lines.append("## Best Epochs")
    lines.append("")
    lines.append("| Channel | Ord MAE | Exp MAE | Within-1 | F1@0.5 | Best Sweep F1 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        epochs = row.get("best_epochs", {})
        lines.append(
            "| {channel} | {ord_mae} | {exp_mae} | {within1} | {f1} | {sweep} |".format(
                channel=row["channel"],
                ord_mae=epochs.get("ordinal_mae", "-"),
                exp_mae=epochs.get("expected_mae", "-"),
                within1=epochs.get("within_1", "-"),
                f1=epochs.get("binary_f1", "-"),
                sweep=epochs.get("binary_best_f1", "-"),
            )
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- For continuous quality scoring, prefer probability expected score over argmax class.")
    lines.append("- `expected_mae` is the most relevant quick metric for continuous score calibration.")
    lines.append("- `binary_best_f1` is kept only to compare against the old pass/fail gate.")
    lines.append("- Channels flagged `weak` should still be usable for embedding features, but not trusted as standalone score heads.")
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Phase 1 ordinal scorer runs")
    parser.add_argument("--output-root", default="asset_quality_scorer/outputs/phase1_ordinal")
    parser.add_argument("--backbone", default="base")
    parser.add_argument("--json-name", default="phase1_summary.json")
    parser.add_argument("--md-name", default="PHASE1_REPORT.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = _resolve_path(args.output_root)
    rows = [_load_channel_summary(output_root, args.backbone, channel) for channel in ALL_CHANNELS]
    report = {
        "output_root": str(output_root),
        "backbone": args.backbone,
        "channels": rows,
    }

    json_path = output_root / args.json_name
    md_path = output_root / args.md_name
    output_root.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2))
    _write_markdown(report, md_path)

    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    for row in rows:
        print(
            f"{row['channel']}: status={row['status']} "
            f"epochs={row.get('epochs_ran', 0)} flag={_quality_flag(row)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
