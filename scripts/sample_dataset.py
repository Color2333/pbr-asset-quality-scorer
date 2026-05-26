"""
Stratified dataset sampler for PBR asset quality scoring.

Reads the full annotation pool and outputs train/val/test splits with
controlled score-level balance across all 4 channels simultaneously.
Each row (asset) appears in exactly one split across all channels.

Strategy
--------
- Exclude "Incorrect (不可用)" tier (broken renders)
- For each (channel, score) bucket:
    * if available < RARE_THRESHOLD  → include ALL (don't waste rare examples)
    * else                           → cap at COMMON_CAP
- Assets are sorted by "scarcity priority" so rare-score assets are included first
- An asset is included if any of its 4 (channel, score) buckets still has capacity
- Split 80/10/10 stratified on normal_map score (rarest, most imbalanced channel)

Usage
-----
    python asset_quality_scorer/scripts/sample_dataset.py
    python asset_quality_scorer/scripts/sample_dataset.py --common-cap 10000 --rare-threshold 5000
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

CHANNELS = ["baseColor", "normal", "roughness", "metallic"]
EXCLUDE_TIERS = {"Incorrect (不可用)"}

SOURCE = Path(__file__).resolve().parents[2] / \
    "asset_quality_scorer/dataset/汇总总表_路径已更新.csv"
OUT_DIR = Path(__file__).resolve().parents[2] / "asset_quality_scorer/dataset"


# ── helpers ──────────────────────────────────────────────────────────────────

def load_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def compute_caps(rows: list[dict], rare_threshold: int, common_cap: int) -> dict:
    """caps[channel][score] = max assets to include for that bucket."""
    caps: dict[str, dict[int, int]] = {}
    for ch in CHANNELS:
        cnt = Counter(int(r[ch]) for r in rows)
        caps[ch] = {}
        for s in range(6):
            avail = cnt.get(s, 0)
            caps[ch][s] = avail if avail < rare_threshold else min(avail, common_cap)
    return caps


def scarcity_priority(row: dict, global_cnt: dict[str, Counter]) -> float:
    """Higher = rarer across channels → included first."""
    return sum(1.0 / max(global_cnt[ch][int(row[ch])], 1) for ch in CHANNELS)


def greedy_select(rows: list[dict], caps: dict, seed: int) -> list[dict]:
    """Include assets in scarcity-priority order, respecting per-bucket caps."""
    # pre-compute global counts for priority scoring
    global_cnt: dict[str, Counter] = {ch: Counter(int(r[ch]) for r in rows) for ch in CHANNELS}

    rows_sorted = sorted(rows, key=lambda r: scarcity_priority(r, global_cnt), reverse=True)

    counts: dict[str, dict[int, int]] = {ch: defaultdict(int) for ch in CHANNELS}
    selected: list[dict] = []

    for row in rows_sorted:
        scores = {ch: int(row[ch]) for ch in CHANNELS}
        # include if any bucket still has capacity
        if any(counts[ch][scores[ch]] < caps[ch][scores[ch]] for ch in CHANNELS):
            selected.append(row)
            for ch in CHANNELS:
                counts[ch][scores[ch]] += 1

    return selected


def stratified_split(rows: list[dict], val_ratio: float, test_ratio: float,
                     stratify_col: str, seed: int) -> tuple[list, list, list]:
    """Split rows stratified by stratify_col score level."""
    rng = random.Random(seed)
    by_score: dict[int, list] = defaultdict(list)
    for r in rows:
        by_score[int(r[stratify_col])].append(r)

    train_all, val_all, test_all = [], [], []
    for score in sorted(by_score):
        bucket = by_score[score][:]
        rng.shuffle(bucket)
        n = len(bucket)
        n_test = max(1, round(n * test_ratio))
        n_val  = max(1, round(n * val_ratio))
        test_all  += bucket[:n_test]
        val_all   += bucket[n_test:n_test + n_val]
        train_all += bucket[n_test + n_val:]

    return train_all, val_all, test_all


def write_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_distribution(rows: list[dict], label: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {label}  (n={len(rows)})")
    print(f"{'─'*60}")
    for ch in CHANNELS:
        cnt = Counter(int(r[ch]) for r in rows)
        row_str = "  ".join(f"{s}:{cnt.get(s,0):5d}" for s in range(6))
        print(f"  {ch:12s}  {row_str}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=str(SOURCE))
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--common-cap", type=int, default=8000,
                   help="Max assets per (channel, score) bucket for common scores")
    p.add_argument("--rare-threshold", type=int, default=5000,
                   help="If available < this, treat as rare (no cap)")
    p.add_argument("--val-ratio",  type=float, default=0.10)
    p.add_argument("--test-ratio", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    source  = Path(args.source)
    out_dir = Path(args.out_dir)

    print(f"Loading {source} …")
    all_rows = load_rows(source)
    print(f"  total rows: {len(all_rows)}")

    # ── filter ────────────────────────────────────────────────────────────────
    rows = [r for r in all_rows if r["tier"] not in EXCLUDE_TIERS]
    n_excluded = len(all_rows) - len(rows)
    print(f"  excluded (tier): {n_excluded}  →  usable: {len(rows)}")

    # ── pool stats ────────────────────────────────────────────────────────────
    print_distribution(rows, "Full usable pool")

    # ── caps ──────────────────────────────────────────────────────────────────
    caps = compute_caps(rows, args.rare_threshold, args.common_cap)
    print(f"\nCaps (rare_threshold={args.rare_threshold}, common_cap={args.common_cap}):")
    for ch in CHANNELS:
        row_str = "  ".join(f"{s}:{caps[ch][s]:5d}" for s in range(6))
        print(f"  {ch:12s}  {row_str}")

    # ── greedy selection ───────────────────────────────────────────────────────
    print("\nRunning greedy stratified selection …")
    selected = greedy_select(rows, caps, args.seed)
    print(f"  selected: {len(selected)} / {len(rows)}")

    print_distribution(selected, "Selected pool")

    # ── split ─────────────────────────────────────────────────────────────────
    train, val, test = stratified_split(
        selected, args.val_ratio, args.test_ratio,
        stratify_col="normal",   # stratify on rarest channel
        seed=args.seed,
    )
    print_distribution(train, f"Train  ({len(train)})")
    print_distribution(val,   f"Val    ({len(val)})")
    print_distribution(test,  f"Test   ({len(test)})")

    # ── write ──────────────────────────────────────────────────────────────────
    fieldnames = list(all_rows[0].keys())
    write_csv(train, out_dir / "sampled_train.csv", fieldnames)
    write_csv(val,   out_dir / "sampled_val.csv",   fieldnames)
    write_csv(test,  out_dir / "sampled_test.csv",  fieldnames)

    # combined summary table with split column
    summary_fieldnames = fieldnames + ["split"]
    for r in train: r["split"] = "train"
    for r in val:   r["split"] = "val"
    for r in test:  r["split"] = "test"
    write_csv(train + val + test, out_dir / "sampled_all.csv", summary_fieldnames)

    print(f"\nSaved to {out_dir}/")
    print(f"  sampled_train.csv  {len(train)}")
    print(f"  sampled_val.csv    {len(val)}")
    print(f"  sampled_test.csv   {len(test)}")
    print(f"  sampled_all.csv    {len(train)+len(val)+len(test)}  (含 split 列)")


if __name__ == "__main__":
    main()
