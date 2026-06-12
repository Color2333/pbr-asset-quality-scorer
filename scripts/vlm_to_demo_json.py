"""Convert a VLM-scorer run's preds.npy into the demo_predictions.json schema
that scripts/demo.py reads. VLM preds.npy holds {channel: array} in the exact
order build_items("test") yields, so names+GT are rebuilt deterministically.

Usage:
    python asset_quality_scorer/scripts/vlm_to_demo_json.py \
        --run vlm_scorer_a_old50k_oldtest --exp-label "Qwen2.5-VL (best)"
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "scripts"))
from vlm_scorer_eval import build_items, CHANNELS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir under outputs/runs/")
    ap.add_argument("--exp-label", default=None, help="display label (defaults to run name)")
    ap.add_argument("--csv", default=None); ap.add_argument("--data-root", default=None)
    args = ap.parse_args()
    run_dir = PKG / "outputs/runs" / args.run
    preds = np.load(run_dir / "preds.npy", allow_pickle=True).item()
    rows = build_items("test", csv_path=args.csv, data_root=args.data_root)
    n = len(rows)
    assert len(preds[CHANNELS[0]]) == n, f"len mismatch {len(preds[CHANNELS[0]])} vs {n}"

    srcc = {ch: round(float(spearmanr(preds[ch], [r['scores'][ch] for r in rows]).statistic), 4)
            for ch in CHANNELS}
    srcc_mean = round(float(np.mean(list(srcc.values()))), 4)
    assets = [{"name": rows[i]["name"],
               "pred": {ch: round(float(preds[ch][i]), 2) for ch in CHANNELS},
               "gt":   {ch: int(rows[i]["scores"][ch]) for ch in CHANNELS}}
              for i in range(n)]
    out = {"exp_id": args.exp_label or args.run, "split": "test",
           "srcc": srcc, "srcc_mean": srcc_mean, "n": n, "assets": assets}
    (run_dir / "demo_predictions.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"per-channel SRCC: {srcc}  mean={srcc_mean}")
    print(f"wrote {run_dir}/demo_predictions.json  ({n} assets)")


if __name__ == "__main__":
    main()
