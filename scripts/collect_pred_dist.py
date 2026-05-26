"""Collect prediction distributions for all channels on the test split."""
import json
import sys
import torch
import numpy as np
from collections import Counter
from pathlib import Path
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "asset_quality_scorer"))

from quality_scorer.constants import ALL_CHANNELS
from quality_scorer.convnext_regression import ConvNeXtRegressionScorer
from quality_scorer.data_v2 import CHANNEL_DEFECT_COLS, TensorCacheCLIPDataset, build_score_lookup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results = {}

for channel in ALL_CHANNELS:
    out_dir = PROJECT_ROOT / f"asset_quality_scorer/outputs/phase2_regression/convnext_base_{channel}_regression_v2"
    ckpt_path = out_dir / "best.pt"
    if not ckpt_path.exists():
        print(f"[skip] {channel}")
        continue

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    has_clip_direct = any(k.startswith("clip_direct") for k in state)
    defect_cols = ckpt.get("defect_cols", CHANNEL_DEFECT_COLS.get(channel, []))
    invalid_max_score = ckpt.get("invalid_max_score", 1)

    model = ConvNeXtRegressionScorer(
        clip_dim=1536, attn_proj_dim=256, attn_heads=4, hidden_dim=512,
        dropout=0.0, n_defect_labels=len(defect_cols),
        freeze_features=False, use_clip_direct=has_clip_direct,
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    score_by_model = build_score_lookup(PROJECT_ROOT / "screening/data_38k", channel)
    ds = TensorCacheCLIPDataset(
        tensor_cache_root=PROJECT_ROOT / "screening/cache_224_tensors",
        clip_feature_path=PROJECT_ROOT / "screening/features/clip_vitl14_openai_base_color_render.pt",
        split_image_root=PROJECT_ROOT / "screening/data_v2" / channel,
        split="test", channel=channel,
        score_by_model=score_by_model,
        invalid_max_score=invalid_max_score, is_train=False,
        manifest_path=PROJECT_ROOT / "screening/channel_store_38k/manifest.csv",
        defect_cols=defect_cols,
    )
    loader = DataLoader(ds, batch_size=256, num_workers=4, pin_memory=True)

    gt_list, pred_list = [], []
    with torch.no_grad():
        for imgs, clips, scores, _, _ in loader:
            pred, _, _ = model(imgs.to(device), clips.to(device))
            gt_list.extend(scores.tolist())
            pred_list.extend(pred.cpu().tolist())

    gt = np.array(gt_list)
    pred = np.array(pred_list)
    pred_rounded = np.clip(np.round(pred), 0, 5).astype(int)
    gt_int = gt.astype(int)

    gt_counts = dict(sorted(Counter(gt_int.tolist()).items()))
    pred_counts = dict(sorted(Counter(pred_rounded.tolist()).items()))

    print(f"{channel} (n={len(gt_list)}):")
    print(f"  GT  : {gt_counts}")
    print(f"  Pred: {pred_counts}")

    results[channel] = {
        "gt_counts": gt_counts,
        "pred_counts": pred_counts,
        "gt": gt_list,
        "pred": pred_list,
        "n": len(gt_list),
    }

out_path = PROJECT_ROOT / "asset_quality_scorer/outputs/score_dist.json"
out_path.write_text(json.dumps(results, indent=2))
print(f"\nSaved → {out_path}")
