"""Build a case-study HTML visualization using original data_v2 PNG images."""
from __future__ import annotations
import base64, json, sys
from io import BytesIO
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "asset_quality_scorer"))

from quality_scorer.constants import ALL_CHANNELS
from quality_scorer.convnext_regression import ConvNeXtRegressionScorer
from quality_scorer.data_v2 import CHANNEL_DEFECT_COLS, TensorCacheCLIPDataset, build_score_lookup

try:
    from PIL import Image
except ImportError:
    import subprocess as _sp, sys as _sys
    _sp.check_call([_sys.executable, "-m", "pip", "install", "pillow", "-q"])
    from PIL import Image


DATA_V2_ROOT = PROJECT_ROOT / "screening" / "data_v2"


def find_image(channel: str, name: str, split: str = "test") -> Path | None:
    """Find original PNG in data_v2/{channel}/{split}/{valid|invalid}/{name}.png."""
    base = DATA_V2_ROOT / channel / split
    for label in ("valid", "invalid"):
        p = base / label / f"{name}.png"
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def img_to_b64(path: Path, size: int = 200) -> str:
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def pick_cases(records: list[dict], n_best: int = 5, n_worst: int = 5) -> list[dict]:
    """For each GT score: best-predicted + worst-predicted. Plus global top errors."""
    cases: list[dict] = []
    seen: set[str] = set()

    by_score: dict[int, list] = {}
    for r in records:
        by_score.setdefault(r["gt"], []).append(r)

    for s in range(6):
        pool = sorted(by_score.get(s, []), key=lambda r: abs(r["pred"] - r["gt"]))
        for r in pool[:n_best]:
            if r["name"] not in seen:
                cases.append({**r, "case_type": "correct"})
                seen.add(r["name"])
        for r in pool[-n_worst:]:
            if r["name"] not in seen:
                cases.append({**r, "case_type": "error"})
                seen.add(r["name"])

    # Add global largest errors not yet included
    for r in sorted(records, key=lambda r: -abs(r["pred"] - r["gt"]))[:16]:
        if r["name"] not in seen:
            cases.append({**r, "case_type": "top_error"})
            seen.add(r["name"])

    return cases


def run_channel(channel: str, device: torch.device) -> list[dict]:
    out_dir = PROJECT_ROOT / f"asset_quality_scorer/outputs/phase2_regression/convnext_base_{channel}_regression_v2"
    ckpt = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    defect_cols = ckpt.get("defect_cols", CHANNEL_DEFECT_COLS.get(channel, []))
    has_clip = any(k.startswith("clip_direct") for k in state)

    model = ConvNeXtRegressionScorer(
        clip_dim=1536, attn_proj_dim=256, attn_heads=4, hidden_dim=512,
        dropout=0.0, n_defect_labels=len(defect_cols),
        freeze_features=False, use_clip_direct=has_clip,
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
        invalid_max_score=ckpt.get("invalid_max_score", 1),
        is_train=False,
        manifest_path=PROJECT_ROOT / "screening/channel_store_38k/manifest.csv",
        defect_cols=defect_cols,
    )
    loader = DataLoader(ds, batch_size=256, num_workers=4, pin_memory=True)
    names = [name for name, _ in ds.samples]

    all_preds, all_gt = [], []
    with torch.no_grad():
        for imgs, clips, scores, _, _ in loader:
            p, _, _ = model(imgs.to(device), clips.to(device))
            all_preds.extend(p.cpu().tolist())
            all_gt.extend(scores.tolist())

    records = [
        {"name": n, "gt": int(g), "pred": float(p)}
        for n, g, p in zip(names, all_gt, all_preds)
    ]
    cases = pick_cases(records, n_best=6, n_worst=6)

    # Attach images from data_v2 originals
    found, missing = 0, 0
    for c in cases:
        img_path = find_image(channel, c["name"])
        if img_path:
            c["img_b64"] = img_to_b64(img_path)
            found += 1
        else:
            c["img_b64"] = ""
            missing += 1

    print(f"  images: {found} found, {missing} missing")
    return cases


def build_html(all_cases: dict[str, list[dict]], out_path: Path) -> None:
    metrics = {
        "normal_map": dict(mae=0.5942, srcc=0.7688, acc=0.5825, w1=89.6, qwk=0.7511),
        "roughness":  dict(mae=0.5341, srcc=0.8280, acc=0.5761, w1=96.1, qwk=0.8094),
        "metallic":   dict(mae=0.6257, srcc=0.8162, acc=0.6355, w1=85.8, qwk=0.8034),
        "base_color": dict(mae=0.4351, srcc=0.7463, acc=0.6674, w1=97.5, qwk=0.7322),
    }
    ch_label = {
        "normal_map": "Normal Map", "roughness": "Roughness",
        "metallic": "Metallic",     "base_color": "Base Color",
    }
    score_colors = ["#ef4444", "#f97316", "#eab308", "#22c55e", "#3b82f6", "#a855f7"]

    def score_badge(s: int) -> str:
        c = score_colors[max(0, min(s, 5))]
        return (f'<span style="background:{c};color:#fff;padding:2px 8px;'
                f'border-radius:999px;font-size:13px;font-weight:700">{s}</span>')

    def case_card(c: dict) -> str:
        gt, pred = c["gt"], c["pred"]
        err = pred - gt
        err_str = f"{err:+.2f}"
        border = "#4ade80" if abs(err) < 0.5 else "#f97316" if abs(err) < 1.5 else "#ef4444"
        err_col = "#4ade80" if abs(err) < 0.5 else "#f87171"
        type_labels = {"correct": "✓ 正确", "error": "✗ 错误", "top_error": "⚠ 大错误"}
        type_cols   = {"correct": "#4ade80", "error": "#f97316", "top_error": "#ef4444"}
        type_label = type_labels.get(c["case_type"], "")
        type_col   = type_cols.get(c["case_type"], "#888")
        pred_rounded = score_badge(int(round(min(max(pred, 0), 5))))
        if c["img_b64"]:
            img_html = (f'<img src="data:image/jpeg;base64,{c["img_b64"]}" '
                        f'style="width:100%;aspect-ratio:1;object-fit:cover;'
                        f'border-radius:7px 7px 0 0;display:block">')
        else:
            img_html = ('<div style="width:100%;aspect-ratio:1;background:#1a1f2e;border-radius:7px 7px 0 0;'
                        'display:flex;align-items:center;justify-content:center;color:#475569">no img</div>')
        name = c["name"]
        short_name = ("…" + name[-30:]) if len(name) > 30 else name
        lines = [
            f'<div style="background:#1e2330;border-radius:8px;border:2px solid {border};',
            'overflow:hidden;width:200px;flex-shrink:0">',
            img_html,
            '<div style="padding:9px 11px">',
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">',
            f'<span style="font-size:12px;color:#94a3b8">GT</span>{score_badge(gt)}',
            f'<span style="font-size:12px;color:#94a3b8">预测</span>{pred_rounded}',
            '</div>',
            f'<div style="font-size:12px;color:#64748b">',
            f'连续值: <span style="color:#e2e8f0">{pred:.2f}</span>',
            f' <span style="color:{err_col};font-weight:600">{err_str}</span>',
            '</div>',
            f'<div style="font-size:10px;color:#374151;margin-top:4px;white-space:nowrap;',
            f'overflow:hidden;text-overflow:ellipsis" title="{name}">{short_name}</div>',
            f'<div style="font-size:11px;color:{type_col};margin-top:3px;font-weight:500">{type_label}</div>',
            '</div></div>',
        ]
        return "".join(lines)

    channel_sections: dict[str, str] = {}
    for ch in ["normal_map", "roughness", "metallic", "base_color"]:
        cases = all_cases.get(ch, [])
        m = metrics[ch]

        by_score_correct: dict[int, list] = {}
        by_score_error:   dict[int, list] = {}
        for c in cases:
            if c["case_type"] == "correct":
                by_score_correct.setdefault(c["gt"], []).append(c)
            elif c["case_type"] in ("error", "top_error"):
                by_score_error.setdefault(c["gt"], []).append(c)

        score_rows_parts = []
        for s in range(6):
            correct = by_score_correct.get(s, [])
            errors  = by_score_error.get(s, [])
            if not correct and not errors:
                continue

            correct_html = "".join(case_card(c) for c in correct)
            errors_html  = "".join(case_card(c) for c in errors)

            row = f'<div style="margin-bottom:24px">'
            row += (f'<div style="font-size:13px;color:#64748b;margin-bottom:10px;'
                    f'font-weight:600;text-transform:uppercase;letter-spacing:.5px">'
                    f'GT Score = {s}'
                    f'</div>')

            if correct_html:
                row += (f'<div style="font-size:11px;color:#4ade80;margin-bottom:6px;'
                        f'font-weight:500">✓ 预测准确（误差 &lt; 0.5）</div>'
                        f'<div style="display:flex;flex-wrap:wrap;gap:10px;margin-bottom:12px">'
                        f'{correct_html}</div>')
            if errors_html:
                row += (f'<div style="font-size:11px;color:#f87171;margin-bottom:6px;'
                        f'font-weight:500">✗ 预测偏差</div>'
                        f'<div style="display:flex;flex-wrap:wrap;gap:10px">'
                        f'{errors_html}</div>')
            row += '</div>'
            score_rows_parts.append(row)

        score_rows = "".join(score_rows_parts)
        srcc_str = f'{m["srcc"]:.3f}'
        mae_str  = f'{m["mae"]:.3f}'
        acc_str  = f'{m["acc"]*100:.1f}%'
        w1_str   = f'{m["w1"]:.1f}%'
        qwk_str  = f'{m["qwk"]:.3f}'
        chlabel  = ch_label[ch]

        channel_sections[ch] = (
            f'<div style="background:#141824;border-radius:14px;padding:24px 26px;'
            f'border:1px solid #2a3142;margin-bottom:28px">'
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;flex-wrap:wrap">'
            f'<h2 style="font-size:19px;color:#f1f5f9;font-weight:700">{chlabel}</h2>'
            f'<span style="font-size:12px;background:#1e3a5f;color:#60a5fa;padding:3px 10px;border-radius:999px">SRCC {srcc_str}</span>'
            f'<span style="font-size:12px;background:#3b1d0a;color:#fb923c;padding:3px 10px;border-radius:999px">MAE {mae_str}</span>'
            f'<span style="font-size:12px;background:#1a3a2a;color:#4ade80;padding:3px 10px;border-radius:999px">Acc {acc_str}</span>'
            f'<span style="font-size:12px;background:#2a1a3a;color:#a78bfa;padding:3px 10px;border-radius:999px">QWK {qwk_str}</span>'
            f'<span style="font-size:12px;background:#1e2330;color:#94a3b8;padding:3px 10px;border-radius:999px">within_1 {w1_str}</span>'
            f'</div>'
            f'<p style="font-size:12px;color:#475569;margin-bottom:20px">'
            f'边框：<span style="color:#4ade80">█ 误差&lt;0.5</span> '
            f'<span style="color:#f97316">█ &lt;1.5</span> '
            f'<span style="color:#ef4444">█ ≥1.5</span></p>'
            f'{score_rows}'
            f'</div>'
        )

    tabs_html = (
        '<div class="tabs">'
        '<div class="tab active" onclick="show(\'normal_map\',this)">Normal Map</div>'
        '<div class="tab" onclick="show(\'roughness\',this)">Roughness</div>'
        '<div class="tab" onclick="show(\'metallic\',this)">Metallic</div>'
        '<div class="tab" onclick="show(\'base_color\',this)">Base Color</div>'
        '</div>'
    )
    sections_html = "".join(
        f'<div class="{"section active" if i == 0 else "section"}" id="{ch}">'
        f'{channel_sections[ch]}</div>'
        for i, ch in enumerate(["normal_map", "roughness", "metallic", "base_color"])
    )

    html = """<!DOCTYPE html>
<html lang="zh"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PBR Scorer — 案例可视化</title>
<style>
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:#0f1117; color:#e2e8f0;
         font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; padding:24px; }
  h1 { text-align:center; padding:16px 0 6px; font-size:22px; color:#f8fafc; }
  .sub { text-align:center; color:#64748b; font-size:13px; margin-bottom:24px; }
  .tabs { display:flex; gap:8px; margin-bottom:22px; }
  .tab { padding:8px 20px; border-radius:8px; cursor:pointer; font-size:13px;
         background:#1e2330; border:1px solid #2a3142; color:#94a3b8; transition:all .15s; }
  .tab.active { background:#1e3a5f; color:#60a5fa; border-color:#3b82f6; font-weight:600; }
  .tab:hover { color:#e2e8f0; }
  .section { display:none; }
  .section.active { display:block; }
</style>
</head><body>
<h1>PBR Quality Scorer — 案例可视化</h1>
<p class="sub">测试集原始图像 · 每个分值展示预测准确和预测偏差的案例</p>
""" + tabs_html + "\n" + sections_html + """
<script>
function show(id, el) {
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  el.classList.add('active');
}
</script>
</body></html>"""

    out_path.write_text(html, encoding="utf-8")
    size_kb = out_path.stat().st_size // 1024
    print(f"Saved -> {out_path}  ({size_kb} KB)")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = PROJECT_ROOT / "asset_quality_scorer/outputs/viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_cases: dict[str, list[dict]] = {}
    for ch in ALL_CHANNELS:
        print(f"[{ch}] running inference + selecting cases...")
        all_cases[ch] = run_channel(ch, device)
        print(f"  -> {len(all_cases[ch])} cases selected")

    build_html(all_cases, out_dir / "cases.html")


if __name__ == "__main__":
    main()
