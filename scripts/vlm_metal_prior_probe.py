"""Phase-1 probe: can a VLM's WORLD-KNOWLEDGE prior separate missing-metal
(漏标) from correctly-all-dielectric assets among near-black metallic maps?

This is the one info source our falsification campaign never touched: the
visual prior (CLIP base_color -> has-metal, AUC 0.80) was trained inside our
own noisy label distribution and gave zero ensemble gain; a VLM instead judges
"should this OBJECT contain metal in reality" — insensitive to how the albedo
was authored. Inputs are white_model + base_color ONLY (render is rendered
FROM the metallic map -> circular). Kill criterion: near-black AUC <= 0.58
(the within-near-black visual ceiling).

Reads the yes-token probability from a single forward pass (no generate, no
text parsing) and dumps the last hidden state at the same position so Phase-3
feature injection needs no re-run.

Usage:
    ~/miniconda3/envs/qwen-vl/bin/python asset_quality_scorer/scripts/vlm_metal_prior_probe.py \
        [--limit 0] [--max-pixels 589824]
"""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

PKG = Path(__file__).resolve().parents[1]
DATA0526 = PKG.parent / "datasets0526"
MODEL_DIR = PKG.parent / "models/Qwen2.5-VL-7B-Instruct"
OUT = PKG / "outputs/vlm_metal_prior"
NEARBLACK_THRESH = 0.02   # same bucket as compute_metallic_stats (<2% non-black = 47% of data)

# Variant ladder. v1 asks "should this contain metal" — probe showed it
# systematically misfires on partial-metal assets (score=4 group, leather sofa
# with brass studs: contains metal=Yes, but all-black metallic is an acceptable
# authoring choice). v2/v3 align the question with the LABEL semantics instead:
# v2 judges the authoring DECISION, v3 judges the visual DAMAGE of getting it wrong.
PROMPTS = {
    "v1_hasmetal": dict(
        text=(
            "You are looking at a 3D asset. The first image is its untextured white-clay "
            "render (shape only); the second image is its base-color (albedo) texture map. "
            "Based on what this OBJECT IS in the real world, should any part of it be made "
            "of metal (e.g. armor, blades, machinery, jewelry, fixtures)? Ignore rendering "
            "style and texture quality — judge only what the object should be physically "
            "made of. Answer with exactly one word: Yes or No."),
        pos=("Yes", "yes"), neg=("No", "no")),
    "v2_error": dict(
        text=(
            "You are looking at a 3D asset. The first image is its untextured white-clay "
            "render (shape only); the second image is its base-color (albedo) texture map. "
            "This asset's metallic map is ENTIRELY BLACK: every part of the object is "
            "rendered as non-metal. Given what this object is, is the all-black metallic "
            "map a reasonable authoring choice (the object needs no metal, or only "
            "negligible metal accents), or is it an authoring ERROR (significant metal "
            "parts were left unmarked)? Answer with exactly one word: Error or OK."),
        pos=("Error", "error"), neg=("OK", "Ok", "ok")),
    "v4_largemetal": dict(
        text=(
            "You are looking at a 3D asset. The first image is its untextured white-clay "
            "render (shape only); the second image is its base-color (albedo) texture map. "
            "Does this object have LARGE metal parts — major surfaces or components that "
            "are clearly metal in the real world (armor plates, blades, machine bodies, "
            "metal frames)? Small accents like studs, buckles, handles or trims do NOT "
            "count. Answer with exactly one word: Yes or No."),
        pos=("Yes", "yes"), neg=("No", "no")),
    "v3_damage": dict(
        text=(
            "You are looking at a 3D asset. The first image is its untextured white-clay "
            "render (shape only); the second image is its base-color (albedo) texture map. "
            "Suppose every metal part of this object were rendered as plain non-metal "
            "(no metallic reflection at all). How much would that hurt the visual "
            "believability of this specific object? If the object has no real metal parts "
            "or only tiny accents, the damage is negligible. Answer with exactly one word: "
            "Severe or Negligible."),
        pos=("Severe", "severe"), neg=("Negligible", "negligible")),
    # ── analysis panel: not ONE pre-digested judgment but the annotator's
    # conditioning VARIABLES (workflow/style/bake), so the scorer can learn the
    # conditional rule itself. Each is a yes/no token-prob read like v4.
    "q2_scan": dict(
        text=(
            "You are looking at a 3D asset. The first image is its untextured white-clay "
            "render (shape only); the second image is its base-color (albedo) texture map. "
            "Is this asset a 3D SCAN (photogrammetry capture of a real object, with baked "
            "photographic texture), rather than a hand-authored 3D model? Signs of a scan: "
            "irregular organic silhouette, photographic noise in the texture, baked-in "
            "shadows, a single fused mesh. Answer with exactly one word: Yes or No."),
        pos=("Yes", "yes"), neg=("No", "no")),
    "q3_stylized": dict(
        text=(
            "You are looking at a 3D asset. The first image is its untextured white-clay "
            "render (shape only); the second image is its base-color (albedo) texture map. "
            "Is this asset STYLIZED (cartoon / hand-painted / low-poly art style with "
            "exaggerated shapes or flat painted colors), rather than realistic? "
            "Answer with exactly one word: Yes or No."),
        pos=("Yes", "yes"), neg=("No", "no")),
    "q4_baked": dict(
        text=(
            "You are looking at a 3D asset. The first image is its untextured white-clay "
            "render (shape only); the second image is its base-color (albedo) texture map. "
            "Does the albedo texture have LIGHTING BAKED IN — shadows, highlights, "
            "ambient occlusion or reflections painted/captured directly in the base color "
            "(instead of a flat, lighting-free albedo as proper PBR requires)? "
            "Answer with exactly one word: Yes or No."),
        pos=("Yes", "yes"), neg=("No", "no")),
    "q5_smallmetal": dict(
        text=(
            "You are looking at a 3D asset. The first image is its untextured white-clay "
            "render (shape only); the second image is its base-color (albedo) texture map. "
            "Does this object have ONLY SMALL metal accents (studs, buckles, handles, "
            "trims, screws) and no large metal surfaces? Answer No if it has no metal at "
            "all OR if it has large metal parts. Answer with exactly one word: Yes or No."),
        pos=("Yes", "yes"), neg=("No", "no")),
    "q6_multimat": dict(
        text=(
            "You are looking at a 3D asset. The first image is its untextured white-clay "
            "render (shape only); the second image is its base-color (albedo) texture map. "
            "Is this object made of MULTIPLE distinct material types (e.g. wood + metal + "
            "fabric), rather than essentially one material? "
            "Answer with exactly one word: Yes or No."),
        pos=("Yes", "yes"), neg=("No", "no")),
    # v5 uses GENERATION (Qwen grounding), not token prob: model outputs bbox_2d
    # JSON for metal regions on the albedo map; score = total box area fraction.
    # Slow (~2s/sample) — run on --sample subsets. Boxes are also the raw material
    # for Phase-4 spatial injection (rasterize -> patch-grid mask).
    "v5_bbox": dict(
        text=(
            "You are looking at a 3D asset. The first image is its untextured white-clay "
            "render (shape only); the second image is its base-color (albedo) texture map. "
            "Locate every region of the SECOND image (the albedo map) that corresponds to "
            "a part of the object that should be METAL in the real world. Report each as "
            'JSON: [{"bbox_2d": [x1, y1, x2, y2], "label": "<part name>"}]. '
            "If nothing should be metal, output []."),
        generate=True),
}


def stratified_sample(rows, n, seed=42):
    """Balanced random subsample for fast prompt iteration (AUC std ~±0.02 @ n=600)."""
    rng = np.random.RandomState(seed)
    pos = [r for r in rows if r["y_missing"]]; neg = [r for r in rows if not r["y_missing"]]
    k = min(n // 2, len(pos), len(neg))
    picked = [pos[i] for i in rng.choice(len(pos), k, replace=False)] + \
             [neg[i] for i in rng.choice(len(neg), k, replace=False)]
    rng.shuffle(picked)
    print(f"stratified sample: {len(picked)} ({k}+{k})")
    return picked


def build_subset(limit: int = 0):
    """Near-black metallic test samples. y=1 漏标嫌疑 (score<=2), y=0 正确全黑 (>=3)."""
    meta = json.loads((PKG / "cache/224/meta.json").read_text())
    names = set(meta["model_names"])
    frac = dict(zip(meta["model_names"], np.load(PKG / "dataset/metallic_nonblack.npy")))
    rows = []
    with open(PKG / "dataset/sampled_all.csv", newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("split") != "test" or not r.get("metallic"):
                continue
            name = r["model"].removeprefix("raw_data/").replace("/", "__").removesuffix(".glb")
            if name not in names or frac[name] >= NEARBLACK_THRESH:
                continue
            bc = DATA0526 / "base_color" / f"{name}.png"
            wm = DATA0526 / "white_model" / f"{name}.png"
            if not (bc.exists() and wm.exists()):
                continue
            score = float(r["metallic"])
            rows.append({"name": name, "metallic": score, "nonblack": float(frac[name]),
                         "y_missing": int(score <= 2), "base_color": str(bc), "white_model": str(wm)})
    if limit:
        rows = rows[:limit]
    n1 = sum(r["y_missing"] for r in rows)
    print(f"near-black test subset: {len(rows)}  (漏标嫌疑 score<=2: {n1} | 正确全黑 score>=3: {len(rows)-n1})")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap subset size (0=all)")
    ap.add_argument("--max-pixels", type=int, default=768 * 768, help="per-image visual token budget")
    ap.add_argument("--variant", default="v1_hasmetal", choices=sorted(PROMPTS))
    ap.add_argument("--quant", default="bf16", choices=["bf16", "int8", "nf4"],
                    help="bf16 needs both GPUs (device_map=auto); int8/nf4 fit one 24G card")
    ap.add_argument("--device", default="auto", help='"auto" or e.g. "cuda:1" (with --quant int8/nf4)')
    ap.add_argument("--no-hidden", action="store_true", help="skip hidden-state dump (variant sweeps)")
    ap.add_argument("--model-dir", default=str(MODEL_DIR), help="local VLM dir (Qwen2.5-VL / Qwen3-VL)")
    ap.add_argument("--out-suffix", default="", help="appended to output tag (e.g. _qwen3)")
    ap.add_argument("--sample", type=int, default=0,
                    help="balanced random subsample for fast iteration (0=full subset)")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="batched forward for token-prob variants (~4x faster at 8)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    spec = PROMPTS[args.variant]
    prompt = spec["text"]
    is_gen = bool(spec.get("generate"))
    tag = "" if args.variant == "v1_hasmetal" else f"_{args.variant}"
    if args.sample:
        tag += f"_s{args.sample}"
    tag += args.out_suffix

    rows = build_subset(args.limit)
    if args.sample:
        rows = stratified_sample(rows, args.sample)

    from transformers import AutoModelForImageTextToText, AutoProcessor
    print(f"loading {args.model_dir} (variant={args.variant}, quant={args.quant}, device={args.device})…")
    kw: dict = {"torch_dtype": torch.bfloat16}
    if args.quant in ("int8", "nf4"):
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = (
            BitsAndBytesConfig(load_in_8bit=True) if args.quant == "int8"
            else BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.bfloat16))
        kw["device_map"] = args.device if args.device != "auto" else "auto"
    elif args.device != "auto":
        kw["device_map"] = args.device          # bf16 single-GPU (e.g. 8B on one 24G card)
    else:
        kw["device_map"] = "auto"
        kw["max_memory"] = {0: "15GiB", 1: "13GiB"}
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, **kw)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_dir, min_pixels=256 * 256, max_pixels=args.max_pixels)

    tok = processor.tokenizer
    if not is_gen:
        yes_ids = sorted({tok.encode(v, add_special_tokens=False)[0]
                          for w in spec["pos"] for v in (w, " " + w)})
        no_ids = sorted({tok.encode(v, add_special_tokens=False)[0]
                         for w in spec["neg"] for v in (w, " " + w)})
        print(f"pos ids {yes_ids}  neg ids {no_ids}")

    def make_msgs(r):
        return [{"role": "user", "content": [
            {"type": "image", "image": Image.open(r["white_model"]).convert("RGB")},
            {"type": "image", "image": Image.open(r["base_color"]).convert("RGB")},
            {"type": "text", "text": prompt}]}]

    p_yes_all, hidden_all, done, raw_all = [], [], [], []
    t0 = time.time()

    if is_gen:
        # ---- bbox grounding path: generate JSON, score = total box area fraction ----
        import re as _re
        for i, r in enumerate(rows):
            msgs = make_msgs(r)
            text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            imgs = [c["image"] for c in msgs[0]["content"] if c["type"] == "image"]
            inputs = processor(text=[text], images=imgs, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            ans = tok.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            area = 0.0
            try:
                boxes = json.loads(_re.search(r"\[.*\]", ans, _re.S).group(0)) if "[" in ans else []
                W = H = 1000.0  # qwen grounding 坐标系按输入分辨率;归一化用相对面积近似
                iw, ih = imgs[1].size
                for b in boxes:
                    x1, y1, x2, y2 = b["bbox_2d"]
                    area += max(0, x2 - x1) * max(0, y2 - y1) / (iw * ih)
            except Exception:
                pass
            p_yes_all.append(min(area, 1.0)); raw_all.append({"name": r["name"], "raw": ans})
            done.append(r)
            if (i + 1) % 50 == 0 or i == len(rows) - 1:
                dt = time.time() - t0
                y = np.array([d["y_missing"] for d in done]); p = np.array(p_yes_all)
                auc = roc_auc_score(y, p) if 0 < y.mean() < 1 else float("nan")
                print(f"  {i+1}/{len(rows)}  {dt/(i+1):.2f}s/样本  running AUC={auc:.4f}", flush=True)
    else:
        # ---- batched token-prob path ----
        B = max(1, args.batch_size)
        for s in range(0, len(rows), B):
            chunk = rows[s:s + B]
            msgs_l = [make_msgs(r) for r in chunk]
            texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs_l]
            imgs = [c["image"] for m in msgs_l for c in m[0]["content"] if c["type"] == "image"]
            inputs = processor(text=texts, images=imgs, padding=True, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=not args.no_hidden)
            last = inputs["attention_mask"].sum(1) - 1                     # [B] 右填充下最后真实位置
            logits = out.logits[torch.arange(len(chunk)), last].float()    # [B,V]
            probs = torch.softmax(logits, dim=-1)
            py = probs[:, yes_ids].sum(1); pn = probs[:, no_ids].sum(1)
            pv = (py / (py + pn).clamp_min(1e-9)).cpu().tolist()
            if not args.no_hidden:
                h = out.hidden_states[-1][torch.arange(len(chunk)), last].float().cpu().numpy()
                hidden_all.extend(h.astype(np.float16))
            p_yes_all.extend(pv); done.extend(chunk)
            if (s // B) % max(1, 100 // B) == 0 or s + B >= len(rows):
                dt = time.time() - t0
                y = np.array([d["y_missing"] for d in done]); p = np.array(p_yes_all)
                auc = roc_auc_score(y, p) if 0 < y.mean() < 1 else float("nan")
                print(f"  {len(done)}/{len(rows)}  {dt/len(done):.2f}s/样本  running AUC={auc:.4f}", flush=True)

    y = np.array([d["y_missing"] for d in done]); p = np.array(p_yes_all)
    auc = roc_auc_score(y, p)
    res = {"variant": args.variant, "quant": args.quant, "n": len(done), "n_missing": int(y.sum()), "auc_missing_vs_correct": round(float(auc), 4),
           "visual_ceiling": 0.58, "passed": bool(auc > 0.58),
           "p_yes_mean_missing": round(float(p[y == 1].mean()), 4),
           "p_yes_mean_correct": round(float(p[y == 0].mean()), 4)}
    print(json.dumps(res, indent=2))

    with open(OUT / f"probe_nearblack_test{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "metallic", "nonblack", "y_missing", "p_yes"])
        w.writeheader()
        for d, pv in zip(done, p_yes_all):
            w.writerow({k: d[k] for k in ("name", "metallic", "nonblack", "y_missing")} | {"p_yes": round(pv, 5)})
    if hidden_all:
        np.save(OUT / f"probe_hidden_states{tag}.npy", np.stack(hidden_all))
    if raw_all:
        with open(OUT / f"probe_raw{tag}.jsonl", "w") as f:
            for d in raw_all:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    (OUT / f"probe_summary{tag}.json").write_text(json.dumps(res, indent=2))
    print(f"saved -> {OUT}/probe_nearblack_test{tag}.csv + probe_summary{tag}.json")


if __name__ == "__main__":
    main()
