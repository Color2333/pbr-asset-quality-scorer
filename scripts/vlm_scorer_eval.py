"""Q-Align-style VLM scorer evaluation: the VLM rates each channel map 0-5;
score = expectation over the six digit-token probabilities (no generation, no
text parsing — same readout philosophy as our EMD head, hosted in a VLM).

Zero-shot (no adapter) or finetuned (--adapter from vlm_scorer_sft.py).
Inputs per item: [channel map, render(+base_color for non-bc channels)] at full
resolution (processor downscales by max-pixels budget). Render is legitimate
here: annotators scored while seeing the render; we mimic the annotator.

Usage:
    CUDA_VISIBLE_DEVICES=1 ~/miniconda3/envs/qwen-vl/bin/python \
      asset_quality_scorer/scripts/vlm_scorer_eval.py \
      --model-dir models/Qwen2.5-VL-7B-Instruct --quant nf4 \
      --sample 150 --out-tag qwen25_zeroshot_smoke
Writes outputs/runs/vlm_scorer_{out-tag}/eval_test.json
"""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

PKG = Path(__file__).resolve().parents[1]
DATA = PKG.parent / "datasets0526"
CHANNELS = ["base_color", "normal_map", "roughness", "metallic"]
SCORE_COL = {"base_color": "baseColor", "normal_map": "normal",
             "roughness": "roughness", "metallic": "metallic"}

CH_PROMPT = {
    "base_color": ("the BASE COLOR (albedo) texture map. A good albedo is clean and "
                   "lighting-free (no baked shadows/highlights), with appropriate detail "
                   "and color; a bad one is blurry, noisy, has baked lighting or wrong colors"),
    "normal_map": ("the NORMAL map. A good normal map has correct bluish encoding and adds "
                   "meaningful, clean surface detail matching the object; a bad one is flat, "
                   "noisy, has wrong tint or artifacts"),
    "roughness":  ("the ROUGHNESS map. A good roughness map has plausible per-material "
                   "variation (different surfaces get different roughness); a bad one is a "
                   "constant value, inverted, or mismatched with the materials"),
    "metallic":   ("the METALLIC map. A good metallic map marks exactly the metal parts of "
                   "the object as white and non-metal as black; it is bad if real metal "
                   "parts are missing (left black), or non-metal parts are wrongly white"),
}


METALLIC_EXPERT_PROMPT = (
    "the METALLIC map using this procedure: (1) From the render and albedo, identify which "
    "parts of this object are METAL in the real world (blades, armor, machine bodies, frames, "
    "fittings). (2) Look at the CLAIM OVERLAY image: regions tinted RED are the parts the "
    "metallic map claims to be metal. (3) Judge the match: if large real-metal parts have NO "
    "red tint, metal is MISSING from the map - score 0-1. If non-metal parts are wrongly red, "
    "score low. If the red regions match the real metal parts well, score high. If the object "
    "genuinely has no metal and nothing is red, that is correct - score 5. Small accents "
    "(studs, buckles, trims) left untinted are a minor flaw only")


def make_claim_overlay(bc_img, me_img):
    """UV-aligned albedo with metallic-claimed regions tinted red — turns the
    cross-image alignment problem into direct visual pattern matching."""
    import numpy as np_
    bc = np_.asarray(bc_img.resize((1024, 1024)), dtype=np_.float32)
    m = np_.asarray(me_img.convert("L").resize((1024, 1024)), dtype=np_.float32)[..., None] / 255.0
    red = np_.zeros_like(bc); red[..., 0] = 255.0
    out = bc * (1 - 0.6 * m) + red * (0.6 * m)
    return Image.fromarray(out.astype("uint8"))


def build_items(split="test", sample=0, seed=42, csv_path=None, data_root=None):
    csv_path = Path(csv_path) if csv_path else PKG / "dataset/sampled_all.csv"
    root = Path(data_root) if data_root else DATA
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("split") != split:
                continue
            name = r["model"].removeprefix("raw_data/").replace("/", "__").removesuffix(".glb")
            paths = {c: root / c / f"{name}.png" for c in CHANNELS + ["render"]}
            if not all(p.exists() for p in paths.values()):
                continue
            try:
                scores = {c: int(r[SCORE_COL[c]]) for c in CHANNELS}
            except Exception:
                continue
            rows.append({"name": name, "scores": scores, "paths": {c: str(p) for c, p in paths.items()}})
    if sample and sample < len(rows):
        rng = np.random.RandomState(seed)
        rows = [rows[i] for i in rng.choice(len(rows), sample, replace=False)]
    print(f"{split} assets: {len(rows)}  (items = x{len(CHANNELS)} channels)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(PKG.parent / "models/Qwen2.5-VL-7B-Instruct"))
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (omit = zero-shot)")
    ap.add_argument("--quant", default="nf4", choices=["bf16", "nf4"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-pixels", type=int, default=448 * 448)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--sample", type=int, default=0, help="random asset subsample (0=all)")
    ap.add_argument("--out-tag", required=True)
    ap.add_argument("--csv", default=None); ap.add_argument("--data-root", default=None)
    ap.add_argument("--channels", nargs="+", default=None, help="subset of channels to eval")
    ap.add_argument("--metallic-overlay", action="store_true",
                    help="metallic items get claim-overlay image + expert prompt (must match training)")
    ap.add_argument("--multi-channel", action="store_true",
                    help="feed ALL channel maps for metallic (must match training)")
    args = ap.parse_args()

    rows = build_items(sample=args.sample, csv_path=args.csv, data_root=args.data_root)
    out_dir = PKG / "outputs/runs" / f"vlm_scorer_{args.out_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForImageTextToText, AutoProcessor
    kw: dict = {"torch_dtype": torch.bfloat16, "device_map": args.device}
    if args.quant == "nf4":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, **kw)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"loaded adapter {args.adapter}")
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_dir, min_pixels=224 * 224,
                                              max_pixels=args.max_pixels)
    tok = processor.tokenizer
    digit_ids = [tok.encode(str(k), add_special_tokens=False)[0] for k in range(6)]
    print(f"digit token ids: {digit_ids}")

    def make_msgs(item, ch):
        if args.multi_channel and ch == "metallic":
            order = ["metallic", "render", "base_color", "normal_map", "roughness"]
            imgs = [Image.open(item["paths"][c]).convert("RGB") for c in order]
            labels = ["the METALLIC map being rated", "the final render of the asset",
                      "the base color (albedo) map", "the normal map", "the roughness map"]
            crit = CH_PROMPT["metallic"]
            body = "; ".join(f"image {i+1} = {l}" for i, l in enumerate(labels))
            text = (f"You are a professional 3D PBR material QA inspector. {body}. "
                    f"Rate the quality of {crit}. "
                    f"Score 0 = broken/missing/wrong, 3 = usable with flaws, 5 = excellent. "
                    f"Answer with exactly one digit (0-5).")
            return [{"role": "user", "content":
                     [{"type": "image", "image": im} for im in imgs] + [{"type": "text", "text": text}]}]
        imgs = [Image.open(item["paths"][ch]).convert("RGB"),
                Image.open(item["paths"]["render"]).convert("RGB")]
        labels = ["the channel map being rated", "the final render of the asset"]
        if ch != "base_color":
            imgs.append(Image.open(item["paths"]["base_color"]).convert("RGB"))
            labels.append("the base color map (context)")
        crit = CH_PROMPT[ch]
        if ch == "metallic" and args.metallic_overlay:
            imgs.append(make_claim_overlay(imgs[2], imgs[0]))
            labels.append("the CLAIM OVERLAY (albedo with metal-claimed regions tinted red)")
            crit = METALLIC_EXPERT_PROMPT
        body = "; ".join(f"image {i+1} = {l}" for i, l in enumerate(labels))
        text = (f"You are a professional 3D PBR material QA inspector. {body}. "
                f"Rate the quality of {crit}. "
                f"Score 0 = broken/missing/wrong, 3 = usable with flaws, 5 = excellent. "
                f"Answer with exactly one digit (0-5).")
        return [{"role": "user", "content":
                 [{"type": "image", "image": im} for im in imgs] + [{"type": "text", "text": text}]}]

    eval_channels = args.channels or CHANNELS
    preds = {c: [] for c in eval_channels}; gts = {c: [] for c in eval_channels}; probs = {}
    t0 = time.time(); n_done = 0
    for ch in eval_channels:
        B = args.batch_size
        for s in range(0, len(rows), B):
            chunk = rows[s:s + B]
            msgs_l = [make_msgs(it, ch) for it in chunk]
            texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                     for m in msgs_l]
            imgs = [c_["image"] for m in msgs_l for c_ in m[0]["content"] if c_["type"] == "image"]
            inputs = processor(text=texts, images=imgs, padding=True, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model(**inputs)
            last = inputs["attention_mask"].sum(1) - 1
            logits = out.logits[torch.arange(len(chunk)), last].float()
            p6 = torch.softmax(logits[:, digit_ids], dim=-1)       # renormalized over 6 digits
            exp_score = (p6 * torch.arange(6, dtype=torch.float32, device=p6.device)).sum(1)
            preds[ch].extend(exp_score.cpu().tolist())
            probs.setdefault(ch, []).extend(p6.float().cpu().tolist())
            gts[ch].extend(it["scores"][ch] for it in chunk)
            n_done += len(chunk)
            if (s // B) % 50 == 0:
                el = time.time() - t0
                total = len(rows) * len(eval_channels)
                print(f"  [{ch}] {n_done}/{total}  {el/max(n_done,1):.2f}s/项  ETA {(total-n_done)*el/max(n_done,1)/3600:.1f}h", flush=True)

    res = {"tag": args.out_tag, "model": args.model_dir, "adapter": args.adapter,
           "quant": args.quant, "n_assets": len(rows)}
    srccs = []
    for c in eval_channels:
        p, g = np.array(preds[c]), np.array(gts[c])
        s_ = float(spearmanr(p, g).statistic); srccs.append(s_)
        res[c] = {"srcc": round(s_, 4), "mae": round(float(np.abs(p - g).mean()), 4)}
    res["srcc_mean"] = round(float(np.mean(srccs)), 4)
    # metallic near-black (if nonblack stats cover these assets)
    meta = json.loads((PKG / "cache/224/meta.json").read_text())
    fr = dict(zip(meta["model_names"], np.load(PKG / "dataset/metallic_nonblack.npy")))
    nb = np.array([fr.get(it["name"], 1.0) < 0.02 for it in rows])
    if nb.sum() > 20 and "metallic" in eval_channels:
        p, g = np.array(preds["metallic"]), np.array(gts["metallic"])
        y = (g[nb] <= 2).astype(int)
        res["metallic_nearblack"] = {"n": int(nb.sum()),
            "srcc": round(float(spearmanr(p[nb], g[nb]).statistic), 4),
            "auc_missing": round(float(roc_auc_score(y, -p[nb])), 4) if 0 < y.mean() < 1 else None}
    print(json.dumps(res, indent=2))
    (out_dir / "eval_test.json").write_text(json.dumps(res, indent=2))
    np.save(out_dir / "preds.npy", {c: np.array(preds[c]) for c in eval_channels}, allow_pickle=True)
    np.save(out_dir / "probs.npy", {c: np.array(probs[c]) for c in eval_channels}, allow_pickle=True)
    print(f"saved -> {out_dir}/eval_test.json")


if __name__ == "__main__":
    main()
