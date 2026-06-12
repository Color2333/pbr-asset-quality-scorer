"""Extract the fine-tuned Qwen's answer-position hidden state (3584-d) per
(asset, channel) on the test split, using the SAME scorer prompt as eval. Lets us
train a head on the FROZEN Qwen features (linear-probe) and compare to the digit
readout. Reuses vlm_scorer_eval's build_items + per-channel message format.

Dumps outputs/qwen_hidden_{tag}.npz: feat_{ch}[N,3584], gt_{ch}[N], names.
Usage: CUDA_VISIBLE_DEVICES=0 python asset_quality_scorer/scripts/extract_qwen_hidden.py \
         --adapter asset_quality_scorer/outputs/runs/vlm_scorer_qwen25_sft_fullcover/best --tag fullcover
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "scripts"))
from vlm_scorer_eval import build_items, CHANNELS, CH_PROMPT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--model-dir", default=str(PKG.parent / "models/Qwen2.5-VL-7B-Instruct"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-pixels", type=int, default=147456)
    args = ap.parse_args()
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    from peft import PeftModel
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, device_map="cuda",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16))
    model = PeftModel.from_pretrained(model, args.adapter); model.eval()
    proc = AutoProcessor.from_pretrained(args.model_dir, min_pixels=224 * 224, max_pixels=args.max_pixels)

    rows = build_items("test"); names = [r["name"] for r in rows]
    feat = {c: np.zeros((len(rows), 3584), np.float16) for c in CHANNELS}
    gt = {c: np.array([r["scores"][c] for r in rows]) for c in CHANNELS}

    def msgs(item, ch):
        imgs = [Image.open(item["paths"][ch]).convert("RGB"), Image.open(item["paths"]["render"]).convert("RGB")]
        labels = ["the channel map being rated", "the final render of the asset"]
        if ch != "base_color":
            imgs.append(Image.open(item["paths"]["base_color"]).convert("RGB")); labels.append("the base color map (context)")
        body = "; ".join(f"image {i+1} = {l}" for i, l in enumerate(labels))
        text = (f"You are a professional 3D PBR material QA inspector. {body}. Rate the quality of "
                f"{CH_PROMPT[ch]}. Score 0 = broken/missing/wrong, 3 = usable with flaws, 5 = excellent. "
                f"Answer with exactly one digit (0-5).")
        return [{"role": "user", "content": [{"type": "image", "image": im} for im in imgs] + [{"type": "text", "text": text}]}], imgs

    for ci, ch in enumerate(CHANNELS):
        for i, r in enumerate(rows):
            m, imgs = msgs(r, ch)
            t = proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            inp = proc(text=[t], images=imgs, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)
            feat[ch][i] = out.hidden_states[-1][0, -1].float().cpu().numpy().astype(np.float16)
            if (i + 1) % 1000 == 0:
                print(f"  {ch} {i+1}/{len(rows)}", flush=True)
        print(f"[{ch}] done", flush=True)
    out = {f"feat_{c}": feat[c] for c in CHANNELS}
    out.update({f"gt_{c}": gt[c] for c in CHANNELS}); out["names"] = np.array(names)
    p = PKG / "outputs" / f"qwen_hidden_{args.tag}.npz"
    np.savez(p, **out); print(f"saved {p}")


if __name__ == "__main__":
    main()
