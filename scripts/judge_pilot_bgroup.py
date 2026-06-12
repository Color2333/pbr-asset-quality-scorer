"""Zero-shot world-knowledge judge pilot for the metallic B-group blind spot.

Hypothesis: our fine-tuned scorers learned "black metallic map -> low score" and
can't tell a CORRECTLY-empty map on a non-metal object (B-group, GT high) from a
genuinely MISSING metallic map (control, GT low). A strong VLM with world
knowledge, asked to reason about the object's materials first, might separate them
zero-shot (no fine-tuning) — proving the gap is reasoning, not pixels.

Sets (near-black metallic both):
  B       = bucket B, suspicion>1.5  (GT high, empty map is CORRECT)
  control = near-black + GT<=1        (GT low, empty map is WRONG/missing)

Success = judge scores B HIGH and control LOW (separation), which our scorers can't.

Usage: CUDA_VISIBLE_DEVICES=0 python asset_quality_scorer/scripts/judge_pilot_bgroup.py
"""
from __future__ import annotations
import json, sys, csv
from pathlib import Path
import numpy as np
import torch
from PIL import Image
PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "scripts"))
from vlm_scorer_eval import build_items

MODEL = str(PKG.parent / "models/Qwen3-VL-8B-Instruct")
N_PER = 206  # balanced sample per set

PROMPT = (
    "You are a PBR material QA expert. You are given three images: image 1 = the "
    "final RENDER of a 3D asset, image 2 = its BASE COLOR (albedo) texture, image 3 "
    "= its METALLIC map (white = metal, black = non-metal).\n"
    "Reason in steps:\n"
    "1. From the render and albedo, identify the object and what real-world materials "
    "it is made of.\n"
    "2. Decide whether this object SHOULD have any metallic surfaces.\n"
    "3. Judge the metallic map. CRUCIAL: if the object is non-metallic (wood, fabric, "
    "plastic, stone, ceramic, leather), then an all-black / empty metallic map is "
    "CORRECT and should score HIGH. Only score low if the object clearly has metal "
    "that the map fails to mark, or the map is noisy/wrong.\n"
    "Output exactly one digit 0-5: 5 = metallic map is correct for this object's "
    "materials, 0 = wrong or missing. Answer with one digit."
)


def main():
    rows = build_items("test")
    gt = {r["name"]: r["scores"]["metallic"] for r in rows}
    paths = {r["name"]: r["paths"] for r in rows}
    names = [r["name"] for r in rows]
    nbf = dict(zip(json.loads((PKG / "cache/224/meta.json").read_text())["model_names"],
                   np.load(PKG / "dataset/metallic_nonblack.npy")))
    cand = {r["name"]: r for r in csv.DictReader(open(PKG / "outputs/label_noise/candidates_metallic.csv"))}
    B = [n for n in names if cand.get(n, {}).get("bucket") == "B"
         and float(cand.get(n, {}).get("suspicion", 0)) > 1.5]
    control = [n for n in names if nbf.get(n, 1) < 0.02 and gt[n] <= 1][:N_PER]
    print(f"B={len(B)}  control={len(control)}")

    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16))
    model.eval()
    proc = AutoProcessor.from_pretrained(MODEL, min_pixels=224 * 224, max_pixels=147456)
    tok = proc.tokenizer
    digit_ids = [tok.encode(str(k), add_special_tokens=False)[0] for k in range(6)]

    def score(name):
        p = paths[name]
        imgs = [Image.open(p["render"]).convert("RGB"), Image.open(p["base_color"]).convert("RGB"),
                Image.open(p["metallic"]).convert("RGB")]
        msgs = [{"role": "user", "content": [{"type": "image", "image": im} for im in imgs]
                 + [{"type": "text", "text": PROMPT}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[text], images=imgs, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**inp).logits[0, -1]
        p6 = torch.softmax(logits[digit_ids].float(), -1)
        return float((p6 * torch.arange(6., device=p6.device)).sum())

    res = {}
    for label, lst in [("B", B), ("control", control)]:
        sc = []
        for i, n in enumerate(lst):
            try:
                sc.append((score(n), gt[n]))
            except Exception as e:
                print("skip", n, e)
            if (i + 1) % 50 == 0:
                print(f"  {label} {i+1}/{len(lst)}", flush=True)
        arr = np.array([s for s, _ in sc])
        res[label] = arr
        print(f"[{label}] n={len(arr)}  judge均值={arr.mean():.2f}  std={arr.std():.2f}  "
              f">=3占比={np.mean(arr>=3)*100:.0f}%  GT均值={np.mean([g for _,g in sc]):.2f}")
    sep = res["B"].mean() - res["control"].mean()
    print(f"\n分离度 (B - control) = {sep:+.2f}   "
          f"{'✅ 判官能用世界知识区分' if sep > 0.7 else '❌ 判官也分不开(世界知识没解B组)'}")
    out = PKG / "outputs/judge_pilot_bgroup.json"
    out.write_text(json.dumps({"B_mean": float(res["B"].mean()), "control_mean": float(res["control"].mean()),
                               "separation": float(sep), "n_B": len(res["B"]), "n_ctrl": len(res["control"])}, indent=2))
    print("saved ->", out)


if __name__ == "__main__":
    main()
