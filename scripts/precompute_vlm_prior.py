"""Precompute the VLM world-knowledge metal prior for EVERY asset in a tensor
cache (train+val+test), for Phase-3 feature injection.

Uses the v4_largemetal prompt — unlike v2_error ("the metallic map IS all
black"), v4's question is factually valid for every asset, near-black or not,
and tied v2 on full-set AUC (0.6313 vs 0.6300). Saves per-asset:
  p_yes  [N]       renormalized yes-probability (scalar prior)
  hidden [N, 3584] last hidden state at the answer position, fp16
aligned to the cache meta.json model_names order, so training can mmap-index
it exactly like the channel tensors.

Usage (GPU1, ~5.5h for 49k @ batch 4):
    CUDA_VISIBLE_DEVICES=1 ~/miniconda3/envs/qwen-vl/bin/python \
        asset_quality_scorer/scripts/precompute_vlm_prior.py \
        [--cache asset_quality_scorer/cache/224] [--image-root datasets0526] \
        [--batch-size 4] [--max-pixels 262144]
Resumable: skips indices already filled in the output memmaps (p_yes init -1).
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_metal_prior_probe import PROMPTS, MODEL_DIR

PKG = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(PKG / "cache/224"))
    ap.add_argument("--image-root", default=str(PKG.parent / "datasets0526"))
    ap.add_argument("--out", default=str(PKG / "dataset/vlm_prior_v4"))
    ap.add_argument("--variant", default="v4_largemetal")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-pixels", type=int, default=262144)
    args = ap.parse_args()

    meta = json.loads((Path(args.cache) / "meta.json").read_text())
    names = meta["model_names"]
    N = len(names)
    root = Path(args.image_root)
    prompt = PROMPTS[args.variant]["text"]
    spec = PROMPTS[args.variant]

    out_p = Path(args.out + "_pyes.npy")
    out_h = Path(args.out + "_hidden.npy")
    if out_p.exists():
        p_yes = np.lib.format.open_memmap(out_p, mode="r+")
        hidden = np.lib.format.open_memmap(out_h, mode="r+")
        print(f"resume: {(p_yes >= 0).sum()}/{N} already done")
    else:
        p_yes = np.lib.format.open_memmap(out_p, mode="w+", dtype=np.float32, shape=(N,))
        p_yes[:] = -1.0
        hidden = np.lib.format.open_memmap(out_h, mode="w+", dtype=np.float16, shape=(N, 3584))

    todo = [i for i in range(N) if p_yes[i] < 0]
    print(f"todo {len(todo)}/{N}  variant={args.variant}")

    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cuda:0",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16))
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_DIR, min_pixels=256 * 256, max_pixels=args.max_pixels)
    tok = processor.tokenizer
    yes_ids = sorted({tok.encode(v, add_special_tokens=False)[0]
                      for w in spec["pos"] for v in (w, " " + w)})
    no_ids = sorted({tok.encode(v, add_special_tokens=False)[0]
                     for w in spec["neg"] for v in (w, " " + w)})

    B = args.batch_size
    t0 = time.time(); ok = skip = 0
    for s in range(0, len(todo), B):
        idxs = todo[s:s + B]
        texts, imgs, kept = [], [], []
        for i in idxs:
            wm = root / "white_model" / f"{names[i]}.png"
            bc = root / "base_color" / f"{names[i]}.png"
            if not (wm.exists() and bc.exists()):
                p_yes[i] = 0.0; skip += 1   # 缺图: 中性先验, 不再重试
                continue
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": Image.open(wm).convert("RGB")},
                {"type": "image", "image": Image.open(bc).convert("RGB")},
                {"type": "text", "text": prompt}]}]
            texts.append(processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            imgs.extend(c["image"] for c in msgs[0]["content"] if c["type"] == "image")
            kept.append(i)
        if not kept:
            continue
        inputs = processor(text=texts, images=imgs, padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        last = inputs["attention_mask"].sum(1) - 1
        logits = out.logits[torch.arange(len(kept)), last].float()
        probs = torch.softmax(logits, dim=-1)
        py = probs[:, yes_ids].sum(1); pn = probs[:, no_ids].sum(1)
        pv = (py / (py + pn).clamp_min(1e-9)).cpu().numpy()
        hv = out.hidden_states[-1][torch.arange(len(kept)), last].float().cpu().numpy()
        for j, i in enumerate(kept):
            p_yes[i] = pv[j]; hidden[i] = hv[j].astype(np.float16)
        ok += len(kept)
        if ok % 2000 < B:
            el = time.time() - t0
            eta = el / max(ok, 1) * (len(todo) - ok - skip)
            print(f"  {ok+skip}/{len(todo)}  {el/max(ok,1):.2f}s/样本  ETA {eta/3600:.1f}h", flush=True)

    p_yes.flush(); hidden.flush()
    done = int((np.asarray(p_yes) >= 0).sum())
    print(f"DONE ok={ok} skip={skip} total_filled={done}/{N}")
    print(f"saved -> {out_p} + {out_h}")


if __name__ == "__main__":
    main()
