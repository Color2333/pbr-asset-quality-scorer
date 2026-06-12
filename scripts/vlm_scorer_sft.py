"""QLoRA SFT: teach a Qwen-VL to score PBR channel maps 0-5 (Q-Align/DeQA recipe).

Supervision = Gaussian-softened 6-bin distribution over the digit tokens (DeQA:
soft labels absorb annotation noise far better than one-hot), loss = CE against
that soft target at the single answer position. Vision tower frozen, LoRA on the
LLM, 4-bit base → fits one 24G card with gradient checkpointing.

Usage:
    CUDA_VISIBLE_DEVICES=0 ~/miniconda3/envs/qwen-vl/bin/python \
      asset_quality_scorer/scripts/vlm_scorer_sft.py \
      --model-dir models/Qwen2.5-VL-7B-Instruct --items 60000 --out qwen25_sft
"""
from __future__ import annotations
import argparse, json, math, random, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_scorer_eval import (CHANNELS, CH_PROMPT, METALLIC_EXPERT_PROMPT,  # noqa: E402
                             build_items, make_claim_overlay)

PKG = Path(__file__).resolve().parents[1]


def soft_target(score: int, sigma: float = 0.75) -> torch.Tensor:
    ks = torch.arange(6, dtype=torch.float32)
    w = torch.exp(-0.5 * ((ks - score) / sigma) ** 2)
    return w / w.sum()


METALLIC_OVERLAY = False   # set from args in main()
MULTI_CHANNEL = False       # set from args in main(): feed ALL channel maps for context

def make_msgs(item, ch):
    if MULTI_CHANNEL and ch == "metallic":
        # full context: let the model reason about object identity from every map
        # (e.g. wood -> empty metallic is CORRECT). Same criteria/prompt as plain;
        # only the available imagery changes, to isolate the effect of multi-channel input.
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
        return imgs, text
    imgs = [Image.open(item["paths"][ch]).convert("RGB"),
            Image.open(item["paths"]["render"]).convert("RGB")]
    labels = ["the channel map being rated", "the final render of the asset"]
    if ch != "base_color":
        imgs.append(Image.open(item["paths"]["base_color"]).convert("RGB"))
        labels.append("the base color map (context)")
    crit = CH_PROMPT[ch]
    if ch == "metallic" and METALLIC_OVERLAY:
        imgs.append(make_claim_overlay(imgs[2], imgs[0]))
        labels.append("the CLAIM OVERLAY (albedo with metal-claimed regions tinted red)")
        crit = METALLIC_EXPERT_PROMPT
    body = "; ".join(f"image {i+1} = {l}" for i, l in enumerate(labels))
    text = (f"You are a professional 3D PBR material QA inspector. {body}. "
            f"Rate the quality of {crit}. "
            f"Score 0 = broken/missing/wrong, 3 = usable with flaws, 5 = excellent. "
            f"Answer with exactly one digit (0-5).")
    return imgs, text


def stratified_items(rows, n_items, seed=0, channels=None):
    """(channel, score) 配额尽量均匀 — tailsamp 的同款动机: 稀有极端分不被淹没."""
    rng = random.Random(seed)
    buckets: dict[tuple, list] = {}
    for it in rows:
        for ch in (channels or CHANNELS):
            buckets.setdefault((ch, it["scores"][ch]), []).append((it, ch))
    for b in buckets.values():
        rng.shuffle(b)
    quota = max(1, n_items // len(buckets))
    picked = []
    for b in buckets.values():
        picked.extend(b[:quota])
    # 不足配额的桶剩余名额用大桶补齐
    if len(picked) < n_items:
        rest = [x for b in buckets.values() for x in b[quota:]]
        rng.shuffle(rest)
        picked.extend(rest[:n_items - len(picked)])
    rng.shuffle(picked)
    from collections import Counter
    cnt = Counter(ch for _, ch in picked)
    print(f"SFT items: {len(picked)}  per-channel {dict(cnt)}")
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(PKG.parent / "models/Qwen2.5-VL-7B-Instruct"))
    ap.add_argument("--items", type=int, default=60000)
    ap.add_argument("--val-assets", type=int, default=120, help="quick-val asset count")
    ap.add_argument("--max-pixels", type=int, default=448 * 448)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--sigma", type=float, default=0.75)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--use-dora", action="store_true", help="DoRA (weight-decomposed LoRA)")
    ap.add_argument("--vision-lora", action="store_true",
                    help="also adapt the vision tower (our maps are far from natural images)")
    ap.add_argument("--eval-every", type=int, default=2500, help="items between quick-vals")
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", default=None); ap.add_argument("--data-root", default=None)
    ap.add_argument("--channels", nargs="+", default=None, help="train only these channels")
    ap.add_argument("--metallic-overlay", action="store_true")
    ap.add_argument("--multi-channel", action="store_true",
                    help="feed ALL channel maps (render+base+normal+roughness+metallic) for metallic")
    ap.add_argument("--extreme-weight", type=float, default=1.0,
                    help="loss multiplier for GT in {0,5} (push the model to commit at extremes)")
    args = ap.parse_args()

    # DDP: when launched via torchrun, each rank trains on a shard and grads are
    # all-reduced manually (robust with 4-bit + grad-checkpointing; no DDP wrapper).
    import os
    import torch.distributed as dist
    DDP = "LOCAL_RANK" in os.environ
    if DDP:
        # long timeout: rank0's quick_val (cephfs, no prefetch) can exceed the 10-min
        # NCCL default while other ranks wait at the post-val barrier -> watchdog abort.
        from datetime import timedelta
        dist.init_process_group("nccl", timeout=timedelta(hours=3))
        local_rank = int(os.environ["LOCAL_RANK"]); world = dist.get_world_size(); rank = dist.get_rank()
        torch.cuda.set_device(local_rank)
    else:
        local_rank, world, rank = 0, 1, 0
    device = f"cuda:{local_rank}"
    is_main = rank == 0

    out_dir = PKG / "outputs/runs" / f"vlm_scorer_{args.out}"
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    global METALLIC_OVERLAY, MULTI_CHANNEL
    METALLIC_OVERLAY = args.metallic_overlay
    MULTI_CHANNEL = args.multi_channel
    train_rows = build_items("train", csv_path=args.csv, data_root=args.data_root)
    val_rows = build_items("val", sample=args.val_assets, seed=7, csv_path=args.csv, data_root=args.data_root)
    chs = args.channels or CHANNELS
    items = stratified_items(train_rows, args.items, channels=chs)
    if DDP:
        # trim to a multiple of (accum*world) so every rank runs identical #accum-steps
        # (keeps manual grad all-reduce in lockstep), then take this rank's shard
        keep = (len(items) // (args.accum * world)) * (args.accum * world)
        items = items[:keep][rank::world]
        if is_main:
            print(f"DDP world={world}: {keep} items -> {len(items)}/rank", flush=True)

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, device_map={"": local_rank},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_use_double_quant=True,
                                               bnb_4bit_compute_dtype=torch.bfloat16))
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    if args.vision_lora:
        # regex: LLM proj 全家桶 + 视觉塔 blocks 的 qkv/proj/mlp(法线贴图/UV图是域外图像, 视觉塔需要适配)
        targets = (r".*\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.(gate_proj|up_proj|down_proj|fc1|fc2))"
                   r"|.*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$")
    else:
        targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
                      target_modules=targets, use_dora=args.use_dora,
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    processor = AutoProcessor.from_pretrained(args.model_dir, min_pixels=224 * 224,
                                              max_pixels=args.max_pixels)
    tok = processor.tokenizer
    digit_ids = [tok.encode(str(k), add_special_tokens=False)[0] for k in range(6)]
    digit_ids_t = torch.tensor(digit_ids, device=device)

    def prepare_item(item, ch):
        """CPU-only: decode 3x 2048^2 PNG + processor — ~40% of item time, prefetchable."""
        imgs, text = make_msgs(item, ch)
        msgs = [{"role": "user", "content":
                 [{"type": "image", "image": im} for im in imgs] + [{"type": "text", "text": text}]}]
        prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return processor(text=[prompt], images=imgs, return_tensors="pt")

    def forward_item(item, ch, train=True):
        inputs = prepare_item(item, ch).to(device)
        out = model(**inputs)
        logits = out.logits[0, -1]                      # 下一个 token = 答案位
        return logits

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0)
    total_steps = math.ceil(len(items) / args.accum)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / max(total_steps * 0.03, 1),
                                              0.5 * (1 + math.cos(math.pi * s / total_steps))))

    def quick_val():
        model.eval()
        preds = {c: [] for c in chs}; gts = {c: [] for c in chs}
        with torch.no_grad():
            for it in val_rows:
                for ch in chs:
                    logits = forward_item(it, ch, train=False)
                    p6 = torch.softmax(logits[digit_ids_t].float(), dim=-1)
                    preds[ch].append(float((p6 * torch.arange(6, device=p6.device)).sum()))
                    gts[ch].append(it["scores"][ch])
        model.train()
        s = {c: float(spearmanr(preds[c], gts[c]).statistic) for c in chs}
        s["mean"] = float(np.mean(list(s.values())))
        return s

    model.train()
    best = -1.0
    trainable = [p for p in model.parameters() if p.requires_grad]
    # prefetch pipeline: decode/preprocess runs in threads while the GPU computes.
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=3)
    DEPTH = 6
    futs = {j: pool.submit(prepare_item, *items[j]) for j in range(min(DEPTH, len(items)))}
    t0 = time.time(); run_loss = 0.0
    for i, (item, ch) in enumerate(items):
        inputs = futs.pop(i).result().to(device)
        if i + DEPTH < len(items):
            futs[i + DEPTH] = pool.submit(prepare_item, *items[i + DEPTH])
        target = soft_target(item["scores"][ch], args.sigma).to(device)
        out = model(**inputs)
        logits = out.logits[0, -1]
        logp = torch.log_softmax(logits.float(), dim=-1)[digit_ids_t]
        ew = args.extreme_weight if item["scores"][ch] in (0, 5) else 1.0
        loss = ew * -(target * logp).sum() / args.accum
        loss.backward()
        run_loss += float(loss) * args.accum
        if (i + 1) % args.accum == 0:
            if DDP:  # average LoRA grads across ranks -> all ranks step identically
                for p in trainable:
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM); p.grad /= world
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if (i + 1) % 500 == 0 and is_main:
            el = time.time() - t0
            print(f"  item {i+1}/{len(items)}  loss={run_loss/500:.4f}  "
                  f"{el/(i+1):.2f}s/项  ETA {(len(items)-i-1)*el/(i+1)/3600:.1f}h", flush=True)
            run_loss = 0.0
        if (i + 1) % args.eval_every == 0 or i == len(items) - 1:
            if is_main:  # only rank0 evals + saves; others wait at the barrier
                s = quick_val()
                print(f"  [val@{i+1}] mean={s['mean']:.4f}  " +
                      " ".join(f"{c.split('_')[0]}={s[c]:.3f}" for c in chs), flush=True)
                model.save_pretrained(out_dir / "last")
                if s["mean"] > best:
                    best = s["mean"]
                    model.save_pretrained(out_dir / "best")
                print(f"  ⭐ new best {best:.4f}", flush=True)
            if DDP:  # non-main ranks waited here while rank0 ran val+save -> resync
                dist.barrier()
    if is_main:
        print(f"done. best val mean SRCC = {best:.4f}  adapters -> {out_dir}/best")
    if DDP:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
