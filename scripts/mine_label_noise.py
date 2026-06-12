"""Mine label-error candidates by cross-model consensus.

Two independent architectures (DINOv2 vision head + Qwen-VL scorer) trained on
the SAME noisy labels. The key signal:

  * Both models land FAR from GT, in the SAME direction  -> the LABEL is suspect
    (not the models — two unrelated inductive biases don't co-hallucinate the
    same error). Suspicion = min(|e_dino|, |e_qwen|) when signs agree, else 0.
    This is the lower bound on how far GT sits from independent consensus.

  * Models disagree with EACH OTHER (|p_dino - p_qwen| large) -> intrinsically
    ambiguous / hard sample. Useful for triage, NOT a label-error signal.

Outputs (per channel + combined):
  outputs/label_noise/candidates_{channel}.csv   ranked suspect labels
  outputs/label_noise/summary.json                counts at thresholds
  outputs/label_noise/review.html                 visual adjudication page
                                                   (images via running demo /api/image)

Usage:  python asset_quality_scorer/scripts/mine_label_noise.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

PKG = Path(__file__).resolve().parents[1]
CH = ["base_color", "normal_map", "roughness", "metallic"]
OUT = PKG / "outputs/label_noise"
# All general (full-channel) scorers. More independent architectures = harder
# evidence that a co-agreed deviation is a LABEL error, not a model quirk.
MODELS = [("DINOv2", "dinov2_large_multitask_emd_all"),
          ("Qwen", "vlm_scorer_a_old50k_oldtest"),
          ("ConvNeXt", "archive/convnext_base_multitask_emd")]
MIN_SUSP = 0.0    # render every asset whose models co-deviate from GT (susp>this);
                  # the page's threshold buttons filter further client-side


def load(run):
    p = PKG / "outputs/runs" / run / "demo_predictions.json"
    return {a["name"]: a for a in json.loads(p.read_text())["assets"]} if p.exists() else None


def nearblack_map():
    """name -> metallic non-black fraction (for flagging known-noisy near-black)."""
    try:
        meta = json.loads((PKG / "cache/224/meta.json").read_text())
        frac = np.load(PKG / "dataset/metallic_nonblack.npy")
        return dict(zip(meta["model_names"], frac))
    except Exception:
        return {}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    loaded = [(s, load(r)) for s, r in MODELS]
    loaded = [(s, a) for s, a in loaded if a is not None]
    short = [s for s, _ in loaded]
    names = sorted(set.intersection(*[set(a) for _, a in loaded]))
    nb = nearblack_map()
    print(f"{len(loaded)} models {short}, {len(names)} common assets\n")

    summary = {}
    review = {}  # channel -> list of top rows for HTML
    for ch in CH:
        rows = []
        for n in names:
            gt = float(loaded[0][1][n]["gt"][ch])
            preds = {s: round(float(a[n]["pred"][ch]), 2) for s, a in loaded}
            errs = [v - gt for v in preds.values()]
            # all models agree in sign -> suspicion = min |error| (lower bound on
            # GT's distance from unanimous consensus); any sign disagreement -> 0
            allpos, allneg = all(e > 0 for e in errs), all(e < 0 for e in errs)
            susp = min(abs(e) for e in errs) if (allpos or allneg) else 0.0
            spread = max(preds.values()) - min(preds.values())  # inter-model disagreement
            frac = nb.get(n, None)
            # bucket: A = models say HIGHER than GT (GT likely too low -> relabel);
            #         B = models say LOWER than GT (for metallic: blind spot on a
            #             correctly-empty map; GT often right -> do NOT relabel)
            bucket = "A" if allpos else ("B" if allneg else "-")
            rows.append({
                "name": n, "gt": gt, **{s: preds[s] for s in short},
                "consensus": round(sum(preds.values()) / len(preds), 2),
                "suspicion": round(susp, 3), "bucket": bucket,
                "direction": "高估GT被压低" if allpos else ("低估GT被抬高" if allneg else "—"),
                "spread": round(spread, 3),
                "nb_frac": (round(float(frac), 4) if frac is not None else ""),
            })
        rows.sort(key=lambda r: r["suspicion"], reverse=True)

        # CSV
        import csv
        cols = ["name", "gt", *short, "consensus", "suspicion", "bucket", "direction",
                "spread", "nb_frac"]
        with open(OUT / f"candidates_{ch}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

        susp_arr = np.array([r["suspicion"] for r in rows])
        hi = [r for r in rows if r["suspicion"] > 1.5]
        summary[ch] = {
            "n": len(rows),
            "suspicion>1.0": int((susp_arr > 1.0).sum()),
            "suspicion>1.5": int((susp_arr > 1.5).sum()),
            "suspicion>2.0": int((susp_arr > 2.0).sum()),
            "pct>1.5": round(float((susp_arr > 1.5).mean()) * 100, 1),
            "top_suspicion": round(float(susp_arr.max()), 2),
            "A>1.5": sum(1 for r in hi if r["bucket"] == "A"),
            "B>1.5": sum(1 for r in hi if r["bucket"] == "B"),
        }
        review[ch] = [r for r in rows if r["suspicion"] > MIN_SUSP]
        s = summary[ch]
        print(f"[{ch:11s}] >1.5:{s['suspicion>1.5']:4d}  "
              f"(A重标 {s['A>1.5']} / B盲区 {s['B>1.5']})  max={s['top_suspicion']}")

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    write_html(review, summary, short)
    print(f"\nsaved -> {OUT}/  (candidates_*.csv, summary.json, review.html)")


def write_html(review, summary, short):
    CH_LAB = {"base_color": "Base Color", "normal_map": "Normal",
              "roughness": "Roughness", "metallic": "Metallic"}
    tabs, panels = [], []
    for ch in CH:
        s = summary[ch]
        tabs.append(f'<button class="tab" onclick="show(\'{ch}\')" id="t_{ch}">'
                    f'{CH_LAB[ch]} <span class=badge>{s["suspicion>1.5"]}</span></button>')
        cards = []
        for r in review[ch]:
            nbtag = (f'<span class="nb">近黑 {r["nb_frac"]}</span>'
                     if ch == "metallic" and r["nb_frac"] != "" and float(r["nb_frac"]) < 0.02 else "")
            imgs = "".join(
                f'<figure><img loading=lazy onclick="lb(this)" '
                f'data-cap="{r["name"]} · {"渲染" if c=="render" else CH_LAB.get(c,c)}" '
                f'src="/img?thumb=1&ch={c}&n={r["name"]}">'
                f'<figcaption>{"渲染" if c=="render" else CH_LAB.get(c,c)}</figcaption></figure>'
                for c in ["render", ch])
            pscores = "".join(f'<span class=p>{m} {r[m]}</span>' for m in short)
            btag = (f'<span class="bk bk{r["bucket"]}">{r["bucket"]}</span>'
                    if r["bucket"] in ("A", "B") else "")
            cards.append(f'''<div class=card data-susp="{r['suspicion']}" data-bk="{r['bucket']}">
  <div class=imgs>{imgs}</div>
  <div class=meta>
    <div class=name title="{r['name']}">{btag}{r['name']}</div>
    <div class=scores><span class=gt>GT {r['gt']:.0f}</span>{pscores}</div>
    <div class=susp>疑似度 <b>{r['suspicion']}</b> · {r['direction']} {nbtag}</div>
    <div class=dis>模型间分歧 {r['spread']}</div>
  </div></div>''')
        panels.append(f'<div class=panel id="p_{ch}" style="display:none">'
                      f'<div class=hint>{len(short)} 个独立架构（{"、".join(short)}）一致偏离 GT。'
                      f'<b>方向</b>把候选分两类：<span class=bk bkA>A</span> <b>模型高/GT低</b>——'
                      f'物体疑似该有金属却被打低分，<b>真·重标候选</b>；'
                      f'<span class=bk bkB>B</span> <b>模型低/GT高</b>——'
                      f'多为正确的空 map 被模型误判（<b>模型盲区，标签往往是对的，勿重标</b>）。'
                      f'共 {s["n"]} 资产，>1.5 的 <b>{s["suspicion>1.5"]}</b> 个'
                      f'（A {s["A>1.5"]} / B {s["B>1.5"]}）。用下方阈值 + 方向按钮筛选。</div>'
                      f'<div class=grid id="g_{ch}">{"".join(cards)}</div>'
                      f'<div class=more id="m_{ch}"></div></div>')
    html = f'''<!doctype html><html lang=zh><meta charset=utf-8>
<title>标签噪声复核 · PBR Scorer</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;font:14px/1.5 system-ui,Segoe UI,sans-serif;background:#0e1117;color:#e6edf3}}
header{{padding:18px 24px;border-bottom:1px solid #21262d}}h1{{margin:0;font-size:18px}}
.sub{{color:#8b949e;font-size:13px;margin-top:4px}}
.tabs{{display:flex;gap:8px;padding:14px 24px;flex-wrap:wrap}}
.tab{{background:#161b22;color:#c9d1d9;border:1px solid #30363d;border-radius:8px;padding:8px 14px;cursor:pointer;font-size:13px}}
.tab.on{{background:#1f6feb;color:#fff;border-color:#1f6feb}}
.badge{{background:#da3633;color:#fff;border-radius:10px;padding:1px 7px;font-size:11px;margin-left:4px}}
.hint{{margin:0 24px 14px;color:#8b949e;font-size:13px;background:#161b22;border:1px solid #21262d;border-radius:8px;padding:10px 14px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px;padding:0 24px 40px}}
.card{{background:#161b22;border:1px solid #21262d;border-radius:10px;overflow:hidden}}
.imgs{{display:flex}}.imgs figure{{margin:0;flex:1;text-align:center}}
.imgs img{{width:100%;aspect-ratio:1;object-fit:cover;background:#000;display:block}}
.imgs figcaption{{font-size:11px;color:#8b949e;padding:3px}}
.meta{{padding:10px 12px}}.name{{font-size:11px;color:#8b949e;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.scores{{margin:6px 0;display:flex;gap:6px;flex-wrap:wrap}}
.gt{{background:#238636;color:#fff;border-radius:5px;padding:1px 8px;font-weight:600}}
.p{{background:#21262d;border-radius:5px;padding:1px 8px}}
.susp{{font-size:13px}}.susp b{{color:#f0883e;font-size:15px}}
.dis{{font-size:12px;color:#6e7681;margin-top:2px}}
.nb{{background:#3d1d1d;color:#ffa198;border-radius:4px;padding:0 6px;font-size:11px;margin-left:6px}}
.threshbar{{display:flex;gap:8px;align-items:center;padding:0 24px 12px;flex-wrap:wrap;color:#8b949e;font-size:13px}}
.tb{{background:#161b22;color:#c9d1d9;border:1px solid #30363d;border-radius:7px;padding:5px 12px;cursor:pointer;font-size:13px}}
.tb.on{{background:#f0883e;color:#0e1117;border-color:#f0883e;font-weight:600}}
#shown{{color:#e6edf3;font-weight:600}}
.lazyhidden{{display:none}}
.bk{{display:inline-block;border-radius:4px;padding:0 6px;font-weight:700;font-size:11px;margin-right:5px}}
.bkA{{background:#1f6feb;color:#fff}}.bkB{{background:#8957e5;color:#fff}}
.imgs img{{cursor:zoom-in}}
#lb{{display:none;position:fixed;inset:0;z-index:99;background:rgba(0,0,0,.9);
  justify-content:center;align-items:center;cursor:zoom-out}}
#lb.on{{display:flex}}
#lb img{{max-width:94vw;max-height:88vh;object-fit:contain;background:#000;border:1px solid #30363d}}
#lbcap{{position:fixed;top:14px;left:0;right:0;text-align:center;color:#e6edf3;font-size:13px;pointer-events:none}}
</style>
<header><h1>🔍 标签噪声复核 — 多模型一致性挖掘</h1>
<div class=sub>{len(short)} 个独立架构（{"、".join(short)}）在同一套噪声标签上训练。它们<b>一致</b>地远离某条 GT，
说明该 GT 很可能是<b>标注错误</b>（无关归纳偏置不会同向同幅幻觉）。这是唯一还能撬动 0.79 标签天花板的杠杆。</div></header>
<div class=tabs>{"".join(tabs)}</div>
<div class=threshbar>疑似度阈值：
  <button class=tb onclick="setThresh(0)" id="th0">全部 &gt;0</button>
  <button class=tb onclick="setThresh(0.5)" id="th0.5">&gt;0.5</button>
  <button class=tb onclick="setThresh(1)" id="th1">&gt;1.0</button>
  <button class=tb onclick="setThresh(1.5)" id="th1.5">&gt;1.5</button>
  <button class=tb onclick="setThresh(2)" id="th2">&gt;2.0</button>
  <span style="width:18px"></span>方向：
  <button class=tb onclick="setDir('all')" id="dall">全部</button>
  <button class=tb onclick="setDir('A')" id="dA">A·重标候选</button>
  <button class=tb onclick="setDir('B')" id="dB">B·模型盲区</button>
  <span style="margin-left:auto">当前显示 <span id=shown>0</span> 个</span>
</div>
{"".join(panels)}
<div id=lb onclick="this.classList.remove('on')"><div id=lbcap></div><img id=lbimg></div>
<script>
const CHS={json.dumps(CH)};
function lb(img){{
  const big=img.src.replace('thumb=1&','');
  document.getElementById('lbimg').src=big;
  document.getElementById('lbcap').textContent=img.dataset.cap;
  document.getElementById('lb').classList.add('on');
}}
document.addEventListener('keydown',e=>{{if(e.key==='Escape')document.getElementById('lb').classList.remove('on');}});
let CUR='metallic', THRESH=1, DIR='all';
function apply(){{
  let n=0;
  document.querySelectorAll('#g_'+CUR+' .card').forEach(c=>{{
    const sv=parseFloat(c.dataset.susp);
    const ok=sv>=THRESH-1e-9 && sv>0 && (DIR==='all' || c.dataset.bk===DIR);
    c.classList.toggle('lazyhidden', !ok); if(ok) n++;
  }});
  document.getElementById('shown').textContent=n;
}}
function setDir(d){{
  DIR=d;
  for(const x of ['all','A','B']) document.getElementById('d'+x).classList.toggle('on', x===d);
  apply();
}}
function show(ch){{
  CUR=ch;
  for(const c of CHS){{
    document.getElementById('p_'+c).style.display = c===ch?'block':'none';
    document.getElementById('t_'+c).classList.toggle('on', c===ch);
  }}
  apply();
}}
function setThresh(t){{
  THRESH=t;
  for(const x of [0,0.5,1,1.5,2]) document.getElementById('th'+x).classList.toggle('on', x===t);
  apply();
}}
setThresh(1); setDir('all'); show('metallic');
</script></html>'''
    (OUT / "review.html").write_text(html)


if __name__ == "__main__":
    main()
