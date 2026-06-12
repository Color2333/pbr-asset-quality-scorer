"""Build a standalone, insight-rich data-overview HTML comparing every scorer
model on the test split. Auto-discovers models from REGISTRY (skips those
without a demo_predictions.json). Computes:

  - per-channel SRCC / acc / within-1 / MAE
  - metallic near-black analysis (SRCC / missing-AUC / detection P-R-F1)
  - metallic confusion matrices
  - cross-model PREDICTION-correlation matrix (per channel)  ← agreement
  - ENSEMBLE gain: pairwise + all-model average vs single-best  ← complementarity
  - per-score MAE breakdown (which score levels each model nails)

Embeds all data inline → one self-contained file. Style mirrors
outputs/scorer_analysis.html.

Usage:  python asset_quality_scorer/scripts/build_scorer_overview.py
Open    outputs/scorer_overview.html
"""
from __future__ import annotations
import json, itertools
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

PKG = Path(__file__).resolve().parents[1]
CH = ["base_color", "normal_map", "roughness", "metallic"]
LAB = {"base_color": "Base Color", "normal_map": "Normal", "roughness": "Roughness", "metallic": "Metallic"}

# label, short, demo_predictions.json path (relative to outputs/runs)
REGISTRY = [
    ("DINOv2-L EMD", "DINOv2", "dinov2_large_multitask_emd_all"),
    ("ConvNeXt-B EMD", "ConvNeXt", "archive/convnext_base_multitask_emd"),
    ("Qwen2.5-VL SFT 60k", "Qwen2.5", "vlm_scorer_a_old50k_oldtest"),
    ("Qwen2.5-VL 全覆盖160k", "Qwen2.5-full", "vlm_scorer_fullcover_oldtest"),
    ("Qwen3-VL SFT", "Qwen3", "vlm_scorer_qwen3_smoke10k_oldtest"),
]


def near_black_map():
    meta = json.loads((PKG / "cache/224/meta.json").read_text())
    frac = np.load(PKG / "dataset/metallic_nonblack.npy")
    return {n: float(frac[i]) for i, n in enumerate(meta["model_names"])}


def load_assets(run):
    p = PKG / "outputs/runs" / run / "demo_predictions.json"
    if not p.exists():
        return None
    return {a["name"]: a for a in json.loads(p.read_text())["assets"]}


def main():
    nb_map = near_black_map()
    models = []
    for label, short, run in REGISTRY:
        a = load_assets(run)
        if a is None:
            print(f"[skip] {label}: no demo_predictions.json")
            continue
        models.append({"label": label, "short": short, "run": run, "A": a})
    # common assets across all loaded models
    names = sorted(set.intersection(*[set(m["A"]) for m in models]))
    print(f"{len(models)} models, {len(names)} common assets")
    nb = np.array([nb_map.get(n, 1.0) < 0.02 for n in names])

    P = {}  # short -> {ch: array}
    G = {ch: np.array([models[0]["A"][n]["gt"][ch] for n in names]) for ch in CH}
    for m in models:
        P[m["short"]] = {ch: np.array([m["A"][n]["pred"][ch] for n in names]) for ch in CH}

    def srcc(a, b): return float(spearmanr(a, b).statistic)

    payload = {"shorts": [m["short"] for m in models], "labels": {m["short"]: m["label"] for m in models},
               "n": len(names), "n_nb": int(nb.sum()), "perch": {}, "metric": {}, "nb": {},
               "cm": {}, "corr": {}, "ens": {}, "perscore": {}, "dist": {}}

    # GT vs predicted label distribution (6-bin %) + std, per channel
    for ch in CH:
        g = G[ch]
        d = {"gt": [round(float((g == k).mean()) * 100, 1) for k in range(6)],
             "gt_std": round(float(g.std()), 2), "models": {}}
        for m in models:
            p = P[m["short"]][ch]; r = np.clip(np.round(p), 0, 5).astype(int)
            d["models"][m["short"]] = {"hist": [round(float((r == k).mean()) * 100, 1) for k in range(6)],
                                       "std": round(float(p.std()), 2), "mean": round(float(p.mean()), 2),
                                       "gt_mean": round(float(g.mean()), 2)}
        payload["dist"][ch] = d

    for m in models:
        s = m["short"]; payload["metric"][s] = {}
        for ch in CH:
            p, g = P[s][ch], G[ch]; r = np.clip(np.round(p), 0, 5).astype(int)
            payload["metric"][s][ch] = {"srcc": round(srcc(p, g), 4), "acc": round(float((r == g).mean()), 3),
                                        "within1": round(float((np.abs(r - g) <= 1).mean()), 3),
                                        "mae": round(float(np.abs(p - g).mean()), 3)}
        # metallic near-black
        p, g = P[s]["metallic"], G["metallic"]; y = (g[nb] <= 2).astype(int)
        pb = (p[nb] < 2.5).astype(int)
        tp = int(((pb == 1) & (y == 1)).sum()); fp = int(((pb == 1) & (y == 0)).sum()); fn = int(((pb == 0) & (y == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0; rec = tp / (tp + fn) if tp + fn else 0
        payload["nb"][s] = {"srcc": round(srcc(p[nb], g[nb]), 4), "srcc_nonblack": round(srcc(p[~nb], g[~nb]), 4),
                            "auc": round(float(roc_auc_score(y, -p[nb])), 4),
                            "prec": round(prec, 3), "rec": round(rec, 3),
                            "f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0}
        # metallic confusion
        cm = np.zeros((6, 6), int)
        for gg, rr in zip(g, np.clip(np.round(p), 0, 5).astype(int)):
            cm[int(gg), int(rr)] += 1
        payload["cm"][s] = cm.tolist()
        # per-score MAE (metallic)
        payload["perscore"][s] = {int(k): round(float(np.abs(p[g == k] - g[g == k]).mean()), 2)
                                  if (g == k).any() else None for k in range(6)}

    # cross-model prediction correlation (per channel) + ensemble gain
    shorts = payload["shorts"]
    for ch in CH:
        payload["corr"][ch] = [[round(srcc(P[a][ch], P[b][ch]), 3) for b in shorts] for a in shorts]
        # all-pairs ensemble + full ensemble
        singles = {s: srcc(P[s][ch], G[ch]) for s in shorts}
        best_single = max(singles.values())
        full = srcc(np.mean([P[s][ch] for s in shorts], axis=0), G[ch])
        pairs = []
        for a, b in itertools.combinations(shorts, 2):
            e = srcc((P[a][ch] + P[b][ch]) / 2, G[ch])
            pairs.append({"pair": f"{a}+{b}", "srcc": round(e, 4),
                          "gain": round(e - max(singles[a], singles[b]), 4)})
        payload["ens"][ch] = {"singles": {s: round(v, 4) for s, v in singles.items()},
                              "best_single": round(best_single, 4),
                              "full": round(full, 4), "full_gain": round(full - best_single, 4),
                              "pairs": sorted(pairs, key=lambda x: -x["gain"])}

    out = PKG / "outputs/scorer_overview.html"
    out.write_text(_HTML.replace("/*DATA*/", json.dumps(payload, ensure_ascii=False)), encoding="utf-8")
    print(f"wrote {out}")
    # console summary
    for ch in CH:
        e = payload["ens"][ch]
        print(f"  {ch:<12} best单模 {e['best_single']:.3f}  全集成 {e['full']:.3f} ({e['full_gain']:+.3f})")


_HTML = r"""<!DOCTYPE html><html lang=zh><head><meta charset=UTF-8>
<title>PBR Scorer — 模型纵览</title><style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#f5f6fa;color:#2d3436;padding-bottom:60px}
h1{padding:26px 32px 4px;font-size:22px}.subtitle{padding:0 32px 18px;color:#636e72;font-size:14px}
.section{background:#fff;border-radius:10px;padding:22px 24px;margin:0 32px 18px;box-shadow:0 1px 4px rgba(0,0,0,.08)}
.section-title{font-size:15px;font-weight:600;margin-bottom:4px}
.section-desc{font-size:13px;color:#636e72;margin-bottom:16px;line-height:1.6}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;margin:0 32px 22px}
.card{background:#fff;border-radius:10px;padding:16px;box-shadow:0 1px 4px rgba(0,0,0,.08);border-left:4px solid}
.card-title{font-size:12px;color:#636e72}.card-value{font-size:24px;font-weight:700;margin:5px 0 2px}.card-sub{font-size:11px;color:#b2bec3}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:8px 10px;text-align:center;border-bottom:1px solid #f0f2f5}
th{background:#f8f9fa;font-weight:600;color:#495057;border-bottom:2px solid #dee2e6}
td.l,th.l{text-align:left;font-weight:600}
.good{color:#00b894}.ok{color:#fdcb6e}.bad{color:#e17055}.win{font-weight:700}
.bar-row{display:flex;align-items:center;margin-bottom:7px;gap:10px}
.bar-label{width:90px;font-size:12px;text-align:right;flex-shrink:0;color:#636e72}
.bar-track{flex:1;height:22px;background:#f0f2f5;border-radius:4px;overflow:hidden}
.bar-fill{height:100%;display:flex;align-items:center;padding-left:8px;font-size:11px;color:#fff;font-weight:600}
.bar-num{width:50px;font-size:12px;text-align:right;flex-shrink:0}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px}
table.conf td,table.heat td{width:50px;height:36px;font-size:12px}
.note{font-size:12px;color:#2d3436;background:#eef6ff;border-left:3px solid #0984e3;padding:10px 14px;border-radius:4px;margin-top:14px;line-height:1.7}
.tag{display:inline-block;padding:1px 7px;border-radius:8px;font-size:11px;font-weight:600;margin-left:6px}
.tag-hi{background:#d8f5e8;color:#00875a}.tag-lo{background:#ffe9e0;color:#c0392b}
h2.blk{font-size:13px;font-weight:600;margin:6px 0 8px;color:#2d3436}
</style></head><body>
<h1>🎨 PBR Scorer — 模型纵览</h1>
<div class=subtitle id=sub>test set · 同口径对比</div>
<div class=cards id=cards></div>

<div class=section><div class=section-title>① 逐通道 SRCC</div>
 <div class=section-desc>排序相关性（主指标）。各通道下并列所有模型。</div><div id=srcc></div></div>

<div class=section><div class=section-title>② 全指标对比</div>
 <div class=section-desc>SRCC / 精确acc / within-1 / MAE。acc 偏低是连续期望读出的特性，<b>within-1 才是产品可用度</b>。</div>
 <div id=tbl></div></div>

<div class=section><div class=section-title>③ 模型间「预测相关性」—— 它们在犯同样的错吗？</div>
 <div class=section-desc>两两之间预测的 Spearman 相关。<b>相关越低 = 信息源越正交 = 集成越可能突破天花板</b>。
   重点看 metallic：视觉模型(DINOv2/ConvNeXt)之间通常 &gt;0.95（同源），与 VLM 之间明显更低（世界知识 vs 像素统计）。</div>
 <div class=grid2 id=corr></div></div>

<div class=section><div class=section-title>④ 集成增益 —— 相关性低的通道能白捡分</div>
 <div class=section-desc>单模型最佳 vs 取平均集成。零训练成本。<b>增益与相关性反相关</b>是核心规律。</div>
 <div id=ens></div></div>

<div class=section><div class=section-title>⑤ Metallic 近黑战场 (47% 数据)</div>
 <div class=section-desc>全黑 metallic 图。SRCC 被噪声标签锁死(~0.56)，<b>漏标 AUC / F1 才是真实可用信号</b>。</div>
 <div id=nb></div></div>

<div class=section><div class=section-title>⑥ Metallic 混淆矩阵 & 逐分 MAE</div>
 <div class=section-desc>行=真分，列=预测(四舍五入)。绿=命中，黄=within-1，红=错档。下方为每个真分档的 MAE。</div>
 <div class=grid2 id=cm></div><div id=perscore style="margin-top:16px"></div></div>

<div class=section><div class=section-title>⑦ 标签分布：GT vs 模型预测</div>
 <div class=section-desc>各分档(0-5)占比。<b>预测分布比 GT 窄 = 向均值塌缩</b>(连续期望读出的固有特性，尤其压缩 0/5 两端)。
   std 越接近 GT、两端越不被削平，说明模型越敢给极端分。</div>
 <div class=grid2 id=dist></div></div>

<script>
const D=/*DATA*/;
const CH=["base_color","normal_map","roughness","metallic"];
const LAB={base_color:"Base Color",normal_map:"Normal",roughness:"Roughness",metallic:"Metallic"};
const COL=["#6c5ce7","#0984e3","#00b894","#e17055","#e84393"];
const S=D.shorts, ci=Object.fromEntries(S.map((s,i)=>[s,i]));
function cls(v,a,b){return v>=a?'good':v>=b?'ok':'bad'}
document.getElementById('sub').textContent=`test set · ${D.n} 共同资产 · ${S.length} 模型 · 同口径`;
// cards
document.getElementById('cards').innerHTML=S.map((s,i)=>{
 const mean=(CH.reduce((a,ch)=>a+D.metric[s][ch].srcc,0)/4).toFixed(4);
 return `<div class=card style="border-left-color:${COL[i]}"><div class=card-title>${D.labels[s]}</div>`+
  `<div class=card-value>${mean}</div><div class=card-sub>mean SRCC</div></div>`;}).join('');
// ① srcc bars
let s='';
for(const ch of CH){s+=`<h2 class=blk>${LAB[ch]}</h2>`;
 S.forEach((sh,i)=>{const v=D.metric[sh][ch].srcc;
  s+=`<div class=bar-row><div class=bar-label>${sh}</div><div class=bar-track>`+
   `<div class=bar-fill style="width:${v*100}%;background:${COL[i]}">${v}</div></div></div>`;});}
document.getElementById('srcc').innerHTML=s;
// ② full table
let h='<table><thead><tr><th class=l>通道</th>';
S.forEach(sh=>h+=`<th colspan=4>${sh}</th>`);h+='</tr><tr><th class=l></th>';
S.forEach(()=>h+='<th>SRCC</th><th>acc</th><th>w-1</th><th>MAE</th>');h+='</tr></thead><tbody>';
for(const ch of CH){h+=`<tr><td class=l>${LAB[ch]}</td>`;
 S.forEach(sh=>{const c=D.metric[sh][ch];
  h+=`<td class=${cls(c.srcc,.8,.6)}>${c.srcc}</td><td>${(c.acc*100).toFixed(0)}%</td>`+
     `<td>${(c.within1*100).toFixed(0)}%</td><td>${c.mae}</td>`;});h+='</tr>';}
document.getElementById('tbl').innerHTML=h+'</tbody></table>';
// ③ correlation heatmaps per channel
function heat(ch){const M=D.corr[ch];let t=`<div><h2 class=blk>${LAB[ch]}</h2><table class=heat><thead><tr><th></th>`;
 S.forEach(sh=>t+=`<th>${sh}</th>`);t+='</tr></thead><tbody>';
 for(let i=0;i<S.length;i++){t+=`<tr><th>${S[i]}</th>`;for(let j=0;j<S.length;j++){const v=M[i][j];
  const a=(v-0.7)/0.3;const bg=i===j?'#eceff4':`rgba(108,92,231,${Math.max(0,Math.min(1,a))*.75+.08})`;
  t+=`<td style="background:${bg};color:${a>.6?'#fff':'#2d3436'}">${i===j?'—':v}</td>`;}t+='</tr>';}
 return t+'</tbody></table></div>';}
document.getElementById('corr').innerHTML=CH.map(heat).join('');
// ④ ensemble
let e='<table><thead><tr><th class=l>通道</th><th>单模最佳</th><th>全集成</th><th>增益</th><th class=l>最佳两两集成</th></tr></thead><tbody>';
for(const ch of CH){const x=D.ens[ch];const bp=x.pairs[0];
 const gtag=x.full_gain>=0.015?'<span class="tag tag-hi">显著</span>':x.full_gain<=0.004?'<span class="tag tag-lo">冗余</span>':'';
 e+=`<tr><td class=l>${LAB[ch]}</td><td>${x.best_single}</td><td class=win>${x.full}</td>`+
  `<td class=${x.full_gain>=0.015?'good':x.full_gain<=0.004?'bad':'ok'}>${x.full_gain>0?'+':''}${x.full_gain}${gtag}</td>`+
  `<td class=l>${bp.pair} → ${bp.srcc} (${bp.gain>0?'+':''}${bp.gain})</td></tr>`;}
e+='</tbody></table><div class=note>规律：<b>通道的集成增益 ∝ 模型间预测的不相关度</b>。'+
 'roughness/base_color 几个视觉模型高度同源(相关&gt;0.93)→ 集成几乎无收益；'+
 'metallic 因 VLM 引入正交的「世界知识」信息源 → 集成增益最大，是唯一能小幅突破单模型天花板的通道。</div>';
document.getElementById('ens').innerHTML=e;
// ⑤ near-black
let n='<table><thead><tr><th class=l>模型</th><th>近黑SRCC</th><th>非黑SRCC</th><th>漏标AUC</th><th>检测精度</th><th>召回</th><th>F1</th></tr></thead><tbody>';
S.forEach(sh=>{const b=D.nb[sh];
 n+=`<tr><td class=l>${D.labels[sh]}</td><td class=${cls(b.srcc,.62,.54)}>${b.srcc}</td>`+
  `<td class=${cls(b.srcc_nonblack,.7,.6)}>${b.srcc_nonblack}</td><td class=${cls(b.auc,.78,.72)}>${b.auc}</td>`+
  `<td>${b.prec}</td><td>${b.rec}</td><td class=win>${b.f1}</td></tr>`;});
document.getElementById('nb').innerHTML=n+`</tbody></table><div class=note>近黑子集 n=${D.n_nb}。SRCC 是被噪声标签锁死的伪目标；<b>漏标 AUC/F1 才是产品指标</b>——把"全黑该不该是金属"当二分类。</div>`;
// ⑥ confusion + per-score
function cmH(sh){const m=D.cm[sh];const mx=Math.max(...m.flat());
 let t=`<div><h2 class=blk>${sh}</h2><table class=conf><thead><tr><th></th>`;
 for(let j=0;j<6;j++)t+=`<th>${j}</th>`;t+='</tr></thead><tbody>';
 for(let i=0;i<6;i++){t+=`<tr><th>${i}</th>`;for(let j=0;j<6;j++){const v=m[i][j],a=v/mx;
  const bg=i===j?`rgba(0,184,148,${.2+.8*a})`:Math.abs(i-j)===1?`rgba(253,203,110,${.15+.6*a})`:`rgba(225,112,85,${.1+.7*a})`;
  t+=`<td style="background:${v?bg:'#fff'}">${v||''}</td>`;}t+='</tr>';}
 return t+'</tbody></table></div>';}
document.getElementById('cm').innerHTML=S.map(cmH).join('');
let ps='<table><thead><tr><th class=l>真分档(metallic)</th>';for(let k=0;k<6;k++)ps+=`<th>${k}</th>`;ps+='</tr></thead><tbody>';
S.forEach(sh=>{ps+=`<tr><td class=l>${sh} MAE</td>`;for(let k=0;k<6;k++){const v=D.perscore[sh][k];
 ps+=`<td class=${v==null?'':cls(-v,-0.8,-1.6)}>${v==null?'-':v}</td>`;}ps+='</tr>';});
document.getElementById('perscore').innerHTML=ps+'</tbody></table>';
// ⑦ distribution histograms (GT vs each model), one card per channel
function distH(ch){const d=D.dist[ch];const series=[{name:'GT',hist:d.gt,std:d.gt_std,col:'#b2bec3'}]
  .concat(S.map((sh,i)=>({name:sh,hist:d.models[sh].hist,std:d.models[sh].std,col:COL[i]})));
 const mx=Math.max(...series.flatMap(s=>s.hist));
 let t=`<div><h2 class=blk>${LAB[ch]} <span style="font-weight:400;color:#636e72">— GT std ${d.gt_std}</span></h2>`;
 // grouped bars per score 0-5
 t+='<div style="display:flex;gap:14px;align-items:flex-end;height:130px;padding:6px 0 2px;border-bottom:1px solid #eee">';
 for(let k=0;k<6;k++){t+='<div style="flex:1;display:flex;flex-direction:column;align-items:center;gap:2px">';
  t+='<div style="display:flex;gap:2px;align-items:flex-end;height:104px">';
  series.forEach(s=>{const hh=s.hist[k]/mx*100;
   t+=`<div title="${s.name}: ${s.hist[k]}%" style="width:9px;height:${hh}%;background:${s.col};border-radius:2px 2px 0 0"></div>`;});
  t+=`</div><div style="font-size:11px;color:#636e72">${k}</div></div>`;}
 t+='</div><div style="margin-top:8px;font-size:11px;color:#636e72">';
 t+=series.map(s=>`<span style="margin-right:12px"><span style="display:inline-block;width:9px;height:9px;background:${s.col};border-radius:2px"></span> ${s.name} <b>σ${s.std}</b></span>`).join('');
 return t+'</div></div>';}
document.getElementById('dist').innerHTML=CH.map(distH).join('');
</script></body></html>"""

if __name__ == "__main__":
    main()
