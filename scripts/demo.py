#!/usr/bin/env python3
"""PBR Quality Scorer — Demo page.

Pick an asset → see its 4 PBR channel maps + the best model's predicted 0-5
quality score for each channel, side-by-side with the human ground-truth score.

Reads precomputed predictions (scripts/predict_demo.py writes them), so the web
server holds NO GPU and never interferes with training.

Usage:
    conda run -n asset-quality-scorer python asset_quality_scorer/scripts/demo.py [--port 7862]
Then open http://localhost:7862
"""
import argparse, csv, io, json
from pathlib import Path
from flask import Flask, jsonify, request, send_file, Response
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "asset_quality_scorer"
IMAGE_ROOT   = PROJECT_ROOT / "datasets0526"
CSV_PATH     = PACKAGE_ROOT / "dataset/sampled_all.csv"
# Switchable models: label → run dir (each must hold demo_predictions.json over
# the SAME old-test 4917 assets, so comparisons are apples-to-apples).
MODELS = {
    "dinov2_emd":  "dinov2_large_multitask_emd_all",
    "qwen_vl_best": "vlm_scorer_a_old50k_oldtest",
    "qwen_fullcover": "vlm_scorer_fullcover_oldtest",
}
DEFAULT_MODEL = "dinov2_emd"
CHANNELS     = ["base_color", "normal_map", "roughness", "metallic"]
CH_LABEL     = {"base_color": "Base Color", "normal_map": "Normal",
                "roughness": "Roughness", "metallic": "Metallic"}

app = Flask(__name__, static_folder=None)


def _name(model_path: str) -> str:
    return model_path.removeprefix("raw_data/").replace("/", "__").removesuffix(".glb")


# ── load metadata (pbrType / tier / finalScore) ──────────────────────────────
META = {}
with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        META[_name(row["model"])] = {
            "pbr":  row.get("pbrType", ""),
            "tier": row.get("tier", ""),
            "final": row.get("finalScore", ""),
        }

# ── load precomputed predictions ─────────────────────────────────────────────
def load_preds(exp: str) -> dict:
    p = PACKAGE_ROOT / "outputs" / "runs" / exp / "demo_predictions.json"
    data = json.loads(p.read_text())
    # attach abs error + metadata, precompute mean err for sorting
    for a in data["assets"]:
        a["err"] = {c: round(a["pred"][c] - a["gt"][c], 2) for c in CHANNELS}
        a["mae"] = round(sum(abs(a["err"][c]) for c in CHANNELS) / len(CHANNELS), 3)
        m = META.get(a["name"], {})
        a["pbr"] = m.get("pbr", ""); a["tier"] = m.get("tier", ""); a["final"] = m.get("final", "")
    return data

# load every available model once; missing ones are skipped gracefully
PREDS_BY_MODEL = {}
for key, exp in MODELS.items():
    try:
        PREDS_BY_MODEL[key] = load_preds(exp)
    except FileNotFoundError:
        print(f"[warn] {key}: no demo_predictions.json at runs/{exp}, skipping")

def get_preds(model: str) -> dict:
    return PREDS_BY_MODEL.get(model, PREDS_BY_MODEL[DEFAULT_MODEL])


# ── image serving ─────────────────────────────────────────────────────────────
@app.route("/img")
def img():
    ch = request.args.get("ch", "render"); n = request.args.get("n", "")
    thumb = request.args.get("thumb") == "1"
    path = IMAGE_ROOT / ch / f"{n}.png"
    if not path.exists():
        return Response(status=404)
    if thumb:
        im = Image.open(path).convert("RGB")
        im.thumbnail((384, 384), Image.LANCZOS)
        buf = io.BytesIO(); im.save(buf, format="JPEG", quality=85); buf.seek(0)
        return send_file(buf, mimetype="image/jpeg")
    return send_file(path, mimetype="image/png")


@app.route("/api/meta")
def api_meta():
    P = get_preds(request.args.get("model", DEFAULT_MODEL))
    return jsonify({"exp": P["exp_id"], "srcc": P["srcc"],
                    "srcc_mean": P["srcc_mean"], "n": P["n"],
                    "models": [{"key": k, "label": PREDS_BY_MODEL[k]["exp_id"]}
                               for k in MODELS if k in PREDS_BY_MODEL],
                    "current": request.args.get("model", DEFAULT_MODEL)})


@app.route("/api/list")
def api_list():
    """Lightweight list for the picker: name + per-channel error + mae + meta."""
    sort = request.args.get("sort", "name")
    flt  = request.args.get("filter", "all")
    items = get_preds(request.args.get("model", DEFAULT_MODEL))["assets"]
    if flt == "metal_wrong":
        items = [a for a in items if abs(a["err"]["metallic"]) >= 2]
    elif flt == "big_err":
        items = [a for a in items if a["mae"] >= 1.0]
    elif flt == "accurate":
        items = [a for a in items if a["mae"] <= 0.4]
    if sort == "mae_desc":
        items = sorted(items, key=lambda a: -a["mae"])
    elif sort == "mae_asc":
        items = sorted(items, key=lambda a: a["mae"])
    elif sort == "metal_err":
        items = sorted(items, key=lambda a: -abs(a["err"]["metallic"]))
    out = [{"name": a["name"], "mae": a["mae"], "pbr": a["pbr"],
            "metal_err": a["err"]["metallic"]} for a in items[:1500]]
    return jsonify({"n_total": len(items), "items": out})


@app.route("/api/asset/<path:name>")
def api_asset(name):
    P = get_preds(request.args.get("model", DEFAULT_MODEL))
    for a in P["assets"]:
        if a["name"] == name:
            return jsonify(a)
    return jsonify({"error": "not found"}), 404


@app.route("/api/compare/<path:name>")
def api_compare(name):
    """Both models' pred for one asset, side by side."""
    out = {"name": name, "models": {}}
    for k in PREDS_BY_MODEL:
        for a in PREDS_BY_MODEL[k]["assets"]:
            if a["name"] == name:
                out["models"][k] = {"label": PREDS_BY_MODEL[k]["exp_id"],
                                    "pred": a["pred"], "err": a["err"]}
                out["gt"] = a["gt"]; out["pbr"] = a["pbr"]
                out["tier"] = a["tier"]; out["final"] = a["final"]
                break
    return jsonify(out)


# ── page ──────────────────────────────────────────────────────────────────────
PAGE = """<!doctype html><html><head><meta charset=utf-8>
<title>PBR Quality Scorer — Demo</title>
<style>
 * { box-sizing: border-box; }
 body { font: 14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif; margin:0;
        background:#0f1115; color:#e6e6e6; }
 header { padding:14px 20px; background:#171a21; border-bottom:1px solid #2a2f3a;
          display:flex; align-items:center; gap:18px; flex-wrap:wrap; }
 header h1 { font-size:17px; margin:0; font-weight:600; }
 .badge { background:#1f2530; padding:3px 10px; border-radius:6px; font-size:12px; color:#9fb3c8; }
 .badge b { color:#6cc04a; }
 main { display:flex; height:calc(100vh - 58px); }
 #side { width:300px; border-right:1px solid #2a2f3a; overflow-y:auto; background:#12151b; flex-shrink:0; }
 .ctrl { padding:10px; border-bottom:1px solid #2a2f3a; position:sticky; top:0; background:#12151b; z-index:2; }
 .ctrl select, .ctrl input, .ctrl button { width:100%; margin:3px 0; padding:6px 8px;
   background:#1f2530; color:#e6e6e6; border:1px solid #2a2f3a; border-radius:6px; font-size:13px; }
 .ctrl button { cursor:pointer; background:#2563eb; border:none; font-weight:600; }
 .ctrl button:hover { background:#1d4ed8; }
 .row { padding:7px 12px; cursor:pointer; border-bottom:1px solid #1c2029; display:flex;
        justify-content:space-between; gap:8px; }
 .row:hover { background:#1a1f29; }
 .row.sel { background:#1e3a5f; }
 .row .nm { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:12px; }
 .row .mae { font-size:11px; padding:1px 6px; border-radius:4px; flex-shrink:0; }
 #content { flex:1; overflow-y:auto; padding:20px; }
 .imgs { display:grid; grid-template-columns:repeat(5,1fr); gap:12px; margin-bottom:20px; }
 .imgcell { text-align:center; }
 .imgcell img { width:100%; border-radius:8px; background:#000; aspect-ratio:1; object-fit:cover; }
 .imgcell .cap { font-size:12px; color:#9fb3c8; margin-top:5px; }
 table { width:100%; border-collapse:collapse; margin-top:6px; }
 th,td { padding:10px 14px; text-align:center; border-bottom:1px solid #2a2f3a; }
 th { color:#9fb3c8; font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:.04em; }
 td.ch { text-align:left; font-weight:600; }
 .score { font-size:20px; font-weight:700; }
 .err-g { color:#6cc04a; } .err-y { color:#e5b338; } .err-r { color:#e5484d; }
 .pill { display:inline-block; padding:2px 9px; border-radius:10px; font-size:12px; font-weight:600; }
 .pill-g { background:#1c3a1c; color:#6cc04a; } .pill-y { background:#3a3318; color:#e5b338; }
 .pill-r { background:#3a1c1e; color:#e5484d; }
 .meta { color:#9fb3c8; font-size:13px; margin-bottom:14px; }
 .meta span { margin-right:16px; } .meta b { color:#e6e6e6; }
 .bar { height:8px; background:#1f2530; border-radius:4px; overflow:hidden; margin-top:4px; }
 .bar > div { height:100%; }
 .empty { color:#667; text-align:center; margin-top:80px; font-size:15px; }
</style></head><body>
<header>
 <h1>🎨 PBR 质量评分 Demo</h1>
 <span class=badge id=expb>model: …</span>
 <span class=badge id=srccb>SRCC: …</span>
 <span class=badge>预测分 vs 人工真分 · test set</span>
</header>
<main>
 <div id=side>
  <div class=ctrl>
   <select id=model onchange="switchModel()" style="font-weight:600;color:#6cc04a"></select>
   <label style="font-size:12px;color:#9fb3c8;display:flex;align-items:center;gap:6px;margin:4px 0">
     <input type=checkbox id=cmp onchange="if(SEL)pick(SEL)" style="width:auto;margin:0"> 对比两个模型</label>
   <input id=search placeholder="🔍 搜索资产名…" oninput="render()">
   <select id=filter onchange="reload()">
    <option value=all>全部</option>
    <option value=accurate>预测准的 (MAE≤0.4)</option>
    <option value=big_err>误差大的 (MAE≥1.0)</option>
    <option value=metal_wrong>metallic 差≥2 分</option>
   </select>
   <select id=sort onchange="reload()">
    <option value=name>排序: 默认</option>
    <option value=mae_desc>排序: 误差大→小</option>
    <option value=mae_asc>排序: 误差小→大</option>
    <option value=metal_err>排序: metallic 误差</option>
   </select>
   <button onclick="randomPick()">🎲 随机一个</button>
   <div style="font-size:11px;color:#667;margin-top:6px" id=count></div>
  </div>
  <div id=list></div>
 </div>
 <div id=content><div class=empty>← 从左侧选一个资产，或点「随机一个」</div></div>
</main>
<script>
const CH=["base_color","normal_map","roughness","metallic"];
const LAB={base_color:"Base Color",normal_map:"Normal",roughness:"Roughness",metallic:"Metallic"};
let ITEMS=[], SEL=null, MODEL='dinov2_emd';

function maeClass(e){e=Math.abs(e); return e<0.5?'g':e<1.5?'y':'r';}
function pill(e){const c=maeClass(e); const s=e>0?'+':''; return `<span class="pill pill-${c}">${s}${e.toFixed(2)}</span>`;}

async function boot(){
 const m=await (await fetch('/api/meta?model='+MODEL)).json();
 const ms=document.getElementById('model');
 if(!ms.options.length){ ms.innerHTML=m.models.map(x=>`<option value="${x.key}">${x.label}</option>`).join(''); ms.value=m.current; }
 document.getElementById('expb').textContent='model: '+m.exp;
 document.getElementById('srccb').innerHTML='SRCC mean: <b>'+m.srcc_mean+'</b> &nbsp; (ba '+m.srcc.base_color+' / no '+m.srcc.normal_map+' / ro '+m.srcc.roughness+' / me '+m.srcc.metallic+')';
 reload();
}
function switchModel(){ MODEL=document.getElementById('model').value; boot().then(()=>{ if(SEL)pick(SEL); }); }
async function reload(){
 const f=document.getElementById('filter').value, s=document.getElementById('sort').value;
 const r=await (await fetch(`/api/list?model=${MODEL}&filter=${f}&sort=${s}`)).json();
 ITEMS=r.items; document.getElementById('count').textContent=r.n_total+' 个资产 (显示前'+ITEMS.length+')';
 render();
}
function render(){
 const q=document.getElementById('search').value.toLowerCase();
 const list=document.getElementById('list'); list.innerHTML='';
 for(const a of ITEMS){
  if(q && !a.name.toLowerCase().includes(q)) continue;
  const d=document.createElement('div'); d.className='row'+(SEL===a.name?' sel':'');
  d.onclick=()=>pick(a.name);
  d.innerHTML=`<span class=nm title="${a.name}">${a.name}</span>`+
    `<span class="mae pill-${maeClass(a.mae)}">${a.mae}</span>`;
  list.appendChild(d);
 }
}
function randomPick(){ if(ITEMS.length) pick(ITEMS[Math.floor(Math.random()*ITEMS.length)].name); }
async function pick(name){
 SEL=name; render();
 const cmp=document.getElementById('cmp').checked;
 const imgs=['render',...CH].map(ch=>{
   const cap = ch==='render' ? '渲染图 (参考)' : LAB[ch];
   return `<div class=imgcell><img loading=lazy src="/img?ch=${ch}&n=${encodeURIComponent(name)}&thumb=1">`+
          `<div class=cap>${cap}</div></div>`;}).join('');
 let head, rows='', meta;
 if(cmp){
   const c=await (await fetch('/api/compare/'+encodeURIComponent(name))).json();
   const keys=Object.keys(c.models);
   head=`<th>通道</th>`+keys.map(k=>`<th>${c.models[k].label}</th>`).join('')+`<th>人工真分</th>`;
   for(const ch of CH){
     const g=c.gt[ch];
     rows+=`<tr><td class=ch>${LAB[ch]}</td>`+
       keys.map(k=>{const p=c.models[k].pred[ch],e=c.models[k].err[ch];
         return `<td class=score>${p.toFixed(2)} <span class="pill pill-${maeClass(e)}" style="font-size:10px">${e>0?'+':''}${e.toFixed(1)}</span></td>`;}).join('')+
       `<td class=score>${g}</td></tr>`;
   }
   meta=c;
 } else {
   const a=await (await fetch(`/api/asset/${encodeURIComponent(name)}?model=${MODEL}`)).json();
   head=`<th>通道</th><th>模型预测</th><th>人工真分</th><th>误差</th>`;
   let sp=0,sg=0;
   for(const ch of CH){
     const p=a.pred[ch], g=a.gt[ch], e=a.err[ch]; sp+=p; sg+=g;
     rows+=`<tr><td class=ch>${LAB[ch]}</td><td class=score>${p.toFixed(2)}</td>`+
       `<td class=score>${g}</td><td class="err-${maeClass(e)}">${pill(e)}</td></tr>`;
   }
   const mp=sp/4,mg=sg/4,me=mp-mg;
   rows+=`<tr style="border-top:2px solid #2a2f3a"><td class=ch>均值</td>`+
     `<td class=score>${mp.toFixed(2)}</td><td class=score>${mg.toFixed(2)}</td>`+
     `<td class="err-${maeClass(me)}">${pill(me)}</td></tr>`;
   meta=a;
 }
 document.getElementById('content').innerHTML=
   `<div class=meta><span>资产: <b>${name}</b></span><span>类型: <b>${meta.pbr||'?'}</b></span>`+
   `<span>${meta.tier||''}</span><span>finalScore: <b>${meta.final||'?'}</b></span></div>`+
   `<div class=imgs>${imgs}</div>`+
   `<table><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table>`;
}
boot();
</script></body></html>"""


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


@app.route("/overview")
def overview():
    """Standalone data-overview page (built by scripts/build_scorer_overview.py).
    Independent route — open in its own tab, does not touch the main demo page."""
    p = PACKAGE_ROOT / "outputs" / "scorer_overview.html"
    if not p.exists():
        return Response("overview not built yet — run scripts/build_scorer_overview.py", status=404)
    return Response(p.read_text(encoding="utf-8"), mimetype="text/html")


@app.route("/labelnoise")
def labelnoise():
    """Label-noise review page (built by scripts/mine_label_noise.py). Standalone
    tab; images load via this server's /api/image. Cross-model consensus picks
    out likely mis-labeled assets — the lever on the 0.79 label ceiling."""
    p = PACKAGE_ROOT / "outputs" / "label_noise" / "review.html"
    if not p.exists():
        return Response("not mined yet — run scripts/mine_label_noise.py", status=404)
    return Response(p.read_text(encoding="utf-8"), mimetype="text/html")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7862)
    args = ap.parse_args()
    print(f"Demo: {len(PREDS_BY_MODEL)} models loaded: " +
          ", ".join(f"{k}({PREDS_BY_MODEL[k]['srcc_mean']})" for k in PREDS_BY_MODEL))
    print(f"  → http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
