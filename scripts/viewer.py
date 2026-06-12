#!/usr/bin/env python3
"""PBR Asset Channel Viewer.

Usage:
    conda run -n asset-quality-scorer python asset_quality_scorer/scripts/viewer.py [--port 7860]
Then open http://localhost:7860 in your browser.
"""
import csv
import io
import json
import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_file
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_ROOT   = PROJECT_ROOT / "datasets0526"
CSV_PATH     = PROJECT_ROOT / "asset_quality_scorer/dataset/sampled_all.csv"
CHANNELS     = ["render", "base_color", "metallic", "roughness", "normal_map",
                "white_model", "white_with_normal"]
THUMB_SIZE   = 320   # px — thumbnail served to grid

app = Flask(__name__, static_folder=None)


# ── data loading ─────────────────────────────────────────────────────────────

def _name(model_path: str) -> str:
    return model_path.removeprefix("raw_data/").replace("/", "__").removesuffix(".glb")

def _int(v):
    try: return int(v)
    except: return None

def _float(v):
    try: return round(float(v), 2)
    except: return None


print("Loading CSV…", end=" ", flush=True)
ASSETS: list[dict] = []
with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        name = _name(row["model"])
        ASSETS.append({
            "name":    name,
            "split":   row.get("split", ""),
            "pbr":     row.get("pbrType", ""),
            "tier":    row.get("tier", ""),
            "source":  name.split("__")[0] if "__" in name else "?",
            "scores": {
                "base_color": _int(row.get("baseColor")),
                "normal_map": _int(row.get("normal")),
                "roughness":  _int(row.get("roughness")),
                "metallic":   _int(row.get("metallic")),
                "render":     _int(row.get("rendered")),
                "final":      _float(row.get("finalScore")),
            },
            "flags": {
                "text":       row.get("hasTextOrPattern", "") == "True",
                "fake_ao":    row.get("baseColorHasFakeAOOrGlow", "") == "True",
                "tint":       row.get("normalHasAbnormalTint", "") == "True",
                "flipped":    row.get("normalIsFlipped", "") == "True",
            },
        })
print(f"{len(ASSETS):,} assets loaded.")

# name → index for fast lookup
NAME_IDX = {a["name"]: i for i, a in enumerate(ASSETS)}


# ── image serving ─────────────────────────────────────────────────────────────

def _img_path(channel: str, name: str) -> Path | None:
    p = IMAGE_ROOT / channel / f"{name}.png"
    return p if p.exists() else None


def _serve_image(path: Path, thumb: bool) -> "Response":
    if thumb:
        img = Image.open(path).convert("RGB")
        img.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=82)
        buf.seek(0)
        return send_file(buf, mimetype="image/jpeg")
    return send_file(path, mimetype="image/png")


RUNS_ROOT = PROJECT_ROOT / "asset_quality_scorer" / "outputs" / "runs"

@app.route("/api/training")
def api_training():
    """Return training curves from all train_log.json files."""
    runs = []
    for log_file in sorted(RUNS_ROOT.glob("*/train_log.json")):
        try:
            data = json.loads(log_file.read_text())
            runs.append({
                "exp_id":  data.get("exp_id", log_file.parent.name),
                "epochs":  data.get("epochs", []),
                "running": not (log_file.parent / "summary.json").exists(),
            })
        except Exception:
            pass
    return jsonify(runs)


TRAINING_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>Training Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#111;color:#e0e0e0}
header{padding:12px 20px;background:#1a1a1a;border-bottom:1px solid #2a2a2a;display:flex;align-items:center;gap:16px}
header h1{font-size:15px;font-weight:600}
.badge{font-size:11px;padding:2px 8px;border-radius:10px;background:#2a3d5a;color:#9bc4f0}
#controls{padding:10px 20px;background:#161616;border-bottom:1px solid #2a2a2a;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
#controls label{font-size:12px;color:#888}
#metric-select,#exp-select{background:#222;border:1px solid #333;color:#ddd;padding:4px 8px;border-radius:5px;font-size:12px}
#auto-refresh{accent-color:#5b8dd9}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(480px,1fr));gap:16px;padding:16px}
.card{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:8px;padding:14px}
.card h3{font-size:12px;color:#888;text-transform:uppercase;letter-spacing:.8px;margin-bottom:10px}
.chart-wrap{position:relative;height:220px}
.no-data{color:#555;font-size:13px;padding:40px;text-align:center}
#status{font-size:11px;color:#666}
</style>
</head>
<body>
<header>
  <h1>📈 Training Dashboard</h1>
  <span class="badge" id="run-count">— runs</span>
  <a href="/" style="margin-left:auto;color:#5b8dd9;font-size:12px">← Asset Viewer</a>
</header>
<div id="controls">
  <label>指标</label>
  <select id="metric-select" onchange="render()">
    <option value="srcc">SRCC ↑（排序一致性）</option>
    <option value="mae">MAE ↓（平均误差）</option>
    <option value="loss">Train Loss ↓</option>
  </select>
  <label style="margin-left:12px">实验</label>
  <select id="exp-select" multiple size="1" style="min-width:200px" onchange="render()"></select>
  <label style="margin-left:12px">
    <input type="checkbox" id="auto-refresh" checked> 自动刷新(30s)
  </label>
  <span id="status"></span>
</div>
<div id="charts" class="grid"><div class="no-data">加载中…</div></div>

<script>
const CHANNELS = ['base_color','normal_map','roughness','metallic'];
const CH_COLORS = {
  base_color:'#5b8dd9', normal_map:'#7ecb8c', roughness:'#f0c060', metallic:'#ff8a8a'
};
const PALETTE = ['#5b8dd9','#7ecb8c','#f0c060','#ff8a8a','#c07af0','#f07a5b'];
let allRuns = []; let charts = {};

async function fetchData() {
  const res = await fetch('/api/training');
  allRuns = await res.json();
  document.getElementById('run-count').textContent = allRuns.length + ' runs';
  // populate exp selector
  const sel = document.getElementById('exp-select');
  const prev = Array.from(sel.selectedOptions).map(o=>o.value);
  sel.innerHTML = '';
  sel.size = Math.min(6, Math.max(2, allRuns.length));
  allRuns.forEach((r,i) => {
    const opt = document.createElement('option');
    opt.value = r.exp_id; opt.textContent = (r.running ? '🔄 ' : '✅ ') + r.exp_id;
    if (prev.length === 0 || prev.includes(r.exp_id)) opt.selected = true;
    sel.appendChild(opt);
  });
  document.getElementById('status').textContent = 'Updated ' + new Date().toLocaleTimeString();
  render();
}

function selectedExps() {
  return Array.from(document.getElementById('exp-select').selectedOptions).map(o=>o.value);
}

function getValues(run, metric, ch) {
  if (!run.epochs || run.epochs.length === 0) return [];
  if (metric === 'loss') return run.epochs.map(e => e.train_loss);
  if (metric === 'srcc' && ch === 'mean') return run.epochs.map(e => e.srcc_mean);
  if (metric === 'mae'  && ch === 'mean') return run.epochs.map(e => e.mae_mean);
  return run.epochs.map(e => e.per_channel?.[ch]?.[metric] ?? null);
}

function makeChart(containerId, title, datasets, yLabel) {
  const existing = charts[containerId];
  if (existing) existing.destroy();
  const canvas = document.getElementById(containerId);
  if (!canvas) return;
  const maxEp = Math.max(...datasets.map(d=>d.data.length), 1);
  charts[containerId] = new Chart(canvas, {
    type: 'line',
    data: {
      labels: Array.from({length: maxEp}, (_,i) => i+1),
      datasets: datasets.map(d => ({
        label: d.label, data: d.data,
        borderColor: d.color, backgroundColor: d.color + '22',
        borderWidth: 2, pointRadius: 2, tension: 0.3, fill: false,
      }))
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: false,
      plugins: {legend:{labels:{color:'#aaa',font:{size:11}}},
                tooltip:{mode:'index',intersect:false}},
      scales: {
        x: {ticks:{color:'#666',font:{size:10}}, grid:{color:'#222'}, title:{display:true,text:'Epoch',color:'#666'}},
        y: {ticks:{color:'#666',font:{size:10}}, grid:{color:'#222'}, title:{display:true,text:yLabel,color:'#666'}},
      }
    }
  });
}

function render() {
  const metric = document.getElementById('metric-select').value;
  const exps   = selectedExps();
  const runs   = allRuns.filter(r => exps.includes(r.exp_id));
  const grid   = document.getElementById('charts');
  grid.innerHTML = '';
  if (runs.length === 0) { grid.innerHTML = '<div class="no-data">请选择实验</div>'; return; }

  // Chart 1: mean metric across all channels
  {
    const card = document.createElement('div'); card.className = 'card';
    const title = metric === 'loss' ? 'Train Loss' : (metric.toUpperCase() + ' 均值');
    card.innerHTML = `<h3>${title}</h3><div class="chart-wrap"><canvas id="c-mean"></canvas></div>`;
    grid.appendChild(card);
    const datasets = runs.map((r,i)=>({
      label: r.exp_id.replace('dinov2_large_multitask_',''),
      data: getValues(r, metric, 'mean'),
      color: PALETTE[i % PALETTE.length],
    }));
    setTimeout(()=>makeChart('c-mean', title, datasets, metric==='loss'?'Loss':'Value'),0);
  }

  // Per-channel charts (not for loss)
  if (metric !== 'loss') {
    CHANNELS.forEach(ch => {
      const card = document.createElement('div'); card.className = 'card';
      const cid = 'c-' + ch;
      card.innerHTML = `<h3>${ch}</h3><div class="chart-wrap"><canvas id="${cid}"></canvas></div>`;
      grid.appendChild(card);
      const datasets = runs.map((r,i)=>({
        label: r.exp_id.replace('dinov2_large_multitask_',''),
        data: getValues(r, metric, ch),
        color: runs.length === 1 ? CH_COLORS[ch] : PALETTE[i % PALETTE.length],
      }));
      setTimeout(()=>makeChart(cid, ch, datasets, metric.toUpperCase()), 0);
    });
  }
}

// auto-refresh
let timer = setInterval(()=>{ if(document.getElementById('auto-refresh').checked) fetchData(); }, 30000);
fetchData();
</script>
</body>
</html>"""


@app.route("/training")
def training():
    return TRAINING_HTML


@app.route("/img")
def img():
    """Query-param based: /img?ch=render&n=<name>&t=1"""
    channel = request.args.get("ch", "")
    name    = request.args.get("n",  "")
    thumb   = request.args.get("t") == "1"
    if channel not in CHANNELS:
        return "bad channel", 400
    p = _img_path(channel, name)
    if p is None:
        return "not found", 404
    return _serve_image(p, thumb)


# ── API ───────────────────────────────────────────────────────────────────────

def _score_ok(score, lo, hi):
    if score is None: return True   # missing scores pass through
    return lo <= score <= hi


@app.route("/api/assets")
def api_assets():
    page      = max(1, int(request.args.get("page", 1)))
    per_page  = min(100, int(request.args.get("per", 40)))
    q         = request.args.get("q", "").strip().lower()

    # score filters — default 0-5 for each
    def sf(key):
        lo = int(request.args.get(f"{key}_lo", 0))
        hi = int(request.args.get(f"{key}_hi", 5))
        return lo, hi
    bc_lo, bc_hi   = sf("bc")
    nm_lo, nm_hi   = sf("nm")
    ro_lo, ro_hi   = sf("ro")
    me_lo, me_hi   = sf("me")
    fi_lo = float(request.args.get("fi_lo", 0))
    fi_hi = float(request.args.get("fi_hi", 5))

    splits  = set(request.args.getlist("split")) or {"train","val","test",""}
    pbrs    = set(request.args.getlist("pbr"))   or {"physical","stylized","uncertain",""}
    sources = set(request.args.getlist("src"))   or set()

    flag_text     = request.args.get("flag_text")
    flag_fake_ao  = request.args.get("flag_fake_ao")
    flag_tint     = request.args.get("flag_tint")
    flag_flipped  = request.args.get("flag_flipped")

    def flag_ok(val, param):
        if param is None: return True
        return val == (param == "1")

    sort_key = request.args.get("sort", "final_desc")

    results = []
    for a in ASSETS:
        s = a["scores"]
        if q and q not in a["name"].lower(): continue
        if a["split"] not in splits: continue
        if a["pbr"] not in pbrs: continue
        if sources and a["source"] not in sources: continue
        if not _score_ok(s["base_color"], bc_lo, bc_hi): continue
        if not _score_ok(s["normal_map"], nm_lo, nm_hi): continue
        if not _score_ok(s["roughness"],  ro_lo, ro_hi): continue
        if not _score_ok(s["metallic"],   me_lo, me_hi): continue
        fi = s["final"]
        if fi is not None and not (fi_lo <= fi <= fi_hi): continue
        f = a["flags"]
        if not flag_ok(f["text"],    flag_text):    continue
        if not flag_ok(f["fake_ao"], flag_fake_ao): continue
        if not flag_ok(f["tint"],    flag_tint):    continue
        if not flag_ok(f["flipped"], flag_flipped): continue
        results.append(a)

    # sort
    def sort_fn(a):
        s = a["scores"]
        if sort_key == "final_asc":  return  (s["final"] or 0)
        if sort_key == "final_desc": return -(s["final"] or 0)
        if sort_key == "me_asc":     return  (s["metallic"] or 0)
        if sort_key == "me_desc":    return -(s["metallic"] or 0)
        if sort_key == "name":       return  a["name"]
        return 0
    results.sort(key=sort_fn)

    total = len(results)
    start = (page - 1) * per_page
    page_data = results[start: start + per_page]

    return jsonify({
        "total": total,
        "page":  page,
        "pages": max(1, (total + per_page - 1) // per_page),
        "assets": [{"name": a["name"], "scores": a["scores"],
                    "flags": a["flags"], "split": a["split"],
                    "pbr": a["pbr"], "source": a["source"]} for a in page_data],
    })


@app.route("/api/asset/<name>")
def api_asset(name: str):
    i = NAME_IDX.get(name)
    if i is None: return jsonify({"error": "not found"}), 404
    a = ASSETS[i]
    channels_exist = {ch: _img_path(ch, name) is not None for ch in CHANNELS}
    return jsonify({**a, "channels": channels_exist})


# ── frontend ──────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PBR Asset Viewer</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#111;color:#e0e0e0;display:flex;height:100vh;overflow:hidden}

/* sidebar */
#sidebar{width:240px;min-width:200px;background:#161616;border-right:1px solid #2a2a2a;display:flex;flex-direction:column;overflow:hidden}
#sidebar-header{padding:13px 14px;background:#1e1e1e;border-bottom:1px solid #2a2a2a;font-size:14px;font-weight:600;letter-spacing:.2px;display:flex;align-items:center;gap:7px}
#filters{flex:1;overflow-y:auto;padding:10px 12px;display:flex;flex-direction:column;gap:14px}
.fg{display:flex;flex-direction:column;gap:5px}
.fg-label{font-size:10px;color:#666;text-transform:uppercase;letter-spacing:1px}
#q{width:100%;background:#222;border:1px solid #333;color:#ddd;padding:6px 9px;border-radius:6px;font-size:13px}
#q:focus{outline:none;border-color:#5b8dd9}
/* score chips */
.score-ch{display:flex;flex-direction:column;gap:3px}
.sc-row{display:flex;align-items:center;gap:5px}
.sc-name{font-size:11px;color:#888;width:76px;flex-shrink:0}
.sc-chips{display:flex;gap:3px}
.sc{width:24px;height:24px;border-radius:4px;border:1px solid #333;background:#222;color:#888;font-size:11px;font-weight:600;cursor:pointer;display:flex;align-items:center;justify-content:center;user-select:none;transition:all .12s}
.sc.active.s0,.sc.active.s1{background:#5c1e1e;border-color:#7a2a2a;color:#ff9090}
.sc.active.s2,.sc.active.s3{background:#4a3800;border-color:#6a5200;color:#f0c060}
.sc.active.s4,.sc.active.s5{background:#1a4020;border-color:#2a6030;color:#6dcc8a}
/* chip groups */
.chips{display:flex;flex-wrap:wrap;gap:4px}
.chip{padding:3px 9px;border-radius:12px;border:1px solid #333;background:#222;color:#888;font-size:11px;cursor:pointer;user-select:none;transition:all .12s;white-space:nowrap}
.chip.active{background:#2a3d5a;border-color:#5b8dd9;color:#9bc4f0}
/* flag chips — 3 state */
.fchip{padding:3px 9px;border-radius:12px;border:1px solid #333;background:#222;color:#888;font-size:11px;cursor:pointer;user-select:none;transition:all .12s}
.fchip.on{background:#1e3a5f;border-color:#4a8ad4;color:#90c0ff}
.fchip.off{background:#4a1515;border-color:#8b3030;color:#ff9090}
/* search results clear */
#sidebar-footer{padding:9px 12px;border-top:1px solid #2a2a2a;display:flex;gap:6px}
#btn-reset{flex:1;padding:7px;border:1px solid #333;background:#222;color:#999;border-radius:6px;cursor:pointer;font-size:12px}
#btn-reset:hover{background:#2a2a2a;color:#ccc}

/* main */
#main{flex:1;display:flex;flex-direction:column;overflow:hidden}
#toolbar{padding:10px 16px;background:#1a1a1a;border-bottom:1px solid #333;display:flex;align-items:center;gap:12px}
#results-info{font-size:13px;color:#888}
#sort-select{background:#2a2a2a;border:1px solid #444;color:#e0e0e0;padding:5px 8px;border-radius:6px;font-size:13px}
#grid-container{flex:1;overflow-y:auto;padding:14px}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px}
.card{background:#1e1e1e;border:1px solid #2e2e2e;border-radius:8px;overflow:hidden;cursor:pointer;transition:transform .15s,border-color .15s}
.card:hover{transform:translateY(-2px);border-color:#5b8dd9}
.card-thumb{width:100%;aspect-ratio:1;overflow:hidden;background:#111}
.card-thumb img{width:100%;height:100%;object-fit:contain;display:block}
.card-info{padding:8px 10px}
.card-name{font-size:11px;color:#888;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:5px}
.score-badges{display:flex;flex-wrap:wrap;gap:3px}
.badge{font-size:10px;padding:2px 5px;border-radius:3px;font-weight:600}
.s0,.s1{background:#5c1e1e;color:#ff8a8a}
.s2,.s3{background:#4a3a10;color:#f0c060}
.s4,.s5{background:#1a4a2a;color:#6dcc8a}
.sn{background:#2a2a2a;color:#666}
.flag-badge{font-size:9px;padding:1px 4px;border-radius:3px;background:#3a2800;color:#f0a020}

/* pagination */
#pagination{padding:10px 16px;border-top:1px solid #333;display:flex;align-items:center;gap:8px;background:#1a1a1a}
#pagination button{background:#2a2a2a;border:1px solid #444;color:#ccc;padding:5px 12px;border-radius:5px;cursor:pointer;font-size:13px}
#pagination button:hover:not(:disabled){background:#333}
#pagination button:disabled{opacity:.35;cursor:default}
#page-info{font-size:13px;color:#888;flex:1;text-align:center}

/* modal */
#modal-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.75);z-index:100;align-items:center;justify-content:center}
#modal-overlay.show{display:flex}
#modal{background:#1a1a1a;border:1px solid #333;border-radius:10px;width:min(1100px,95vw);max-height:92vh;overflow:hidden;display:flex;flex-direction:column}
#modal-header{padding:14px 18px;border-bottom:1px solid #333;display:flex;align-items:center;gap:10px}
#modal-title{font-size:14px;font-weight:600;flex:1;word-break:break-all}
#modal-close{background:none;border:none;color:#888;font-size:22px;cursor:pointer;padding:0 4px;line-height:1}
#modal-close:hover{color:#fff}
#modal-scores{padding:10px 18px;border-bottom:1px solid #2a2a2a;display:flex;flex-wrap:wrap;gap:8px}
.mscore{font-size:12px;color:#ccc}
.mscore span{font-weight:600}
#modal-body{overflow-y:auto;padding:14px 18px}
#channel-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px}
.ch-card{background:#111;border-radius:6px;overflow:hidden}
.ch-label{font-size:11px;color:#888;padding:4px 8px;background:#1a1a1a;display:flex;justify-content:space-between}
.ch-img-wrap{aspect-ratio:1;overflow:hidden;cursor:zoom-in;position:relative}
.ch-img-wrap img{width:100%;height:100%;object-fit:contain;display:block}
.ch-missing{width:100%;aspect-ratio:1;display:flex;align-items:center;justify-content:center;color:#444;font-size:12px}
#lightbox{display:none;position:fixed;inset:0;background:rgba(0,0,0,.9);z-index:200;align-items:center;justify-content:center;cursor:zoom-out}
#lightbox.show{display:flex}
#lightbox img{max-width:95vw;max-height:95vh;object-fit:contain}
</style>
</head>
<body>

<!-- sidebar -->
<div id="sidebar">
  <div id="sidebar-header">⬡ PBR Asset Viewer</div>
  <div id="filters">

    <div class="fg">
      <div class="fg-label">搜索</div>
      <input type="text" id="q" placeholder="model name…" oninput="debounce()">
    </div>

    <div class="fg">
      <div class="fg-label">质量分（点选，可多选）</div>
      <div class="score-ch">
        <div class="sc-row">
          <span class="sc-name">base_color</span>
          <div class="sc-chips" id="sc-bc"></div>
        </div>
        <div class="sc-row">
          <span class="sc-name">normal_map</span>
          <div class="sc-chips" id="sc-nm"></div>
        </div>
        <div class="sc-row">
          <span class="sc-name">roughness</span>
          <div class="sc-chips" id="sc-ro"></div>
        </div>
        <div class="sc-row">
          <span class="sc-name">metallic</span>
          <div class="sc-chips" id="sc-me"></div>
        </div>
      </div>
    </div>

    <div class="fg">
      <div class="fg-label">Split</div>
      <div class="chips" id="chips-split">
        <span class="chip active" data-v="train">train</span>
        <span class="chip active" data-v="val">val</span>
        <span class="chip active" data-v="test">test</span>
      </div>
    </div>

    <div class="fg">
      <div class="fg-label">PBR 类型</div>
      <div class="chips" id="chips-pbr">
        <span class="chip active" data-v="physical">physical</span>
        <span class="chip active" data-v="stylized">stylized</span>
        <span class="chip active" data-v="uncertain">uncertain</span>
      </div>
    </div>

    <div class="fg">
      <div class="fg-label">来源</div>
      <div class="chips" id="chips-src">
        <span class="chip active" data-v="sketchfab">sketchfab</span>
        <span class="chip active" data-v="3d66">3d66</span>
        <span class="chip active" data-v="sketchfab-objaverse">objaverse</span>
        <span class="chip active" data-v="games">games</span>
        <span class="chip active" data-v="unreal">unreal</span>
        <span class="chip active" data-v="abo">abo</span>
        <span class="chip active" data-v="pbrmax">pbrmax</span>
        <span class="chip active" data-v="kitbash">kitbash</span>
        <span class="chip active" data-v="megascan">megascan</span>
      </div>
    </div>

    <div class="fg">
      <div class="fg-label">Defect（蓝=有 红=无 灰=不限）</div>
      <div class="chips">
        <span class="fchip" id="f_text"    onclick="toggleFlag('text')"   >含文字</span>
        <span class="fchip" id="f_fake_ao" onclick="toggleFlag('fake_ao')">伪AO</span>
        <span class="fchip" id="f_tint"    onclick="toggleFlag('tint')"   >法线异色</span>
        <span class="fchip" id="f_flipped" onclick="toggleFlag('flipped')">法线翻转</span>
      </div>
    </div>

  </div>
  <div id="sidebar-footer">
    <button id="btn-reset" onclick="resetFilters()">重置所有筛选</button>
  </div>
</div>

<!-- main -->
<div id="main">
  <div id="toolbar">
    <span id="results-info">—</span>
    <select id="sort-select" onchange="fetchPage(1)">
      <option value="final_desc">总分 ↓</option>
      <option value="final_asc" >总分 ↑</option>
      <option value="me_desc"   >metallic ↓</option>
      <option value="me_asc"    >metallic ↑</option>
      <option value="name"      >名称 A-Z</option>
    </select>
  </div>
  <div id="grid-container"><div id="grid"></div></div>
  <div id="pagination">
    <button id="btn-prev" onclick="changePage(-1)" disabled>‹ 上一页</button>
    <span id="page-info">—</span>
    <button id="btn-next" onclick="changePage(1)">下一页 ›</button>
  </div>
</div>

<!-- detail modal -->
<div id="modal-overlay" onclick="if(event.target===this)closeModal()">
  <div id="modal">
    <div id="modal-header">
      <div id="modal-title">—</div>
      <button id="modal-close" onclick="closeModal()">✕</button>
    </div>
    <div id="modal-scores"></div>
    <div id="modal-body">
      <div id="channel-grid"></div>
    </div>
  </div>
</div>

<!-- lightbox -->
<div id="lightbox" onclick="closeLightbox()">
  <img id="lb-img" src="">
</div>

<script>
const CHANNELS = ['render','base_color','metallic','roughness','normal_map','white_model','white_with_normal'];
const CH_LABELS = {render:'Render',base_color:'Base Color',metallic:'Metallic',
  roughness:'Roughness',normal_map:'Normal Map',white_model:'White Model',
  white_with_normal:'White+Normal'};
const SCORE_CH = {base_color:'sc-bc',normal_map:'sc-nm',roughness:'sc-ro',metallic:'sc-me'};
const SC_KEYS  = {base_color:'bc',normal_map:'nm',roughness:'ro',metallic:'me'};

let state = {page:1, total:0, pages:1};
let flagState = {text:null, fake_ao:null, tint:null, flipped:null};
let debTimer = null;

// ── init score chips ──────────────────────────────────────────────────────────
function initScoreChips() {
  Object.entries(SCORE_CH).forEach(([ch, cid]) => {
    const wrap = document.getElementById(cid);
    const sc = SC_KEYS[ch];
    for (let i=0; i<=5; i++) {
      const el = document.createElement('div');
      el.className = `sc s${i} active`; el.textContent = i;
      el.dataset.sc = sc; el.dataset.v = i;
      el.onclick = () => { el.classList.toggle('active'); debounce(); };
      wrap.appendChild(el);
    }
  });
}

// ── chip group toggles ────────────────────────────────────────────────────────
document.querySelectorAll('.chips .chip').forEach(el => {
  el.onclick = () => { el.classList.toggle('active'); debounce(); };
});

function activeVals(groupId) {
  return Array.from(document.querySelectorAll(`#${groupId} .chip.active`)).map(e=>e.dataset.v);
}
function activeScores(sc) {
  return Array.from(document.querySelectorAll(`.sc[data-sc="${sc}"].active`)).map(e=>+e.dataset.v);
}

// ── flag 3-state ──────────────────────────────────────────────────────────────
function toggleFlag(key) {
  const el = document.getElementById('f_'+key);
  if      (flagState[key] === null)  { flagState[key]=true;  el.classList.add('on');  el.classList.remove('off'); }
  else if (flagState[key] === true)  { flagState[key]=false; el.classList.add('off'); el.classList.remove('on');  }
  else                               { flagState[key]=null;  el.classList.remove('on','off'); }
  debounce();
}

// ── build API params ──────────────────────────────────────────────────────────
function buildParams(page) {
  const p = new URLSearchParams();
  p.set('page', page); p.set('per', 40);
  const q = document.getElementById('q').value.trim();
  if (q) p.set('q', q);

  // score chips → lo/hi from selected set
  Object.entries(SC_KEYS).forEach(([ch, sc]) => {
    const sel = activeScores(sc);
    if (sel.length === 0) { p.set(sc+'_lo', 99); p.set(sc+'_hi', -1); }  // impossible → 0 results
    else { p.set(sc+'_lo', Math.min(...sel)); p.set(sc+'_hi', Math.max(...sel)); }
  });

  activeVals('chips-split').forEach(v => p.append('split', v));
  activeVals('chips-pbr').forEach(v => p.append('pbr', v));
  activeVals('chips-src').forEach(v => {
    // restore full source name from shortened display
    const map = {'objaverse':'sketchfab-objaverse'};
    p.append('src', map[v]||v);
  });
  if (flagState.text    !== null) p.set('flag_text',    flagState.text    ?'1':'0');
  if (flagState.fake_ao !== null) p.set('flag_fake_ao', flagState.fake_ao ?'1':'0');
  if (flagState.tint    !== null) p.set('flag_tint',    flagState.tint    ?'1':'0');
  if (flagState.flipped !== null) p.set('flag_flipped', flagState.flipped ?'1':'0');
  p.set('sort', document.getElementById('sort-select').value);
  return p;
}

function debounce() { clearTimeout(debTimer); debTimer = setTimeout(()=>fetchPage(1), 300); }

function badgeClass(s) { return (s===null||s===undefined) ? 'sn' : 's'+s; }
function scoreColor(s) {
  if (s===null||s===undefined) return '#555';
  return s<=1 ? '#ff8a8a' : s<=3 ? '#f0c060' : '#6dcc8a';
}

async function fetchPage(page) {
  state.page = page;
  const resp = await fetch('/api/assets?'+buildParams(page));
  const data = await resp.json();
  state.total = data.total; state.pages = data.pages;
  document.getElementById('results-info').textContent =
    `${data.total.toLocaleString()} 个资产  ·  第 ${data.page}/${data.pages} 页`;
  document.getElementById('page-info').textContent = `${data.page} / ${data.pages}`;
  document.getElementById('btn-prev').disabled = data.page <= 1;
  document.getElementById('btn-next').disabled = data.page >= data.pages;

  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  data.assets.forEach(a => {
    const s = a.scores;
    const flagHtml = Object.entries(a.flags).filter(([,v])=>v)
      .map(([k])=>({text:'文字',fake_ao:'伪AO',tint:'法线异色',flipped:'翻转'}[k]))
      .map(t=>`<span class="flag-badge">${t}</span>`).join('');
    const badgesHtml = Object.entries(SC_KEYS).map(([ch]) => {
      const v = s[ch];
      return `<span class="badge ${badgeClass(v)}" title="${ch}">${ch[0].toUpperCase()}:${v??'?'}</span>`;
    }).join('');
    const card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = `
      <div class="card-thumb">
        <img src="/img?ch=render&n=${encodeURIComponent(a.name)}&t=1" loading="lazy" alt="">
      </div>
      <div class="card-info">
        <div class="card-name" title="${a.name}">${a.name}</div>
        <div class="score-badges">${badgesHtml}</div>
        ${flagHtml?`<div class="score-badges" style="margin-top:3px">${flagHtml}</div>`:''}
      </div>`;
    card.onclick = () => openModal(a.name);
    grid.appendChild(card);
  });
}

function changePage(delta) { fetchPage(Math.max(1, Math.min(state.pages, state.page+delta))); }

function resetFilters() {
  document.getElementById('q').value = '';
  document.querySelectorAll('.sc,.chip').forEach(e=>e.classList.add('active'));
  flagState = {text:null,fake_ao:null,tint:null,flipped:null};
  ['text','fake_ao','tint','flipped'].forEach(k=>document.getElementById('f_'+k).classList.remove('on','off'));
  document.getElementById('sort-select').value = 'final_desc';
  fetchPage(1);
}

async function openModal(name) {
  const resp = await fetch('/api/asset/'+encodeURIComponent(name));
  const a = await resp.json();
  document.getElementById('modal-title').textContent = name;
  const s = a.scores;
  document.getElementById('modal-scores').innerHTML =
    `<span class="mscore">总分 <span style="color:${scoreColor(s.final)}">${s.final??'?'}</span></span>`
    + Object.entries(SCORE_CH).map(([ch,k])=>
        `<span class="mscore">${CH_LABELS[ch]} <span style="color:${scoreColor(s[ch])}">${s[ch]??'?'}</span></span>`).join('')
    + (a.tier ? `<span class="mscore" style="color:#888">${a.tier}</span>` : '')
    + (a.pbr  ? `<span class="mscore" style="color:#888">${a.pbr}</span>` : '');

  const cg = document.getElementById('channel-grid');
  cg.innerHTML = '';
  CHANNELS.forEach(ch => {
    const exists = a.channels[ch];
    const url = `/img?ch=${ch}&n=${encodeURIComponent(name)}`;
    const scoreVal = a.scores[{base_color:'base_color',normal_map:'normal_map',
      roughness:'roughness',metallic:'metallic',render:'render'}[ch]];
    const scoreTag = scoreVal !== undefined && scoreVal !== null
      ? `<span class="badge ${badgeClass(scoreVal)}">${scoreVal}</span>` : '';
    const div = document.createElement('div');
    div.className = 'ch-card';
    div.innerHTML = exists
      ? `<div class="ch-label"><span>${CH_LABELS[ch]}</span>${scoreTag}</div>
         <div class="ch-img-wrap" onclick="openLightbox('${url}')">
           <img src="${url}&t=1" loading="lazy" alt="${ch}">
         </div>`
      : `<div class="ch-label">${CH_LABELS[ch]}</div>
         <div class="ch-missing">missing</div>`;
    cg.appendChild(div);
  });
  document.getElementById('modal-overlay').classList.add('show');
}

function closeModal() { document.getElementById('modal-overlay').classList.remove('show'); }

function openLightbox(url) {
  document.getElementById('lb-img').src = url;
  document.getElementById('lightbox').classList.add('show');
}
function closeLightbox() { document.getElementById('lightbox').classList.remove('show'); }

document.addEventListener('keydown', e => {
  if (e.key === 'Escape') { closeLightbox(); closeModal(); }
});

// init
initScoreChips();
// re-attach chip listeners after DOM init
document.querySelectorAll('#chips-split .chip,#chips-pbr .chip,#chips-src .chip').forEach(el=>{
  el.onclick = () => { el.classList.toggle('active'); debounce(); };
});
fetchPage(1);
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return HTML


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", default="0.0.0.0")
    args = p.parse_args()
    print(f"Starting viewer on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
