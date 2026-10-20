"""SlothSort Flask app: the lazy person's file-triage dashboard + JSON API.

Routes:
  GET  /            dashboard (single-page UI, inline template)
  POST /api/scan    score a list of file-metadata dicts -> recommendations
  GET  /api/demo    scored SIMULATED ~/Downloads set (real model inference)
  POST /api/act     simulate archive/delete with undo (in-memory, nothing touches disk)
  GET  /api/health  model kind, dataset status

SlothSort never reads file contents and never touches real files: /api/act
only flips in-memory labels on the simulated set, with a full undo stack.
"""
import copy

from flask import Flask, jsonify, render_template_string, request

from classifier import get_model, classify, classify_batch, load_dataset
from features import featurize
from recommender import recommend
from demo_data import DEMO_FILES, SIMULATED_NOTE

app = Flask(__name__)

# In-memory simulation state: name -> "active" | "archived" | "deleted".
_sim_state = {f["name"]: "active" for f in DEMO_FILES}
_undo_stack = []


def _scored_files():
    out = []
    for meta, cls in zip(DEMO_FILES, classify_batch(DEMO_FILES)):
        rec = recommend(meta, cls)
        rec["name"] = meta["name"]
        rec["size_mb"] = meta["size_mb"]
        rec["age_days"] = meta["age_days"]
        rec["days_since_access"] = meta["days_since_access"]
        rec["state"] = _sim_state.get(meta["name"], "active")
        out.append(rec)
    return out


@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/api/scan", methods=["POST"])
def api_scan():
    data = request.get_json(force=True) or {}
    files = data.get("files", [])
    if not isinstance(files, list) or not files:
        return jsonify({"error": "POST JSON like {\"files\": [{\"name\":..., \"size_mb\":..., \"age_days\":..., \"days_since_access\":...}]}"}), 400
    # Validate first, then score all valid rows in one forward pass.
    valid, results = [], []
    for meta in files[:200]:
        if not isinstance(meta, dict):
            results.append({"name": "?", "error": "bad input: not an object"})
            continue
        try:
            featurize(meta)
            valid.append(meta)
            results.append(None)
        except (ValueError, TypeError) as e:
            results.append({"name": str(meta.get("name", "?")),
                            "error": f"bad input: {e}"})
    for meta, cls in zip(valid, classify_batch(valid)):
        rec = recommend(meta, cls)
        rec["name"] = str(meta.get("name", "file"))
        results[results.index(None)] = rec
    return jsonify({"simulated_input": True, "results": results})


@app.route("/api/demo")
def api_demo():
    """Scored simulated ~/Downloads set — real model inference, labeled SIMULATED."""
    return jsonify({"simulated": True, "note": SIMULATED_NOTE, "results": _scored_files()})


@app.route("/api/act", methods=["POST"])
def api_act():
    """Simulate archive/delete/undo on the demo set. Nothing touches real files."""
    data = request.get_json(force=True) or {}
    action = data.get("action")
    name = data.get("name")
    if action not in ("archive", "delete", "undo"):
        return jsonify({"error": "action must be archive, delete, or undo"}), 400
    if action == "undo":
        if not _undo_stack:
            return jsonify({"ok": False, "error": "nothing to undo"})
        prev = _undo_stack.pop()
        _sim_state.update(prev)
        return jsonify({"ok": True, "undone": True, "state": copy.deepcopy(_sim_state)})
    if name not in _sim_state:
        return jsonify({"error": f"unknown file: {name}"}), 404
    _undo_stack.append(copy.deepcopy(_sim_state))
    _sim_state[name] = "archived" if action == "archive" else "deleted"
    return jsonify({"ok": True, "name": name, "new_state": _sim_state[name],
                    "undo_depth": len(_undo_stack)})


@app.route("/api/health")
def api_health():
    m = get_model()
    try:
        X, y = load_dataset()
        rows = len(X)
    except FileNotFoundError:
        rows = 0
    return jsonify({"status": "ok", "model": m["kind"],
                    "training_rows": rows, "demo_files": len(DEMO_FILES),
                    "undo_depth": len(_undo_stack)})


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SlothSort — the AI that cleans your Downloads folder so you don't have to</title>
<style>
:root{--bg:#10160f;--card:#1a2318;--ink:#eef3e6;--muted:#9aa88f;--keep:#5cc46a;--arch:#e8b13c;--del:#ef6a5e;--acc:#7bc96f}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1040px;margin:0 auto;padding:24px}
h1{margin:0 0 4px;font-size:34px}.sub{color:var(--muted);margin:0 0 16px}
.sim{background:#3a2f10;border:1px solid var(--arch);color:var(--arch);border-radius:8px;padding:10px 14px;margin:16px 0;font-size:14px}
.priv{background:#10281a;border:1px solid var(--keep);color:var(--keep);border-radius:8px;padding:10px 14px;margin:16px 0;font-size:14px}
.card{background:var(--card);border-radius:12px;padding:20px;margin:16px 0}
.stats{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0}
.stat{flex:1;min-width:140px;background:#10160f;border-radius:10px;padding:12px;text-align:center}
.stat b{font-size:26px;display:block}
button{background:var(--acc);border:0;border-radius:8px;padding:8px 14px;font-weight:700;cursor:pointer;color:#0b1f0a;margin:2px}
button.ghost{background:transparent;border:1px solid #3a4a35;color:var(--ink)}
button.danger{background:transparent;border:1px solid var(--del);color:var(--del)}
button:disabled{opacity:.4;cursor:default}
table{width:100%;border-collapse:collapse;font-size:14px}
th,td{text-align:left;padding:9px 8px;border-bottom:1px solid #2a3527;vertical-align:top}
.badge{display:inline-block;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:800;color:#0b1f0a}
.b-KEEP{background:var(--keep)}.b-ARCHIVE{background:var(--arch)}.b-DELETE{background:var(--del)}
.confbar{height:8px;border-radius:4px;background:#2a3527;overflow:hidden;min-width:90px;margin-top:4px}
.confbar>div{height:100%}
.reason{color:var(--muted);font-size:12.5px;margin-top:4px}
.strike td{opacity:.45;text-decoration:line-through}
.note{font-size:12px;color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}
label{font-size:12px;color:var(--muted);display:block;margin-bottom:4px}
input{width:100%;padding:8px;border-radius:8px;border:1px solid #3a4a35;background:#10160f;color:var(--ink)}
.review{color:var(--arch);font-size:12px;font-weight:700}
</style>
</head>
<body><div class="wrap">
<h1>🦥 SlothSort</h1>
<p class="sub">Your Downloads folder has 400+ files you've been "meaning to sort" since 2023. SlothSort triages them for you — with TabPFN, an open-source tabular foundation model running locally on your CPU. No cloud, no API keys, no file contents ever read. The 24-file demo folder scores in a single ~9-second forward pass.</p>
<div class="priv">🔒 <b>100% local &amp; private:</b> SlothSort sees only file <i>metadata</i> (name, size, dates) — never contents. Nothing leaves your machine; it works fully offline.</div>
<div class="sim">⚠️ <b>SIMULATED data:</b> the model trained on 6,000 synthetic file records generated locally (seeded RNG), and the folder below is an invented ~/Downloads set. Scores are illustrative.</div>

<div class="card">
<h3>Simulated ~/Downloads — triaged</h3>
<div class="stats">
<div class="stat"><b id="s-keep" style="color:var(--keep)">–</b>keep</div>
<div class="stat"><b id="s-arch" style="color:var(--arch)">–</b>archive</div>
<div class="stat"><b id="s-del" style="color:var(--del)">–</b>delete</div>
<div class="stat"><b id="s-freed">–</b>reclaimable</div>
<div class="stat"><b id="s-model">–</b>model</div>
</div>
<p><button onclick="scan()">🦥 Sort my (simulated) folder</button>
<button class="ghost" id="undobtn" onclick="act('undo')" disabled>↩ Undo</button></p>
<table id="ftable"><thead><tr><th>File</th><th>Verdict</th><th>Why</th><th>Do it</th></tr></thead><tbody></tbody></table>
<p class="note" id="meta"></p>
</div>

<div class="card">
<h3>Score your own file metadata</h3>
<p class="note">Paste metadata (not the file!) — SlothSort never needs the actual file.</p>
<div class="grid">
<div><label>Filename</label><input id="f-name" value="old-installer-v3.exe"></div>
<div><label>Size (MB)</label><input id="f-size" type="number" value="85"></div>
<div><label>Age (days)</label><input id="f-age" type="number" value="300"></div>
<div><label>Days since opened</label><input id="f-acc" type="number" value="300"></div>
</div>
<p><button onclick="scoreOne()">Score it</button></p>
<div id="one"></div>
</div>

<div class="card">
<h3>How it works</h3>
<ol>
<li><b>Metadata features</b> — size, age, days since opened, file-type category, filename clutter signals (final/copy/v2/…), hidden/temp flags. Contents never touched.</li>
<li><b>TabPFN</b> (open-source tabular foundation model, local CPU) classifies each file KEEP / ARCHIVE / DELETE from 6,000 simulated file records.</li>
<li><b>Act</b> — one click archives or deletes (simulated here, undo included); the real script moves files, always with an undo log.</li>
</ol>
<p class="note">Built Oct 20, 2026 for the DEV Hacktoberfest AI Challenge (Week 3: Slow and Steady). MIT licensed.</p>
</div>
</div>
<script>
function fmtMB(m){return m>=1024?(m/1024).toFixed(1)+" GB":m>=1?m.toFixed(m<10?1:0)+" MB":Math.round(m*1024)+" KB"}
function badge(p){return `<span class="badge b-${p}">${p}</span>`}
function confColor(p){return p==="KEEP"?"var(--keep)":p==="ARCHIVE"?"var(--arch)":"var(--del)"}
async function scan(){
 const r=await fetch("/api/demo").then(r=>r.json());
 const tb=document.querySelector("#ftable tbody");tb.innerHTML="";
 let k=0,a=0,d=0,freed=0;
 r.results.forEach(f=>{
  if(f.prediction==="KEEP")k++;else if(f.prediction==="ARCHIVE"){a++;freed+=f.size_mb}else{d++;freed+=f.size_mb}
  const tr=document.createElement("tr");
  if(f.state!=="active")tr.className="strike";
  tr.innerHTML=`<td><b>${f.name}</b><br><span class="note">${fmtMB(f.size_mb)} · ${f.age_days}d old · opened ${f.days_since_access}d ago</span></td>
  <td>${badge(f.prediction)}${f.needs_review?'<br><span class="review">⚠ low confidence — review</span>':""}<div class="confbar"><div style="width:${f.confidence_pct}%;background:${confColor(f.prediction)}"></div></div><span class="note">${f.confidence_pct}%</span></td>
  <td><div class="reason">${f.reasons.map(x=>"• "+x).join("<br>")}</div></td>
  <td>${f.state==="active"?`<button class="ghost" onclick="act('archive','${f.name.replace(/'/g,"")}')">📦 Archive</button><button class="danger" onclick="act('delete','${f.name.replace(/'/g,"")}')">🗑 Delete</button>`:`<span class="note">${f.state}d</span>`}</td>`;
  tb.appendChild(tr);
 });
 document.getElementById("s-keep").textContent=k;
 document.getElementById("s-arch").textContent=a;
 document.getElementById("s-del").textContent=d;
 document.getElementById("s-freed").textContent=fmtMB(freed);
 document.getElementById("s-model").textContent=r.results[0].model_kind==="tabpfn"?"TabPFN":"fallback";
 document.getElementById("meta").textContent=r.results[0].data_note+" · "+r.results[0].model;
 document.getElementById("undobtn").disabled=false;
}
async function act(action,name){
 const r=await fetch("/api/act",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({action,name})}).then(r=>r.json());
 if(r.ok)scan();else alert(r.error||"failed");
}
async function scoreOne(){
 const body={files:[{name:document.getElementById("f-name").value,size_mb:+document.getElementById("f-size").value,age_days:+document.getElementById("f-age").value,days_since_access:+document.getElementById("f-acc").value}]};
 const r=await fetch("/api/scan",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)}).then(r=>r.json());
 const f=r.results[0];
 document.getElementById("one").innerHTML=f.error?`<p>${f.error}</p>`:
 `<p>${badge(f.prediction)} <b>${f.confidence_pct}%</b> — ${f.action}: ${f.action_detail}</p><div class="reason">${f.reasons.map(x=>"• "+x).join("<br>")}</div><p class="note">${f.model} · ${f.data_note}</p>`;
}
scan();
</script>
</body></html>
"""


if __name__ == "__main__":
    app.run(debug=True)
