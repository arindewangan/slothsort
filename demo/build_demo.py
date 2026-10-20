"""Bake REAL classifier predictions for the simulated demo set into demo/index.html.

Run:  python demo/build_demo.py
Out:  demo/index.html  (fully client-side — no backend needed on GitHub Pages)

The baked JSON records the true model kind ("tabpfn" or "fallback") and the
build date, and the page labels everything SIMULATED/PRECOMPUTED. Nothing is
ever presented as a TabPFN prediction unless TabPFN actually produced it.
"""
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from classifier import classify, classify_batch, get_model, load_dataset, tabpfn_available  # noqa: E402
from recommender import recommend  # noqa: E402
from demo_data import DEMO_FILES, SIMULATED_NOTE  # noqa: E402
import data.build_dataset as build_dataset  # noqa: E402

OUT_HTML = os.path.join(HERE, "index.html")
BUILD_DATE = datetime.date.today().isoformat()


def main():
    if not os.path.exists(os.path.join(ROOT, "data", "file_history.csv")):
        print("building simulated dataset...")
        build_dataset.main()
    m = get_model()
    print(f"model kind: {m['kind']} (tabpfn_available={tabpfn_available()})")
    X, y = load_dataset()
    print(f"training rows: {len(X)}")
    results = []
    for meta, cls in zip(DEMO_FILES, classify_batch(DEMO_FILES)):
        rec = recommend(meta, cls)
        rec["name"] = meta["name"]
        rec["size_mb"] = meta["size_mb"]
        rec["age_days"] = meta["age_days"]
        rec["days_since_access"] = meta["days_since_access"]
        results.append(rec)
    payload = {
        "build_date": BUILD_DATE,
        "model_kind": m["kind"],
        "model_label": results[0]["model"],
        "data_note": results[0]["data_note"],
        "simulated_note": SIMULATED_NOTE,
        "context_rows": m["n_train"],
        "results": results,
    }
    html = TEMPLATE.replace("__PAYLOAD__", json.dumps(payload))
    with open(OUT_HTML, "w") as f:
        f.write(html)
    kinds = {}
    for r in results:
        kinds[r["prediction"]] = kinds.get(r["prediction"], 0) + 1
    print(f"baked {len(results)} predictions -> {OUT_HTML} "
          f"(KEEP={kinds.get('KEEP',0)} ARCHIVE={kinds.get('ARCHIVE',0)} DELETE={kinds.get('DELETE',0)})")


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SlothSort — AI file triage for the chronically lazy (demo)</title>
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
.review{color:var(--arch);font-size:12px;font-weight:700}
.tip{cursor:help;border-bottom:1px dotted var(--muted)}
</style>
</head>
<body><div class="wrap">
<h1>🦥 SlothSort <span class="note">demo</span></h1>
<p class="sub">Your Downloads folder has 400+ files you've been "meaning to sort" since 2023. SlothSort triages them for you — with TabPFN, an open-source tabular foundation model running locally on your CPU. The 24-file demo folder below scores in a single ~9-second forward pass.</p>
<div class="priv">🔒 <b>100% local &amp; private:</b> SlothSort sees only file <i>metadata</i> (name, size, dates) — never contents. Nothing leaves your machine; it works fully offline, set up once, never pay.</div>
<div class="sim">⚠️ <b>PRECOMPUTED + SIMULATED demo</b> (baked <span id="bdate"></span>): predictions below were computed by <b id="mkind"></b> on an <b>invented</b> ~/Downloads set — no real files were scanned. The live Flask app (<code>app.py</code>) scores the same set on request.</div>

<div class="card">
<h3>Simulated ~/Downloads — triaged in seconds</h3>
<div class="stats">
<div class="stat"><b id="s-keep" style="color:var(--keep)">–</b>keep</div>
<div class="stat"><b id="s-arch" style="color:var(--arch)">–</b>archive</div>
<div class="stat"><b id="s-del" style="color:var(--del)">–</b>delete</div>
<div class="stat"><b id="s-freed">–</b>reclaimable</div>
</div>
<p><button class="ghost" id="undobtn" onclick="undo()" disabled>↩ Undo last action</button>
<button class="ghost" onclick="resetAll()">Reset demo</button></p>
<table id="ftable"><thead><tr><th>File</th><th>Verdict <span class="note tip" title="Predicted by the baked model. Confidence below 55% is flagged for manual review.">ⓘ</span></th><th>Why <span class="note tip" title="Reasons are derived from the file's own metadata — not invented model internals.">ⓘ</span></th><th>Do it</th></tr></thead><tbody></tbody></table>
<p class="note" id="meta"></p>
</div>

<div class="card">
<h3>How it works</h3>
<ol>
<li><b>Metadata features</b> — size, age, days since opened, file-type category, filename clutter signals (<i>final</i>, <i>copy</i>, <i>v2</i>…), hidden/temp flags. Contents never touched.</li>
<li><b>TabPFN</b> (open-source tabular foundation model, local CPU, no API key) classifies each file <b>KEEP / ARCHIVE / DELETE</b> from 6,000 simulated file records.</li>
<li><b>Act</b> — one click archives or deletes, always with undo. The real script moves files on your machine; this demo only simulates.</li>
</ol>
<p class="note">Built Oct 20, 2026 for the DEV Hacktoberfest AI Challenge (Week 3: Slow and Steady). MIT licensed. Repo: <a href="https://github.com/arindewangan/slothsort" style="color:var(--acc)">github.com/arindewangan/slothsort</a></p>
</div>
</div>
<script>
const P = __PAYLOAD__;
const state = {}; P.results.forEach(f=>state[f.name]="active");
const undoStack = [];
function fmtMB(m){return m>=1024?(m/1024).toFixed(1)+" GB":m>=1?m.toFixed(m<10?1:0)+" MB":Math.round(m*1024)+" KB"}
function badge(p){return `<span class="badge b-${p}">${p}</span>`}
function confColor(p){return p==="KEEP"?"var(--keep)":p==="ARCHIVE"?"var(--arch)":"var(--del)"}
function render(){
 const tb=document.querySelector("#ftable tbody");tb.innerHTML="";
 let k=0,a=0,d=0,freed=0;
 P.results.forEach(f=>{
  const st=state[f.name];
  if(st==="active"){if(f.prediction==="KEEP")k++;else if(f.prediction==="ARCHIVE"){a++;freed+=f.size_mb}else{d++;freed+=f.size_mb}}
  const tr=document.createElement("tr");if(st!=="active")tr.className="strike";
  const safe=f.name.replace(/'/g,"");
  tr.innerHTML=`<td><b>${f.name}</b><br><span class="note">${fmtMB(f.size_mb)} · ${f.age_days}d old · opened ${f.days_since_access}d ago</span></td>
  <td>${badge(f.prediction)}${f.needs_review?'<br><span class="review">⚠ low confidence — review</span>':""}<div class="confbar"><div style="width:${f.confidence_pct}%;background:${confColor(f.prediction)}"></div></div><span class="note">${f.confidence_pct}% confident</span></td>
  <td><div class="reason" title="Derived from this file's metadata">${f.reasons.map(x=>"• "+x).join("<br>")}</div></td>
  <td>${st==="active"?`<button class="ghost" onclick="doAct('${safe}','archived')">📦 Archive</button><button class="danger" onclick="doAct('${safe}','deleted')">🗑 Delete</button>`:`<span class="note">${st}</span>`}</td>`;
  tb.appendChild(tr);
 });
 document.getElementById("s-keep").textContent=k;
 document.getElementById("s-arch").textContent=a;
 document.getElementById("s-del").textContent=d;
 document.getElementById("s-freed").textContent=fmtMB(freed);
 document.getElementById("undobtn").disabled=!undoStack.length;
}
function doAct(name,to){undoStack.push({...state});state[name]=to;render()}
function undo(){if(!undoStack.length)return;const p=undoStack.pop();Object.keys(state).forEach(k=>state[k]=p[k]);render()}
function resetAll(){Object.keys(state).forEach(k=>state[k]="active");undoStack.length=0;render()}
document.getElementById("bdate").textContent=P.build_date;
document.getElementById("mkind").textContent=P.model_kind==="tabpfn"?"TabPFN (open-source, local CPU)":"the labeled rule-based fallback (TabPFN was unavailable at bake time)";
document.getElementById("meta").textContent=P.data_note+" · "+P.model_label+" · context: "+P.context_rows+" simulated rows · "+P.simulated_note;
render();
</script>
</body></html>
"""

if __name__ == "__main__":
    main()
