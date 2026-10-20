# 🦥 SlothSort — the AI that cleans your Downloads folder so you don't have to

Your Downloads folder has 400+ files you've been "meaning to sort" since 2023.
SlothSort triages every one in seconds: a **TabPFN** classifier (open-source
tabular foundation model, local CPU — no cloud, no API keys) scores each file
from its metadata alone into **KEEP / ARCHIVE / DELETE**, with confidence and
human-readable reasons. One click archives or deletes (always with undo).

**Privacy is the point:** SlothSort sees only file *metadata* (name, size,
dates) — never contents. The file list never leaves your machine. Works fully
offline. Set up once, never pay.

Built for the **DEV Hacktoberfest AI Challenge — Week 3: Slow and Steady**
("build something with open-weight models or open-source AI that rewards
laziness"), Oct 19–25, 2026.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Build the simulated training set (seeded, reproducible)
python data/build_dataset.py

# 2. Bake the client-side demo with REAL classifier predictions
python demo/build_demo.py

# 3. Run the live Flask app
python app.py          # -> http://127.0.0.1:5000

# 4. Run the tests
pytest tests/ -q
```

**TabPFN is optional — and pinned to 2.x.** `requirements.txt` pins `tabpfn>=2.0,<3`:
TabPFN 2.x is the last line with freely downloadable open weights; TabPFN ≥ 3
requires a Prior Labs account, license acceptance, and `TABPFN_TOKEN`, which
would break the "no accounts, no keys" promise. If `tabpfn` isn't installed
(or weights can't load), set `TABPFN_MODEL=fallback` (or just run — it
auto-detects) and the app runs a *labeled* rule-based fallback. Fallback
outputs are explicitly marked `"model_kind": "fallback"` / "NOT TabPFN"
everywhere — never presented as TabPFN predictions.

## Architecture

```
slothsort/
├── app.py               # Flask: dashboard, /api/scan, /api/demo, /api/act, /api/health
├── features.py          # metadata -> 6-dim feature vector (contents never read)
├── classifier.py        # TabPFN (local CPU, env-gated) + honest labeled fallback
├── recommender.py       # class -> action + metadata-derived reasons
├── demo_data.py         # SIMULATED ~/Downloads set (invented names/sizes/dates)
├── data/
│   ├── build_dataset.py # generates 6,000-row SIMULATED file_history.csv (seed 42)
│   └── file_history.csv # generated; git-ignored in real use, kept here for repro
├── demo/
│   ├── build_demo.py    # precomputes REAL predictions -> demo/index.html
│   └── index.html       # fully client-side demo (GitHub Pages ready)
└── tests/
    └── test_slothsort.py
```

**The pipeline:** `data/build_dataset.py` synthesizes 6,000 file records with
common-sense "digital hoarding" labels + 8% label noise. At request time,
`classifier.py` fits TabPFN in-context on a seeded, class-stratified 999-row
context (CPU-friendly) and returns P(KEEP/ARCHIVE/DELETE). `recommender.py`
turns the winning class into an action plus deterministic reasons drawn from
the file's own metadata. Low-confidence calls (<55%) are flagged for manual
review instead of being auto-acted on.

## Honest demo-mode disclosure

- **All training data is simulated.** `file_history.csv` is generated locally by
  `data/build_dataset.py` (seeded RNG); it is not real user file data, and the
  UI + API say so on every response (`data_note`).
- **The demo folder is simulated.** `demo_data.py` is an invented ~/Downloads
  set — plausible names, fictional everything. No real files were scanned.
- **The GitHub Pages demo is precomputed.** `demo/build_demo.py` runs the real
  classifier over the simulated set and bakes the outputs into
  `demo/index.html`, stamped with the build date and the true model kind.
  Predictions labeled "TabPFN" were produced by TabPFN; if the bake ran in
  fallback mode, the page says so instead.
- **`/api/act` simulates.** Archive/delete/undo flip in-memory labels on the
  simulated set only. Nothing touches real files.

## Why open innovation matters here

A file tidier that uploads your file list to a cloud API is a non-starter —
the list *is* the private data. SlothSort works because TabPFN is open-weight:
the whole model runs on your CPU, offline, free forever. That isn't a feature
bullet; it's the reason the product can exist at all.

## License

MIT — see `LICENSE`.
