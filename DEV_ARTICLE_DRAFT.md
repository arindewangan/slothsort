# SlothSort: I taught an open-source AI to clean my Downloads folder so I'd never have to 🦥

*DEV Hacktoberfest AI Challenge — Week 3: Slow and Steady. Cover image: `assets/banner.png`. Tags: `devchallenge, hf26challenge, ai, opensource`.*

---

My Downloads folder is where files go to be forgotten. Installers from 2024. `report-final(1).pdf` sitting next to `report-final.pdf`. Screenshots of things I no longer remember screenshotting. Every few months I open the folder, sigh, and close it again — the digital equivalent of shoving everything into a closet before guests arrive.

So I built **SlothSort** — a file tidier that does the procrastinated chore for you, using **TabPFN**, the open-source tabular foundation model, running entirely on your own CPU. Point it at a folder listing and it triages every file into **KEEP / ARCHIVE / DELETE** with a confidence score and a plain-English reason — the 24-file demo folder scores in a single ~9-second forward pass — then archives or deletes in one click, always with undo, because laziness should never be irreversible.

**🔗 Live demo:** https://arindewangan.github.io/slothsort/demo/ (precomputed TabPFN outputs on a simulated folder, clearly labeled — works with zero backend)
**💻 Repo:** https://github.com/arindewangan/slothsort
**🎬 Video:** https://youtu.be/VIDEO_ID (~20s)

## What it does

Open the dashboard and SlothSort has already triaged a simulated ~/Downloads folder — 24 files, the usual suspects:

- **ChromeSetup.exe** (untouched 400 days) → **DELETE** — "It's an installer — you can re-download it any time"
- **report-final(1).pdf** → **DELETE** — "Suspicious name — smells like a duplicate"
- **trip-goa-2024.mp4** (1.8 GB, unopened 480 days) → **ARCHIVE** — "Cold media file worth keeping, not worth the clutter"
- **report-final.pdf** (opened 3 days ago) → **KEEP** — "still in active use"

The dashboard shows a verdict badge, a confidence bar, and the reasons for every file. One click archives or deletes (simulated in the demo, with full undo), and a stats row totals what you'd reclaim. There's also a "score your own file metadata" box — paste a filename, size, and dates, and get a verdict. Note the deliberate design choice: **you paste metadata, never the file**. SlothSort never reads file contents, because it never needs to.

## How it's built

A Python/Flask backend with a vanilla JS frontend. The pipeline:

1. `data/build_dataset.py` generates **6,000 simulated file records** locally (seeded RNG, reproducible) — filename clutter signals, sizes, ages, access gaps — labeled KEEP/ARCHIVE/DELETE by common-sense "digital hoarding" heuristics with ~8% label noise, so the model has to generalize instead of memorizing a rule list.
2. At request time, **TabPFN classifies each file in-context** on a seeded, class-stratified 999-row context (333 per class) — one forward pass per batch, local CPU, no gradient training, no API key. Low-confidence calls (<55%) are flagged "review manually" rather than auto-acted on.
3. `recommender.py` turns the winning class into an action plus deterministic reasons drawn from the file's *own metadata*. The model decides the class; the file explains itself.
4. The frontend is vanilla HTML/CSS/JS. The GitHub Pages demo bakes **real precomputed TabPFN outputs** (via `demo/build_demo.py`, stamped with the build date) so it works with zero backend.

## The honest hard parts

**Getting TabPFN running at all.** The build machine's `/tmp` is a 512MB tmpfs, and the default torch wheel (~800MB) died mid-unpack with "No space left on device" — on a disk with gigabytes free. The fix was a CPU-only torch install with `TMPDIR` pointed at real disk, plus `--ignore-installed` to dodge a Debian-pinned `typing_extensions` that pip refused to touch. Then the real surprise: current TabPFN (v9.x) requires a Prior Labs account + license acceptance + API token just to download weights — a non-starter for a "no accounts, no keys" project. So SlothSort runs **TabPFN 2.2.1**, the last version with freely downloadable open weights. None of this is in any tutorial; all of it is the actual tax of "runs on your CPU."

**Making the synthetic data non-trivial.** My first label rules were so clean a decision stump could ace them — which would make TabPFN decorative, not load-bearing. The 8% label noise and overlapping rules (a big cold media file that's *also* recently opened, a cluttered name on a warm document) exist specifically so the classifier has to weigh evidence like a real triage decision.

**Refusing to fake explanations.** It would have been easy to dress up rule outputs as "the model's reasoning." Instead the architecture is explicit: TabPFN outputs class probabilities; the reasons come from the file's metadata. The UI tooltips say exactly that. An explanation you can't trace to something real is marketing, not AI.

## What I'm proud of

- **Real open-source-AI core:** TabPFN inference on real (simulated, disclosed) data, reproducible end-to-end from the repo — no API keys, no proprietary models, no fake "AI" labels. If TabPFN isn't installed, the app runs a *labeled* fallback and says "NOT TabPFN" on every output.
- **Privacy as architecture, not a bullet point:** metadata-only features mean the file list never needs to leave the machine. That's the whole reason open weights matter here (more below).
- **Laziness with a safety net:** every destructive action is one click *with undo* — the demo keeps a full undo stack, and the design assumes you'll never double-check the AI.
- **An honest demo:** precomputed vs. live is labeled, simulated data is labeled, the build date is stamped, low-confidence verdicts ask for review.

## What I learned

TabPFN does 3-class in-context learning out of the box with no architecture changes — the same `TabPFNClassifier` that did binary risk scoring in my Week 2 entry (LatePay) handled KEEP/ARCHIVE/DELETE with just a stratified context. The technical takeaway: for small tabular problems, **data plumbing beats model choice** — the interesting work was all in the label design and the feature encoding (log-size, clutter signals), not the classifier. And the process takeaway: labeling your demo "simulated" costs nothing and buys everything. Nobody trusts a file-deleting AI that won't show its work.

## What's next

- **Real folder scanning:** a `pathlib` walker that reads actual `~/Downloads` metadata into the same feature pipeline, with the undo log persisted to disk.
- **Learn from overrides:** when you undo the AI's verdict, feed the correction back as context — TabPFN's in-context learning makes this nearly free.
- **Per-user thresholds:** let the chronically anxious set "never auto-delete above 90% confidence."
- **Scheduled sloth mode:** a weekly background pass that quietly archives the cold stuff and reports what it did — maximum laziness, minimum surprise.

## How it uses open-source AI

SlothSort's decision engine is **TabPFN** (Apache 2.0, Prior Labs) — an open-source tabular foundation model. I generate a simulated file-history dataset locally with a seeded script, and TabPFN classifies each file's metadata vector in-context on the local CPU at inference time. No proprietary models or API keys are involved anywhere in the pipeline — which is exactly why this product can exist: a file tidier that phones your file list to a cloud API is a privacy non-starter, but an open-weight model on your own machine never sends anything anywhere. It works offline, it costs nothing, and nobody's file list is the product.

---

*Built with: Python, Flask, TabPFN (open-source), JavaScript. License: MIT.*

---

## Writing-quality checklist

- [x] Hook lands in the first 3 lines (pain → project, no throat-clearing).
- [x] Every AI claim grounded in code/data (files and functions named).
- [x] Honest limitations stated (simulated data, precomputed demo, fallback labeling, /tmp install fight).
- [x] No marketing fluff ("revolutionary"/"game-changer" absent).
- [ ] All four links verified live: demo, repo, video, cover image (VIDEO_ID + article slug pending publish).
- [x] Tags: `devchallenge, hf26challenge, ai, opensource`.
- [ ] Read aloud once before publishing.
- [x] "How it uses open-source AI" names model, license, data source.
- [x] Non-US-centric examples (Goa trip video, Bengaluru author context in profile).
