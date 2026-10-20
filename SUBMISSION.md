# SlothSort — submission (DEV Hacktoberfest AI Challenge, Week 3: Slow and Steady)

- **Repo:** https://github.com/arindewangan/slothsort
- **Live demo:** https://arindewangan.github.io/slothsort/demo/
  (precomputed classifier outputs on a SIMULATED ~/Downloads set, clearly labeled; fully client-side)
- **Video:** https://youtu.be/VIDEO_ID (brag video, ~20s)
- **DEV article:** https://dev.to/arin_dewangan_6110e41aadb/ARTICLE_SLUG
  (tags: devchallenge, hf26challenge, ai, opensource)

**Partner-category claim:** Best Use of TabPFN — TabPFN (open-source tabular
foundation model) is the entire decision engine: local-CPU in-context
classification of every file into KEEP / ARCHIVE / DELETE. No proprietary
models or API keys anywhere in the pipeline.

**Honesty notes for judges:**
- Training data is SIMULATED (6,000 synthetic file records, seeded RNG).
- Demo predictions are PRECOMPUTED by the real classifier and labeled with the build date.
- If TabPFN was unavailable at bake time, the demo says "fallback" instead — never faked.
