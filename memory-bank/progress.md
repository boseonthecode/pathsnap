# PathSnap — Progress Tracker

> Update this file after every completed implementation step.
> Move steps from **To Do** to **In Progress** to **Completed**; include the completion date and a one-line verification note.
> **Last updated:** 2026-11-18
> **Data audit:** 2026-11-18 — verified dataset against plan; findings and doc corrections recorded in `architecture.md` under `## Updates`.

---

## ⏭️ To Do

- [ ] Step 03 — Trip loader and validator
- [ ] Step 04 — Dataset inventory / EDA script
- [ ] Step 05 — Geometry and projection utilities
- [ ] Step 06 — Reference label alignment
- [ ] Step 07 — Candidate generation with spatial index
- [ ] Step 08 — Feature extraction
- [ ] Step 09 — Feature cache and versioning
- [ ] Step 10 — Metrics module
- [ ] Step 11 — Training dataset assembly
- [ ] Step 12 — First three model families
- [ ] Step 13 — Remaining model families (≥ 5 total)
- [ ] Step 14 — Cross-validation and model comparison
- [ ] Step 15 — Model comparison plots
- [ ] Step 16 — Bayesian hyperparameter optimisation
- [ ] Step 17 — Tuned vs. untuned comparison and export
- [ ] Step 18 — HMM emission and transition models
- [ ] Step 19 — Viterbi decoding and route stitching
- [ ] Step 20 — Decoder comparison
- [ ] Step 21 — Decoder visualisation and failure analysis
- [ ] Step 22 — Artifact loading service
- [ ] Step 23 — FastAPI core endpoints
- [ ] Step 24 — API safeguards, configuration, and logging
- [ ] Step 25 — Static Leaflet UI shell
- [ ] Step 26 — Interactive noise and sampling controls
- [ ] Step 27 — Two-laptop deployment and CORS verification
- [ ] Step 28 — Folium map generation
- [ ] Step 29 — End-to-end verification across all ten test trips
- [ ] Step 30 — README and deployment instructions
- [ ] Step 31 — Technical report
- [ ] Step 32 — Leakage, security, and repository QA

---

## 🔄 In Progress

_None._

---

## ✅ Completed

- [x] Step 01 — Repository scaffolding (2026-11-18) — folder skeleton, `requirements.txt`, `.gitignore`, placeholder README, package import smoke test; `pytest tests/test_package.py` passed and all dependencies installed.
- [x] Step 02 — Configuration module (2026-11-18) — `pathsnap/config.py` with seeds, candidate/alignment/HMM defaults, UTM override, and `TRAIN_DATA_DIR`/`TEST_DATA_DIR`/`EVAL_DATA_DIR`/`ARTIFACT_DIR`/`RESULTS_DIR` env overrides; missing data dirs raise `FileNotFoundError`; `pytest tests/test_config.py` 5 passed.
