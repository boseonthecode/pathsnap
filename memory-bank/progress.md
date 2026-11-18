# PathSnap — Progress Tracker

> Update this file after every completed implementation step.
> Move steps from **To Do** to **In Progress** to **Completed**; include the completion date and a one-line verification note.
> **Last updated:** 2026-11-18
> **Data audit:** 2026-11-18 — verified dataset against plan; findings and doc corrections recorded in `architecture.md` under `## Updates`.

---

## ⏭️ To Do

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
- [x] Step 03 — Trip loader and validator (2026-11-18) — `pathsnap/io.py` loads the four raw files with shape/dtype/finite/monotonic/bounds validation; route discontinuities and repeated arcs are diagnostics not fatal; all 80 train + 10 test trips load, corrupt fixtures are rejected, and the audit counts (11 discontinuities, 10 repeated) reproduce; `pytest tests/test_io.py` 10 passed.
- [x] Step 04 — Dataset inventory / EDA script (2026-11-18) — `scripts/01_eda.py` + `pathsnap/eda.py` write `results/eda_summary.csv` (90 rows, 25 columns, no nulls), `results/eda_report.md`, and `results/folium/eda_00000001.html` with GPS/network/reference layers. Report flags 11 discontinued routes, 10 repeated-arc routes, and severe mismatch trips `00000033`/`00000054`; `pytest tests/test_eda.py` 5 passed.
- [x] Step 05 — Geometry and projection utilities (2026-11-18) — `pathsnap/geometry.py` adds per-trip UTM `Projection` (round-trip <1 cm), bearings, geodesic and projected distances, point-to-segment projection, arc lengths, directed graph, and `build_route_polyline` with explicit breaks or directed shortest-path bridging. EDA now reuses these utilities. `pytest tests/test_geometry.py` 13 passed.
- [x] Step 06 — Reference label alignment (2026-11-18) — `pathsnap/labels.py` implements monotonic DP alignment over route positions (`_monotonic_align`), emits labels only for observations within 50 m, returns coverage/backward-step/low-coverage diagnostics, and preserves repeated arcs by route position. All ten student-test trips align at ≥99% and `00000033`/`00000054` are flagged low-coverage; `pytest tests/test_labels.py` 5 passed.
- [x] Step 07 — Candidate generation with spatial index (2026-11-18) — `pathsnap/candidates.py` builds an STRtree over projected arcs, returns top-k candidates with distance/projection/along-fraction/heading metadata, falls back 50→100→200 m, and reports miss rate over alignable observations only. Student-test miss rate ≤1% with every alignable observation covered; `pytest tests/test_candidates.py` 7 passed.
- [x] Step 08 — Feature extraction (2026-11-18) — `pathsnap/features.py` defines the stable 10-feature order and extracts distance, log-distance, heading difference, speed consistency, rank, along-fraction, arc length, time gap, and node-degree context; empty candidate sets yield a typed empty matrix. Output is finite and bit-identical across runs; `pytest tests/test_features.py` 5 passed.
