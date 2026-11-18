# PathSnap — Product Requirements Document

> **Status:** MVP specification for a 7-hour hackathon build.
> **Source of truth for context:** `docs/problem/*.md` is authoritative for the hackathon; `docs/idea.md` is the conceptual pitch. Where they conflict (synthetic/OSMnx data vs. provided trips, Folium/Flask vs. FastAPI/Leaflet), the problem documents and the user's confirmed decisions win.
> **Last updated:** 2026-11-18

---

## 1. Product Vision

PathSnap reconstructs the road a vehicle actually travelled from noisy GPS observations by combining a learned candidate scorer with a road-network-aware HMM/Viterbi decoder, and presents the result live on an interactive map with measurable accuracy. The system is built for a two-laptop evaluation demo (API on Laptop 1, browser UI on Laptop 2) and doubles as a reproducible research harness for comparing map-matching methods.

---

## 2. Target Users and Personas

### Primary persona — Maya, ML student/researcher
- Runs EDA and experiments on the 80 training trips.
- Trains at least five model families, tunes the best one with Bayesian optimisation, and compares nearest-snap, classifier-only, and classifier+HMM outputs.
- Needs reproducible scripts, saved metrics/plots, and a clear comparison table for the technical report.
- Cares about: Macro F1, Top-1 accuracy, route continuity, log loss, inference time, candidate miss rate.

### Secondary persona — Hackathon evaluator/judge
- Opens the UI on Laptop 2, sees the ten evaluation trips, selects one, and expects a predicted route plus server-side metrics within a few seconds.
- Must not see or influence ground-truth labels.
- Cares about: the API working for all ten trips, honest server-side scoring, visible API logs, and a complete repo/report.

### Tertiary persona — General demo audience
- Watches the map and moves noise/sampling sliders.
- Needs immediately legible layers: raw GPS, candidates, predicted route, and live metrics.
- Cares about: visual clarity, responsiveness, and the "hubris of nearest-snap vs. PathSnap" story.

---

## 3. Business Objectives and Success Metrics

### Objectives
1. Deliver a correct, measurable, demo-able map-matching system for the PES PathSnap hackathon.
2. Satisfy every mandatory deliverable: GitHub repo, trained artifacts, API, UI, visualisations, and PDF report.
3. Produce defensible experimental evidence for model selection, Bayesian tuning, and HMM decoding.
4. Keep evaluation ground truth server-side and prevent leakage.

### Success metrics (MVP acceptance)
| Metric | Target |
| :--- | :--- |
| Models trained and compared | ≥ 5 distinct algorithm families |
| Observation-level Top-1 accuracy (student-test) | Reported honestly; ≥ nearest-snap baseline |
| Macro F1 (candidate classification) | Reported per model with trip-grouped CV |
| Route continuity (classifier + HMM) | ≥ 0.90 on representative trips |
| Candidate miss rate | Measured and reported on alignable observations |
| Alignment coverage | Reported per trip; unalignable observations and anomalous reference routes flagged |
| Evaluation demo | All ten reserved trips return a valid predicted route and metrics |
| Bayesian optimisation | Search space, trials, and tuned-vs-untuned comparison recorded |
| HMM comparison | Classifier-only vs classifier+HMM table produced |
| Report | All required sections, real figures, and Folium maps present |
| Leakage | No reference route reaches the UI/prediction response |
| Latency | Typical trip returns from `/predict` within the demo budget (~≤ 30 s) |

### Non-goals for success
- State-of-the-art accuracy on unseen cities.
- Production-grade uptime, scaling, or multi-tenant security.
- Real-time streaming inference.

---

## 4. Full Feature List (MVP Scope)

All items below are **MVP** unless explicitly marked otherwise.

### F1. Dataset ingestion and validation
- Load per-trip `.track`, `.nodes`, `.arcs`, and `.route` files.
- Validate row shapes, coordinate plausibility, timestamp ordering, and route arc-id bounds.
- Partition awareness: 80 training trips, 10 student-testing trips, evaluator trips mountable at runtime.

### F2. Exploratory data analysis
- Inventory (trip counts, points, route lengths, graph sizes).
- Missing/invalid observation checks, sampling-interval statistics, gap outliers.
- Road-network connectivity, isolated nodes, and disconnected-component summaries.
- Reference-route quality: node discontinuities, repeated arcs, route/track length ratio, and GPS-to-route coverage.
- Candidate-miss-rate baseline on alignable observations.
- Rendered summary tables, plots, and at least one Folium EDA map.

### F3. Candidate generation
- Per-trip metric projection (UTM).
- Spatial index over arc geometries (`shapely.STRtree`).
- Top-k nearest arcs within an adaptive radius (default 50 m, expand to 100 m).
- Projection point, perpendicular distance, along-arc fraction, and heading information per candidate.

### F4. Feature engineering and labels
- Features per `(observation, candidate)` pair: distance, log-distance, heading difference, speed consistency, candidate rank, along-arc fraction, arc length, temporal gap, and topology context.
- Reference labels by robust monotonic sequence alignment of observations onto the reference route (dynamic programming / constrained progress), not naive nearest projection.
- Handle repeated route arcs and self-overlapping routes by tracking route position, not arc ID alone; mark observations too far from the route as unalignable and report coverage.
- Deterministic, cached feature matrices for reproducible training.

### F5. Model training and evaluation (≥ 5 models)
- Logistic Regression, distance-weighted KNN, ExtraTrees/Random Forest, LightGBM/HistGradientBoosting, and a shallow MLP at minimum; RBF SVM optional if time permits.
- Trip-grouped cross-validation on the 80 training trips.
- Metrics: Top-1 accuracy, Macro F1, precision/recall, log loss, candidate miss rate, route continuity, training/inference time.

### F6. Bayesian hyperparameter optimisation
- Optuna search on the selected model with a justified search space.
- Group-aware CV by trip.
- Recorded trials, best parameters, computational cost, and tuned-vs-untuned comparison.

### F7. HMM and Viterbi route reconstruction
- Hidden states: candidate arcs per observation.
- Emission: calibrated model probability (or Gaussian distance for the baseline).
- Transition: exponential penalty on the difference between GPS displacement and directed network path distance.
- Viterbi in log-space, invalid-transition handling, missing-candidate fallback, GPS-gap splitting, and shortest-path stitching of skipped arcs.

### F8. Model and inference artifact export
- Saved model, feature specification, candidate/HMM configuration, and version metadata.
- Fresh-process load and predict verification.

### F9. FastAPI backend
- Endpoints for health, trip listing, observation retrieval, and prediction.
- Server-side reference scoring; no ground-truth geometry in responses.
- Configurable evaluation data directory and CORS for LAN access.
- Structured logs suitable for evaluator inspection.

### F10. Static Leaflet UI
- Plain HTML/CSS/JS served independently of the API.
- Trip list, map, layer controls, legend, and metrics panel.
- Configurable API base URL for two-laptop deployment.

### F11. Interactive noise and sampling controls
- Sliders for GPS noise and sampling interval, plus a method selector.
- Server-side deterministic perturbation using a seed.
- Live re-query and metric refresh without exposing ground truth.

### F12. Folium and static map exports
- EDA maps and result maps with GPS observations, road network, reference route (development only), and predicted route.
- Distinguishable colours, legend, layer controls, and appropriate zoom.

### F13. Method comparison and visualisations
- Nearest-snap vs classifier-only vs classifier+HMM comparison table.
- CV F1, Top-1, Macro F1, log-loss, confusion matrices, Bayesian trials, and HMM comparison plots.
- All figures generated from actual stored results.

### F14. Technical report assets and PDF
- EDA narrative, model comparison table, per-model Q&A, Bayesian optimisation, HMM comparison, Folium maps, and API/UI screenshots.
- Exportable to a single PDF.

### F15. Documentation and deployment
- README with installation, single-machine run, and two-laptop LAN deployment.
- Requirements pinning, artifact instructions, and reproducibility notes.

---

## 5. Explicit Out-of-Scope Items

- Real-time or streaming GPS matching (no online fixed-lag Viterbi).
- Downloading or rebuilding road networks from OSM at runtime (use provided per-trip networks only).
- Synthetic trajectory generation and OSMnx-based network construction described in `docs/idea.md`; the provided per-trip dataset is the sole data source.
- Multi-city or cross-trip global road-network persistence.
- Mobile applications; the UI is browser-based only.
- User accounts, roles, registration, or admin panels.
- Cloud hosting, containers-as-a-service, or production orchestration.
- Editing or manually correcting ground-truth routes.
- Hardcoding predictions or route IDs for any evaluation trip.
- Exposing evaluator ground truth to the UI or prediction pipeline.

---

## 6. User Journey — Primary Persona (Maya, ML student/researcher)

1. **Setup:** Clones the repo, installs requirements, and reads `README.md` and `memory-bank/` docs.
2. **Understand the data:** Runs the EDA script to inspect trip inventory, gaps, graph connectivity, reference-route quality/coverage, and candidate miss rate.
3. **Build features:** Generates candidates and features for the training trips, caches them, verifies monotonic label alignment, and reviews flagged anomalous trips.
4. **Train and compare:** Runs trip-grouped CV for at least five models, reviews the comparison CSV/plots, and selects the best family.
5. **Tune:** Runs Bayesian optimisation, compares tuned vs untuned performance, and exports the chosen artifact.
6. **Decode:** Runs nearest-snap, classifier-only, and classifier+HMM on the student-testing trips; reviews accuracy, continuity, and failure cases.
7. **Demo:** Starts the API on Laptop 1 and the UI on Laptop 2, selects a trip, inspects the route and metrics, then varies noise/sampling to show robustness.
8. **Report:** Generates Folium maps and figures, writes the report sections, and verifies every metric against the stored results.
9. **Submit:** Confirms no secrets or evaluator labels are committed and pushes the public repository.

### Secondary journey — Evaluator
1. Opens the UI on Laptop 2.
2. Sees ten reserved trip options.
3. Selects one; the UI sends the trip ID to the API on Laptop 1.
4. The API runs the full pipeline, scores against internal reference data, and returns the route plus aggregate metrics.
5. The evaluator inspects the map, metrics, and API logs; labels never reach the browser.

---

## 7. Assumptions and Constraints

- The provided dataset contains 80 training and 10 student-testing trips; the ten evaluator trips will be mounted separately.
- Each trip is self-contained: node IDs and arc IDs are row indices in that trip's files.
- Reference routes are chronological arc sequences used as labels, but they are **not guaranteed** to be node-connected or to cover the whole track. The data audit found 11/90 trips with 1–3 arc discontinuities, 10/90 with repeated arcs, and 2/80 training trips with severe GPS-to-route mismatch. The pipeline must tolerate, flag, and report these cases rather than failing or silently mislabelling.
- Naive nearest-segment projection is not reliably monotonic on self-overlapping routes (observed median ~98.5%, as low as 67%); label alignment must enforce route-order progress and expose alignment coverage.
- Time budget is approximately 7 hours; scope must be ruthlessly prioritised.
- No database or authentication is required for the local demo.
- All learned preprocessing and tuning use training/student-testing data only; evaluator data remains untouched.
