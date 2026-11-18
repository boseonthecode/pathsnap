# PathSnap — Implementation Plan

> Sequential build plan for a ~7-hour hackathon. Each step is sized for one AI session and includes a concrete verification test.
> No code appears in this document; it is intentionally implementation-agnostic.
> **Last updated:** 2026-11-18
> **Auth note:** Authentication is explicitly out of scope for the local demo, so there is no auth phase. Phase E covers API validation, CORS, logging, and ground-truth isolation instead.

---

## Guiding Rules

1. Complete steps in order; each step assumes the previous ones are done and verified.
2. Never use evaluator data for training, tuning, feature selection, or threshold selection.
3. Keep the API and UI runnable at all times after Step 18.
4. Commit only after a step's verification test passes.
5. Update `memory-bank/progress.md` and `memory-bank/architecture.md` after every completed step.

---

## Phase A — Scaffolding and Data Foundations

### Step 01 — Repository scaffolding
Create the package and folder skeleton: `pathsnap/`, `scripts/`, `api/`, `ui/`, `artifacts/`, `results/` (with subfolders), `tests/`, `report/`, and `memory-bank/`. Add `requirements.txt`, `.gitignore`, and a placeholder `README.md`.
**Test:** The folder tree matches the architecture document; `pip install -r requirements.txt` succeeds; `python -c "import pathsnap"` succeeds.

### Step 02 — Configuration module
Add a central configuration module for dataset paths, artifact paths, candidate radius/top-k defaults, UTM selection, random seeds, and HMM defaults. Support environment overrides for `EVAL_DATA_DIR`, `ARTIFACT_DIR`, and `RESULTS_DIR`.
**Test:** A unit test confirms defaults load, environment overrides take effect, and missing directories raise a clear error.

### Step 03 — Trip loader and validator
Implement loading for `.track`, `.nodes`, `.arcs`, and `.route`, with validation of column counts, numeric dtypes, finite values, monotonic timestamps, arc-id bounds, and node-id bounds. Report route discontinuities and repeated arcs as diagnostics rather than fatal errors. Support discovering all trips in a partition directory.
**Test:** Run the loader over all 90 provided trips; every trip loads with expected file shapes and no validation failures; the 11 trips with route discontinuities and 10 with repeated arcs are reported as diagnostics. Deliberately corrupt a copy (arc id out of range) and confirm validation rejects it.

### Step 04 — Dataset inventory / EDA script
Create `scripts/01_eda.py` that reports trip counts per partition, observation counts, file sizes and availability, dtypes/column names, route lengths, graph sizes, sampling-interval statistics, gap outliers, coordinate bounds, road-network connectivity/isolated nodes/disconnected components, route node discontinuities, repeated route arcs, route/track length ratio, and GPS-to-route coverage. Write `results/eda_summary.csv`, a short data-quality markdown report, and at least one Folium EDA map (GPS + network + reference route).
**Test:** The CSV has one row per trip (90 rows); required columns are non-null; the report flags the known anomalies (`00000033` and `00000054` severe GPS–route mismatch, 11 trips with route discontinuities, 10 trips with repeated arcs); the Folium HTML exists and contains the GPS, network, and reference layers.

### Step 05 — Geometry and projection utilities
Implement per-trip projection from WGS84 to the appropriate UTM zone, inverse projection, bearing calculation, point-to-segment projection, and route-polyline construction from ordered arcs. When consecutive reference arcs are not node-connected, bridge them with a directed shortest path or insert an explicit break; never silently concatenate a jump.
**Test:** Unit tests verify a known bearing (~90° east), a known short distance within tolerance, projection round-trip accuracy under 1 cm, and that route-polyline vertex order matches arc order; a fixture with a route discontinuity produces the documented bridge/break, and a repeated-arc route preserves both occurrences.

## Phase B — Candidates, Labels, Features, Metrics

### Step 06 — Reference label alignment
Implement robust monotonic sequence alignment of each GPS observation to the reference route: use a dynamic-programming / constrained-progress alignment over route positions rather than naive nearest-segment projection. Preserve route order, support repeated arcs and self-overlapping/out-and-back routes, and mark observations whose best route distance exceeds a configurable threshold as unalignable. Expose diagnostics for backward steps, alignment coverage, and anomalous trips.
**Test:** On the ten student-test trips, alignment coverage (observations within 50 m of the reference route) is ≥ 99% and route positions are non-decreasing by construction; every emitted label is a valid arc ID; a synthetic repeated-arc/self-overlap fixture produces the expected positions; training trips `00000033` and `00000054` are flagged as low-coverage rather than silently mislabelled.

### Step 07 — Candidate generation with spatial index
Build an `STRtree` over projected arc geometries per trip. For each observation, return top-k candidates within an adaptive radius, including distance, projection point, along-arc fraction, heading, and arc metadata. Handle empty candidate sets explicitly.
**Test:** On the ten student-test trips, at least 99% of alignable observations have ≥ 1 candidate within the adaptive radius (base 50 m, fallback up to 100–200 m); candidate miss rate is computed only on alignable observations and reported alongside alignment coverage; a known point returns its true arc among the top-k; anomalous training trips are reported separately rather than failing the test.

### Step 08 — Feature extraction
Implement per-`(observation, candidate)` features: perpendicular distance, log-distance, heading difference, GPS speed vs. arc-implied speed, candidate rank, along-arc fraction, arc length, temporal gap, and topology/degree context. Define a stable, documented feature order.
**Test:** The feature matrix has the documented columns, no NaN/infinite values, a row count equal to total candidates, and identical output across two runs with a fixed seed.

### Step 09 — Feature cache and versioning
Add caching of candidates/features per trip to disk with a version key derived from configuration and feature specification. Implement load-or-build behavior and reject stale caches.
**Test:** Building twice reuses the cache; changing the feature list invalidates it; cached and freshly built matrices are numerically identical.

### Step 10 — Metrics module
Implement observation-level Top-1 accuracy, Macro F1, precision/recall, log loss, candidate miss rate, route continuity, route precision/recall, route mismatch fraction, alignment coverage, and timing helpers. Define exact reference-vs-prediction alignment semantics for repeated arcs, non-contiguous references, and unalignable observations; report accuracy on alignable points with coverage alongside it.
**Test:** Perfect predictions yield accuracy/F1/continuity of 1.0 and zero log loss; a hand-constructed wrong-route case yields the expected lower scores; a repeated-arc fixture scores by sequence position, not first occurrence; log loss matches scikit-learn on a small fixture; coverage is zero-safe.

## Phase C — Models and Tuning

### Step 11 — Training dataset assembly
Assemble train/test datasets from cached features, with binary candidate labels, route positions, and trip identifiers. Exclude unalignable observations from training labels, report excluded counts, and exclude or downweight training trips whose alignment coverage is below a documented threshold. Implement trip-grouped train/validation folds while keeping student-test trips strictly separate. Fit any learned scalers/encoders on training folds only.
**Test:** Fold membership is disjoint by trip; no trip appears in both train and validation; student-test features are never included in training or scaler fitting; excluded-observation and low-coverage-trip counts are reported; a label from a repeated-arc trip maps to the correct route occurrence.

### Step 12 — First three model families
Add Logistic Regression, distance-weighted KNN, and ExtraTrees/Random Forest with sensible initial configurations. Support probability outputs and save per-model configurations.
**Test:** Each model trains on a small subset, produces probabilities in `[0, 1]`, and writes a metrics row. Training and inference times are recorded.

### Step 13 — Remaining model families (≥ 5 total)
Add LightGBM/HistGradientBoosting and a shallow MLP. Optionally add an RBF SVM only if runtime permits. Enforce a minimum of five distinct families.
**Test:** The comparison CSV contains at least five rows with distinct family names; every model reloads from its saved configuration and predicts on a held-out sample.

### Step 14 — Cross-validation and model comparison
Create `scripts/03_train.py` to run trip-grouped CV for all models, record fold-level and aggregate metrics, and produce `results/model_comparison.csv`. Select the final model using Macro F1 and route-relevant metrics, not training accuracy.
**Test:** The CSV has fold-level and aggregate rows for all models; rerunning with the same seed reproduces identical metrics; a written selection note justifies the chosen model.

### Step 15 — Model comparison plots
Generate CV Macro F1 comparison, student-test Top-1/Macro F1 comparison, log-loss comparison, confusion matrices, timing comparison, and any model-specific diagnostic plots.
**Test:** All required PNG files exist and are non-empty; plot titles/axes are labelled; figures derive from the stored results CSV rather than hardcoded numbers.

### Step 16 — Bayesian hyperparameter optimisation
Implement `scripts/04_tune.py` with Optuna, a justified search space, trip-grouped CV, a recorded objective, a trial/time budget, and persistence of trials and best parameters.
**Test:** At least ten completed trials are recorded unless the time budget expires; `results/optuna_trials.csv` contains parameters and objective values; the best configuration can be refit successfully.

### Step 17 — Tuned vs. untuned comparison and export
Refit the best configuration, evaluate against the untuned model on identical folds and student-test data, and export the final model artifact plus `config.json` (feature order, model params, thresholds, versions).
**Test:** The comparison CSV reports CV Macro F1, Top-1, Macro F1, log loss, and inference time for both models; a fresh Python process loads the artifact and produces predictions identical to the training process.

## Phase D — HMM / Viterbi

### Step 18 — HMM emission and transition models
Implement classifier-based emission probabilities, a Gaussian-distance baseline emission, and transition scores based on the gap between GPS displacement and directed shortest-path network distance. Represent invalid transitions explicitly.
**Test:** Unit tests on a toy three-arc graph confirm valid connected transitions score higher than disconnected ones; emission probabilities are normalised; one-way direction violations receive the configured penalty.

### Step 19 — Viterbi decoding and route stitching
Implement log-space Viterbi over candidates, handling missing candidates, invalid transitions, GPS gaps, and disconnected segments. Stitch chosen candidates with shortest paths to produce a continuous arc sequence.
**Test:** On a toy graph the decoded path matches the hand-computed optimum; on a real trip the output arc sequence is contiguous (consecutive arcs share nodes) and covers the reference route with high continuity.

### Step 20 — Decoder comparison
Create `scripts/05_hmm.py` to compare nearest-snap, classifier-only, and classifier+HMM across the student-testing trips. Record pointwise accuracy, route continuity, route precision/recall, alignment coverage, and inference time in `results/hmm_comparison.csv`. Tune decoder parameters (β, penalties, gap thresholds) using training and student-testing data only; never evaluator data.
**Test:** The comparison CSV covers all ten student-test trips and all three methods; each method returns a prediction per observation or a documented skip; alignment coverage is reported per trip (all test trips are ≥ 99% alignable); HMM continuity is reported alongside any regression cases; decoder tuning records exclude evaluator trips.

### Step 21 — Decoder visualisation and failure analysis
Plot accuracy/continuity by method and noise/gap conditions; produce at least one case study where HMM improves the route and one where it does not.
**Test:** Figures exist and reference the stored comparison data; the failure analysis names specific trips and explains the observed behaviour.

## Phase E — API and UI

### Step 22 — Artifact loading service
Build an API service layer that loads the model, config, HMM settings, trip metadata, and reference scorer once at startup. Expose a prediction pipeline function usable by both tests and endpoints.
**Test:** Service startup succeeds with exported artifacts; a direct service call returns a route and metrics for a sample trip; missing artifacts produce a clear startup error.

### Step 23 — FastAPI core endpoints
Implement `/health`, `/trips`, `/trips/{trip_id}/observations`, and `/predict`. `/trips` returns identifiers and metadata only; `/observations` returns raw GPS without labels; `/predict` returns predicted arcs, route geometry, confidence, and server-side aggregate metrics.
**Test:** FastAPI `TestClient` tests confirm success responses, correct schemas, 404 for unknown trips, and absence of any reference-route/ground-truth field in every response.

### Step 24 — API safeguards, configuration, and logging
Add strict request validation, configurable evaluation directory, CORS handling, deterministic seeds, structured request logs, and explicit ground-truth isolation. Document the CORS/env configuration for LAN use.
**Test:** Invalid method/parameter values return 422; `/predict` response JSON contains no route labels; CORS preflight from the UI origin succeeds; logs show request ID, trip ID, method, and latency but never ground-truth data.

### Step 25 — Static Leaflet UI shell
Build `ui/index.html`, `ui/style.css`, `ui/app.js`, and `ui/config.js`. Implement the trip list, API health indicator, map, layer controls, legend, and metrics panel using plain JavaScript and Leaflet.
**Test:** Serving `ui/` over HTTP loads the page; the trip list populates from the API; selecting a trip draws GPS points and a predicted route; browser console has no errors.

### Step 26 — Interactive noise and sampling controls
Add sliders for noise sigma and sampling interval, a method selector, and a seed/new-trip action. Re-query `/predict` on change and refresh layers and metrics without leaking labels.
**Test:** Moving each control triggers a new request with the expected parameters; returned route and metrics update; rapid changes do not crash or show stale results; no reference geometry appears in the UI network traffic.

### Step 27 — Two-laptop deployment and CORS verification
Document and verify single-machine and two-laptop setups: API bound to all interfaces, UI base URL configurable, CORS origin correct, and evaluation directory mountable.
**Test:** On a simulated second origin, the UI successfully calls the API; changing `apiBaseUrl` redirects requests; the API starts with a custom `EVAL_DATA_DIR` and lists those trips.

## Phase F — Polish, Maps, Report, QA

### Step 28 — Folium map generation
Create `scripts/07_folium.py` to generate maps with GPS observations, road-network geometry, reference route (development data only), and predicted route, using distinct colours, markers, legends, and layer controls.
**Test:** At least one HTML map is generated and non-empty, contains the expected layer names, and opens without errors; a report screenshot is produced.

### Step 29 — End-to-end verification across all ten test trips
Create `scripts/06_e2e.py` to call the API for every student-test trip under default settings, validating response schemas, route continuity, and metric presence. Record a summary table.
**Test:** All ten trips return HTTP 200 with valid routes and finite metrics; the summary table shows no missing or null values.

### Step 30 — README and deployment instructions
Write installation, data layout, training, tuning, HMM evaluation, API, UI, and two-laptop deployment instructions. Include artifact generation commands and a troubleshooting section.
**Test:** A fresh shell following only the README can install dependencies, load artifacts, start the API, serve the UI, and complete one prediction.

### Step 31 — Technical report
Assemble the PDF: EDA and preprocessing (including reference-route anomalies and coverage), model comparison table, per-model Q&A, model visualisations, Bayesian optimisation, HMM/Viterbi, Folium maps, API/UI screenshots, and a CMD screenshot of a successful API response on the receiving laptop. Every number must cite a generated CSV/figure.
**Test:** The PDF exists, contains all required sections, at least one Folium map screenshot, the API/UI/CMD screenshots, and every metric matches the stored results files; the reference-quality anomalies and alignment coverage are documented.

### Step 32 — Leakage, security, and repository QA
Audit for hardcoded predictions, evaluator data usage, exposed secrets, missing `.gitignore` entries, large-file issues, and untracked artifacts. Confirm the repository structure, document the known reference-route anomalies, and verify public-readiness.
**Test:** A repository search finds no hardcoded route IDs or secrets; evaluator data is untouched; the data-audit anomalies are documented; `git status`/artifact checks are clean; the final test suite passes.

---

## Scope Cut Order (if time runs short)

1. Drop the optional RBF SVM.
2. Reduce report prose depth while keeping required sections.
3. Skip decoder β tuning (use justified defaults).
4. Reduce UI polish (keep trip list, map, metrics, sliders).
5. Reduce the number of Folium variants (keep at least one).

**Never cut:** five models, classifier-vs-HMM comparison, working API/UI with server-side scoring, evaluation integrity, and the required report sections.
