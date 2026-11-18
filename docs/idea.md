# PathSnap: Map Matching for Noisy GPS Trajectories

> Recover the road a vehicle actually drove on from noisy GPS points, and show it live on an interactive map.

**Format:** Hackathon (1-2 days) | **Stack:** Python | **Highlights:** HMM/Viterbi matcher, interactive map demo, learned candidate scorer

---

## 1. The Problem in One Paragraph

GPS points drift because of urban canyons, multipath reflections and cheap receivers. Plotted raw, a car appears to drive through buildings or on the wrong parallel street. Snapping each point to its nearest road fails because it ignores connectivity, one-way streets and path continuity. **PathSnap** takes a noisy trajectory `{p1, ..., pn}` and a road graph, and returns the most likely connected sequence of road segments the vehicle traveled.

## 2. Our Idea

Build a map matcher that is **correct, measurable and demo-able**:

1. A classic **HMM + Viterbi** matcher as the reliable baseline (Newson & Krumm, 2009).
2. A **learned candidate scorer** that replaces the hand-tuned emission model with a small ML model, then compare it against the baseline.
3. A **web map demo** where judges drag a noise slider and watch raw points, candidates and the recovered route update live.
4. A **synthetic ground-truth benchmark** so every claim ("we are X% accurate") is backed by numbers.

**Pitch line:** *"We don't just snap points to roads. We prove which route the car took, show how confident we are, and let you break it with noise to see how it holds up."*

---

## 3. How It Works

### 3.1 Pipeline

```
Road graph (OSMnx) ──┐
                     ├─► Candidate search ─► Scoring (emission + transition) ─► Viterbi ─► Route stitching ─► Map demo
GPS trajectory ──────┘      (R-tree)           (HMM or learned model)
```

### 3.2 Baseline: HMM + Viterbi

- **Hidden states:** candidate road segments (or projected points on them) for each GPS observation.
- **Candidate selection:** for each point, query a spatial index (R-tree via `shapely.STRtree` or `rtree`) for segments within a radius (e.g. 50-100 m), keep the top-k nearest.
- **Emission probability:** Gaussian on the distance between the GPS point and its projection on the segment, `P(p | s) ~ exp(-d^2 / 2*sigma^2)`.
- **Transition probability:** exponential on `| great_circle(p_t, p_t+1) - network_path_distance(s_t, s_t+1) |`. Paths that closely match the straight-line movement are favored.
- **Decoding:** Viterbi in log-space to avoid underflow; stitch the chosen segments with shortest paths to form a continuous route.
- **Road rules:** use a directed graph so one-way streets and turn constraints are respected automatically.

### 3.3 Learned Candidate Scorer (the ML twist)

Keep Viterbi for guaranteed connectivity, but let a model decide how plausible each candidate is.

- **Features per (point, candidate):** distance to segment, heading difference between the GPS bearing and the segment bearing, speed vs. road speed limit, road class, segment length, position along the segment.
- **Model:** start with gradient boosting (LightGBM/XGBoost) or logistic regression; it trains in seconds on synthetic data and is easy to explain. Stretch goal: a small GRU/LSTM over the point sequence.
- **Training labels:** synthetic trajectories with known true segments (see section 4). Label = candidate is the true segment.
- **Integration:** the model's predicted probability replaces the Gaussian emission term in Viterbi. Transitions stay graph-based, so output is always a valid route.
- **Why this is a good story:** a hybrid (ML scores, graph logic enforces physics) beats a pure neural net on interpretability and validity, and the comparison is honest and measurable.

### 3.4 Real-World Messiness Handled

| Issue | Handling |
|---|---|
| Large noise / outliers | Drop points with no candidate in radius; widen radius adaptively |
| GPS gaps (tunnels) | Bridge with shortest path between last and next matched segments |
| Stationary jitter (red lights) | Collapse clusters of near-zero-speed points |
| Parallel roads / overpasses | Heading and speed features in the emission term |
| Sparse sampling | Transition model leans on network distance; report accuracy vs. sampling interval |

---

## 4. Data Strategy (No Dataset Needed)

Real labeled trajectories are hard to get in 48 hours, so we **generate our own ground truth**:

1. Download a city road graph with **OSMnx** (a dense area gives a hard, interesting test).
2. Sample random origin-destination pairs and compute the shortest-path route; this is the **true route**.
3. Walk along the route at realistic speeds, emitting a point every `dt` seconds.
4. Add **Gaussian noise** (sigma = 5 to 50 m), random **outliers**, and **dropout gaps**.
5. Keep the true segment IDs as labels for both evaluation and ML training.

Optional: test on a few real GPX traces (e.g. recorded on a phone) as a qualitative sanity check.

---

## 5. Evaluation

Report these on held-out synthetic trajectories:

- **Segment accuracy:** fraction of points matched to the correct segment.
- **Route mismatch fraction (RMF):** length of wrongly added/missed route divided by true route length.
- **Precision / recall** on the set of segments in the recovered route.
- **Runtime:** ms per trajectory and per point.

**Experiments / plots:**

1. Accuracy vs. noise sigma (nearest-snap baseline vs. HMM vs. learned scorer).
2. Accuracy vs. sampling interval (1 s, 5 s, 15 s, 30 s).
3. Ablation: HMM with and without heading/speed features.

A simple **nearest-road snapping baseline** is included to show *why* the problem is hard and how much we improve.

---

## 6. Interactive Demo

A single-page web map (Folium for speed, or Flask/FastAPI + Leaflet for interactivity):

- **Layers:** raw GPS points (red), candidate segments (faint), matched route (green), true route (dashed, if synthetic).
- **Controls:** noise slider, sampling-interval slider, method toggle (nearest snap / HMM / learned), "new random trip" button.
- **Confidence view:** color the route by per-segment confidence; highlight ambiguous stretches where two routes were nearly tied.
- **Live metrics panel:** accuracy and RMF update as sliders move.
- **Wow moment:** crank the noise up and show nearest-snap falling apart while PathSnap keeps the route.

---

## 7. Tech Stack

| Purpose | Tools |
|---|---|
| Road network | `osmnx`, `networkx` |
| Geometry / spatial index | `shapely`, `geopandas`, `rtree` or `STRtree`, `pyproj` |
| Matching | custom Python (NumPy) Viterbi |
| ML | `scikit-learn`, `lightgbm` (optional `pytorch` for stretch GRU) |
| Demo | `folium` or `Flask` + Leaflet, `plotly` for metric charts |
| Benchmark reference (optional) | Valhalla / OSRM map-matching endpoints |

---

## 8. 48-Hour Plan

**Day 1: Core**
- [ ] Load a city graph with OSMnx; set up projection and spatial index.
- [ ] Synthetic trajectory generator with noise, outliers and gaps.
- [ ] Nearest-snap baseline and evaluation metrics.
- [ ] HMM + Viterbi matcher with route stitching.
- [ ] First accuracy-vs-noise plot.

**Day 2: Differentiators + Demo**
- [ ] Generate a training set; train the learned candidate scorer; plug into Viterbi.
- [ ] Run comparison experiments and ablation.
- [ ] Build the interactive map with sliders and method toggle.
- [ ] Confidence coloring and live metrics.
- [ ] Polish: README, 2-minute pitch, screenshots/GIF.

**Cut list if time runs short (in order):** stretch GRU model, confidence coloring, ablation study, outlier handling polish.

---

## 9. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Shortest-path calls make Viterbi slow | Limit candidates to top-k, cache distances, cap path search by a cutoff distance |
| Learned model overfits synthetic noise | Vary noise and sampling during training; test on unseen areas/trips; show honest comparison even if HMM wins in places |
| Broken connectivity in OSM data | Use the largest strongly connected component; handle unreachable transitions with a large penalty |
| Demo too slow live | Precompute a few trips; keep the live-noise demo on short trajectories |

---

## 10. Stretch Ideas

- **Online matching:** fixed-lag Viterbi for streaming GPS with small delay.
- **Applications on top:** per-segment travel-time estimation, detecting missing roads from consistent off-network traces, flagging unusual driving behavior.
- **Particle filter** alternative for tracking multiple route hypotheses.

---

## 11. Expected Deliverables

1. Python package/notebook with the matcher, generator and evaluator.
2. Results: comparison table and plots (accuracy vs. noise and sampling interval).
3. Interactive map demo.
4. Short README and a 2-minute pitch script.

## 12. References

- Newson, P. & Krumm, J. (2009). *Hidden Markov Map Matching Through Noise and Sparseness.* ACM SIGSPATIAL.
- Boeing, G. (2017). *OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks.*
- Valhalla / OSRM / GraphHopper documentation for map-matching baselines.
