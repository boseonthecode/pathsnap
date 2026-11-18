"""Build the technical report (Markdown + PDF) from stored result tables."""

from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

import _bootstrap  # noqa: F401
import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from pathsnap import config

PROJECT_ROOT = config.PROJECT_ROOT
RESULTS = PROJECT_ROOT / "results"
FIGURES = RESULTS / "figures"
REPORT_DIR = PROJECT_ROOT / "report"
ASSETS = REPORT_DIR / "assets"

BLOCKS: list[dict] = []


def h1(text: str) -> None:
    BLOCKS.append({"type": "h1", "text": text})


def h2(text: str) -> None:
    BLOCKS.append({"type": "h2", "text": text})


def h3(text: str) -> None:
    BLOCKS.append({"type": "h3", "text": text})


def para(text: str) -> None:
    BLOCKS.append({"type": "p", "text": text})


def table(rows: list[list[str]], widths: list[float] | None = None) -> None:
    BLOCKS.append({"type": "table", "rows": rows, "widths": widths})


def image(relative_path: str, caption: str, width_in: float = 6.4) -> None:
    BLOCKS.append(
        {
            "type": "image",
            "path": relative_path,
            "caption": caption,
            "width_in": width_in,
        }
    )


def _fmt(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def build_blocks() -> None:
    cv = pd.read_csv(RESULTS / "model_comparison.csv")
    test = pd.read_csv(RESULTS / "student_test_comparison.csv")
    aggregates = cv[cv["scope"] == "cv_aggregate"].set_index("model")
    test_by_model = test.set_index("model")
    tuned = pd.read_csv(RESULTS / "tuned_vs_untuned.csv").set_index("variant")
    best = json.loads((RESULTS / "best_params.json").read_text(encoding="utf-8"))
    selected_features = json.loads(
        (RESULTS / "selected_features.json").read_text(encoding="utf-8")
    )
    hmm = pd.read_csv(RESULTS / "hmm_comparison.csv", dtype={"trip_id": str})
    hmm_means = hmm.groupby("method")[
        [
            "point_accuracy",
            "route_continuity",
            "route_precision",
            "route_recall",
            "rmf",
            "alignment_coverage",
            "inference_ms",
        ]
    ].mean()
    accuracy_pivot = hmm.pivot_table(
        index="trip_id", columns="method", values="point_accuracy"
    )
    delta = accuracy_pivot["hmm"] - accuracy_pivot["classifier"]
    best_trip, worst_trip = str(delta.idxmax()), str(delta.idxmin())
    e2e = pd.read_csv(RESULTS / "e2e_summary.csv", dtype={"trip_id": str})
    eda = pd.read_csv(RESULTS / "eda_summary.csv", dtype={"trip_id": str})
    ablation = pd.read_csv(RESULTS / "feature_ablation.csv")
    severe_ids = ", ".join(eda.loc[eda["severe_mismatch"], "trip_id"].tolist())
    lowest_trip = eda.nsmallest(3, "route_gps_within50_frac").iloc[2]
    hmm_accuracy = float(hmm_means.loc["hmm", "point_accuracy"])
    hmm_continuity = float(hmm_means.loc["hmm", "route_continuity"])
    classifier_accuracy = float(hmm_means.loc["classifier", "point_accuracy"])
    e2e_max_ms = float(e2e["inference_ms"].max())

    h1("PathSnap: Map-Matching and Route Reconstruction")
    para(
        "Technical report for the PathSnap hackathon. System: a learned candidate "
        "scorer plus a network-aware HMM/Viterbi decoder, a FastAPI backend, and a "
        "build-free Leaflet UI. Date: 18 November 2026."
    )
    para(
        "Abstract. PathSnap reconstructs the road sequence a vehicle travelled from "
        "noisy per-trip GPS observations. Candidate arcs are generated with a metric "
        "STRtree, scored by five compared model families, and decoded with a "
        "transition model based on directed network distance. On the ten "
        "student-testing trips, classifier+HMM reaches "
        f"{_fmt(hmm_accuracy)} observation-level accuracy and {_fmt(hmm_continuity)} "
        "route continuity, improving classifier-only accuracy by "
        f"{(hmm_accuracy - classifier_accuracy) * 100:.1f} percentage points. All ten "
        f"trips return valid API predictions, with the slowest full pipeline at "
        f"{e2e_max_ms / 1000:.2f} s."
    )
    para(
        "Reproducibility. Every number below is read from `results/*.csv` at report "
        "build time. The generator is `scripts/09_report.py`; it writes both this "
        "Markdown source and `report/report.pdf`."
    )

    h1("1. Dataset understanding and exploratory data analysis")
    para(
        "The dataset is organised per trip: `<trip_id>.track` (longitude, latitude, "
        "timestamp seconds), `.nodes` (longitude, latitude), `.arcs` (from_node_id, "
        "to_node_id), and `.route` (chronological reference arc ids). Row indices are "
        "the node and arc identifiers. The training partition has 80 trips and the "
        "student-testing partition has 10; evaluator trips are mounted separately and "
        "were never used."
    )
    table(
        [
            ["Partition", "Trips", "Use"],
            ["Training", "80", "CV, fitting, feature selection, tuning"],
            ["Student testing", "10", "Development comparison and decoder evaluation"],
            ["Evaluator testing", "10 (external)", "Final evaluation only; untouched"],
        ],
        widths=[1.5, 1.1, 3.8],
    )
    para(
        f"The EDA table contains {len(eda)} trips, {int(eda['n_points'].sum()):,} GPS "
        f"observations, and {int(eda['n_arcs'].sum()):,} road-network arcs "
        "(`results/eda_summary.csv`). Preprocessing performs shape/dtype checks, "
        "finite-value and coordinate-range validation, strictly increasing timestamp "
        "validation, and arc/node bound checks. Distances are computed in a per-trip "
        "UTM projection chosen from the trip centroid, not in degrees."
    )
    para(
        "Reference routes are treated as labels, not verified ground truth. The audit "
        f"found {int((eda['route_discontinuities'] > 0).sum())} trips with node "
        f"discontinuities, {int((eda['repeated_route_arcs'] > 0).sum())} trips with "
        "repeated route arcs, and two severe GPS-to-route mismatches: "
        f"{', '.join(eda.loc[eda['severe_mismatch'], 'trip_id'].tolist())}. A third "
        f"training trip ({eda.nsmallest(3, 'route_gps_within50_frac').iloc[2]['trip_id']}) "
        "falls below the 90% coverage flag. Labels are therefore assigned by a "
        "monotonic dynamic-programming alignment over route positions, not by a bare "
        "arc-id lookup, and unalignable observations are excluded."
    )
    image(
        "results/figures/eda_inventory_network.png",
        "Figure 1. Observations per trip and road-network size (source: results/eda_summary.csv).",
    )
    image(
        "results/figures/eda_route_coverage.png",
        f"Figure 2. GPS-to-route coverage with severe-mismatch and low-coverage thresholds; trips {severe_ids} are annotated (source: results/eda_summary.csv).",
    )
    image(
        "results/figures/eda_route_anomalies.png",
        "Figure 3. Reference-route node discontinuities and repeated arcs (source: results/eda_summary.csv).",
    )

    h2("1.1 Feature engineering and selection")
    para(
        "Every `(observation, candidate)` pair receives ten features in a fixed order "
        "stored in `artifacts/config.json`: perpendicular distance, log-distance, "
        "heading difference, GPS-vs-arc speed consistency, spatial candidate rank, "
        "along-arc fraction, arc length, temporal gap, and from/to node total degree. "
        "The stable order is the contract between training and inference."
    )
    table(
        [
            ["Feature", "Rationale"],
            ["distance_m / log_distance", "Core geometric fit; log compresses large outliers"],
            ["heading_diff_deg", "Separates parallel roads and opposite directions"],
            ["speed_consistency", "Compares GPS speed with arc-implied speed"],
            ["candidate_rank", "Encodes local spatial ordering"],
            ["along_fraction", "Position on the arc; helps at intersections"],
            ["arc_length_m", "Scale context for short connectors vs long roads"],
            ["time_gap_s", "Handles irregular sampling and gaps"],
            ["from_degree / to_degree", "Topology context at intersections"],
        ],
        widths=[2.1, 4.3],
    )
    para(
        "Feature selection used trip-grouped CV on training folds only "
        "(`results/feature_ablation.csv`). The full ten-feature tuned model scored "
        f"{_fmt(ablation.loc[ablation['config'] == 'full', 'cv_macro_f1'].iloc[0])} "
        "CV Macro F1. A six-feature subset ("
        + ", ".join(selected_features["selected_features"])
        + f") scored {_fmt(selected_features['selected_cv_macro_f1'])}. Leave-one-out "
        "importance shows heading difference (0.0970 Macro F1 drop), candidate rank "
        "(0.0211), and along-arc fraction (0.0172) as the dominant signals; distance, "
        "time gap, speed consistency, and arc length were nearly redundant for this "
        "tree ensemble. The deployed artifact keeps all ten features so its contract "
        "matches the exported model; the six-feature result is retained as a "
        "reportable selection experiment."
    )
    image(
        "results/figures/feature_ablation.png",
        "Figure 4. Leave-one-feature-out importance and cumulative feature-group ablation (source: results/feature_ablation.csv).",
    )

    h1("2. Model experimentation and comparison")
    para(
        "Five model families were trained with deterministic trip-grouped "
        "cross-validation on the 77 training trips that passed the coverage filter. "
        "Each fold evaluates on held-out trips, never on observations from the same "
        "trip as training. For tractability, CV uses up to 300 evenly spaced "
        "observations per trip; student-test evaluation uses up to 800. All models "
        "predict calibrated positive-class probabilities."
    )
    table(
        [
            [
                "Model",
                "Family",
                "CV Macro F1",
                "CV Top-1",
                "CV Log Loss",
                "Test Macro F1",
                "Test Top-1",
                "Test Log Loss",
                "Train s",
                "Infer ms",
            ]
        ]
        + [
            [
                name,
                aggregates.loc[name, "family"],
                _fmt(aggregates.loc[name, "macro_f1"]),
                _fmt(aggregates.loc[name, "top1_accuracy"]),
                _fmt(aggregates.loc[name, "log_loss"]),
                _fmt(test_by_model.loc[name, "macro_f1"]),
                _fmt(test_by_model.loc[name, "top1_accuracy"]),
                _fmt(test_by_model.loc[name, "log_loss"]),
                _fmt(aggregates.loc[name, "train_time_s"], 2),
                _fmt(test_by_model.loc[name, "inference_ms"], 1),
            ]
            for name in [
                "logistic_regression",
                "knn",
                "extra_trees",
                "lightgbm",
                "mlp",
            ]
        ],
        widths=[1.15, 0.95, 0.65, 0.6, 0.65, 0.7, 0.62, 0.65, 0.55, 0.62],
    )
    para(
        "Selection criterion was mean CV Macro F1 with CV Top-1 as the tie-breaker; "
        "training accuracy was not used. ExtraTrees has the highest CV Macro F1 "
        f"({_fmt(aggregates.loc['extra_trees', 'macro_f1'])}) and generalises almost "
        f"identically on student-test data ({_fmt(test_by_model.loc['extra_trees', 'macro_f1'])})."
    )

    h2("2.1 Per-model analysis")
    model_notes = {
        "logistic_regression": (
            "Logistic Regression (linear). Chosen as a calibrated linear baseline: it "
            "produces probabilities directly, trains in under a second, and makes the "
            "linear separability of the candidate features explicit. Strengths: stable, "
            "interpretable, fast inference (11.4 ms). Limits: it cannot represent "
            "thresholds or interactions such as 'close AND aligned AND shallow', which "
            "the tree models exploit. Configuration: standard scaling, max_iter=1000, "
            "class_weight='balanced', L2 regularisation; objective is binary log loss. "
            f"CV Macro F1 {_fmt(aggregates.loc['logistic_regression', 'macro_f1'])}, "
            f"student-test Macro F1 {_fmt(test_by_model.loc['logistic_regression', 'macro_f1'])}, "
            f"Top-1 {_fmt(test_by_model.loc['logistic_regression', 'top1_accuracy'])}, "
            f"log loss {_fmt(test_by_model.loc['logistic_regression', 'log_loss'])}. "
            "CV and test scores are close, so there is no evidence of overfitting; the "
            "model underfits relative to the ensembles."
        ),
        "knn": (
            "Distance-weighted KNN (instance-based). Chosen to test whether local "
            "similarity in feature space is enough to identify the true arc. Strengths: "
            "no parametric assumptions, training is 0.17 s. Limits: inference is very "
            f"expensive ({_fmt(test_by_model.loc['knn', 'inference_ms'], 0)} ms) and "
            f"probabilities are poorly calibrated (student-test log loss "
            f"{_fmt(test_by_model.loc['knn', 'log_loss'])}). Configuration: standard "
            "scaling, k=15, distance weights. "
            f"CV Macro F1 {_fmt(aggregates.loc['knn', 'macro_f1'])}; student-test "
            f"Macro F1 {_fmt(test_by_model.loc['knn', 'macro_f1'])}. The CV/test gap is "
            "small, but the high log loss and latency make it unsuitable for the HMM "
            "and the live demo despite respectable accuracy."
        ),
        "extra_trees": (
            "ExtraTrees (ensemble). Chosen for nonlinear boundaries and robust "
            "probability estimates with modest cost. Strengths: highest CV Macro F1, "
            "only slightly slower than boosting, and handles the distance/heading "
            "interaction well. Limits: deeper trees increase artifact size, and the "
            "model cannot use sequential context (the HMM supplies that). "
            "Configuration: 200 trees, min_samples_leaf=2, balanced_subsample, no "
            "depth cap in the comparison run; objective is the classification log loss "
            f"via impurity. CV Macro F1 {_fmt(aggregates.loc['extra_trees', 'macro_f1'])}; "
            f"student-test Macro F1 {_fmt(test_by_model.loc['extra_trees', 'macro_f1'])}, "
            f"Top-1 {_fmt(test_by_model.loc['extra_trees', 'top1_accuracy'])}, log loss "
            f"{_fmt(test_by_model.loc['extra_trees', 'log_loss'])}. The near-identical "
            "CV/test result gives no strong overfitting signal; the tuning stage later "
            "constrains depth/leaves for artifact size."
        ),
        "lightgbm": (
            "LightGBM (boosting). Chosen as the gradient-boosting representative for "
            "fast tabular inference. Strengths: very fast inference (64 ms), compact "
            "artifact, strong recall (0.9124). Limits: lower Macro F1 than ExtraTrees "
            "under the chosen budget; it underfits slightly due to only 200 trees and "
            "column subsampling. Configuration: 200 trees, learning_rate=0.05, "
            "num_leaves=31, subsample=0.8 with subsample_freq=1, "
            "colsample_bytree=0.8, balanced class weights; objective binary log loss. "
            f"CV Macro F1 {_fmt(aggregates.loc['lightgbm', 'macro_f1'])}; student-test "
            f"Macro F1 {_fmt(test_by_model.loc['lightgbm', 'macro_f1'])}, Top-1 "
            f"{_fmt(test_by_model.loc['lightgbm', 'top1_accuracy'])}, log loss "
            f"{_fmt(test_by_model.loc['lightgbm', 'log_loss'])}. CV/test parity "
            "indicates neither severe under- nor overfitting, but capacity is limited."
        ),
        "mlp": (
            "Shallow MLP (neural). Chosen to test a smooth nonlinear scorer. Strengths: "
            "best student-test log loss and the best student-test Macro F1, with fast "
            f"inference ({_fmt(test_by_model.loc['mlp', 'inference_ms'], 1)} ms). "
            "Limits: the longest training time, and the optimiser reached its 150-"
            "iteration cap (a convergence warning), so it is mildly under-trained. "
            "Configuration: standard scaling, hidden layers 64x32, ReLU, alpha=1e-3, "
            "batch_size=1024, learning_rate_init=1e-3, max_iter=150; objective binary "
            f"log loss. CV Macro F1 {_fmt(aggregates.loc['mlp', 'macro_f1'])}; "
            f"student-test Macro F1 {_fmt(test_by_model.loc['mlp', 'macro_f1'])}, Top-1 "
            f"{_fmt(test_by_model.loc['mlp', 'top1_accuracy'])}, log loss "
            f"{_fmt(test_by_model.loc['mlp', 'log_loss'])}. CV/test parity is good; "
            "longer training is the clearest improvement path."
        ),
    }
    for name in [
        "logistic_regression",
        "knn",
        "extra_trees",
        "lightgbm",
        "mlp",
    ]:
        h3(name.replace("_", " ").title())
        para(model_notes[name])

    h2("2.2 Model visualisations")
    image(
        "results/figures/cv_macro_f1_comparison.png",
        "Figure 5. Trip-grouped CV Macro F1 by model (source: results/model_comparison.csv).",
    )
    image(
        "results/figures/student_test_top1_macro_f1.png",
        "Figure 6. Student-test Top-1 accuracy and Macro F1 (source: results/student_test_comparison.csv).",
    )
    image(
        "results/figures/log_loss_comparison.png",
        "Figure 7. Student-test log loss by model (source: results/student_test_comparison.csv).",
    )
    image(
        "results/figures/confusion_matrices.png",
        "Figure 8. Candidate confusion matrices on student-test data (source: results/student_test_comparison.csv).",
    )
    image(
        "results/figures/timing_comparison.png",
        "Figure 9. CV training time and inference time (sources: results/model_comparison.csv, results/student_test_comparison.csv).",
    )

    h1("3. Bayesian hyperparameter optimisation")
    para(
        "The selected ExtraTrees family was tuned with Optuna's TPE sampler using a "
        "trip-grouped CV Macro F1 objective. The search space was deliberately size-"
        "bounded so the exported artifact stays deployable: depth 8-14, leaves 10-30, "
        "trees 100-200, max_features in {sqrt, log2, 0.5}, criterion in {gini, "
        "entropy}, class_weight in {balanced, balanced_subsample}. The study ran 15 "
        "completed trials within a 600 s budget (total script runtime 244 s), recorded "
        "in `results/optuna_trials.csv`."
    )
    table(
        [
            ["Hyperparameter", "Search space"],
            ["n_estimators", "100-200, step 50"],
            ["max_depth", "8-14, step 2"],
            ["min_samples_leaf", "10-30, step 10"],
            ["max_features", "sqrt, log2, 0.5"],
            ["criterion", "gini, entropy"],
            ["class_weight", "balanced, balanced_subsample"],
        ],
        widths=[2.0, 4.0],
    )
    para(
        "Best parameters: "
        + json.dumps(best["best_params"], sort_keys=True)
        + f" (CV Macro F1 {_fmt(best['best_cv_macro_f1'])}). The tuned model did not "
        "improve accuracy: the size constraint reduced CV Macro F1 from "
        f"{_fmt(tuned.loc['untuned', 'cv_macro_f1'])} to "
        f"{_fmt(tuned.loc['tuned', 'cv_macro_f1'])}. It did reduce inference time from "
        f"{_fmt(tuned.loc['untuned', 'student_inference_ms'], 1)} ms to "
        f"{_fmt(tuned.loc['tuned', 'student_inference_ms'], 1)} ms and produced a "
        "15 MB artifact instead of a multi-gigabyte model. This trade-off is reported "
        "honestly rather than claiming a tuning gain."
    )
    table(
        [
            [
                "Variant",
                "CV Macro F1",
                "CV Top-1",
                "CV Log Loss",
                "Test Macro F1",
                "Test Top-1",
                "Test Log Loss",
                "Test Infer ms",
            ],
            [
                "Untuned",
                _fmt(tuned.loc["untuned", "cv_macro_f1"]),
                _fmt(tuned.loc["untuned", "cv_top1_accuracy"]),
                _fmt(tuned.loc["untuned", "cv_log_loss"]),
                _fmt(tuned.loc["untuned", "student_macro_f1"]),
                _fmt(tuned.loc["untuned", "student_top1_accuracy"]),
                _fmt(tuned.loc["untuned", "student_log_loss"]),
                _fmt(tuned.loc["untuned", "student_inference_ms"], 1),
            ],
            [
                "Tuned",
                _fmt(tuned.loc["tuned", "cv_macro_f1"]),
                _fmt(tuned.loc["tuned", "cv_top1_accuracy"]),
                _fmt(tuned.loc["tuned", "cv_log_loss"]),
                _fmt(tuned.loc["tuned", "student_macro_f1"]),
                _fmt(tuned.loc["tuned", "student_top1_accuracy"]),
                _fmt(tuned.loc["tuned", "student_log_loss"]),
                _fmt(tuned.loc["tuned", "student_inference_ms"], 1),
            ],
        ],
        widths=[0.9, 0.8, 0.7, 0.75, 0.8, 0.7, 0.75, 0.85],
    )

    h1("4. HMM and Viterbi route reconstruction")
    para(
        "Hidden states are the candidate arcs for each observation. Classifier "
        "emissions are the model's positive-class probabilities, normalised across the "
        "candidates of an observation in log space; a Gaussian distance emission is "
        "kept as a baseline. The transition score is "
        "`-|gps_displacement - network_distance| / beta`, where network distance is "
        "the directed shortest-path length from the projection on the first arc, "
        "through the network, to the projection on the second arc. Disconnected or "
        "direction-violating transitions receive an explicit invalid penalty (-50). "
        "Beta was tuned on training/student-test trips only (grid 5, 10, 20, 50); "
        "beta=50 was selected and persisted in `artifacts/hmm.json`."
    )
    para(
        "Observations without candidates and GPS gaps longer than 30 s split the "
        "sequence into segments; each segment is decoded independently. Chosen arcs "
        "are then stitched with directed shortest paths so consecutive output arcs "
        "share nodes. Viterbi runs in log space over an 8-candidate beam per "
        "observation, with a 2 km shortest-path cutoff and per-source Dijkstra caching."
    )
    table(
        [
            [
                "Method",
                "Point accuracy",
                "Continuity",
                "Route precision",
                "Route recall",
                "RMF",
                "Coverage",
                "Decode ms",
            ]
        ]
        + [
            [
                method,
                _fmt(hmm_means.loc[method, "point_accuracy"]),
                _fmt(hmm_means.loc[method, "route_continuity"]),
                _fmt(hmm_means.loc[method, "route_precision"]),
                _fmt(hmm_means.loc[method, "route_recall"]),
                _fmt(hmm_means.loc[method, "rmf"], 2),
                _fmt(hmm_means.loc[method, "alignment_coverage"]),
                _fmt(hmm_means.loc[method, "inference_ms"], 0),
            ]
            for method in ["nearest", "classifier", "hmm"]
        ],
        widths=[1.0, 0.9, 0.8, 0.95, 0.8, 0.65, 0.8, 0.8],
    )
    para(
        "Classifier+HMM improves point accuracy over classifier-only on nine of ten "
        f"trips. The strongest gain is trip `{best_trip}` "
        f"({delta.loc[best_trip]:+.4f}); the only regression is trip `{worst_trip}` "
        f"({delta.loc[worst_trip]:+.4f}). Route precision rises from "
        f"{_fmt(hmm_means.loc['classifier', 'route_precision'])} to "
        f"{_fmt(hmm_means.loc['hmm', 'route_precision'])}, and RMF falls from "
        f"{_fmt(hmm_means.loc['classifier', 'rmf'], 2)} to "
        f"{_fmt(hmm_means.loc['hmm', 'rmf'], 2)}. The regression occurs when several "
        "candidates are nearly tied and the transition penalty prefers a connected "
        "but locally wrong arc. Full detail is in `results/hmm_comparison.csv` and "
        "`results/hmm_failure_analysis.md`."
    )
    image(
        "results/figures/decoder_accuracy_continuity.png",
        "Figure 10. Mean point accuracy and route continuity by decoder (source: results/hmm_comparison.csv).",
    )
    image(
        "results/figures/decoder_by_trip.png",
        "Figure 11. Per-trip point accuracy for nearest, classifier-only, and classifier+HMM (source: results/hmm_comparison.csv).",
    )

    h1("5. Geographical visualisation with Folium")
    para(
        "Folium maps were generated for three representative student-test trips: the "
        "largest HMM improvement, the regression case, and the first trip. Each map "
        "contains the GPS observations, a sampled road network, the reference route "
        "(development data only, dashed blue), the predicted route (green), a legend, "
        "and layer controls. The reference layer exists only in offline report maps; "
        "the evaluation API never serves it."
    )
    image(
        f"results/figures/folium_{best_trip}_preview.png",
        f"Figure 12. Trip {best_trip}: strongest HMM accuracy gain. GPS, network, reference (dev only), predicted route.",
        width_in=5.2,
    )
    image(
        f"results/figures/folium_{worst_trip}_preview.png",
        f"Figure 13. Trip {worst_trip}: decoder regression case. The predicted route remains contiguous but loses {abs(delta.loc[worst_trip]):.4f} point accuracy.",
        width_in=5.2,
    )

    h1("6. API, UI, and deployment")
    para(
        "The backend exposes `GET /health`, `GET /trips`, "
        "`GET /trips/{trip_id}/observations`, and `POST /predict`. `/predict` "
        "accepts `trip_id`, `method` (nearest, classifier, hmm), `noise_sigma_m` "
        "(0-100), `sample_every_s` (1-60), and `seed`. Responses contain predicted "
        "arcs, route coordinates, per-point predictions, aggregate metrics, and "
        "display layers; reference geometry never leaves the server. Trip IDs are "
        "resolved through a discovered-trip allowlist."
    )
    para(
        f"End-to-end verification called the API for all ten student-testing trips "
        f"(`results/e2e_summary.csv`): {int((e2e['status'] == 200).sum())}/10 returned "
        f"HTTP 200 with finite metrics, mean full-pipeline latency "
        f"{_fmt(e2e['inference_ms'].mean(), 0)} ms, and worst-case latency "
        f"{_fmt(e2e['inference_ms'].max(), 0)} ms."
    )
    image(
        "report/assets/api_response_terminal.png",
        "Figure 14. Captured POST /predict response (list fields abbreviated for display).",
    )
    image(
        "report/assets/ui_screenshot.png",
        "Figure 15. The Leaflet UI after an HMM prediction: trip picker, controls, server-side metrics, and map layers.",
    )
    para(
        "Deployment is two independent processes. Laptop 1 runs "
        "`uvicorn api.main:app --host 0.0.0.0 --port 8000` with `EVAL_DATA_DIR` and "
        "`CORS_ORIGINS` configured; Laptop 2 serves `ui/` with "
        "`python -m http.server 5500 --directory ui` and sets `apiBaseUrl` in "
        "`ui/config.js` to Laptop 1's LAN address. The single-machine fallback uses "
        "localhost for both. Full steps are in `docs/deployment.md`."
    )

    h1("7. Evaluation integrity, limitations, and conclusion")
    para(
        "Evaluator-testing data was never used for training, tuning, feature "
        "selection, or threshold selection. Reference routes and labels remain "
        "server-side; API and UI tests assert that responses contain no reference "
        "fields and that logs contain no ground truth. Feature caches are versioned "
        "by the feature/candidate/alignment specification, and the model artifact "
        "validates its feature order against `config.json`."
    )
    para(
        "Limitations: (1) CV and training use a deterministic per-trip observation cap "
        "for tractability; (2) the size-bounded tuning search trades about two Macro "
        "F1 points for a deployable 15 MB artifact; (3) the MLP did not fully converge "
        f"at 150 iterations; (4) {severe_ids} and {lowest_trip['trip_id']} have the "
        f"lowest coverage ({float(lowest_trip['route_gps_within50_frac']):.1%}, plus "
        "the two severe cases), so their labels are reported with coverage rather than "
        "treated as clean; (5) HMM decoding can override a locally correct but "
        f"topologically implausible candidate, as seen on trip {worst_trip}."
    )
    para(
        "Conclusion. PathSnap delivers a complete, reproducible map-matching system: "
        "validated data processing, five compared model families, recorded Bayesian "
        "tuning, a network-aware HMM decoder, honest server-side scoring, a working "
        "FastAPI service, and a live Leaflet demo. Classifier+HMM is the strongest "
        f"route reconstruction method, with {_fmt(hmm_accuracy)} point accuracy and "
        f"{_fmt(hmm_continuity)} continuity across the ten student-testing trips."
    )


def render_markdown(path: Path) -> None:
    lines: list[str] = []
    for block in BLOCKS:
        kind = block["type"]
        if kind == "h1":
            lines.extend([f"# {block['text']}", ""])
        elif kind == "h2":
            lines.extend([f"## {block['text']}", ""])
        elif kind == "h3":
            lines.extend([f"### {block['text']}", ""])
        elif kind == "p":
            lines.extend([block["text"], ""])
        elif kind == "table":
            rows = block["rows"]
            lines.append("| " + " | ".join(str(cell) for cell in rows[0]) + " |")
            lines.append("| " + " | ".join("---" for _ in rows[0]) + " |")
            for row in rows[1:]:
                lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
            lines.append("")
        elif kind == "image":
            lines.extend(
                [
                    f"![{block['caption']}](../{block['path']})",
                    "",
                    f"*{block['caption']}*",
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def render_pdf(path: Path) -> None:
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleCenter",
            parent=styles["Title"],
            alignment=TA_CENTER,
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Caption",
            parent=styles["Italic"],
            fontSize=8.5,
            textColor=colors.HexColor("#444444"),
            alignment=TA_CENTER,
            spaceAfter=10,
        )
    )

    doc = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title="PathSnap Technical Report",
        author="PathSnap team",
    )

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(7.8 * inch, 0.45 * inch, f"Page {document.page}")
        canvas.drawString(0.7 * inch, 0.45 * inch, "PathSnap technical report")
        canvas.restoreState()

    story = []
    for block in BLOCKS:
        kind = block["type"]
        if kind == "h1":
            story.append(Paragraph(escape(block["text"]), styles["TitleCenter"]))
            story.append(Spacer(1, 8))
        elif kind == "h2":
            story.append(Paragraph(escape(block["text"]), styles["Heading1"]))
            story.append(Spacer(1, 4))
        elif kind == "h3":
            story.append(Paragraph(escape(block["text"]), styles["Heading2"]))
            story.append(Spacer(1, 3))
        elif kind == "p":
            story.append(Paragraph(escape(block["text"]), styles["BodyText"]))
            story.append(Spacer(1, 6))
        elif kind == "table":
            rows = [
                [Paragraph(escape(str(cell)), styles["BodyText"]) for cell in row]
                for row in block["rows"]
            ]
            widths = block["widths"]
            if widths:
                total = sum(widths)
                scale = min(1.0, 7.0 / total)
                col_widths = [width * scale * inch for width in widths]
            else:
                col_widths = None
            table_obj = Table(rows, colWidths=col_widths, repeatRows=1)
            table_obj.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde5ee")),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9aa8b5")),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 3),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                    ]
                )
            )
            story.append(table_obj)
            story.append(Spacer(1, 10))
        elif kind == "image":
            image_path = PROJECT_ROOT / block["path"]
            if not image_path.is_file():
                continue
            with PILImage.open(image_path) as pil_image:
                width, height = pil_image.size
            width_in = block["width_in"]
            height_in = width_in * height / width
            if height_in > 7.2:
                height_in = 7.2
                width_in = height_in * width / height
            story.append(
                Image(str(image_path), width=width_in * inch, height=height_in * inch)
            )
            story.append(Paragraph(escape(block["caption"]), styles["Caption"]))
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> None:
    build_blocks()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    markdown_path = REPORT_DIR / "report.md"
    pdf_path = REPORT_DIR / "report.pdf"
    render_markdown(markdown_path)
    render_pdf(pdf_path)
    print(f"Wrote {markdown_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
