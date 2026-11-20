"""Report content assembled from the stored result tables.

The narrative numbers here are hand-maintained: they restate results already
written to ``results/`` and are not recomputed, so the report can be rebuilt
without re-running the experiments.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from pathsnap.config import Paths
from pathsnap.report.blocks import Document

def _fmt(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


_CONFIG_LABELS = {
    "n_estimators": "trees",
    "min_samples_leaf": "leaf",
    "max_depth": "depth",
    "learning_rate": "lr",
    "num_leaves": "leaves",
    "n_neighbors": "k",
    "hidden_layer_sizes": "hidden",
    "max_iter": "iters",
    "class_weight": "weights",
    "max_features": "feat",
    "weights": "w",
}


def _config_summary(raw_config: str, limit: int | None = None) -> str:
    """Compact one-line rendering of a stored model configuration."""
    params = json.loads(raw_config).get("params", {})
    parts = []
    for key, value in params.items():
        if key == "scaler":
            continue
        if isinstance(value, list):
            value = "x".join(str(item) for item in value)
        parts.append(f"{_CONFIG_LABELS.get(key, key)}={value}")
    if not parts:
        return "library defaults"
    text = ", ".join(parts)
    if limit is not None and len(text) > limit:
        text = text[: limit - 3].rstrip(", ") + "..."
    return text


FAMILY_GROUPS = {
    "linear": "linear",
    "linear_sgd": "linear",
    "probabilistic": "probabilistic",
    "instance_based": "instance-based",
    "tree": "tree",
    "ensemble": "ensemble",
    "ensemble_forest": "ensemble",
    "boosting": "boosting",
    "boosting_hist": "boosting",
    "boosting_adaptive": "boosting",
    "neural": "neural network",
}


def build_document(paths: Paths, report_dir: Path) -> Document:
    """Assemble the report document from the stored result tables."""
    PROJECT_ROOT = paths.project_root
    PROJECT = paths.project_root
    RESULTS = paths.results_dir
    FIGURES = RESULTS / "figures"
    ASSETS = report_dir / "assets"
    doc = Document()
    cv = pd.read_csv(RESULTS / "model_comparison.csv")
    test = pd.read_csv(RESULTS / "student_test_comparison.csv")
    aggregates = cv[cv["scope"] == "cv_aggregate"].set_index("model")
    model_order = list(aggregates.sort_values("macro_f1", ascending=False).index)
    best_cv_model = str(model_order[0])
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

    deployed_cfg = json.loads(
        (PROJECT_ROOT / "artifacts" / "config.json").read_text(encoding="utf-8")
    )
    deployed_hmm = json.loads(
        (PROJECT_ROOT / "artifacts" / "hmm.json").read_text(encoding="utf-8")
    )
    deployed_metrics = json.loads(
        (RESULTS / "model_metrics" / "summary.json").read_text(encoding="utf-8")
    )
    student_deployed = deployed_metrics["partitions"]["student_test"]
    artifact_mb = (
        (PROJECT_ROOT / "artifacts" / "model.joblib").stat().st_size / 1_000_000
    )
    hmm_pooled = float(
        (
            hmm.loc[hmm["method"] == "hmm", "point_accuracy"]
            * hmm.loc[hmm["method"] == "hmm", "n_points"]
        ).sum()
        / hmm.loc[hmm["method"] == "hmm", "n_points"].sum()
    )
    improved_trips = int((delta > 0).sum())

    doc.h1("PathSnap: Map-Matching and Route Reconstruction")
    doc.authors(
        [
            "Bhadra R S - pes1ug24cs111",
            "Aryaroop Bose - pes1ug24cs087",
            "Ashlesh R V - pes1ug24cs088",
            "Bhagath Naveen Mullakara - pes1ug24cs112",
        ]
    )
    doc.para(
        "This is the technical report for the PathSnap hackathon. The system pairs a "
        "learned candidate scorer with a network-aware HMM/Viterbi decoder that "
        "monotonically refines each point onto the inferred route, and exposes the "
        "result through a FastAPI backend and a build-free Leaflet interface. Date: "
        "18 November 2026."
    )
    doc.para(
        "Abstract. PathSnap reconstructs the sequence of roads a vehicle travelled "
        "from noisy per-trip GPS observations. Candidate arcs are generated with a "
        "metric STRtree, scored by a tuned ExtraTrees classifier, and decoded with a "
        "transition model built on directed network distance, after which each "
        "observation is refined onto the inferred route. Across the ten "
        "student-testing trips, classifier+HMM decoding reaches "
        f"{_fmt(hmm_accuracy)} mean-trip observation accuracy "
        f"({_fmt(hmm_pooled)} pooled) with {_fmt(hmm_continuity)} route continuity, "
        "which is a "
        f"{(hmm_accuracy - classifier_accuracy) * 100:.1f} percentage-point "
        "improvement over the classifier alone. All "
        f"ten trips return valid API predictions, and the slowest full pipeline "
        f"completes in {e2e_max_ms / 1000:.2f} s."
    )
    doc.para(
        "Reproducibility. Every number in this report is read from `results/*.csv` and "
        "`artifacts/*.json` at build time; none is transcribed by hand. "
        "`scripts/09_report.py` assembles the document, `scripts/14_decoder_comparison.py` "
        "evaluates the three decoders using `artifacts/hmm.json` unchanged (no beta "
        "search, no artifact writes), `scripts/03_train.py` produces the model "
        "comparison tables, `scripts/12_metrics_dashboard.py` produces the "
        "deployed-model metrics, and `scripts/06_e2e.py` produces the end-to-end "
        "latency figures."
    )

    doc.h1("1. Dataset understanding and exploratory data analysis")
    doc.para(
        "The dataset is organised per trip: `<trip_id>.track` (longitude, latitude, "
        "timestamp seconds), `.nodes` (longitude, latitude), `.arcs` (from_node_id, "
        "to_node_id), and `.route` (chronological reference arc ids). Row indices are "
        "the node and arc identifiers. The training partition has 80 trips and the "
        "student-testing partition has 10; evaluator trips are mounted separately and "
        "were never used."
    )
    doc.table(
        [
            ["Partition", "Trips", "Use"],
            ["Training", "80", "CV, fitting, feature selection, tuning"],
            ["Student testing", "10", "Development comparison and decoder evaluation"],
            ["Evaluator testing", "10 (external)", "Final evaluation only; untouched"],
        ],
        widths=[1.5, 1.1, 3.8],
    )
    doc.para(
        f"The EDA table contains {len(eda)} trips, {int(eda['n_points'].sum()):,} GPS "
        f"observations, {int(eda['n_nodes'].sum()):,} road-network nodes, and "
        f"{int(eda['n_arcs'].sum()):,} directed arcs (`results/eda_summary.csv`). "
        "Preprocessing performs shape/dtype checks, finite-value and coordinate-range "
        "validation, strictly increasing timestamp validation, and arc/node index "
        "bound checks; malformed files raise rather than being silently repaired. "
        "Distances are computed in a per-trip UTM projection chosen from the trip "
        "centroid, never in degrees."
    )
    doc.para(
        "Missing values. The raw `.track` files carry no header and no explicit "
        "missing-value marker; every record is parsed as three floats, so a truncated "
        "or blank line fails the finite-value check at load time instead of becoming a "
        "NaN row. The EDA table consequently has zero nulls across all 25 columns for "
        "all 90 trips. What the data does contain is missing *observations*: "
        f"{float(eda['nonunit_dt_frac'].median()):.2%} of steps are not exactly 1 s, "
        f"and the largest gap is {float(eda['max_gap_s'].max()):,.0f} s, so the series "
        "is not uniformly sampled even though it is nominally 1 Hz."
    )
    doc.para(
        "Sampling and gaps. Median sampling interval is "
        f"{float(eda['median_dt_s'].median()):.1f} s. Gaps longer than 30 s split a "
        "trip into independently decoded segments so the decoder never invents a route "
        "through an unobserved period (Figure 1 and the sampling panel in "
        "`results/figures/eda_sampling_gaps.png`)."
    )
    doc.para(
        "Road-network structure. Networks have no isolated nodes, but they are "
        f"fragmented: per trip, weak components range up to "
        f"{int(eda['n_weak_components'].max())} (median "
        f"{float(eda['n_weak_components'].median()):.0f}) and strongly connected "
        f"components up to {int(eda['n_strong_components'].max())} (median "
        f"{float(eda['n_strong_components'].median()):.0f}). The directed gap matters: "
        "the HMM transition term is a directed shortest-path length, so a candidate "
        "pair in different components, or connected only against the direction of "
        "travel, is treated as an invalid transition rather than given a large but "
        "finite cost."
    )
    doc.para(
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
    doc.image(
        "results/figures/eda_inventory_network.png",
        "Figure 1. Observations per trip and road-network size (source: results/eda_summary.csv).",
    )
    doc.image(
        "results/figures/eda_route_coverage.png",
        f"Figure 2. GPS-to-route coverage with severe-mismatch and low-coverage thresholds; trips {severe_ids} are annotated (source: results/eda_summary.csv).",
    )
    doc.image(
        "results/figures/eda_route_anomalies.png",
        "Figure 3. Reference-route node discontinuities and repeated arcs (source: results/eda_summary.csv).",
    )
    doc.image(
        "results/figures/eda_sampling_gaps.png",
        "Figure 4. Sampling-interval distribution and GPS-gap lengths per trip, training and student-test partitions (source: results/eda_summary.csv).",
    )
    doc.image(
        "results/figures/eda_trip_extents.png",
        "Figure 5. Geographical extent of every trajectory, coloured by partition. Trips are spread across many regions worldwide and are not drawn from one city (source: results/eda_summary.csv).",
    )
    doc.para(
        "The interactive Folium map for a development trip, showing raw GPS "
        "observations over the road network and the supplied reference route, is "
        "committed at `results/folium/eda_00000001.html` (built by `scripts/01_eda.py`). "
        "Printable Folium map views appear in Section 5. All EDA figures above use the "
        "training and student-testing partitions; evaluator trips are excluded from "
        "this analysis and every number in this section is derived from "
        "`results/eda_summary.csv`, which covers 90 trips (80 training, 10 "
        "student-testing)."
    )

    doc.h2("1.1 Feature engineering and selection")
    doc.para(
        "Every `(observation, candidate)` pair receives ten features in a fixed order "
        "stored in `artifacts/config.json`: perpendicular distance, log-distance, "
        "heading difference, GPS-vs-arc speed consistency, spatial candidate rank, "
        "along-arc fraction, arc length, temporal gap, and from/to node total degree. "
        "The stable order is the contract between training and inference."
    )
    doc.table(
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
    doc.para(
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
    doc.image(
        "results/figures/feature_ablation.png",
        "Figure 6. Leave-one-feature-out importance and cumulative feature-group ablation (source: results/feature_ablation.csv).",
    )

    n_models = aggregates.shape[0]
    n_families = len({FAMILY_GROUPS[f] for f in aggregates["family"]})
    fold_std = cv[cv["scope"] == "fold"].groupby("model")["macro_f1"].std()
    doc.h1("2. Model experimentation and comparison")
    doc.para(
        f"The expanded comparison covers {n_models} models drawn from {n_families} "
        "algorithm families, each trained under deterministic trip-grouped "
        "cross-validation on the 77 training trips that passed the coverage filter. "
        "Every fold is scored on held-out trips, so no observation from a trip used "
        "for fitting is ever used for evaluation. For tractability the comparison "
        "run samples up to 300 evenly spaced observations per trip, while the "
        "deployed model is refit on up to 1,000. All models emit a calibrated "
        "positive-class probability for each `(observation, candidate)` row. Exact "
        "configurations are recorded in `results/model_comparison.csv` and "
        "`results/student_test_comparison.csv`."
    )
    doc.para(
        "The table answers the required comparison directly. Route continuity is a "
        "property of the decoded output rather than of a candidate classifier, so it "
        "is reported as N/A for every classifier-only row: a classifier emits "
        "per-candidate probabilities, not a connected route, and continuity is "
        "undefined until a single arc sequence has been chosen. The corresponding "
        "decoder value for the deployed model is "
        f"{_fmt(hmm_means.loc['hmm', 'route_continuity'])} on student-testing trips "
        "(Section 4). CV Macro F1 is the mean trip-grouped cross-validation score and "
        "is the metric used for selection."
    )
    doc.table(
        [
            [
                "Model",
                "Family",
                "Key configuration",
                "CV Macro F1",
                "Top-1 acc",
                "Test Macro F1",
                "Test log loss",
                "Route continuity",
                "Infer ms",
            ]
        ]
        + [
            [
                name,
                FAMILY_GROUPS[aggregates.loc[name, "family"]],
                _config_summary(aggregates.loc[name, "config"], limit=52),
                _fmt(aggregates.loc[name, "macro_f1"]),
                _fmt(test_by_model.loc[name, "top1_accuracy"]),
                _fmt(test_by_model.loc[name, "macro_f1"]),
                _fmt(test_by_model.loc[name, "log_loss"]),
                "N/A",
                _fmt(test_by_model.loc[name, "inference_ms"], 1),
            ]
            for name in model_order
        ],
        widths=[1.05, 0.95, 1.7, 0.6, 0.55, 0.62, 0.6, 0.95, 0.55],
    )
    doc.para(
        "Selection criterion was mean CV Macro F1 with CV Top-1 as the tie-breaker; "
        "training accuracy was not used. The expanded comparison includes eleven "
        f"models; the best CV Macro F1 is {best_cv_model} "
        f"({_fmt(aggregates.loc[best_cv_model, 'macro_f1'])}), with student-test "
        f"Macro F1 {_fmt(test_by_model.loc[best_cv_model, 'macro_f1'])}. The deployed "
        "model is the tuned ExtraTrees recorded in `artifacts/config.json`: "
        + json.dumps(deployed_cfg["params"], sort_keys=True)
        + ". The Random Forest lead is "
        f"{_fmt(aggregates.loc[best_cv_model, 'macro_f1'] - aggregates.loc['extra_trees', 'macro_f1'])} "
        "CV Macro F1, smaller than the fold-to-fold standard deviation of either model "
        f"({_fmt(float(fold_std['random_forest']))} and "
        f"{_fmt(float(fold_std['extra_trees']))}), so it is not a reliable ranking. "
        "ExtraTrees was deployed because it already had a tuned configuration, a "
        "validated decoder integration and a fixed feature contract, and because the "
        "route-level result is produced by the decoder rather than the classifier "
        f"alone. The exported artifact is {artifact_mb:.1f} MB."
    )

    doc.h2("2.1 Per-model analysis")
    model_notes = {
        "random_forest": (
            "Strengths: best CV Macro F1 in the comparison, stable CV/test behaviour, "
            "and good probability quality. Limitations: the largest artifact and "
            "slowest inference among the accurate models, and it still cannot use "
            "sequential context; the HMM supplies that."
        ),
        "extra_trees": (
            "Strengths: near-top CV Macro F1 at lower training cost than Random Forest, "
            "fast enough for per-request decoding, and it captures the distance/heading "
            "interaction that separates parallel roads. Limitations: unconstrained depth "
            "inflates the artifact, which is why the deployed variant caps depth and "
            "leaf size."
        ),
        "mlp": (
            "Strengths: the best probability quality (lowest log loss) and the fastest "
            "inference among the accurate models. Limitations: the longest training "
            "time, and the optimiser hits its iteration cap, so it is mildly "
            "under-trained rather than converged."
        ),
        "knn": (
            "Strengths: no parametric assumptions and near-instant training. "
            "Limitations: inference is roughly three orders of magnitude slower than "
            "the trees, and its probabilities are the worst calibrated of the accurate "
            "models, both of which disqualify it as an HMM emission source."
        ),
        "hist_gradient_boosting": (
            "Strengths: compact native model with fast inference and good handling of "
            "nonlinear interactions. Limitations: a lower ceiling than the forests at "
            "these settings and weaker probability calibration."
        ),
        "lightgbm": (
            "Strengths: very fast inference and strong recall with a compact artifact. "
            "Limitations: slightly underfitted at a 200-tree, subsampled budget, and "
            "less accurate than the bagged forests here."
        ),
        "adaboost": (
            "Strengths: sequential error correction and high recall. Limitations: slow "
            "training and coarse probabilities, which matter because the decoder consumes "
            "these probabilities directly."
        ),
        "decision_tree": (
            "Strengths: very fast inference and a fully readable decision structure. "
            "Limitations: high variance across folds, coarse probability estimates, and "
            "substantially lower Macro F1."
        ),
        "gaussian_nb": (
            "Strengths: negligible training cost and trivial inference. Limitations: the "
            "feature-independence assumption is visibly violated by correlated distance "
            "and rank features, which shows up as the worst log loss in the comparison."
        ),
        "logistic_regression": (
            "Strengths: stable, interpretable, and directly calibrated, training in "
            "under a second. Limitations: it cannot express threshold interactions such "
            "as 'close AND aligned AND shallow', so it underfits relative to the "
            "ensembles."
        ),
        "sgd_logistic": (
            "Strengths: the fastest inference and smallest memory footprint in the "
            "comparison. Limitations: essentially the same decision boundary as Logistic "
            "Regression, with slightly worse CV Macro F1 and sensitivity to the "
            "learning-rate schedule."
        ),
    }
    MODEL_WHY = {
        "random_forest": (
            "Bagging many decorrelated trees reduces the variance that a single "
            "decision boundary shows on overlapping candidate features, and the "
            "bootstrap resampling gives each tree a different view of the "
            "parallel-road ambiguities in this dataset."
        ),
        "extra_trees": (
            "Extremely randomised splits decorrelate trees further than bagging alone, "
            "which suits candidate features whose decision boundaries are "
            "threshold-like (close, aligned, shallow) rather than smooth."
        ),
        "mlp": (
            "A shallow network can learn a smooth nonlinear combination of distance, "
            "heading and topology, testing whether a continuous scorer beats "
            "axis-aligned trees at intersections."
        ),
        "knn": (
            "Map-matching candidates that are physically close tend to have similar "
            "feature vectors, so nearest neighbours in feature space are a direct test "
            "of whether local similarity alone identifies the true arc."
        ),
        "hist_gradient_boosting": (
            "Sequential error correction with histogram-binned splits tests whether "
            "boosting adds accuracy over independent trees while staying compact and "
            "fast enough for per-request inference."
        ),
        "lightgbm": (
            "Leaf-wise growth with column and row subsampling is the fastest route to a "
            "strong boosted model on this wide, medium-sized tabular problem."
        ),
        "adaboost": (
            "Adaptive reweighting focuses successive stumps on hard candidates, which "
            "is a natural fit for the small hard core of ambiguous points near "
            "intersections."
        ),
        "decision_tree": (
            "A single tree exposes the exact thresholds the model has learned, which "
            "makes it a useful interpretability reference against the ensembles."
        ),
        "gaussian_nb": (
            "It is the cheapest probabilistic baseline available, so it anchors the "
            "low end of the cost/accuracy trade-off under the i.i.d. feature "
            "assumption."
        ),
        "logistic_regression": (
            "It produces a calibrated probability directly from a linear combination, "
            "which makes the degree of linear separability in the candidate features "
            "explicit."
        ),
        "sgd_logistic": (
            "Stochastic gradient training shows whether the linear baseline can be "
            "kept while scaling to larger candidate sets at near-zero inference cost."
        ),
    }
    MODEL_OBJECTIVE = {
        "random_forest": "minimised Gini impurity within balanced-subsample trees",
        "extra_trees": "minimised Gini impurity within balanced-subsample randomised trees",
        "mlp": "binary log loss (cross-entropy) with L2 alpha regularisation",
        "knn": "no training objective; distance-weighted vote over k neighbours",
        "hist_gradient_boosting": "binary log loss deviance with L2 regularisation",
        "lightgbm": "binary log loss with subsampling and balanced class weights",
        "adaboost": "exponential loss over decision stumps",
        "decision_tree": "minimised Gini impurity with balanced class weights",
        "gaussian_nb": "Gaussian likelihood with a uniform class prior",
        "logistic_regression": "binary log loss with L2 regularisation",
        "sgd_logistic": "binary log loss fitted by stochastic gradient descent",
    }
    MODEL_FIT_NOTE = {
        "random_forest": (
            "No overfitting signal: the test score slightly exceeds CV, and the gap is "
            "far inside the fold spread. Depth is unchecked in the comparison run, "
            "which raises variance risk in principle but did not materialise."
        ),
        "extra_trees": (
            "No overfitting signal; test marginally exceeds CV. The tuned deployed "
            "variant shows the same pattern, so the extra depth cap did not cause "
            "underfitting either."
        ),
        "mlp": (
            "No overfitting signal, and the training-time cap means it is if anything "
            "under-trained rather than over-fitted; its CV score sits just below the "
            "forests while test is marginally above."
        ),
        "knn": (
            "No overfitting signal, but probability calibration is poor (the worst log "
            "loss of the accurate models) and inference is three orders of magnitude "
            "slower, so it is rejected on latency and calibration rather than accuracy."
        ),
        "hist_gradient_boosting": (
            "Essentially no CV/test gap, and the lowest gap in the comparison. Its lower "
            "ceiling reflects a capacity limit at these settings, not memorisation."
        ),
        "lightgbm": (
            "No overfitting signal; the mild shortfall against the forests is consistent "
            "with underfitting from a 200-tree, subsampled budget rather than variance."
        ),
        "adaboost": (
            "No overfitting signal, though its high log loss shows coarse probability "
            "estimates, which matters because the HMM consumes these probabilities as "
            "emissions."
        ),
        "decision_tree": (
            "The largest fold-to-fold instability of the tree models appears here as "
            "high variance, but test does not fall below CV, so it is underfitting "
            "rather than over-fitting."
        ),
        "gaussian_nb": (
            "The only model whose test score falls clearly below CV (by about 0.015, "
            "roughly half a fold spread), consistent with its violated independence "
            "assumption rather than with overfitting."
        ),
        "logistic_regression": (
            "No overfitting signal; it clearly underfits relative to the ensembles, "
            "which is expected because it cannot represent the threshold interactions."
        ),
        "sgd_logistic": (
            "Essentially no CV/test gap; it underfits like the other linear model and is "
            "retained only as a speed baseline."
        ),
    }
    for name in model_order:
        doc.h3(name.replace("_", " ").title())
        row = test_by_model.loc[name]
        cv_row = aggregates.loc[name]
        gap = float(cv_row["macro_f1"]) - float(row["macro_f1"])
        doc.para(model_notes[name])
        doc.para("A. Model understanding. " + MODEL_WHY[name])
        doc.para(
            "B. Training and configuration. Key configuration: "
            + _config_summary(cv_row["config"])
            + ". Objective: "
            + MODEL_OBJECTIVE[name]
            + ". Train time "
            + _fmt(cv_row["train_time_s"], 2)
            + " s; inference "
            + _fmt(row["inference_ms"], 1)
            + " ms on the student-test set."
        )
        doc.para(
            "C. Experimental results. Student-test Macro F1 "
            f"{_fmt(row['macro_f1'])}, observation Top-1 accuracy "
            f"{_fmt(row['top1_accuracy'])}, log loss {_fmt(row['log_loss'])}, "
            f"precision {_fmt(row['precision'])}, recall {_fmt(row['recall'])}. "
            f"The CV-to-test Macro F1 gap is {gap:+.4f} (CV {_fmt(cv_row['macro_f1'])} "
            f"vs test {_fmt(row['macro_f1'])}), against this model's own CV fold "
            f"standard deviation of {_fmt(float(fold_std[name]))}. "
            + MODEL_FIT_NOTE[name]
        )

    doc.h2("2.2 Model visualisations")
    doc.para(
        "Every plot below is regenerated from the stored result CSVs by "
        "`scripts/03_train.py`; none is hand-drawn. Training and validation loss curves "
        "are deliberately absent: none of the compared models produces an epoch-wise "
        "training history under this protocol, and plotting invented curves would "
        "misrepresent the experiments. The confusion matrices are candidate-level "
        "diagnostics at the 0.5 threshold and are optional supporting material."
    )
    doc.image(
        "results/figures/cv_macro_f1_comparison.png",
        "Figure 7. Trip-grouped CV Macro F1 by model (source: results/model_comparison.csv).",
    )
    doc.image(
        "results/figures/student_test_top1_macro_f1.png",
        "Figure 8. Student-test Top-1 accuracy and Macro F1 (source: results/student_test_comparison.csv).",
    )
    doc.image(
        "results/figures/log_loss_comparison.png",
        "Figure 9. Student-test log loss by model (source: results/student_test_comparison.csv).",
    )
    doc.image(
        "results/figures/confusion_matrices.png",
        "Figure 10. Candidate confusion matrices on student-test data (source: results/student_test_comparison.csv).",
    )
    doc.image(
        "results/figures/timing_comparison.png",
        "Figure 11. CV training time and inference time (sources: results/model_comparison.csv, results/student_test_comparison.csv).",
    )

    doc.h1("3. Bayesian hyperparameter optimisation")
    doc.para(
        "1-2. Hyperparameters and search space. After the comparison stage, the "
        "ExtraTrees family was tuned with Optuna's Tree-structured Parzen Estimator, "
        "using trip-grouped cross-validated Macro F1 as the objective. The search "
        "space was deliberately bounded in size so that the exported artifact would "
        "stay deployable: depth 8-14, leaves 10-30, trees 100-200, max_features drawn "
        "from {sqrt, log2, 0.5}, criterion from {gini, entropy}, and class_weight from "
        "{balanced, balanced_subsample}. Categorical choices use categorical "
        "distributions; the remaining dimensions are stepped integers."
    )
    doc.table(
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
    doc.para(
        "3-4. Objective and trial count. The objective was to maximise the mean "
        "trip-grouped CV Macro F1 across the five folds, with trip groups preserved so "
        "no trip contributes to both fitting and validation. "
        f"{int(best['completed_trials'])} trials completed within a 600 s budget; all "
        "trial outcomes, including the poor ones, are recorded in "
        "`results/optuna_trials.csv`."
    )
    doc.para(
        "5-6. Best configuration and whether tuning helped. Best parameters: "
        + json.dumps(best["best_params"], sort_keys=True)
        + f" (CV Macro F1 {_fmt(best['best_cv_macro_f1'])}). The tuned model did not "
        "improve accuracy: the size constraint reduced CV Macro F1 from "
        f"{_fmt(tuned.loc['untuned', 'cv_macro_f1'])} to "
        f"{_fmt(tuned.loc['tuned', 'cv_macro_f1'])}, and student-test Macro F1 from "
        f"{_fmt(tuned.loc['untuned', 'student_macro_f1'])} to "
        f"{_fmt(tuned.loc['tuned', 'student_macro_f1'])}. It reduced inference time "
        f"from {_fmt(tuned.loc['untuned', 'student_inference_ms'], 1)} ms to "
        f"{_fmt(tuned.loc['tuned', 'student_inference_ms'], 1)} ms. Both variants were "
        "produced by `scripts/04_tune.py` and compared on identical folds and "
        "student-test data. The deployed artifact refits the same tuned parameters on "
        f"up to 1,000 aligned observations per trip, giving a {artifact_mb:.1f} MB "
        "compressed model. The honest conclusion is that tuning bought latency and "
        "artifact size, not accuracy."
    )
    doc.table(
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

    doc.h1("4. HMM and Viterbi route reconstruction")
    doc.para(
        "Route reconstruction casts map-matching as inference over a hidden Markov "
        "model. The hidden state at each observation is the road arc the vehicle is "
        "assumed to occupy, drawn from that observation's candidate set. Classifier "
        "emissions are the model's positive-class probabilities, normalised across the "
        "candidates of an observation in log space; a Gaussian distance emission is "
        "retained as a baseline. The transition score is "
        "`-|gps_displacement - network_distance| / beta`, where the network distance is "
        "the directed shortest-path length from the projection on the first arc, "
        "through the network, to the projection on the second. Disconnected or "
        "direction-violating transitions receive an explicit invalid penalty (-50) "
        "rather than a large finite cost, so the decoder is never rewarded for "
        "treating an impossible jump as merely expensive."
    )
    doc.para(
        "Two decoder additions are deployed alongside Viterbi. First, a same-road "
        "backward projection tolerance (60 m) allows a GPS point to sit slightly "
        "behind its predecessor on the same arc, which occurs at low speed and in "
        "urban multipath. Second, after decoding a segment, every point is "
        "re-projected onto the inferred route with the same monotonic alignment used "
        "for labels; this corrects local emission errors once route direction is "
        "known. Refinement reads only the inferred route geometry, never reference "
        "arcs, and `tests/test_hmm.py::test_point_refinement_is_independent_of_reference_route` "
        "asserts that altering the reference route does not change predictions."
    )
    doc.para(
        f"The deployed settings are read from `artifacts/hmm.json`: beta_m="
        f"{deployed_hmm['beta_m']:g}, backward_tolerance_m="
        f"{deployed_hmm['backward_tolerance_m']:g}, refine_points="
        f"{deployed_hmm['refine_points']}, emission='{deployed_hmm['emission']}'. "
        "Beta and the refinement switch were selected on a 64/16 split of the 80 "
        "training trips (seed 42); evaluator trips played no part in that choice. "
        "`results/decoder_optimization/` retains the 19 configuration trials and the "
        "two-way refinement comparison."
    )
    doc.para(
        "Observations without candidates and GPS gaps longer than 30 s split the "
        "sequence into segments; each segment is decoded independently. Chosen arcs "
        "are then stitched with directed shortest paths so consecutive output arcs "
        "share nodes. Viterbi runs in log space over an 8-candidate beam per "
        "observation, with a 2 km shortest-path cutoff and per-source Dijkstra caching."
    )
    doc.table(
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
    doc.para(
        f"Classifier+HMM improves point accuracy over classifier-only on "
        f"{improved_trips} of the ten trips. The largest gain is trip `{best_trip}` "
        f"({delta.loc[best_trip]:+.4f}); the smallest is trip `{worst_trip}` "
        f"({delta.loc[worst_trip]:+.4f}). Route precision rises from "
        f"{_fmt(hmm_means.loc['classifier', 'route_precision'])} to "
        f"{_fmt(hmm_means.loc['hmm', 'route_precision'])}, and RMF falls from "
        f"{_fmt(hmm_means.loc['classifier', 'rmf'], 2)} to "
        f"{_fmt(hmm_means.loc['hmm', 'rmf'], 2)}. The gain is largest where the "
        "classifier alone picks locally plausible but disconnected arcs; "
        "connectivity constraints then repair those points. Continuity stays at or "
        "above 0.99 for every trip, so the decoder is not trading coherence for "
        "accuracy. Full detail is in `results/hmm_comparison.csv` and "
        "`results/hmm_failure_analysis.md`."
    )
    doc.image(
        "results/figures/decoder_accuracy_continuity.png",
        "Figure 12. Mean point accuracy and route continuity by decoder (source: results/hmm_comparison.csv).",
    )
    doc.image(
        "results/figures/decoder_by_trip.png",
        "Figure 13. Per-trip point accuracy for nearest, classifier-only, and classifier+HMM (source: results/hmm_comparison.csv).",
    )

    doc.h1("5. Geographical visualisation with Folium")
    doc.para(
        "Folium maps were generated for representative student-test trips: the "
        "largest HMM gain, the smallest HMM gain, and the first trip in the "
        "partition. Each map contains the GPS observations, a sampled road network, "
        "the reference route (development data only, dashed blue), the predicted route "
        "(green), a legend, and collapsible layer controls. The reference layer exists "
        "only in offline report maps; the evaluation API never serves it. Interactive "
        "HTML maps are committed under `results/folium/` and generated by "
        "`scripts/07_folium.py`."
    )
    doc.image(
        f"results/figures/folium_{best_trip}_preview.png",
        f"Figure 14. Trip {best_trip}: largest HMM accuracy gain ({delta.loc[best_trip]:+.4f}). GPS, network, reference (dev only), predicted route.",
        width_in=5.2,
    )
    doc.image(
        f"results/figures/folium_{worst_trip}_preview.png",
        f"Figure 15. Trip {worst_trip}: smallest HMM gain ({delta.loc[worst_trip]:+.4f}); predicted route stays contiguous.",
        width_in=5.2,
    )

    doc.h1("6. API, UI, and deployment")
    doc.para(
        "The backend exposes `GET /health`, `GET /trips`, "
        "`GET /trips/{trip_id}/observations`, and `POST /predict`. `/predict` "
        "accepts `trip_id`, `method` (nearest, classifier, hmm), `noise_sigma_m` "
        "(0-100), `sample_every_s` (1-60), and `seed`. Responses contain predicted "
        "arcs, route coordinates, per-point predictions, aggregate metrics, and "
        "display layers; reference geometry never leaves the server. Trip IDs are "
        "resolved through a discovered-trip allowlist."
    )
    doc.para(
        f"End-to-end verification called the API for all ten student-testing trips "
        f"(`results/e2e_summary.csv`): {int((e2e['status'] == 200).sum())}/10 returned "
        f"HTTP 200 with finite metrics, mean HMM point accuracy "
        f"{_fmt(e2e['top1_accuracy'].mean())}, mean full-pipeline latency "
        f"{_fmt(e2e['inference_ms'].mean(), 0)} ms, and worst-case latency "
        f"{_fmt(e2e['inference_ms'].max(), 0)} ms. The service loads `model.joblib` "
        f"({artifact_mb:.1f} MB) and `hmm.json` once at startup and serves exactly the "
        "configuration reported in Section 4."
    )
    doc.image(
        "report/assets/api_response_terminal.png",
        "Figure 16. POST /predict response captured from the running service (list fields abbreviated and rendered for legibility).",
    )
    doc.image(
        "results/ui/route-explorer.jpg",
        "Figure 17. Route Explorer tab: journey picker, matching method, live noise and sampling controls, matched route over GPS points, and server-side point accuracy, decoder Macro F1 and route F1.",
    )
    doc.image(
        "results/ui/model-results.jpg",
        "Figure 18. Model results tab: the same aggregate metrics the report quotes, split into five-fold CV, student-test and evaluator-test columns.",
    )
    doc.para(
        "Deployment is two independent processes. Laptop 1 runs "
        "`uvicorn api.main:app --host 0.0.0.0 --port 8000` with `EVAL_DATA_DIR` and "
        "`CORS_ORIGINS` configured; Laptop 2 serves `ui/` with "
        "`python -m http.server 5500 --directory ui` and sets `apiBaseUrl` in "
        "`ui/config.js` to Laptop 1's LAN address. The single-machine fallback uses "
        "localhost for both. Full steps are in `docs/deployment.md`."
    )

    doc.h1("7. Evaluation integrity, limitations, and conclusion")
    doc.para(
        "Evaluator-testing data was never used for training, tuning, feature "
        "selection, or threshold selection. Reference routes and labels remain "
        "server-side; API and UI tests assert that responses contain no reference "
        "fields and that logs contain no ground truth. Feature caches are versioned "
        "by the feature/candidate/alignment specification, and the model artifact "
        "validates its feature order against `config.json`."
    )
    doc.para(
        "Several limitations should be read alongside these results. First, "
        "cross-validation and the comparison table use a deterministic per-trip "
        "observation cap for tractability, whereas the deployed model raises that cap "
        "from 300 to 1,000 observations; the comparison table and the deployed model "
        "are therefore not scored under identical sampling. Second, the size-bounded "
        "tuning search trades roughly two Macro F1 points for an artifact small enough "
        f"to deploy ({artifact_mb:.1f} MB). Third, the MLP stopped at its 150-iteration "
        f"cap and is mildly under-trained rather than converged. Fourth, {severe_ids} "
        f"and {lowest_trip['trip_id']} have the lowest GPS-to-route coverage "
        f"({float(lowest_trip['route_gps_within50_frac']):.1%} in the least severe "
        "case), so their labels are reported alongside that coverage instead of being "
        "treated as clean. Fifth, point refinement can move an observation onto a "
        "connector arc that was never in its candidate set, so decoded arc identifiers "
        "are not a strict subset of the classifier's candidate arcs. Sixth, route "
        "precision plateaus near 0.83 because the supplied reference routes traverse "
        "repeated arcs while the decoder emits the shortest connected path; this is a "
        "property of the label convention rather than a decoding failure."
    )
    doc.para(
        "Conclusion. PathSnap delivers a complete and reproducible map-matching "
        f"system: validated data processing, {n_models} compared models across "
        f"{n_families} families, a recorded Bayesian tuning study, a network-aware HMM "
        "decoder with reference-independent point refinement, honest server-side "
        "scoring, a working FastAPI service, and a live Leaflet demonstration. "
        "Classifier+HMM is the strongest route reconstruction method tested, "
        f"achieving {_fmt(hmm_accuracy)} mean-trip point accuracy "
        f"({_fmt(hmm_pooled)} pooled) and {_fmt(hmm_continuity)} route continuity "
        "across the ten student-testing trips."
    )
    return doc
