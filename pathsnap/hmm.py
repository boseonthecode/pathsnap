"""HMM emissions, network transitions, Viterbi decoding, and route stitching.

Hidden states are candidate arcs for each observation. Emissions come from the
learned classifier probability or a Gaussian distance baseline; transitions use
the directed network distance between candidates and are decoded in log space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import networkx as nx
import numpy as np
from scipy.special import logsumexp

from pathsnap import config
from pathsnap.candidates import Candidate, CandidateSet
from pathsnap.geometry import (
    Projection,
    arc_lengths_m,
    build_directed_graph,
    projection_for_trip,
)
from pathsnap.io import Trip
from pathsnap.labels import align_to_inferred_route


@dataclass(frozen=True)
class HMMConfig:
    """Decoder configuration persisted in ``artifacts/hmm.json``."""

    top_k: int = config.HMM_TOP_K
    radius_m: float = config.HMM_RADIUS_M
    max_radius_m: float = config.HMM_MAX_RADIUS_M
    beta_m: float = config.HMM_BETA_M
    invalid_penalty: float = config.HMM_INVALID_PENALTY
    gap_split_s: float = config.HMM_GAP_SPLIT_S
    emission: str = config.HMM_EMISSION
    gaussian_sigma_m: float = config.HMM_GAUSSIAN_SIGMA_M
    max_path_m: float = config.HMM_MAX_PATH_M
    backward_tolerance_m: float = 0.0
    distance_weight: float = 0.0
    refine_points: bool = False

    def __post_init__(self) -> None:
        if self.beta_m <= 0 or self.gaussian_sigma_m <= 0:
            raise ValueError("beta_m and gaussian_sigma_m must be positive")
        if self.backward_tolerance_m < 0 or self.distance_weight < 0:
            raise ValueError("backward_tolerance_m and distance_weight must be nonnegative")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if self.radius_m <= 0:
            raise ValueError("radius_m must be positive")
        if self.emission not in {"classifier", "gaussian"}:
            raise ValueError(f"Unknown emission type: {self.emission!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_k": self.top_k,
            "radius_m": self.radius_m,
            "max_radius_m": self.max_radius_m,
            "beta_m": self.beta_m,
            "invalid_penalty": self.invalid_penalty,
            "gap_split_s": self.gap_split_s,
            "emission": self.emission,
            "gaussian_sigma_m": self.gaussian_sigma_m,
            "max_path_m": self.max_path_m,
            "backward_tolerance_m": self.backward_tolerance_m,
            "distance_weight": self.distance_weight,
            "refine_points": self.refine_points,
        }


@dataclass(frozen=True)
class DecodeResult:
    trip_id: str
    method: str
    predicted_arcs: np.ndarray
    point_arc_ids: np.ndarray
    point_confidences: np.ndarray
    segment_count: int
    n_unbridged_transitions: int


class NetworkDistance:
    """Directed node/arc distance lookup with per-process caching."""

    def __init__(self, trip: Trip, max_path_m: float = config.HMM_MAX_PATH_M):
        self.trip = trip
        self.max_path_m = max_path_m
        self.graph = build_directed_graph(trip)
        # One Dijkstra per source node; targets within the cutoff are cached.
        self._source_cache: dict[int, dict[int, float]] = {}

    def node_distance(self, source: int, target: int) -> float | None:
        if source == target:
            return 0.0
        distances = self._source_cache.get(source)
        if distances is None:
            try:
                distances = nx.single_source_dijkstra_path_length(
                    self.graph, source, cutoff=self.max_path_m, weight="length_m"
                )
            except nx.NodeNotFound:
                distances = {}
            self._source_cache[source] = distances
        return distances.get(target)

    def candidate_distance(
        self, first: Candidate, second: Candidate, backward_tolerance_m: float = 0.0
    ) -> float | None:
        """Network distance from the first projection through to the second."""
        tail = (1.0 - first.along_fraction) * first.arc_length_m
        head = second.along_fraction * second.arc_length_m

        if first.arc_id == second.arc_id:
            backward_m = (first.along_fraction - second.along_fraction) * first.arc_length_m
            if 0 < backward_m <= backward_tolerance_m:
                # A small projection reversal can be observation noise while
                # the vehicle stays on the same directed road.
                return 0.0
            if second.along_fraction >= first.along_fraction:
                return (second.along_fraction - first.along_fraction) * first.arc_length_m
            # Re-entering the same directed arc requires a loop.
            source = int(self.trip.arcs[first.arc_id, 1])
            target = int(self.trip.arcs[second.arc_id, 0])
        else:
            source = int(self.trip.arcs[first.arc_id, 1])
            target = int(self.trip.arcs[second.arc_id, 0])

        middle = self.node_distance(source, target)
        if middle is None:
            return None
        return tail + middle + head


def normalize_log_scores(log_scores: np.ndarray) -> np.ndarray:
    """Normalise arbitrary log weights into log probabilities."""
    log_scores = np.asarray(log_scores, dtype=np.float64)
    if log_scores.size == 0:
        return log_scores
    return log_scores - logsumexp(log_scores)


def gaussian_log_emissions(
    distances_m: np.ndarray, sigma_m: float = config.HMM_GAUSSIAN_SIGMA_M
) -> np.ndarray:
    """Gaussian distance emissions, normalised across the candidates."""
    distances_m = np.asarray(distances_m, dtype=np.float64)
    log_weights = -(distances_m**2) / (2.0 * sigma_m**2)
    return normalize_log_scores(log_weights)


def classifier_log_emissions(probabilities: np.ndarray) -> np.ndarray:
    """Classifier probabilities converted to normalised log emissions."""
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0)
    return normalize_log_scores(np.log(probabilities))


def transition_log_probability(
    network_distance_m: float | None,
    gps_displacement_m: float,
    beta_m: float,
    invalid_penalty: float,
) -> float:
    """Log transition score from GPS-vs-network displacement mismatch."""
    if network_distance_m is None:
        return -float(invalid_penalty)
    return -abs(gps_displacement_m - network_distance_m) / float(beta_m)


def transition_log_matrix(
    previous: Sequence[Candidate],
    current: Sequence[Candidate],
    gps_displacement_m: float,
    network: NetworkDistance,
    config: HMMConfig,
) -> np.ndarray:
    matrix = np.empty((len(previous), len(current)), dtype=np.float64)
    for i, first in enumerate(previous):
        for j, second in enumerate(current):
            network_distance = network.candidate_distance(
                first, second, config.backward_tolerance_m
            )
            matrix[i, j] = transition_log_probability(
                network_distance,
                gps_displacement_m,
                config.beta_m,
                config.invalid_penalty,
            )
    return matrix


def viterbi_log(
    emissions: Sequence[np.ndarray],
    transitions: Sequence[np.ndarray],
) -> list[int]:
    """Log-space Viterbi over per-observation emission and transition matrices."""
    n_steps = len(emissions)
    if n_steps == 0:
        return []
    backpointers = [
        np.full(emissions[t].shape[0], -1, dtype=np.int32) for t in range(n_steps)
    ]
    scores = emissions[0].astype(np.float64, copy=True)

    for t in range(1, n_steps):
        candidate_scores = scores[:, None] + transitions[t - 1]
        best_previous = np.argmax(candidate_scores, axis=0)
        scores = emissions[t] + candidate_scores[best_previous, np.arange(emissions[t].shape[0])]
        backpointers[t] = best_previous

    path = [0] * n_steps
    state = int(np.argmax(scores))
    for t in range(n_steps - 1, -1, -1):
        path[t] = state
        if t > 0:
            state = int(backpointers[t][state])
    return path


def _segments(
    trip: Trip, groups: dict[int, list[Candidate]], gap_split_s: float
) -> list[list[int]]:
    segments: list[list[int]] = []
    current: list[int] = []
    for obs_index in range(trip.n_points):
        if obs_index not in groups:
            if current:
                segments.append(current)
                current = []
            continue
        if current:
            gap = float(trip.track[obs_index, 2] - trip.track[current[-1], 2])
            if gap > gap_split_s:
                segments.append(current)
                current = []
        current.append(obs_index)
    if current:
        segments.append(current)
    return segments


def _gps_steps(trip: Trip) -> np.ndarray:
    projection = projection_for_trip(trip)
    track_xy = projection.to_xy(trip.track[:, :2])
    if trip.n_points < 2:
        return np.zeros(trip.n_points, dtype=np.float64)
    steps = np.linalg.norm(np.diff(track_xy, axis=0), axis=1)
    return np.r_[steps, steps[-1]]


def _emissions_for(
    candidates: Sequence[Candidate],
    probabilities: np.ndarray | None,
    config: HMMConfig,
) -> np.ndarray:
    if config.emission == "gaussian":
        distances = np.array([candidate.distance_m for candidate in candidates])
        return gaussian_log_emissions(distances, config.gaussian_sigma_m)
    if config.emission == "classifier":
        if probabilities is None:
            raise ValueError("Classifier emission requires candidate probabilities")
        scores = classifier_log_emissions(probabilities)
        if config.distance_weight:
            distances = np.array([candidate.distance_m for candidate in candidates])
            scores = scores + config.distance_weight * gaussian_log_emissions(
                distances, config.gaussian_sigma_m
            )
        return normalize_log_scores(scores)
    raise ValueError(f"Unknown emission type: {config.emission!r}")


def _validate_candidates(trip: Trip, candidate_set: CandidateSet) -> None:
    """Boundary checks on candidate indices before any decoding happens."""
    for candidate in candidate_set.candidates:
        if not 0 <= candidate.obs_index < trip.n_points:
            raise ValueError(
                f"Candidate observation index {candidate.obs_index} outside "
                f"[0, {trip.n_points})"
            )
        if not 0 <= candidate.arc_id < trip.n_arcs:
            raise ValueError(
                f"Candidate arc id {candidate.arc_id} outside [0, {trip.n_arcs})"
            )


def _probability_lookup(
    candidate_set: CandidateSet, probabilities: np.ndarray
) -> dict[tuple[int, int], float]:
    """Map ``(obs_index, arc_id)`` to its probability, validating the array shape."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    expected = (len(candidate_set.candidates),)
    if probabilities.shape != expected:
        raise ValueError(
            f"probabilities has shape {probabilities.shape}, expected {expected}"
        )
    return {
        (candidate.obs_index, candidate.arc_id): float(probability)
        for candidate, probability in zip(candidate_set.candidates, probabilities)
    }


def decode_hmm(
    trip: Trip,
    candidate_set: CandidateSet,
    probabilities: np.ndarray | None = None,
    config: HMMConfig | None = None,
    network: NetworkDistance | None = None,
) -> DecodeResult:
    """Run segment-wise Viterbi and stitch the chosen arcs into a route."""
    config = config or HMMConfig()
    groups = candidate_set.by_observation()
    point_arc_ids = np.full(trip.n_points, -1, dtype=np.int32)
    point_confidences = np.zeros(trip.n_points, dtype=np.float64)
    if not groups:
        return DecodeResult(
            trip.trip_id,
            "hmm",
            np.empty(0, dtype=np.int32),
            point_arc_ids,
            point_confidences,
            0,
            0,
        )

    _validate_candidates(trip, candidate_set)
    probability_lookup = (
        _probability_lookup(candidate_set, probabilities)
        if probabilities is not None
        else {}
    )

    network = network or NetworkDistance(trip, config.max_path_m)
    node_arc_lookup = _node_arc_lookup(trip)
    gps_steps = _gps_steps(trip)
    segments = _segments(trip, groups, config.gap_split_s)

    stitched: list[int] = []
    unbridged = 0
    for segment in segments:
        chosen_arcs = []
        emission_rows: list[np.ndarray] = []
        transition_rows: list[np.ndarray] = []
        for position, obs_index in enumerate(segment):
            candidates = groups[obs_index]
            if config.emission == "classifier":
                probs = np.array(
                    [probability_lookup[(obs_index, c.arc_id)] for c in candidates]
                )
            else:
                probs = None
            emission_rows.append(_emissions_for(candidates, probs, config))
            if position > 0:
                previous_obs = segment[position - 1]
                transition_rows.append(
                    transition_log_matrix(
                        groups[previous_obs],
                        candidates,
                        gps_steps[previous_obs],
                        network,
                        config,
                    )
                )

        chosen = viterbi_log(emission_rows, transition_rows)
        for position, (obs_index, candidate_index) in enumerate(zip(segment, chosen)):
            candidate = groups[obs_index][candidate_index]
            point_arc_ids[obs_index] = candidate.arc_id
            point_confidences[obs_index] = float(
                np.exp(emission_rows[position][candidate_index])
            )
            chosen_arcs.append(candidate.arc_id)
        segment_route, segment_unbridged = stitch_arc_sequence(
            trip, chosen_arcs, network=network, node_arc_lookup=node_arc_lookup
        )
        if config.refine_points and segment_route:
            # Re-align to the INFERRED route only. This uses no reference arcs.
            # It corrects local emission/heading errors once route direction is known.
            refined = align_to_inferred_route(trip, segment_route, segment)
            for position, obs_index in enumerate(segment):
                arc_id = int(refined[position])
                point_arc_ids[obs_index] = arc_id
                # A connector may not have been in the point's candidate set.
                point_confidences[obs_index] = next((
                    float(np.exp(emission_rows[position][j]))
                    for j, candidate in enumerate(groups[obs_index])
                    if candidate.arc_id == arc_id
                ), 0.0)
        # Missing observations/time gaps are intentional chain breaks. Do not
        # invent a shortest-path route through an unobserved interval.
        if stitched and segment_route:
            unbridged += 1
        stitched.extend(segment_route)
        unbridged += segment_unbridged
    return DecodeResult(
        trip_id=trip.trip_id,
        method="hmm",
        predicted_arcs=np.asarray(stitched, dtype=np.int32),
        point_arc_ids=point_arc_ids,
        point_confidences=point_confidences,
        segment_count=len(segments),
        n_unbridged_transitions=unbridged,
    )


def decode_nearest(trip: Trip, candidate_set: CandidateSet) -> DecodeResult:
    """Nearest-snap baseline: rank-0 candidate per observation."""
    groups = candidate_set.by_observation()
    point_arc_ids = np.full(trip.n_points, -1, dtype=np.int32)
    chosen_arcs: list[int] = []
    for obs_index, candidates in groups.items():
        if not candidates:
            continue
        best = candidates[0]  # candidates are rank-ordered by (distance, arc_id)
        point_arc_ids[obs_index] = best.arc_id
        chosen_arcs.append(best.arc_id)
    stitched, unbridged = stitch_arc_sequence(trip, chosen_arcs)
    return DecodeResult(
        trip_id=trip.trip_id,
        method="nearest",
        predicted_arcs=np.asarray(stitched, dtype=np.int32),
        point_arc_ids=point_arc_ids,
        point_confidences=np.zeros(trip.n_points, dtype=np.float64),
        segment_count=0,
        n_unbridged_transitions=unbridged,
    )


def decode_classifier(
    trip: Trip, candidate_set: CandidateSet, probabilities: np.ndarray
) -> DecodeResult:
    """Classifier-only baseline: highest-probability candidate per observation."""
    groups = candidate_set.by_observation()
    _validate_candidates(trip, candidate_set)
    probability_lookup = _probability_lookup(candidate_set, probabilities)
    point_arc_ids = np.full(trip.n_points, -1, dtype=np.int32)
    point_confidences = np.zeros(trip.n_points, dtype=np.float64)
    chosen_arcs: list[int] = []
    for obs_index, candidates in groups.items():
        best = max(
            candidates,
            key=lambda candidate: probability_lookup[(obs_index, candidate.arc_id)],
        )
        point_arc_ids[obs_index] = best.arc_id
        point_confidences[obs_index] = probability_lookup[(obs_index, best.arc_id)]
        chosen_arcs.append(best.arc_id)
    stitched, unbridged = stitch_arc_sequence(trip, chosen_arcs)
    return DecodeResult(
        trip_id=trip.trip_id,
        method="classifier",
        predicted_arcs=np.asarray(stitched, dtype=np.int32),
        point_arc_ids=point_arc_ids,
        point_confidences=point_confidences,
        segment_count=0,
        n_unbridged_transitions=unbridged,
    )


def _node_arc_lookup(
    trip: Trip, projection: Projection | None = None
) -> dict[tuple[int, int], int]:
    """Map each directed node pair to its shortest parallel arc, measured in metres."""
    lengths = arc_lengths_m(trip, projection)
    lookup: dict[tuple[int, int], tuple[float, int]] = {}
    for arc_id, (from_node, to_node) in enumerate(trip.arcs):
        key = (int(from_node), int(to_node))
        length = float(lengths[arc_id])
        current = lookup.get(key)
        if current is None or length < current[0]:
            lookup[key] = (length, arc_id)
    return {key: value[1] for key, value in lookup.items()}


def stitch_arc_sequence(
    trip: Trip,
    arcs: Sequence[int],
    network: NetworkDistance | None = None,
    node_arc_lookup: dict[tuple[int, int], int] | None = None,
) -> tuple[list[int], int]:
    """Connect consecutive chosen arcs with directed shortest paths."""
    if not arcs:
        return [], 0
    lookup = node_arc_lookup if node_arc_lookup is not None else _node_arc_lookup(trip)
    graph = network.graph if network is not None else build_directed_graph(trip)

    stitched: list[int] = [int(arcs[0])]
    unbridged = 0
    for arc in arcs[1:]:
        arc = int(arc)
        last = stitched[-1]
        if arc == last:
            continue
        if int(trip.arcs[last, 1]) == int(trip.arcs[arc, 0]):
            stitched.append(arc)
            continue
        try:
            node_path = nx.shortest_path(
                graph,
                int(trip.arcs[last, 1]),
                int(trip.arcs[arc, 0]),
                weight="length_m",
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            unbridged += 1
            stitched.append(arc)
            continue
        for from_node, to_node in zip(node_path[:-1], node_path[1:]):
            connector = lookup.get((int(from_node), int(to_node)))
            if connector is None or connector == stitched[-1]:
                continue
            stitched.append(int(connector))
        if stitched[-1] != arc:
            stitched.append(arc)
    return stitched, unbridged
