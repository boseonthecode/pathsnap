"""Tests for HMM emissions, network transitions, Viterbi, and stitching."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pathsnap import config
from pathsnap.candidates import Candidate, generate_candidates
from pathsnap.hmm import (
    HMMConfig,
    NetworkDistance,
    classifier_log_emissions,
    decode_hmm,
    gaussian_log_emissions,
    stitch_arc_sequence,
    transition_log_probability,
    viterbi_log,
)
from pathsnap.io import iter_trips, load_trip
from pathsnap.metrics import route_continuity
from conftest import write_trip

TOY_NODES = np.array(
    [[0.0, 0.0], [0.0, 0.001], [0.0, 0.002], [0.01, 0.01], [0.0101, 0.01]]
)
TOY_ARCS = np.array([[0, 1], [1, 2], [3, 4]])
TOY_TRACK = np.array([[0.00005, 0.0005, 1.0], [0.00005, 0.0015, 2.0]])


def _toy_trip(tmp_path: Path):
    trip_dir = write_trip(
        tmp_path / "train",
        "00000050",
        nodes=TOY_NODES,
        arcs=TOY_ARCS,
        track=TOY_TRACK,
        route=np.array([0, 1]),
    )
    return load_trip(trip_dir)


def _candidate(trip_id: str, obs: int, arc_id: int, arc_length: float) -> Candidate:
    return Candidate(
        trip_id=trip_id,
        obs_index=obs,
        arc_id=arc_id,
        distance_m=5.0,
        proj_lon=0.0,
        proj_lat=0.0,
        along_fraction=0.5,
        heading_diff_deg=0.0,
        rank=0,
        arc_length_m=arc_length,
    )


def test_connected_transition_scores_higher_than_disconnected(tmp_path: Path) -> None:
    trip = _toy_trip(tmp_path)
    network = NetworkDistance(trip)
    first = _candidate(trip.trip_id, 0, 0, 111.0)
    connected = _candidate(trip.trip_id, 1, 1, 111.0)
    isolated = _candidate(trip.trip_id, 1, 2, 111.0)

    valid_distance = network.candidate_distance(first, connected)
    invalid_distance = network.candidate_distance(first, isolated)

    assert valid_distance is not None
    assert invalid_distance is None
    valid_score = transition_log_probability(valid_distance, 111.0, 10.0, 50.0)
    invalid_score = transition_log_probability(invalid_distance, 111.0, 10.0, 50.0)
    assert valid_score > invalid_score
    assert invalid_score == pytest.approx(-50.0)


def test_one_way_direction_violation_has_no_path(tmp_path: Path) -> None:
    nodes = np.array([[0.0, 0.0], [0.0, 0.001], [0.0, 0.002]])
    arcs = np.array([[0, 1], [1, 2]])
    track = np.array([[0.00005, 0.0005, 1.0], [0.00005, 0.0015, 2.0]])
    trip = load_trip(
        write_trip(
            tmp_path / "train",
            "00000051",
            nodes=nodes,
            arcs=arcs,
            track=track,
            route=np.array([0, 1]),
        )
    )
    network = NetworkDistance(trip)
    downstream = _candidate(trip.trip_id, 0, 1, 111.0)  # arc 1 -> 2
    upstream = _candidate(trip.trip_id, 1, 0, 111.0)  # arc 0 -> 1

    assert network.candidate_distance(downstream, upstream) is None


def test_emissions_are_normalised() -> None:
    gaussian = gaussian_log_emissions(np.array([0.0, 10.0]), sigma_m=20.0)
    assert np.exp(gaussian).sum() == pytest.approx(1.0)
    assert gaussian[0] > gaussian[1]

    classifier = classifier_log_emissions(np.array([0.9, 0.1]))
    assert np.exp(classifier).sum() == pytest.approx(1.0)


def test_viterbi_matches_hand_computed_optimum() -> None:
    emissions = [
        np.array([0.0, -1.0]),
        np.array([0.0, -2.0]),
        np.array([-2.0, 0.0]),
    ]
    free_transitions = [np.zeros((2, 2)), np.zeros((2, 2))]

    assert viterbi_log(emissions, free_transitions) == [0, 0, 1]

    penalised = [np.zeros((2, 2)), np.array([[0.0, -100.0], [-100.0, 0.0]])]
    assert viterbi_log(emissions, penalised) == [0, 0, 0]


def test_stitching_bridges_skipped_arcs(tmp_path: Path) -> None:
    trip = load_trip(
        write_trip(tmp_path / "train", "00000052", route=np.array([0, 1, 2]))
    )

    stitched, unbridged = stitch_arc_sequence(trip, [0, 2])

    assert stitched == [0, 1, 2]
    assert unbridged == 0


@pytest.mark.slow
def test_real_trip_hmm_route_is_contiguous_and_covers_observations() -> None:
    trip = next(iter_trips(config.TEST_DATA_DIR, "test"))
    candidates = generate_candidates(trip)

    result = decode_hmm(
        trip, candidates, config=HMMConfig(emission="gaussian")
    )

    predicted = result.predicted_arcs
    assert predicted.size > 0
    assert np.all(predicted >= 0) and np.all(predicted < trip.n_arcs)
    assert route_continuity(predicted, trip.arcs) >= 0.9
    assert np.mean(result.point_arc_ids >= 0) >= 0.95
