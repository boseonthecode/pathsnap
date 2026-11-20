"""Tests for versioned artifact caching."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pathsnap.cache import (
    compute_cache_version,
    load_or_build_trip_artifacts,
    load_trip_artifacts,
)
from pathsnap.io import load_trip


def test_second_build_reuses_cache(tiny_trip_dir: Path, tmp_path: Path) -> None:
    trip = load_trip(tiny_trip_dir)
    cache_dir = tmp_path / "cache"

    first = load_or_build_trip_artifacts(trip, cache_dir=cache_dir)
    second = load_or_build_trip_artifacts(trip, cache_dir=cache_dir)

    assert first.from_cache is False
    assert second.from_cache is True
    np.testing.assert_allclose(first.features.features, second.features.features)
    np.testing.assert_array_equal(
        first.candidates.candidates[0].arc_id, second.candidates.candidates[0].arc_id
    )


def test_cache_version_is_deterministic_and_feature_sensitive() -> None:
    assert compute_cache_version() == compute_cache_version()
    assert compute_cache_version() != compute_cache_version(
        feature_names=("distance_m", "heading_diff_deg")
    )


def test_mismatched_version_is_treated_as_stale(
    tiny_trip_dir: Path, tmp_path: Path
) -> None:
    trip = load_trip(tiny_trip_dir)
    cache_dir = tmp_path / "cache"
    version = compute_cache_version()

    load_or_build_trip_artifacts(trip, cache_dir=cache_dir, version=version)

    assert load_trip_artifacts(trip.trip_id, cache_dir, "deadbeefdeadbeef") is None


def test_cached_features_are_identical_to_fresh_build(
    tiny_trip_dir: Path, tmp_path: Path
) -> None:
    trip = load_trip(tiny_trip_dir)
    cache_dir = tmp_path / "cache"

    fresh = load_or_build_trip_artifacts(
        trip, cache_dir=cache_dir, use_cache=False
    )
    cached = load_or_build_trip_artifacts(trip, cache_dir=cache_dir)

    assert cached.from_cache is True
    np.testing.assert_array_equal(fresh.features.features, cached.features.features)
    np.testing.assert_array_equal(
        fresh.features.observation_indices, cached.features.observation_indices
    )
    np.testing.assert_array_equal(fresh.features.arc_ids, cached.features.arc_ids)
    np.testing.assert_array_equal(
        fresh.alignment.positions, cached.alignment.positions
    )
    assert fresh.candidates.miss_rate == cached.candidates.miss_rate


def test_corrupt_cache_is_rebuilt(tiny_trip_dir: Path, tmp_path: Path) -> None:
    trip = load_trip(tiny_trip_dir)
    cache_dir = tmp_path / "cache"
    version = compute_cache_version()
    load_or_build_trip_artifacts(trip, cache_dir=cache_dir, version=version)

    npz_path = cache_dir / version / f"{trip.trip_id}.npz"
    npz_path.write_bytes(b"not a valid npz")

    rebuilt = load_or_build_trip_artifacts(trip, cache_dir=cache_dir, version=version)

    assert rebuilt.from_cache is False
    assert rebuilt.features.n_rows > 0


def test_truncated_cache_archive_is_a_miss(tiny_trip_dir: Path, tmp_path: Path) -> None:
    from pathsnap.cache import compute_cache_version

    trip = load_trip(tiny_trip_dir)
    cache_dir = tmp_path / "cache"
    version = compute_cache_version()
    cached = load_or_build_trip_artifacts(trip, cache_dir=cache_dir, version=version)
    assert cached.from_cache is False

    archive = cache_dir / version / f"{trip.trip_id}.npz"
    archive.write_bytes(archive.read_bytes()[:64])

    assert load_trip_artifacts(trip.trip_id, cache_dir, version) is None
