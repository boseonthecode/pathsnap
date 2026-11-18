"""Versioned on-disk cache for per-trip candidates, features, and alignment.

The cache key is a hash of the feature specification, candidate/alignment
configuration, and package version. A cache whose manifest does not match the
requested version is treated as stale and rebuilt.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import pathsnap
from pathsnap import config
from pathsnap.candidates import Candidate, CandidateSet, generate_candidates
from pathsnap.features import FEATURE_NAMES, FeatureMatrix, extract_features
from pathsnap.io import Trip
from pathsnap.labels import AlignmentResult, align_observations


@dataclass(frozen=True)
class TripArtifacts:
    """Everything the training pipeline needs for one trip."""

    candidates: CandidateSet
    features: FeatureMatrix
    alignment: AlignmentResult
    from_cache: bool = False


def compute_cache_version(
    feature_names: Sequence[str] = FEATURE_NAMES,
    *,
    top_k: int = config.CANDIDATE_TOP_K,
    base_radius_m: float = config.CANDIDATE_RADIUS_M,
    max_radius_m: float = config.CANDIDATE_MAX_RADIUS_M,
    align_max_distance_m: float = config.ALIGN_MAX_DISTANCE_M,
) -> str:
    """Deterministic cache key derived from the feature and matching spec."""
    payload: dict[str, Any] = {
        "package_version": pathsnap.__version__,
        "feature_names": list(feature_names),
        "top_k": top_k,
        "base_radius_m": base_radius_m,
        "max_radius_m": max_radius_m,
        "align_max_distance_m": align_max_distance_m,
        "utm_zone_override": config.UTM_ZONE_OVERRIDE,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def build_trip_artifacts(trip: Trip) -> TripArtifacts:
    """Run alignment → candidates → features for one trip."""
    alignment = align_observations(trip)
    candidates = generate_candidates(trip, alignable=alignment.alignable)
    features = extract_features(trip, candidates)
    return TripArtifacts(candidates=candidates, features=features, alignment=alignment)


def _cache_paths(cache_dir: Path, version: str, trip_id: str) -> tuple[Path, Path]:
    run_dir = cache_dir / version
    return run_dir / f"{trip_id}.npz", run_dir / f"{trip_id}.json"


def save_trip_artifacts(
    artifacts: TripArtifacts,
    cache_dir: Path,
    version: str,
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> None:
    npz_path, manifest_path = _cache_paths(cache_dir, version, artifacts.candidates.trip_id)
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    candidates = artifacts.candidates
    features = artifacts.features
    alignment = artifacts.alignment
    np.savez_compressed(
        npz_path,
        candidate_obs=np.array([c.obs_index for c in candidates.candidates], dtype=np.int32),
        candidate_arc=np.array([c.arc_id for c in candidates.candidates], dtype=np.int32),
        candidate_distance=np.array([c.distance_m for c in candidates.candidates], dtype=np.float64),
        candidate_proj_lon=np.array([c.proj_lon for c in candidates.candidates], dtype=np.float64),
        candidate_proj_lat=np.array([c.proj_lat for c in candidates.candidates], dtype=np.float64),
        candidate_along=np.array([c.along_fraction for c in candidates.candidates], dtype=np.float64),
        candidate_heading_diff=np.array(
            [c.heading_diff_deg for c in candidates.candidates], dtype=np.float64
        ),
        candidate_rank=np.array([c.rank for c in candidates.candidates], dtype=np.int32),
        candidate_arc_length=np.array(
            [c.arc_length_m for c in candidates.candidates], dtype=np.float64
        ),
        feature_matrix=features.features,
        feature_observation_indices=features.observation_indices,
        feature_arc_ids=features.arc_ids,
        feature_ranks=features.ranks,
        alignment_positions=alignment.positions,
        alignment_arc_ids=alignment.arc_ids,
        alignment_distances=alignment.distances_m,
        alignment_alignable=alignment.alignable,
    )

    manifest = {
        "version": version,
        "trip_id": candidates.trip_id,
        "feature_names": list(feature_names),
        "candidate_meta": {
            "n_observations": candidates.n_observations,
            "n_alignable": candidates.n_alignable,
            "miss_count": candidates.miss_count,
            "miss_rate": candidates.miss_rate,
        },
        "alignment_meta": {
            "coverage": alignment.coverage,
            "backward_steps": alignment.backward_steps,
            "low_coverage": alignment.low_coverage,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_trip_artifacts(
    trip_id: str,
    cache_dir: Path,
    version: str,
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> TripArtifacts | None:
    """Load a matching cache entry, or return ``None`` if missing/stale/corrupt."""
    npz_path, manifest_path = _cache_paths(cache_dir, version, trip_id)
    if not npz_path.is_file() or not manifest_path.is_file():
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if manifest.get("version") != version:
        return None
    if tuple(manifest.get("feature_names", ())) != tuple(feature_names):
        return None

    try:
        data = np.load(npz_path)
        candidate_meta = manifest["candidate_meta"]
        alignment_meta = manifest["alignment_meta"]

        candidates = tuple(
            Candidate(
                trip_id=trip_id,
                obs_index=int(data["candidate_obs"][i]),
                arc_id=int(data["candidate_arc"][i]),
                distance_m=float(data["candidate_distance"][i]),
                proj_lon=float(data["candidate_proj_lon"][i]),
                proj_lat=float(data["candidate_proj_lat"][i]),
                along_fraction=float(data["candidate_along"][i]),
                heading_diff_deg=float(data["candidate_heading_diff"][i]),
                rank=int(data["candidate_rank"][i]),
                arc_length_m=float(data["candidate_arc_length"][i]),
            )
            for i in range(data["candidate_arc"].shape[0])
        )
        candidate_set = CandidateSet(
            trip_id=trip_id,
            candidates=candidates,
            n_observations=int(candidate_meta["n_observations"]),
            n_alignable=int(candidate_meta["n_alignable"]),
            miss_count=int(candidate_meta["miss_count"]),
            miss_rate=float(candidate_meta["miss_rate"]),
        )
        feature_matrix = FeatureMatrix(
            trip_id=trip_id,
            feature_names=tuple(feature_names),
            observation_indices=data["feature_observation_indices"],
            arc_ids=data["feature_arc_ids"],
            ranks=data["feature_ranks"],
            features=data["feature_matrix"],
        )
        alignment = AlignmentResult(
            trip_id=trip_id,
            positions=data["alignment_positions"],
            arc_ids=data["alignment_arc_ids"],
            distances_m=data["alignment_distances"],
            alignable=data["alignment_alignable"],
            coverage=float(alignment_meta["coverage"]),
            backward_steps=int(alignment_meta["backward_steps"]),
            low_coverage=bool(alignment_meta["low_coverage"]),
        )
    except (OSError, KeyError, ValueError):
        return None

    return TripArtifacts(
        candidates=candidate_set,
        features=feature_matrix,
        alignment=alignment,
        from_cache=True,
    )


def load_or_build_trip_artifacts(
    trip: Trip,
    cache_dir: Path | None = None,
    version: str | None = None,
    use_cache: bool = True,
) -> TripArtifacts:
    """Return cached artifacts when the version matches, otherwise build and save."""
    cache_dir = config.get_paths().cache_dir if cache_dir is None else cache_dir
    version = compute_cache_version() if version is None else version

    if use_cache:
        cached = load_trip_artifacts(trip.trip_id, cache_dir, version)
        if cached is not None:
            return cached

    artifacts = build_trip_artifacts(trip)
    save_trip_artifacts(artifacts, cache_dir, version)
    return artifacts
