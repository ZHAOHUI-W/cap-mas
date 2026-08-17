"""Deterministic, leakage-safe reduction of P5.6 decision-time evidence."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from capmas.contracts.calibration import FEATURE_SCHEMA_VERSION, CandidateFeatureSnapshot
from capmas.evaluation.feature_snapshots import FEATURE_GROUPS_V1

REDUCTION_POLICY_VERSION = "p56b.reduction.v1"

_SCENE_SUPPORT_FEATURES = (
    "scene_freshness",
    "scene_confidence",
    "target_visibility",
    "track_confidence",
    "identity_confidence",
    "pose_reliability",
    "grasp_quality",
    "reachability",
    "clearance",
)
_FIXED_DIMENSIONS = (
    "scene_grounding",
    "action_feasibility",
    "collision_risk",
    "expected_latency_risk",
    "recovery_cost_risk",
)


@dataclass(frozen=True)
class ReducedDimension:
    """One auditable feature after correlation-group reduction."""

    name: str
    value: float | None
    status: Literal["present", "unknown"]
    sources: tuple[str, ...]
    coverage: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("reduced dimension name must not be empty")
        if self.status not in {"present", "unknown"}:
            raise ValueError("reduced dimension status is invalid")
        if not 0.0 <= self.coverage <= 1.0:
            raise ValueError("reduced dimension coverage must be in [0, 1]")
        if self.value is not None and not 0.0 <= self.value <= 1.0:
            raise ValueError("reduced dimension value must be in [0, 1]")
        if self.status == "present" and self.value is None:
            raise ValueError("present reduced dimension must have a value")
        if self.status == "unknown" and (self.value is not None or self.sources):
            raise ValueError("unknown reduced dimension must have no value or sources")
        if self.status == "present" and not self.sources:
            raise ValueError("present reduced dimension must retain sources")

    def to_dict(self) -> dict[str, object]:
        return {
            "coverage": self.coverage,
            "name": self.name,
            "sources": list(self.sources),
            "status": self.status,
            "value": self.value,
        }


@dataclass(frozen=True)
class ReducedFeatureVector:
    """Versioned reduction of a candidate's immutable feature snapshot."""

    episode_id: str
    candidate_id: str
    candidate_fingerprint: str
    family_id: str
    feature_schema_version: str
    dimensions: tuple[ReducedDimension, ...]
    policy_version: str = REDUCTION_POLICY_VERSION

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (
                self.episode_id,
                self.candidate_id,
                self.candidate_fingerprint,
                self.family_id,
                self.feature_schema_version,
                self.policy_version,
            )
        ):
            raise ValueError("reduced feature vector identity must be non-empty")
        if tuple(dimension.name for dimension in self.dimensions) != _FIXED_DIMENSIONS:
            raise ValueError("reduced feature dimensions do not match the V1 policy")

    def dimension(self, name: str) -> ReducedDimension:
        for dimension in self.dimensions:
            if dimension.name == name:
                return dimension
        raise KeyError(f"unknown reduced dimension: {name}")

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_fingerprint": self.candidate_fingerprint,
            "candidate_id": self.candidate_id,
            "dimensions": [dimension.to_dict() for dimension in self.dimensions],
            "episode_id": self.episode_id,
            "family_id": self.family_id,
            "feature_schema_version": self.feature_schema_version,
            "policy_version": self.policy_version,
        }


def reduce_feature_snapshot(snapshot: CandidateFeatureSnapshot) -> ReducedFeatureVector:
    """Reduce only evidence captured at the Arbiter decision boundary."""

    _validate_snapshot(snapshot)
    dimensions = (
        _scene_grounding(snapshot.features, snapshot.feature_status),
        _action_feasibility(snapshot.features, snapshot.feature_status),
        _optional_bounded(
            "collision_risk",
            snapshot.features["collision_risk"],
            snapshot.feature_status["collision_risk"],
            ("collision_risk",),
        ),
        _optional_transformed(
            "expected_latency_risk",
            snapshot.features["expected_latency_ms"],
            snapshot.feature_status["expected_latency_ms"],
            "expected_latency_ms",
            _latency_risk,
        ),
        _optional_transformed(
            "recovery_cost_risk",
            snapshot.features["recovery_cost"],
            snapshot.feature_status["recovery_cost"],
            "recovery_cost",
            _recovery_risk,
        ),
    )
    return ReducedFeatureVector(
        episode_id=snapshot.episode_id,
        candidate_id=snapshot.candidate_id,
        candidate_fingerprint=snapshot.candidate_fingerprint,
        family_id=snapshot.family_id,
        feature_schema_version=snapshot.feature_schema_version,
        dimensions=dimensions,
    )


def _validate_snapshot(snapshot: CandidateFeatureSnapshot) -> None:
    if snapshot.feature_schema_version != FEATURE_SCHEMA_VERSION:
        raise ValueError("feature schema version is not supported by the V1 reducer")
    if dict(snapshot.correlation_groups) != FEATURE_GROUPS_V1:
        raise ValueError("correlation groups do not match the V1 reducer policy")
    for name, status in snapshot.feature_status.items():
        value = snapshot.features[name]
        if status == "invalid":
            raise ValueError(f"feature {name} is invalid")
        if status == "unknown" and value is not None:
            raise ValueError(f"unknown feature {name} must remain None")
        if status == "present" and value is None:
            raise ValueError(f"present feature {name} must have a value")
        if value is not None and not math.isfinite(value):
            raise ValueError(f"feature {name} must be finite")

    bounded = (*_SCENE_SUPPORT_FEATURES, "static_verifier_pass_rate", "static_verifier_coverage", "rehearsal_success_rate", "collision_risk")
    for name in bounded:
        value = snapshot.features[name]
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"feature {name} must be in [0, 1]")
    for name in ("expected_latency_ms", "recovery_cost"):
        value = snapshot.features[name]
        if value is not None and value < 0.0:
            raise ValueError(f"feature {name} must not be negative")

    coverage_status = snapshot.feature_status["static_verifier_coverage"]
    pass_status = snapshot.feature_status["static_verifier_pass_rate"]
    if coverage_status == "present" and pass_status != "present":
        raise ValueError("static verifier coverage requires a pass rate")
    if coverage_status == "unknown" and pass_status == "present":
        raise ValueError("static verifier pass rate requires coverage")


def _scene_grounding(
    features: Mapping[str, float | None], statuses: Mapping[str, str]
) -> ReducedDimension:
    known = tuple(
        name for name in _SCENE_SUPPORT_FEATURES if statuses[name] == "present"
    )
    if not known:
        return _unknown("scene_grounding")
    value = sum(_present_value(features, name) for name in known) / len(known)
    return ReducedDimension(
        name="scene_grounding",
        value=value,
        status="present",
        sources=known,
        coverage=len(known) / len(_SCENE_SUPPORT_FEATURES),
    )


def _action_feasibility(
    features: Mapping[str, float | None], statuses: Mapping[str, str]
) -> ReducedDimension:
    if statuses["static_verifier_coverage"] == "present":
        coverage = _present_value(features, "static_verifier_coverage")
        if coverage > 0.0:
            return ReducedDimension(
                name="action_feasibility",
                value=_present_value(features, "static_verifier_pass_rate"),
                status="present",
                sources=("static_verifier_pass_rate",),
                coverage=coverage,
            )
    return _optional_bounded(
        "action_feasibility",
        features["rehearsal_success_rate"],
        statuses["rehearsal_success_rate"],
        ("rehearsal_success_rate",),
    )


def _optional_bounded(
    name: str,
    value: float | None,
    status: str,
    sources: tuple[str, ...],
) -> ReducedDimension:
    if status == "unknown":
        return _unknown(name)
    if status != "present" or value is None:
        raise ValueError(f"feature {sources[0]} has an invalid status")
    return ReducedDimension(name, value, "present", sources, 1.0)


def _optional_transformed(
    name: str,
    value: float | None,
    status: str,
    source: str,
    transform: callable,
) -> ReducedDimension:
    if status == "unknown":
        return _unknown(name)
    if status != "present" or value is None:
        raise ValueError(f"feature {source} has an invalid status")
    return ReducedDimension(name, transform(value), "present", (source,), 1.0)


def _latency_risk(milliseconds: float) -> float:
    return 1.0 - math.exp(-milliseconds / 1000.0)


def _recovery_risk(cost: float) -> float:
    return 1.0 - math.exp(-cost)


def _present_value(features: Mapping[str, float | None], name: str) -> float:
    value = features[name]
    if value is None:
        raise ValueError(f"present feature {name} must have a value")
    return value


def _unknown(name: str) -> ReducedDimension:
    return ReducedDimension(name, None, "unknown", (), 0.0)


__all__ = [
    "REDUCTION_POLICY_VERSION",
    "ReducedDimension",
    "ReducedFeatureVector",
    "reduce_feature_snapshot",
]
