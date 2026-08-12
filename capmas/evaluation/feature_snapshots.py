"""Leakage-safe, decision-time candidate feature projections."""

from __future__ import annotations

import time
from collections.abc import Callable

from capmas.contracts.calibration import (
    CalibrationCollectionContext,
    CandidateFeatureSnapshot,
)
from capmas.contracts.candidates import GraphCandidate, subgraph_fingerprint
from capmas.verification.evidence import summarize_verifier_results

FEATURE_GROUPS_V1 = {
    "scene_freshness": "scene_grounding",
    "scene_confidence": "scene_grounding",
    "target_visibility": "scene_grounding",
    "track_confidence": "scene_grounding",
    "identity_confidence": "scene_grounding",
    "pose_reliability": "scene_grounding",
    "grasp_quality": "scene_grounding",
    "reachability": "scene_grounding",
    "clearance": "scene_grounding",
    "static_verifier_pass_rate": "action_feasibility",
    "static_verifier_coverage": "action_feasibility",
    "rehearsal_success_rate": "action_feasibility",
    "collision_risk": "cost_risk",
    "expected_latency_ms": "cost_risk",
    "recovery_cost": "cost_risk",
}

_PERCEPTION_FEATURES = (
    "scene_freshness",
    "scene_confidence",
    "target_visibility",
    "track_confidence",
    "identity_confidence",
    "pose_reliability",
)
_GEOMETRY_FEATURES = (
    "grasp_quality",
    "reachability",
    "clearance",
    "collision_risk",
)


def capture_feature_snapshot(
    candidate: GraphCandidate,
    context: CalibrationCollectionContext,
    *,
    map_version: int | None = None,
    clock: Callable[[], int] = time.time_ns,
) -> CandidateFeatureSnapshot:
    """Project only evidence available when the Arbiter made its decision."""

    fingerprint = subgraph_fingerprint(candidate.subgraph)
    features: dict[str, float | None] = {name: None for name in FEATURE_GROUPS_V1}
    refs: list[str] = []
    providers: dict[str, str] = {}
    evidence = candidate.evidence

    if evidence is not None:
        _validate_evidence_scene(evidence.scene_version, candidate.parent_scene_version)
        refs.extend(evidence.evidence_refs)
        if evidence.provider is not None:
            providers["evidence"] = evidence.provider
        available = set(evidence.available_metrics)

        if "perception" in available:
            if evidence.perception is None:
                raise ValueError("declared perception evidence is missing")
            refs.extend(evidence.perception.evidence_refs)
            for name in _PERCEPTION_FEATURES:
                features[name] = getattr(evidence.perception, name)

        if "geometry" in available:
            if evidence.geometry is None:
                raise ValueError("declared geometry evidence is missing")
            geometry = evidence.geometry
            _validate_evidence_scene(geometry.scene_version, candidate.parent_scene_version)
            if geometry.candidate_fingerprint != fingerprint:
                raise ValueError("geometry candidate fingerprint does not match candidate")
            if map_version is not None and geometry.map_version != map_version:
                raise ValueError("geometry map version does not match collection context")
            providers["geometry"] = geometry.provider
            refs.extend(reference.uri for reference in geometry.artifact_refs)
            for name in _GEOMETRY_FEATURES:
                dimension = getattr(geometry, name)
                features[name] = dimension.score if dimension.status != "unknown" else None

        if "verifier" in available:
            if evidence.verifier is None:
                raise ValueError("declared verifier evidence is missing")
            verifier = evidence.verifier
            _validate_evidence_scene(verifier.scene_version, candidate.parent_scene_version)
            if verifier.candidate_fingerprint != fingerprint:
                raise ValueError("verifier candidate fingerprint does not match candidate")
            providers["verifier"] = verifier.provider
            refs.extend(
                reference
                for result in verifier.static_results
                for reference in result.evidence_refs
            )
            pass_rate, coverage = summarize_verifier_results(verifier.static_results)
            if coverage > 0.0:
                features["static_verifier_pass_rate"] = pass_rate
                features["static_verifier_coverage"] = coverage

        for metric, feature in (
            ("rehearsal", "rehearsal_success_rate"),
            ("latency", "expected_latency_ms"),
            ("recovery", "recovery_cost"),
        ):
            if metric in available:
                features[feature] = getattr(evidence, feature)

    statuses = {
        name: "present" if value is not None else "unknown"
        for name, value in features.items()
    }
    rewrite = candidate.rewrite_report
    return CandidateFeatureSnapshot(
        episode_id=context.episode_id,
        episode_epoch=context.episode_epoch,
        family_id=context.family_id,
        candidate_id=candidate.candidate_id,
        candidate_fingerprint=fingerprint,
        scene_version=candidate.parent_scene_version,
        map_version=map_version,
        feature_schema_version=context.feature_schema_version,
        captured_at_ns=clock(),
        collection_lane=context.collection_lane,
        features=features,
        feature_status=statuses,
        correlation_groups=FEATURE_GROUPS_V1,
        memory_skill_version=context.memory_skill_version,
        robot_skill_version=context.robot_skill_version,
        selection_probability=candidate.confidence,
        evidence_refs=tuple(dict.fromkeys(refs)),
        evidence_providers=providers,
        rewrite_metadata={
            "raw_fingerprint": rewrite.raw_fingerprint,
            "normalized_fingerprint": rewrite.normalized_fingerprint,
            "changed": rewrite.changed,
            "rewrite_count": rewrite.rewrite_count,
            "operations": list(rewrite.operations),
        },
    )


def _validate_evidence_scene(evidence_scene_version: int | None, candidate_scene_version: int) -> None:
    if evidence_scene_version is not None and evidence_scene_version != candidate_scene_version:
        raise ValueError("evidence scene version does not match candidate")


__all__ = ["FEATURE_GROUPS_V1", "capture_feature_snapshot"]
