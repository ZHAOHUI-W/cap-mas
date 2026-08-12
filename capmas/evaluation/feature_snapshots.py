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
    rewrite = candidate.rewrite_report
    _validate_rewrite_lineage(candidate, fingerprint)
    features: dict[str, float | None] = {name: None for name in FEATURE_GROUPS_V1}
    refs: list[str] = []
    providers: dict[str, str] = {}
    evidence = candidate.evidence
    effective_map_version = map_version

    if evidence is not None:
        refs.extend(evidence.evidence_refs)
        if evidence.provider is not None:
            providers["evidence"] = evidence.provider
        available = set(evidence.available_metrics)
        for metric, value in (
            ("perception", evidence.perception),
            ("geometry", evidence.geometry),
            ("verifier", evidence.verifier),
        ):
            if metric in available and value is None:
                raise ValueError(f"declared {metric} evidence is missing")
        if available:
            _validate_evidence_scene(evidence.scene_version, candidate.parent_scene_version)

        if "perception" in available:
            assert evidence.perception is not None
            refs.extend(evidence.perception.evidence_refs)
            for name in _PERCEPTION_FEATURES:
                features[name] = getattr(evidence.perception, name)

        if "geometry" in available:
            assert evidence.geometry is not None
            geometry = evidence.geometry
            _validate_evidence_scene(geometry.scene_version, candidate.parent_scene_version)
            if geometry.candidate_fingerprint != fingerprint:
                raise ValueError("geometry candidate fingerprint does not match candidate")
            if map_version is not None and geometry.map_version != map_version:
                raise ValueError("geometry map version does not match collection context")
            if effective_map_version is None:
                effective_map_version = geometry.map_version
            providers["geometry"] = geometry.provider
            providers["geometry_version"] = geometry.provider_version
            refs.extend(reference.uri for reference in geometry.artifact_refs)
            for name in _GEOMETRY_FEATURES:
                dimension = getattr(geometry, name)
                features[name] = dimension.score if dimension.status != "unknown" else None

        if "verifier" in available:
            assert evidence.verifier is not None
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
    return CandidateFeatureSnapshot(
        episode_id=context.episode_id,
        episode_epoch=context.episode_epoch,
        family_id=context.family_id,
        candidate_id=candidate.candidate_id,
        candidate_fingerprint=fingerprint,
        scene_version=candidate.parent_scene_version,
        map_version=effective_map_version,
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


def _validate_rewrite_lineage(candidate: GraphCandidate, fingerprint: str) -> None:
    """Require reports to identify both the raw and effective candidate graphs."""

    rewrite = candidate.rewrite_report
    if rewrite.normalized_fingerprint != fingerprint:
        raise ValueError("rewrite normalized fingerprint does not match candidate")
    raw = candidate.raw_subgraph
    raw_fingerprint = subgraph_fingerprint(raw) if raw is not None else fingerprint
    if rewrite.raw_fingerprint != raw_fingerprint:
        raise ValueError("rewrite raw fingerprint does not match candidate")


def _validate_evidence_scene(evidence_scene_version: int | None, candidate_scene_version: int) -> None:
    if evidence_scene_version is None or evidence_scene_version != candidate_scene_version:
        raise ValueError("evidence scene version does not match candidate")


__all__ = ["FEATURE_GROUPS_V1", "capture_feature_snapshot"]
