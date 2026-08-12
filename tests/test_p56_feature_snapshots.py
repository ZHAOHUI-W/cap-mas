from __future__ import annotations

from dataclasses import replace

import pytest

from capmas.contracts.action import SkillCall
from capmas.contracts.calibration import CalibrationCollectionContext
from capmas.contracts.candidates import (
    CandidateEvidence,
    CandidateRewriteReport,
    EvidenceDimension,
    GeometryEvidence,
    GraphCandidate,
    PerceptionEvidence,
    rewrite_report_for,
    subgraph_fingerprint,
)
from capmas.contracts.core import SkillRef
from capmas.contracts.graph import SubgraphNodeSpec, SubgraphSpec
from capmas.evaluation.feature_snapshots import (
    FEATURE_GROUPS_V1,
    capture_feature_snapshot,
)
from capmas.verification.evidence import VerifierEvidence, VerifierPredicateEvidence


def _candidate(evidence: CandidateEvidence | None = None) -> GraphCandidate:
    node = SubgraphNodeSpec(
        node_id="act",
        description="act",
        skill_calls=(SkillCall(SkillRef("noop", "1.0.0"), {}),),
        postconditions=("done",),
        proposed_by="test",
    )
    subgraph = SubgraphSpec(
        subgraph_id="pick",
        subgoal_id="pick",
        description="pick",
        nodes=(node,),
        edges=(),
        entry_node="act",
        success_nodes=("act",),
        failure_nodes=("act",),
    )
    return GraphCandidate(
        candidate_id="candidate-a",
        subgraph=subgraph,
        parent_scene_version=7,
        producer_agent="test",
        evidence=evidence,
        rewrite_report=rewrite_report_for(subgraph, subgraph),
    )


def _context() -> CalibrationCollectionContext:
    return CalibrationCollectionContext(
        episode_id="episode-1",
        episode_epoch=2,
        family_id="family-1",
        feature_schema_version="p56.feature.v1",
        memory_skill_version="memory-v1",
        robot_skill_version="robot-v1",
    )


def _dimension(name: str, status: str, score: float | None) -> EvidenceDimension:
    return EvidenceDimension(name, status, score, None, "observed")  # type: ignore[arg-type]


def _mixed_evidence() -> CandidateEvidence:
    candidate = _candidate()
    fingerprint = subgraph_fingerprint(candidate.subgraph)
    verifier = VerifierEvidence(
        candidate_fingerprint=fingerprint,
        scene_version=7,
        pass_rate=1 / 3,
        coverage=1.0,
        provider="typed-verifier-v1",
        captured_at_ns=20,
        static_results=(
            VerifierPredicateEvidence("static-pass", "static", "pass", 1.0, "ok"),
            VerifierPredicateEvidence("static-fail", "static", "fail", 1.0, "blocked"),
        ),
        dynamic_results=(
            VerifierPredicateEvidence("dynamic-fail", "dynamic", "fail", 1.0, "failed"),
        ),
    )
    return CandidateEvidence(
        verifier_pass_rate=1 / 3,
        rehearsal_success_rate=0.75,
        ood_success_rate=1.0,
        expected_latency_ms=12.5,
        recovery_cost=3.0,
        evidence_refs=("evidence://base",),
        perception=PerceptionEvidence(
            scene_freshness=0.9,
            scene_confidence=0.8,
            target_visibility=0.7,
            track_confidence=0.6,
            identity_confidence=0.5,
            pose_reliability=0.4,
            evidence_refs=("perception://frame",),
        ),
        geometry=GeometryEvidence(
            grasp_quality=_dimension("grasp_quality", "pass", 0.9),
            reachability=_dimension("reachability", "pass", 0.8),
            clearance=_dimension("clearance", "unknown", None),
            collision_risk=_dimension("collision_risk", "fail", 0.2),
            candidate_fingerprint=fingerprint,
            scene_version=7,
            map_version=3,
            map_backend="local-map",
            provider="geometry-v1",
            provider_version="1.0",
            captured_at_ns=21,
            latency_ms=4.0,
            artifact_refs=(),
        ),
        available_metrics=(
            "perception",
            "geometry",
            "verifier",
            "rehearsal",
            "latency",
            "recovery",
        ),
        scene_version=7,
        provider="evidence-envelope-v1",
        captured_at_ns=22,
        verifier=verifier,
    )


def test_feature_snapshot_uses_static_verifier_and_excludes_dynamic_and_ood() -> None:
    candidate = _candidate(_mixed_evidence())

    snapshot = capture_feature_snapshot(candidate, _context(), map_version=3, clock=lambda: 50)

    assert snapshot.features["static_verifier_pass_rate"] == 0.5
    assert snapshot.features["static_verifier_coverage"] == 1.0
    assert "dynamic_verifier_pass_rate" not in snapshot.features
    assert "ood_success_rate" not in snapshot.features
    assert set(snapshot.features) == set(FEATURE_GROUPS_V1)
    assert snapshot.captured_at_ns == 50
    assert "evidence://base" in snapshot.evidence_refs
    assert snapshot.evidence_providers["geometry"] == "geometry-v1"


def test_feature_groups_v1_is_the_exact_published_schema() -> None:
    assert FEATURE_GROUPS_V1 == {
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


def test_feature_snapshot_preserves_unknown_dimensions() -> None:
    snapshot = capture_feature_snapshot(_candidate(), _context())

    assert snapshot.features["scene_freshness"] is None
    assert snapshot.feature_status["scene_freshness"] == "unknown"
    assert snapshot.correlation_groups["scene_freshness"] == "scene_grounding"
    assert all(value is None for value in snapshot.features.values())


def test_snapshot_rewrite_lineage_requires_populated_matching_fingerprints() -> None:
    candidate = _candidate()
    fingerprint = subgraph_fingerprint(candidate.subgraph)
    empty_default = replace(
        candidate,
        rewrite_report=CandidateRewriteReport("", ""),
    )

    with pytest.raises(ValueError, match="rewrite normalized fingerprint"):
        capture_feature_snapshot(empty_default, _context())

    mismatched = replace(
        candidate,
        rewrite_report=CandidateRewriteReport(
            raw_fingerprint=fingerprint,
            normalized_fingerprint="f" * 64,
        ),
    )
    with pytest.raises(ValueError, match="rewrite normalized fingerprint"):
        capture_feature_snapshot(mismatched, _context())

    mismatched_raw = replace(
        candidate,
        raw_subgraph=candidate.subgraph,
        rewrite_report=CandidateRewriteReport(
            raw_fingerprint="e" * 64,
            normalized_fingerprint=fingerprint,
        ),
    )
    with pytest.raises(ValueError, match="rewrite raw fingerprint"):
        capture_feature_snapshot(mismatched_raw, _context())

    without_rewrite = replace(
        candidate,
        rewrite_report=CandidateRewriteReport(
            raw_fingerprint=fingerprint,
            normalized_fingerprint=fingerprint,
        ),
    )
    snapshot = capture_feature_snapshot(without_rewrite, _context())

    assert snapshot.candidate_fingerprint == fingerprint
    assert snapshot.rewrite_metadata["raw_fingerprint"] == fingerprint
    assert snapshot.rewrite_metadata["normalized_fingerprint"] == fingerprint


def test_snapshot_geometry_lineage_inherits_map_and_preserves_provider_version() -> None:
    snapshot = capture_feature_snapshot(_candidate(_mixed_evidence()), _context())

    assert snapshot.map_version == 3
    assert snapshot.evidence_providers["geometry"] == "geometry-v1"
    assert snapshot.evidence_providers["geometry_version"] == "1.0"


def test_snapshot_rejects_declared_evidence_without_aggregate_scene_version() -> None:
    evidence = CandidateEvidence(
        available_metrics=("rehearsal",),
        rehearsal_success_rate=1.0,
    )

    with pytest.raises(ValueError, match="evidence scene version"):
        capture_feature_snapshot(_candidate(evidence), _context())


def test_static_verifier_with_only_unknown_results_remains_unknown() -> None:
    candidate = _candidate()
    fingerprint = subgraph_fingerprint(candidate.subgraph)
    verifier = VerifierEvidence(
        candidate_fingerprint=fingerprint,
        scene_version=7,
        pass_rate=1.0,
        coverage=0.5,
        provider="typed-verifier-v1",
        captured_at_ns=20,
        static_results=(
            VerifierPredicateEvidence("unobserved", "static", "unknown", None, "unknown"),
        ),
        dynamic_results=(
            VerifierPredicateEvidence("dynamic-pass", "dynamic", "pass", 1.0, "ok"),
        ),
    )
    evidence = CandidateEvidence(
        verifier_pass_rate=1.0,
        available_metrics=("verifier",),
        verifier=verifier,
        scene_version=7,
    )

    snapshot = capture_feature_snapshot(_candidate(evidence), _context())

    assert snapshot.features["static_verifier_pass_rate"] is None
    assert snapshot.features["static_verifier_coverage"] is None
    assert snapshot.feature_status["static_verifier_pass_rate"] == "unknown"


def test_snapshot_rejects_scene_or_fingerprint_mismatch() -> None:
    stale = CandidateEvidence(
        available_metrics=("rehearsal",),
        rehearsal_success_rate=1.0,
        scene_version=6,
    )

    with pytest.raises(ValueError, match="scene version"):
        capture_feature_snapshot(_candidate(stale), _context())

    wrong_geometry = _mixed_evidence()
    geometry = wrong_geometry.geometry
    assert geometry is not None
    wrong_geometry = CandidateEvidence(
        **{
            **wrong_geometry.__dict__,
            "geometry": GeometryEvidence(
                grasp_quality=geometry.grasp_quality,
                reachability=geometry.reachability,
                clearance=geometry.clearance,
                collision_risk=geometry.collision_risk,
                candidate_fingerprint="f" * 64,
                scene_version=7,
                map_version=3,
                map_backend=geometry.map_backend,
                provider=geometry.provider,
                provider_version=geometry.provider_version,
                captured_at_ns=geometry.captured_at_ns,
                latency_ms=geometry.latency_ms,
            ),
        }
    )
    with pytest.raises(ValueError, match="geometry candidate fingerprint"):
        capture_feature_snapshot(_candidate(wrong_geometry), _context(), map_version=3)

    with pytest.raises(ValueError, match="geometry map version"):
        capture_feature_snapshot(_candidate(_mixed_evidence()), _context(), map_version=4)

    verifier = wrong_geometry.verifier
    assert verifier is not None
    wrong_verifier = CandidateEvidence(
        **{
            **wrong_geometry.__dict__,
            "geometry": geometry,
            "verifier": VerifierEvidence(
                candidate_fingerprint="e" * 64,
                scene_version=verifier.scene_version,
                pass_rate=verifier.pass_rate,
                coverage=verifier.coverage,
                provider=verifier.provider,
                captured_at_ns=verifier.captured_at_ns,
                static_results=verifier.static_results,
                dynamic_results=verifier.dynamic_results,
                source_verification=verifier.source_verification,
            ),
        }
    )
    with pytest.raises(ValueError, match="verifier candidate fingerprint"):
        capture_feature_snapshot(_candidate(wrong_verifier), _context(), map_version=3)


def test_declared_typed_lane_missing_object_fails_closed() -> None:
    with pytest.raises(ValueError, match="perception"):
        capture_feature_snapshot(
            _candidate(CandidateEvidence(available_metrics=("perception",))),
            _context(),
        )

    with pytest.raises(ValueError, match="verifier"):
        capture_feature_snapshot(
            _candidate(CandidateEvidence(available_metrics=("verifier",))),
            _context(),
        )
