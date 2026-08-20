from __future__ import annotations

from dataclasses import replace

from capmas.agents.arbiter import CandidateArbiter
from capmas.contracts.action import SkillCall
from capmas.contracts.candidates import (
    CandidateEvidence,
    EvidenceDimension,
    GeometryEvidence,
    GraphCandidate,
    subgraph_fingerprint,
)
from capmas.contracts.core import SkillRef
from capmas.contracts.graph import CheckpointSpec, MotionIntent, SubgraphNodeSpec, SubgraphSpec
from capmas.contracts.scene import SceneSnapshot


def _scene() -> SceneSnapshot:
    return SceneSnapshot("episode", 1, 4, 100, 101, {})


def _candidate(candidate_id: str, *, description: str = "candidate") -> GraphCandidate:
    node = SubgraphNodeSpec(
        "action",
        description,
        skill_calls=(SkillCall(SkillRef("noop", "1.0.0"), {}),),
        postconditions=("done",),
        motion_intent=MotionIntent("move", target_pose_wxyz_xyz=(1.0, 0, 0, 0, 0.4, 0, 0.3)),
    )
    subgraph = SubgraphSpec(
        "pick",
        "pick",
        description,
        (node,),
        (),
        "action",
        ("action",),
        ("action",),
        checkpoints=(CheckpointSpec("check", node.postconditions),),
    )
    return GraphCandidate(candidate_id, subgraph, 4, "policy", strategy="balanced")


def _dimension(name: str, status: str, score: float | None) -> EvidenceDimension:
    return EvidenceDimension(name, status, score, 0.5, status)


def _geometry(candidate: GraphCandidate, *, reachability: EvidenceDimension | None = None,
              collision: EvidenceDimension | None = None, clearance: EvidenceDimension | None = None,
              grasp: EvidenceDimension | None = None, map_version: int | None = 3) -> GeometryEvidence:
    return GeometryEvidence(
        grasp_quality=grasp or _dimension("grasp_quality", "pass", 0.8),
        reachability=reachability or _dimension("reachability", "pass", 0.8),
        clearance=clearance or _dimension("clearance", "pass", 0.8),
        collision_risk=collision or _dimension("collision_risk", "pass", 0.1),
        candidate_fingerprint=subgraph_fingerprint(candidate.subgraph),
        scene_version=4,
        map_version=map_version,
        map_backend="sparse_voxel",
        provider="test",
        provider_version="1",
        captured_at_ns=100,
        latency_ms=1.0,
    )


def _with_geometry(candidate: GraphCandidate, geometry: GeometryEvidence) -> GraphCandidate:
    return replace(
        candidate,
        evidence=CandidateEvidence(
            geometry=geometry,
            available_metrics=("geometry",),
            scene_version=4,
        ),
    )


def test_geometry_evidence_can_change_the_arbiter_winner() -> None:
    weak = _candidate("weak", description="weak")
    strong = _candidate("strong", description="strong")
    weak_geometry = _geometry(
        weak,
        grasp=_dimension("grasp_quality", "pass", 0.1),
        clearance=_dimension("clearance", "pass", 0.1),
    )
    strong_geometry = _geometry(strong)

    result = CandidateArbiter(current_map_version=3).select(
        (_with_geometry(weak, weak_geometry), _with_geometry(strong, strong_geometry)),
        _scene(),
    )

    assert result.selected is not None
    assert result.selected.candidate_id == "strong"
    assert result.score_breakdowns["strong"]["geometry"] > result.score_breakdowns["weak"]["geometry"]


def test_collision_failure_is_a_hard_gate_before_soft_scoring() -> None:
    candidate = _candidate("blocked")
    geometry = _geometry(
        candidate,
        collision=_dimension("collision_risk", "fail", 1.0),
    )

    result = CandidateArbiter(current_map_version=3).select(
        (_with_geometry(candidate, geometry),),
        _scene(),
    )

    assert result.selected is None
    assert result.rejections[0].code == "GEOMETRY_GATE"


def test_typed_semantic_abstention_selects_nothing() -> None:
    candidates = (_candidate("policy-0"), _candidate("safety"))

    result = CandidateArbiter().abstain(
        candidates,
        "CANDIDATE_SEMANTIC_EQUIVALENCE",
        "duplicate motion program",
    )

    assert result.selected is None
    assert result.considered == candidates
    assert result.selection_basis == "candidate_semantic_equivalence"
    assert result.rejections[0].candidate_id == "candidate-wave"
    assert result.rejections[0].code == "CANDIDATE_SEMANTIC_EQUIVALENCE"


def test_unknown_geometry_dimension_is_excluded_not_converted_to_zero() -> None:
    candidate = _candidate("unknown")
    geometry = _geometry(
        candidate,
        grasp=_dimension("grasp_quality", "unknown", None),
        reachability=_dimension("reachability", "unknown", None),
        clearance=_dimension("clearance", "unknown", None),
    )
    result = CandidateArbiter(current_map_version=3).select(
        (_with_geometry(candidate, geometry),),
        _scene(),
    )

    assert result.selected == _with_geometry(candidate, geometry)
    assert "geometry" in result.score_breakdowns["unknown"]
    assert "grasp_quality" not in result.score_breakdowns["unknown"]
    assert "reachability" not in result.score_breakdowns["unknown"]


def test_stale_geometry_fingerprint_and_map_are_rejected() -> None:
    candidate = _candidate("stale")
    geometry = _geometry(candidate, map_version=2)
    stale_fingerprint = replace(geometry, candidate_fingerprint="wrong")

    fingerprint_result = CandidateArbiter(current_map_version=3).select(
        (_with_geometry(candidate, stale_fingerprint),),
        _scene(),
    )
    assert fingerprint_result.selected is None
    assert fingerprint_result.rejections[0].code == "STALE_EVIDENCE"

    map_result = CandidateArbiter(current_map_version=3).select(
        (_with_geometry(candidate, geometry),),
        _scene(),
    )
    assert map_result.selected is None
    assert map_result.rejections[0].code == "STALE_EVIDENCE"
