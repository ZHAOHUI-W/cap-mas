from __future__ import annotations

import pytest

from capmas.contracts.action import SkillCall
from capmas.contracts.candidates import (
    CandidateEvidence,
    EvidenceDimension,
    GeometryEvidence,
)
from capmas.contracts.core import ArtifactRef, SkillRef
from capmas.contracts.graph import (
    CheckpointSpec,
    MotionIntent,
    SubgraphNodeSpec,
    SubgraphSpec,
)
from capmas.graph.serialization import local_subgraph_from_dict, local_subgraph_to_dict


def _subgraph(*, motion_intent: MotionIntent | None) -> SubgraphSpec:
    node = SubgraphNodeSpec(
        node_id="pick-action",
        description="pick the bowl",
        skill_calls=(
            SkillCall(
                SkillRef("sample_grasp_pose", "1.0.0"),
                {"object_name": "bowl", "use_multiview": True},
            ),
        ),
        postconditions=("object_in_gripper(bowl)",),
        motion_intent=motion_intent,
    )
    return SubgraphSpec(
        subgraph_id="pick",
        subgoal_id="pick",
        description="pick the bowl",
        nodes=(node,),
        edges=(),
        entry_node="pick-action",
        success_nodes=("pick-action",),
        failure_nodes=("pick-action",),
        checkpoints=(CheckpointSpec("pick-check", node.postconditions),),
    )


def test_motion_intent_round_trip_through_strict_graph_codec() -> None:
    intent = MotionIntent(
        kind="grasp",
        object_track_id="bowl-track",
        approach_vector_xyz=(0.0, 0.0, -1.0),
        standoff_m=0.04,
        target_pose_wxyz_xyz=(1.0, 0.0, 0.0, 0.0, 0.4, 0.1, 0.2),
    )

    encoded = local_subgraph_to_dict(_subgraph(motion_intent=intent))
    restored = local_subgraph_from_dict(encoded)

    assert restored.nodes[0].motion_intent == intent
    assert encoded["subgraph"]["nodes"][0]["motion_intent"]["kind"] == "grasp"


def test_evidence_dimension_requires_three_state_status_and_bounded_score() -> None:
    assert EvidenceDimension("grasp_quality", "pass", 0.9, 0.5, "aligned")
    assert EvidenceDimension("reachability", "fail", 0.0, 0.5, "outside workspace")
    unknown = EvidenceDimension("clearance", "unknown", None, None, "map timeout")
    assert unknown.score is None

    with pytest.raises(ValueError, match="status"):
        EvidenceDimension("clearance", "missing", None, None, "bad status")
    with pytest.raises(ValueError, match="score"):
        EvidenceDimension("clearance", "pass", 1.2, None, "bad score")
    with pytest.raises(ValueError, match="unknown dimension score"):
        EvidenceDimension("clearance", "unknown", 0.0, None, "unknown is not zero")


def test_geometry_evidence_preserves_candidate_scene_map_and_provider_provenance() -> None:
    geometry = GeometryEvidence(
        grasp_quality=EvidenceDimension("grasp_quality", "pass", 0.8, 0.5, "aligned"),
        reachability=EvidenceDimension("reachability", "unknown", None, None, "no ik"),
        clearance=EvidenceDimension("clearance", "pass", 0.7, 0.1, "clear"),
        collision_risk=EvidenceDimension("collision_risk", "pass", 0.1, 0.5, "clear"),
        candidate_fingerprint="candidate-fingerprint",
        scene_version=7,
        map_version=3,
        map_backend="sparse_voxel",
        provider="reference_motion_preview",
        provider_version="1",
        captured_at_ns=123,
        latency_ms=2.5,
        artifact_refs=(ArtifactRef("artifact://preview", "application/json"),),
    )
    evidence = CandidateEvidence(
        geometry=geometry,
        available_metrics=("geometry",),
        scene_version=7,
        provider="reference_motion_preview",
        captured_at_ns=123,
    )

    assert evidence.geometry is geometry
    assert "geometry" in evidence.available_metrics
    assert geometry.used_privileged_state is False
    assert geometry.artifact_refs[0].uri == "artifact://preview"

    with pytest.raises(ValueError, match="geometry evidence"):
        CandidateEvidence(available_metrics=("geometry",))
