from __future__ import annotations

from dataclasses import replace
import pytest

from capmas.backends.capx import CAPXTypedSkill
from capmas.contracts.action import SkillCall
from capmas.contracts.candidates import GraphCandidate, subgraph_fingerprint
from capmas.contracts.core import SkillRef
from capmas.contracts.graph import (
    CheckpointSpec,
    MotionIntent,
    SubgraphNodeSpec,
    SubgraphSpec,
)
from capmas.graph.normalizer import CandidateNormalizer, normalize_motion_intent
from capmas.skills.registry import SkillRegistry


def _registry() -> SkillRegistry:
    def sample_grasp_pose(object_name: str, use_multiview: bool = True):
        return object_name, use_multiview

    def goto_pose(position, quaternion_wxyz, z_approach=0.0):
        return position, quaternion_wxyz, z_approach

    registry = SkillRegistry()
    for reference, function in (
        (SkillRef("sample_grasp_pose", "1.0.0"), sample_grasp_pose),
        (SkillRef("goto_pose", "1.0.0"), goto_pose),
    ):
        registry.register(reference, CAPXTypedSkill(reference, function))
    return registry


def _subgraph(*, args: dict[str, object], intent: MotionIntent | None = None) -> SubgraphSpec:
    node = SubgraphNodeSpec(
        node_id="action",
        description="pick object",
        skill_calls=(SkillCall(SkillRef("sample_grasp_pose", "1.0.0"), args),),
        postconditions=("object_in_gripper(bowl)",),
        motion_intent=intent,
    )
    return SubgraphSpec(
        subgraph_id="pick",
        subgoal_id="pick",
        description="pick object",
        nodes=(node,),
        edges=(),
        entry_node="action",
        success_nodes=("action",),
        failure_nodes=("action",),
        checkpoints=(CheckpointSpec("check", node.postconditions),),
    )


def _candidate(subgraph: SubgraphSpec) -> GraphCandidate:
    return GraphCandidate("candidate", subgraph, 1, "policy-0")


def test_normalizer_derives_grasp_intent_from_registered_call() -> None:
    normalized = normalize_motion_intent(_subgraph(args={"object_name": "bowl"}).nodes[0])

    assert normalized.motion_intent == MotionIntent(
        kind="grasp",
        object_track_id="bowl",
    )


def test_candidate_normalizer_rejects_unregistered_geometry_arguments() -> None:
    candidate = _candidate(
        _subgraph(args={"object_name": "bowl", "approach": "top"})
    )

    with pytest.raises(ValueError, match="unregistered.*approach"):
        CandidateNormalizer(_registry()).normalize(candidate)


def test_normalized_fingerprint_changes_when_effective_approach_changes() -> None:
    first_raw = _subgraph(
        args={"object_name": "bowl"},
        intent=MotionIntent("grasp", "bowl", approach_vector_xyz=(0.0, 0.0, -1.0)),
    )
    second_raw = _subgraph(
        args={"object_name": "bowl"},
        intent=MotionIntent("grasp", "bowl", approach_vector_xyz=(1.0, 0.0, 0.0)),
    )
    first = replace(first_raw, nodes=(normalize_motion_intent(first_raw.nodes[0]),))
    second = replace(second_raw, nodes=(normalize_motion_intent(second_raw.nodes[0]),))

    assert subgraph_fingerprint(first) != subgraph_fingerprint(second)


def test_candidate_normalizer_preserves_raw_and_effective_fingerprints() -> None:
    candidate = _candidate(_subgraph(args={"object_name": "bowl"}))

    normalized = CandidateNormalizer(_registry()).normalize(candidate)

    assert normalized.raw_subgraph == candidate.subgraph
    assert normalized.subgraph.nodes[0].motion_intent is not None
    assert normalized.rewrite_report.raw_fingerprint == subgraph_fingerprint(candidate.subgraph)
    assert normalized.rewrite_report.normalized_fingerprint == subgraph_fingerprint(
        normalized.subgraph
    )
