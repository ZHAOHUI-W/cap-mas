from __future__ import annotations

from dataclasses import replace

from capmas.contracts.candidates import CandidateEvidence, GraphCandidate, subgraph_fingerprint
from capmas.contracts.graph import CheckpointSpec, SubgraphNodeSpec, SubgraphSpec
from capmas.perception.effective_motion import EffectiveMotionProgram, EffectiveMotionSegment


def _candidate(candidate_id: str, producer_agent: str, description: str) -> GraphCandidate:
    node = SubgraphNodeSpec(
        node_id="action",
        description=description,
        skill_calls=(),
        postconditions=("done",),
    )
    subgraph = SubgraphSpec(
        subgraph_id="sg_pick",
        subgoal_id="pick",
        description=description,
        nodes=(node,),
        edges=(),
        entry_node="action",
        success_nodes=("action",),
        failure_nodes=("action",),
        checkpoints=(CheckpointSpec("done", ("done",)),),
    )
    return GraphCandidate(candidate_id, subgraph, 4, producer_agent)


def _program(candidate: GraphCandidate, *, semantic_signature: str) -> EffectiveMotionProgram:
    segment = EffectiveMotionSegment(
        segment_id="grasp_approach",
        kind="grasp_approach",
        source_subgraph_id="sg_pick",
        source_node_id="action",
        start_pose_wxyz_xyz=(1.0, 0.0, 0.0, 0.0, 0.40, 0.00, 0.35),
        end_pose_wxyz_xyz=(1.0, 0.0, 0.0, 0.0, 0.40, 0.00, 0.30),
        approach_vector_xyz=(0.0, 0.0, -1.0),
        approach_distance_m=0.05,
        payload_track_id="object-6",
    )
    return EffectiveMotionProgram(
        candidate_fingerprint=subgraph_fingerprint(candidate.subgraph),
        execution_graph_fingerprint="graph-fingerprint",
        program_fingerprint=f"program-{candidate.candidate_id}",
        decision_scene_version=4,
        selected_subgraph_id="sg_pick",
        segments=(segment,),
        semantic_signature=semantic_signature,
    )


def _candidates() -> tuple[GraphCandidate, GraphCandidate]:
    return (
        _candidate("policy-0", "policy-0", "direct pick"),
        _candidate("safety", "safety", "conservative pick"),
    )


def test_policy_names_do_not_make_duplicate_programs_diverse() -> None:
    from capmas.agents.candidate_diversity import CandidateDiversityValidator

    candidates = _candidates()
    decision = CandidateDiversityValidator().inspect(
        tuple(_program(candidate, semantic_signature="same-motion") for candidate in candidates),
        candidates,
    )

    assert decision.requires_regeneration is True
    assert all(item.candidate_semantic_equivalent for item in decision.identifiability)
    assert decision.required_difference_fields == (
        "grasp_pose_or_offset",
        "approach",
        "lift",
        "transfer",
        "place_or_release",
    )


def test_different_programs_with_equal_evidence_are_not_identifiable() -> None:
    from capmas.agents.candidate_diversity import CandidateDiversityValidator

    first, second = _candidates()
    equal_evidence = CandidateEvidence(
        rehearsal_success_rate=0.5,
        available_metrics=("rehearsal",),
    )
    candidates = (replace(first, evidence=equal_evidence), replace(second, evidence=equal_evidence))
    decision = CandidateDiversityValidator().inspect(
        (
            _program(candidates[0], semantic_signature="first-motion"),
            _program(candidates[1], semantic_signature="second-motion"),
        ),
        candidates,
    )

    assert decision.requires_regeneration is False
    assert all(item.candidate_evidence_identical for item in decision.identifiability)
    assert not any(item.selection_identifiable for item in decision.identifiability)
