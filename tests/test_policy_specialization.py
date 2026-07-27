from __future__ import annotations

from dataclasses import replace
import json

from capmas.agents.arbiter import CandidateArbiter
from capmas.agents.policy import CallableGraphPolicyAgent
from capmas.contracts.action import SkillCall
from capmas.contracts.candidates import CandidateEvidence, GraphCandidate, PerceptionEvidence
from capmas.contracts.core import SkillRef
from capmas.contracts.graph import CheckpointSpec, SubgraphNodeSpec, SubgraphSpec
from capmas.contracts.scene import ObjectTrack, SceneSnapshot, SceneUncertainty
from capmas.runtime.llm_scheduler import LLMGraphScheduler
from capmas.runtime.episode_runner import to_jsonable
from capmas.verification.libero import libero_candidate_evidence


def _scene(version: int = 4) -> SceneSnapshot:
    return SceneSnapshot("episode", 1, version, 1, 1, {})


def _subgraph(subgraph_id: str, description: str = "candidate") -> SubgraphSpec:
    node_id = f"{subgraph_id}-action"
    node = SubgraphNodeSpec(
        node_id=node_id,
        description=description,
        skill_calls=(SkillCall(SkillRef("noop", "1.0.0"), {}),),
        postconditions=(f"done({subgraph_id})",),
        proposed_by=description,
    )
    return SubgraphSpec(
        subgraph_id=subgraph_id,
        subgoal_id=subgraph_id,
        description=description,
        nodes=(node,),
        edges=(),
        entry_node=node_id,
        success_nodes=(node_id,),
        failure_nodes=(node_id,),
        checkpoints=(CheckpointSpec(f"{subgraph_id}-checkpoint", node.postconditions),),
    )


class _Manager:
    name = "manager"

    def propose_graph(self, _task: str, _scene: SceneSnapshot):
        first = _subgraph("first", "manager")
        second = _subgraph("second", "manager")
        from capmas.contracts.graph import MissionEdge, MissionGraph

        return MissionGraph(
            mission_id="mission",
            task="test task",
            subgraphs=(first, second),
            edges=(MissionEdge("first", "second", "success"),),
            bindings=(),
            entry_subgraph="first",
            success_subgraphs=("second",),
            failure_subgraphs=("first", "second"),
            parent_scene_version=4,
        )


def _agent(name: str, description: str) -> CallableGraphPolicyAgent:
    agent = CallableGraphPolicyAgent(
        lambda _subgoal, _scene, _context: _subgraph("first", description)
    )
    agent.name = name  # type: ignore[attr-defined]
    return agent


def test_scheduler_retains_raw_and_normalized_candidate_views() -> None:
    scheduler = LLMGraphScheduler(
        _Manager(),
        {"first": (_agent("policy-a", "raw intent"),)},
        require_policy_proposals=False,
        candidate_scene_rewriter=lambda subgraph, _scene: replace(
            subgraph,
            description="normalized execution",
        ),
    )

    result = scheduler.compile("test task", _scene())
    candidate = result.arbitrations["first"].considered[0]

    assert candidate.raw_subgraph is not None
    assert candidate.raw_subgraph.description == "raw intent"
    assert candidate.subgraph.description == "normalized execution"
    assert candidate.rewrite_report.changed is True
    assert candidate.rewrite_report.raw_fingerprint != candidate.rewrite_report.normalized_fingerprint


def test_safety_arbiter_prefers_perception_evidence_over_nominal_confidence() -> None:
    weak_perception = GraphCandidate(
        "weak",
        _subgraph("pick", "fast nominal"),
        4,
        "policy-fast",
        confidence=0.95,
        strategy="efficient",
        evidence=CandidateEvidence(
            verifier_pass_rate=1.0,
            perception=PerceptionEvidence(
                scene_freshness=0.4,
                scene_confidence=0.5,
                target_visibility=0.7,
                track_confidence=0.5,
                identity_confidence=0.5,
                pose_reliability=0.5,
            ),
            available_metrics=("verifier", "perception"),
        ),
    )
    strong_perception = GraphCandidate(
        "strong",
        _subgraph("pick", "safe grounded"),
        4,
        "policy-safe",
        confidence=0.70,
        strategy="safety",
        evidence=CandidateEvidence(
            verifier_pass_rate=0.95,
            perception=PerceptionEvidence(
                scene_freshness=1.0,
                scene_confidence=1.0,
                target_visibility=1.0,
                track_confidence=0.95,
                identity_confidence=1.0,
                pose_reliability=0.95,
            ),
            available_metrics=("verifier", "perception"),
        ),
    )

    result = CandidateArbiter().select((weak_perception, strong_perception), _scene())

    assert result.selected == strong_perception
    assert result.selection_basis == "evidence_score"


def test_evidence_score_ignores_legacy_candidate_confidence() -> None:
    weak = GraphCandidate(
        "weak",
        _subgraph("pick", "high self confidence"),
        4,
        "policy-a",
        confidence=1.0,
        evidence=CandidateEvidence(
            perception=PerceptionEvidence(
                scene_freshness=0.0,
                scene_confidence=0.0,
                target_visibility=0.0,
                track_confidence=0.0,
                identity_confidence=0.0,
                pose_reliability=0.0,
            ),
            available_metrics=("perception",),
            scene_version=4,
        ),
    )
    strong = GraphCandidate(
        "strong",
        _subgraph("pick", "grounded evidence"),
        4,
        "policy-b",
        confidence=0.0,
        evidence=CandidateEvidence(
            perception=PerceptionEvidence(
                scene_freshness=1.0,
                scene_confidence=1.0,
                target_visibility=1.0,
                track_confidence=1.0,
                identity_confidence=1.0,
                pose_reliability=1.0,
            ),
            available_metrics=("perception",),
            scene_version=4,
        ),
    )

    result = CandidateArbiter().select((weak, strong), _scene())

    assert result.selected == strong
    assert "confidence" not in result.score_breakdowns["weak"]
    assert "confidence" not in result.score_breakdowns["strong"]


def test_safety_arbiter_rejects_stale_perception_evidence() -> None:
    candidate = GraphCandidate(
        "stale",
        _subgraph("pick"),
        4,
        "policy-safe",
        strategy="safety",
        evidence=CandidateEvidence(
            perception=PerceptionEvidence(
                scene_freshness=0.0,
                scene_confidence=1.0,
                target_visibility=1.0,
                track_confidence=1.0,
                identity_confidence=1.0,
                pose_reliability=1.0,
            ),
            available_metrics=("perception",),
            scene_version=3,
        ),
    )

    result = CandidateArbiter().select((candidate,), _scene())

    assert result.selected is None
    assert result.rejections[0].code == "STALE_EVIDENCE"


def test_libero_evidence_provider_reads_scene_tracks_without_backend_access() -> None:
    subgraph = _subgraph("place")
    node = replace(
        subgraph.nodes[0],
        postconditions=("object_at_target(akita_black_bowl,plate)",),
    )
    subgraph = replace(subgraph, nodes=(node,))
    candidate = GraphCandidate(
        "place-safe",
        subgraph,
        4,
        "policy-safe",
        strategy="safety",
    )
    scene = SceneSnapshot(
        "episode",
        1,
        4,
        1,
        1,
        {},
        objects=(
            ObjectTrack("akita_black_bowl", "akita black bowl", (1, 0, 0, 0, 0, 0, 0), 0.95, 1),
            ObjectTrack("plate", "plate", (1, 0, 0, 0, 0, 0, 0), 0.90, 1),
        ),
        freshness_ms=50.0,
        uncertainty=SceneUncertainty(scene_confidence=0.9),
    )

    evidence = libero_candidate_evidence(candidate, scene)

    assert evidence.available_metrics == ("perception",)
    assert evidence.perception is not None
    assert evidence.perception.scene_freshness == 0.95
    assert evidence.perception.target_visibility == 1.0
    assert evidence.perception.track_confidence == 0.90
    assert evidence.perception.identity_confidence == 1.0
    assert evidence.scene_version == 4
    assert evidence.provider == "libero_scene_snapshot"


def test_evidence_tie_break_is_distinct_from_confidence_fallback() -> None:
    first = GraphCandidate(
        "first",
        _subgraph("pick", "first"),
        4,
        "policy-a",
        confidence=1.0,
        evidence=CandidateEvidence(
            perception=PerceptionEvidence(
                scene_freshness=1.0,
                scene_confidence=1.0,
                target_visibility=1.0,
                track_confidence=1.0,
                identity_confidence=1.0,
                pose_reliability=1.0,
            ),
            available_metrics=("perception",),
            scene_version=4,
        ),
    )
    second = GraphCandidate(
        "second",
        _subgraph("pick", "second"),
        4,
        "policy-b",
        confidence=0.0,
        evidence=CandidateEvidence(
            perception=PerceptionEvidence(
                scene_freshness=1.0,
                scene_confidence=1.0,
                target_visibility=1.0,
                track_confidence=1.0,
                identity_confidence=1.0,
                pose_reliability=1.0,
            ),
            available_metrics=("perception",),
            scene_version=4,
        ),
    )

    result = CandidateArbiter().select((first, second), _scene())

    assert result.selected is not None
    assert result.tie_broken is True
    assert result.selection_basis == "evidence_tie_break"


def test_unavailable_rehearsal_is_not_scored_as_zero_quality() -> None:
    candidate = GraphCandidate(
        "perception-only",
        _subgraph("pick"),
        4,
        "policy-safe",
        strategy="safety",
        evidence=CandidateEvidence(
            perception=PerceptionEvidence(
                scene_freshness=1.0,
                scene_confidence=1.0,
                target_visibility=1.0,
                track_confidence=1.0,
                identity_confidence=1.0,
                pose_reliability=1.0,
            ),
            available_metrics=("perception",),
        ),
    )

    result = CandidateArbiter().select((candidate,), _scene())

    assert result.selected == candidate
    assert "rehearsal" not in result.score_breakdowns["perception-only"]
    assert "ood" not in result.score_breakdowns["perception-only"]


def test_arbitration_artifact_serializes_raw_views_and_score_breakdown() -> None:
    scheduler = LLMGraphScheduler(
        _Manager(),
        {"first": (_agent("policy-0", "raw intent"),)},
        require_policy_proposals=False,
        candidate_evidence_provider=lambda candidate, _scene: CandidateEvidence(
            perception=PerceptionEvidence(
                scene_freshness=1.0,
                scene_confidence=1.0,
                target_visibility=1.0,
                track_confidence=1.0,
                identity_confidence=1.0,
                pose_reliability=1.0,
            ),
            available_metrics=("perception",),
        ),
        candidate_scene_rewriter=lambda subgraph, _scene: replace(
            subgraph,
            description="normalized",
        ),
    )

    result = scheduler.compile("test task", _scene())
    payload = json.dumps(to_jsonable(result.arbitrations))

    assert "raw_subgraph" in payload
    assert "normalized_fingerprint" in payload
    assert "score_breakdowns" in payload
