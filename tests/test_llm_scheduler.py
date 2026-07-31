from __future__ import annotations

from dataclasses import replace

import pytest

from capmas.agents.policy import CallableGraphPolicyAgent
from capmas.contracts.action import SkillCall
from capmas.contracts.agent import AgentArtifact, AgentContext
from capmas.contracts.candidates import CandidateEvidence, subgraph_fingerprint
from capmas.contracts.core import SkillRef
from capmas.contracts.graph import (
    CheckpointSpec,
    MissionEdge,
    MissionGraph,
    SubgraphNodeSpec,
    SubgraphSpec,
)
from capmas.contracts.scene import SceneSnapshot
from capmas.backends.capx import CAPXTypedSkill
from capmas.runtime.llm_scheduler import LLMGraphScheduler, LLMGraphScheduleError
from capmas.evaluation.rehearsal_evidence import RehearsalEvidence
from capmas.skills.registry import SkillRegistry


def _scene(version: int = 4) -> SceneSnapshot:
    return SceneSnapshot("episode", 1, version, 1, 1, {})


def _subgraph(subgraph_id: str, *, description: str = "candidate") -> SubgraphSpec:
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


def _graph(scene_version: int = 4) -> MissionGraph:
    first = _subgraph("first", description="manager")
    second = _subgraph("second", description="manager")
    return MissionGraph(
        mission_id="mission",
        task="test task",
        subgraphs=(first, second),
        edges=(MissionEdge("first", "second", "success"),),
        bindings=(),
        entry_subgraph="first",
        success_subgraphs=("second",),
        failure_subgraphs=("first", "second"),
        parent_scene_version=scene_version,
    )


class _Manager:
    name = "manager"

    def propose_graph(self, _task: str, _scene: SceneSnapshot) -> MissionGraph:
        return _graph()


def _agent(name: str, description: str) -> CallableGraphPolicyAgent:
    agent = CallableGraphPolicyAgent(
        lambda _subgoal, _scene, _context: _subgraph("first", description=description)
    )
    agent.name = name  # type: ignore[attr-defined]
    return agent


def test_llm_scheduler_fans_out_read_only_candidates_and_arbiter_selects_one() -> None:
    scheduler = LLMGraphScheduler(
        _Manager(),
        {"first": (_agent("policy-a", "short"), _agent("policy-b", "preferred"))},
        require_policy_proposals=False,
        max_workers=2,
    )

    compiled = scheduler.compile("test task", _scene())

    assert compiled.graph.subgraph("first").description == "preferred"
    assert len(compiled.arbitrations["first"].considered) == 2
    assert compiled.arbitrations["first"].selected is not None
    assert compiled.arbitrations["first"].selected.candidate_id == "first:policy-b:1"
    assert all(candidate.confidence is None for candidate in compiled.arbitrations["first"].considered)
    assert compiled.arbitrations["first"].selection_basis == "confidence_fallback"
    assert compiled.graph.subgraph("second").description == "manager"


def test_llm_scheduler_can_attach_evidence_before_arbitration() -> None:
    def evidence(candidate, _scene):
        preferred = candidate.subgraph.description == "preferred"
        return CandidateEvidence(
            verifier_pass_rate=1.0 if preferred else 0.2,
            rehearsal_success_rate=0.9 if preferred else 0.1,
            ood_success_rate=0.8 if preferred else 0.1,
            expected_latency_ms=100,
        )

    scheduler = LLMGraphScheduler(
        _Manager(),
        {"first": (_agent("policy-a", "short"), _agent("policy-b", "preferred"))},
        require_policy_proposals=False,
        max_workers=2,
        candidate_evidence_provider=evidence,
    )

    compiled = scheduler.compile("test task", _scene())

    arbitration = compiled.arbitrations["first"]
    assert arbitration.selected is not None
    assert arbitration.selected.subgraph.description == "preferred"
    assert arbitration.selection_basis == "evidence_score"
    assert all(candidate.evidence is not None for candidate in arbitration.considered)


def _rehearsal_evidence(candidate, scene, score: float) -> RehearsalEvidence:
    return RehearsalEvidence(
        candidate_id=candidate.candidate_id,
        candidate_fingerprint=subgraph_fingerprint(candidate.subgraph),
        seed=1,
        scene_version=scene.scene_version,
        success=score > 0.5,
        score=score,
        latency_ms=2.0,
    )


def test_llm_scheduler_online_rehearsal_can_change_live_selection() -> None:
    def rehearsal(items, scene):
        return {
            candidate.candidate_id: _rehearsal_evidence(
                candidate,
                scene,
                1.0 if candidate.subgraph.description == "short" else 0.0,
            )
            for candidate in items
        }

    scheduler = LLMGraphScheduler(
        _Manager(),
        {"first": (_agent("policy-a", "short"), _agent("policy-b", "preferred"))},
        require_policy_proposals=False,
        max_workers=2,
        rehearsal_mode="online_bounded",
        rehearsal_evidence_provider=rehearsal,
    )

    compiled = scheduler.compile("test task", _scene())

    assert compiled.graph.subgraph("first").description == "short"
    report = compiled.rehearsal_reports["first"]
    assert report.live == compiled.arbitrations["first"]
    assert report.baseline.selected is not None
    assert report.baseline.selected.subgraph.description == "preferred"
    assert report.would_change_selection is True


def test_llm_scheduler_shadow_rehearsal_keeps_live_selection() -> None:
    def rehearsal(items, scene):
        return {
            candidate.candidate_id: _rehearsal_evidence(
                candidate,
                scene,
                1.0 if candidate.subgraph.description == "short" else 0.0,
            )
            for candidate in items
        }

    scheduler = LLMGraphScheduler(
        _Manager(),
        {"first": (_agent("policy-a", "short"), _agent("policy-b", "preferred"))},
        require_policy_proposals=False,
        max_workers=2,
        rehearsal_mode="shadow",
        rehearsal_evidence_provider=rehearsal,
    )

    compiled = scheduler.compile("test task", _scene())

    assert compiled.graph.subgraph("first").description == "preferred"
    report = compiled.rehearsal_reports["first"]
    assert report.live == report.baseline
    assert report.evidence_aware is not None
    assert report.evidence_aware.selected is not None
    assert report.evidence_aware.selected.subgraph.description == "short"


def test_llm_scheduler_rehearsal_provider_failure_keeps_baseline() -> None:
    def rehearsal(_items, _scene):
        raise RuntimeError("worker unavailable")

    scheduler = LLMGraphScheduler(
        _Manager(),
        {"first": (_agent("policy-a", "short"),)},
        require_policy_proposals=False,
        rehearsal_mode="online_bounded",
        rehearsal_evidence_provider=rehearsal,
    )

    compiled = scheduler.compile("test task", _scene())

    report = compiled.rehearsal_reports["first"]
    assert compiled.arbitrations["first"] == report.baseline
    assert report.fallback_reason == "provider_error: RuntimeError: worker unavailable"


def test_llm_scheduler_scene_aware_rewriter_receives_current_scene() -> None:
    scheduler = LLMGraphScheduler(
        _Manager(),
        {"first": (_agent("policy-a", "short"),)},
        require_policy_proposals=False,
        candidate_scene_rewriter=lambda subgraph, scene: replace(
            subgraph,
            description=f"scene-{scene.scene_version}",
        ),
    )

    compiled = scheduler.compile("test task", _scene())

    assert compiled.graph.subgraph("first").description == "scene-4"


def test_llm_scheduler_enriches_empty_action_postconditions_before_arbitration() -> None:
    def enrich(subgraph, _scene, _strategy):
        node = subgraph.node(subgraph.entry_node)
        enriched = replace(node, postconditions=("scene_fresh(2000)",))
        return replace(
            subgraph,
            nodes=tuple(enriched if item.node_id == node.node_id else item for item in subgraph.nodes),
            checkpoints=(CheckpointSpec("fresh", ("scene_fresh(2000)",)),),
        )

    empty_agent = CallableGraphPolicyAgent(
        lambda _subgoal, _scene, _context: replace(
            _subgraph("first", description="empty"),
            nodes=(replace(_subgraph("first").node("first-action"), postconditions=()),),
            checkpoints=(CheckpointSpec("first-checkpoint", ("scene_fresh(2000)",)),),
        )
    )
    empty_agent.name = "policy-empty"  # type: ignore[attr-defined]
    scheduler = LLMGraphScheduler(
        _Manager(),
        {"first": (empty_agent,)},
        require_policy_proposals=False,
        condition_enricher=enrich,
    )

    compiled = scheduler.compile("test task", _scene())

    assert compiled.graph.subgraph("first").node("first-action").postconditions == (
        "scene_fresh(2000)",
    )


def test_llm_scheduler_does_not_execute_when_required_policy_proposal_fails() -> None:
    failing = CallableGraphPolicyAgent(
        lambda _subgoal, _scene, _context: (_ for _ in ()).throw(RuntimeError("bad model"))
    )
    failing.name = "failing-policy"  # type: ignore[attr-defined]
    scheduler = LLMGraphScheduler(
        _Manager(),
        {"first": (failing,)},
        require_policy_proposals=True,
    )

    with pytest.raises(LLMGraphScheduleError, match="no valid policy candidate"):
        scheduler.compile("test task", _scene())


def test_llm_scheduler_failure_retains_candidate_skill_diagnostics() -> None:
    def goto_pose(position, quaternion_wxyz, z_approach=0.0):
        return position, quaternion_wxyz, z_approach

    registry = SkillRegistry()
    skill = CAPXTypedSkill(SkillRef("goto_pose", "1.0.0"), goto_pose)
    registry.register(SkillRef("goto_pose", "1.0.0"), skill)

    invalid_node = SubgraphNodeSpec(
        node_id="first-action",
        description="invalid candidate",
        skill_calls=(
            SkillCall(
                SkillRef("goto_pose", "1.0.0"),
                {"approach_z": 0.1},
            ),
        ),
        postconditions=("done(first)",),
        proposed_by="policy-a",
    )
    invalid = replace(
        _subgraph("first", description="invalid candidate"),
        nodes=(invalid_node,),
    )
    agent = CallableGraphPolicyAgent(
        lambda _subgoal, _scene, _context: invalid
    )
    agent.name = "policy-a"  # type: ignore[attr-defined]

    def validate_skills(graph, context) -> None:
        subgraph = graph.subgraphs[0]
        contract = subgraph.to_action_contract(subgraph.entry_node, context)
        registry.validate_contract(contract)

    scheduler = LLMGraphScheduler(
        _Manager(),
        {"first": (agent,)},
        require_policy_proposals=True,
        skill_validator=validate_skills,
    )

    with pytest.raises(LLMGraphScheduleError) as error:
        scheduler.compile("test task", _scene())

    failure = error.value.proposal_failures[0]
    assert failure.candidate_id == "first:policy-a:0"
    assert failure.diagnostics["node_id"] == "first-action"
    assert failure.diagnostics["skill_id"] == "goto_pose"
    assert failure.diagnostics["args"] == {"approach_z": 0.1}
    assert "position" in failure.diagnostics["expected_signature"]
    assert failure.diagnostics["missing_arguments"] == ("position", "quaternion_wxyz")
    assert failure.diagnostics["unexpected_arguments"] == ("approach_z",)


def test_llm_scheduler_rejects_stale_manager_graph() -> None:
    class _StaleManager(_Manager):
        def propose_graph(self, _task: str, _scene: SceneSnapshot) -> MissionGraph:
            return replace(_graph(), parent_scene_version=3)

    scheduler = LLMGraphScheduler(_StaleManager(), {})

    with pytest.raises(LLMGraphScheduleError, match="stale"):
        scheduler.compile("test task", _scene())
