from __future__ import annotations

import time

from capmas.agents.policy import CallableGraphPolicyAgent
from capmas.contracts.action import SkillCall
from capmas.contracts.candidates import CandidateEvidence, subgraph_fingerprint
from capmas.contracts.core import SkillRef
from capmas.contracts.experiment import ExperimentRunConfig
from capmas.contracts.graph import CheckpointSpec, MissionGraph, SubgraphNodeSpec, SubgraphSpec
from capmas.contracts.scene import SceneSnapshot
from capmas.graph.normalizer import CandidateNormalizer
from capmas.runtime.llm_scheduler import LLMGraphScheduler


def _scene() -> SceneSnapshot:
    return SceneSnapshot("episode", 1, 4, 100, 101, {})


def _subgraph() -> SubgraphSpec:
    node = SubgraphNodeSpec(
        "action",
        "grasp bowl",
        skill_calls=(
            SkillCall(
                SkillRef("sample_grasp_pose", "1.0.0"),
                {"object_name": "bowl"},
            ),
        ),
        postconditions=("done",),
    )
    return SubgraphSpec(
        "first",
        "first",
        "grasp bowl",
        (node,),
        (),
        "action",
        ("action",),
        ("action",),
        checkpoints=(CheckpointSpec("check", node.postconditions),),
    )


def _manager() -> object:
    subgraph = _subgraph()
    graph = MissionGraph(
        "mission",
        "grasp bowl",
        (subgraph,),
        (),
        (),
        "first",
        ("first",),
        ("first",),
        parent_scene_version=4,
    )

    class Manager:
        name = "manager"

        def propose_graph(self, _task: str, _scene: SceneSnapshot) -> MissionGraph:
            return graph

    return Manager()


def _policy() -> CallableGraphPolicyAgent:
    agent = CallableGraphPolicyAgent(lambda _artifact, _scene, _context: _subgraph())
    agent.name = "policy-0"  # type: ignore[attr-defined]
    return agent


def test_scheduler_provider_receives_normalized_effective_candidate() -> None:
    seen: list[str] = []

    def provider(candidate, scene):
        assert scene.scene_version == 4
        assert candidate.subgraph.nodes[0].motion_intent is not None
        seen.append(subgraph_fingerprint(candidate.subgraph))
        return CandidateEvidence(verifier_pass_rate=1.0, available_metrics=("verifier",))

    result = LLMGraphScheduler(
        _manager(),
        {"first": (_policy(),)},
        require_policy_proposals=True,
        candidate_normalizer=CandidateNormalizer(),
        candidate_evidence_provider=provider,
    ).compile("grasp bowl", _scene())

    candidate = result.arbitrations["first"].considered[0]
    assert seen == [candidate.rewrite_report.normalized_fingerprint]
    assert candidate.evidence is not None


def test_slow_evidence_provider_times_out_without_blocking_scheduler() -> None:
    def slow_provider(_candidate, _scene):
        time.sleep(0.2)
        return CandidateEvidence(verifier_pass_rate=1.0, available_metrics=("verifier",))

    started = time.monotonic()
    result = LLMGraphScheduler(
        _manager(),
        {"first": (_policy(),)},
        require_policy_proposals=True,
        candidate_evidence_provider=slow_provider,
        candidate_evidence_timeout_ms=5,
    ).compile("grasp bowl", _scene())
    elapsed = time.monotonic() - started

    assert elapsed < 0.15
    assert result.candidate_evidence_timeouts == ("first:policy-0:0",)


def test_experiment_config_records_geometry_controls_without_credentials() -> None:
    config = ExperimentRunConfig(
        run_id="run-geometry",
        task_id="task",
        task="grasp bowl",
        seed=1,
        protocol="staged",
        proposal_mode="ready_wave",
        execution_mode="rolling",
        model="gpt-5.5",
        endpoint_host="api.example.test",
        policy_agents=2,
        max_workers=2,
        llm_deadline_ms=60_000,
        llm_max_output_tokens=1000,
        llm_max_retries=1,
        llm_proposal_retries=0,
        schema_mode="strict_provider_schema",
        geometry_mode="online_bounded",
        geometry_deadline_ms=50,
        preview_backend="reference_motion_preview",
        privilege_mode="realistic_sensor",
        artifact_dir="outputs/phase5/run",
    )

    payload = config.to_dict()
    assert payload["geometry_mode"] == "online_bounded"
    assert payload["geometry_deadline_ms"] == 50
    assert payload["privilege_mode"] == "realistic_sensor"
    assert "api_key" not in payload
