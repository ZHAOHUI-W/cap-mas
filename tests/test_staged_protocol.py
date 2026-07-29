from __future__ import annotations

import json
from dataclasses import replace
from threading import Barrier

import pytest

from capmas.agents.policy import CallableGraphPolicyAgent
from capmas.agents.manager import LLMTopologyManager
from capmas.contracts.action import SkillCall
from capmas.contracts.agent import AgentArtifact, AgentContext
from capmas.contracts.core import SkillRef
from capmas.contracts.graph import (
    CheckpointSpec,
    GraphEdge,
    MissionBinding,
    MissionEdge,
    PortSpec,
    SubgraphNodeSpec,
    SubgraphSpec,
)
from capmas.contracts.scene import SceneSnapshot
from capmas.contracts.staged import MissionTopology, TopologySubgoal
from capmas.graph.serialization import local_subgraph_to_dict
from capmas.graph.staged import TopologyValidator, topology_from_dict, topology_to_dict
from capmas.llm.prompts import (
    build_staged_policy_request,
    build_topology_request,
    mission_graph_response_schema,
    mission_topology_response_schema,
    subgraph_response_schema,
)
from capmas.llm.protocol import LLMRequest, LLMResponse
from capmas.llm.staged_decoder import LocalSubgraphDecoder, MissionTopologyDecoder
from capmas.runtime.llm_scheduler import LLMGraphScheduler, LLMGraphScheduleError


def _scene(version: int = 4) -> SceneSnapshot:
    return SceneSnapshot("episode", 1, version, 1, 1, {})


def _subgraph(subgraph_id: str, subgoal_id: str | None = None, predicate: str = "done") -> SubgraphSpec:
    node_id = f"{subgraph_id}_action"
    node = SubgraphNodeSpec(
        node_id=node_id,
        description=f"execute {subgraph_id}",
        skill_calls=(SkillCall(SkillRef("noop", "1.0.0"), {}),),
        postconditions=(predicate,),
        proposed_by="staged-policy",
    )
    return SubgraphSpec(
        subgraph_id=subgraph_id,
        subgoal_id=subgoal_id or subgraph_id,
        description=f"subgoal {subgraph_id}",
        nodes=(node,),
        edges=(),
        entry_node=node_id,
        success_nodes=(node_id,),
        failure_nodes=(node_id,),
        checkpoints=(CheckpointSpec(f"{subgraph_id}-checkpoint", (predicate,)),),
    )


def _topology(version: int = 4) -> MissionTopology:
    return MissionTopology(
        mission_id="mission",
        task="test task",
        subgoals=(
            TopologySubgoal("first", "first", "first step", success_predicates=("done",)),
            TopologySubgoal(
                "second",
                "second",
                "second step",
                depends_on=("first",),
                success_predicates=("done",),
            ),
        ),
        edges=(MissionEdge("first", "second", "success"),),
        entry_subgraph="first",
        success_subgraphs=("second",),
        failure_subgraphs=("first", "second"),
        parent_scene_version=version,
    )


def test_staged_schemas_remove_repeated_mission_graph_fields() -> None:
    topology = mission_topology_response_schema()
    local = subgraph_response_schema()

    assert "subgoals" in topology["properties"]
    assert "subgraphs" not in topology["properties"]
    assert topology["properties"]["subgoals"]["items"]["properties"][
        "execution_kind"
    ]["enum"] == ["physical_action", "checkpoint_only"]
    assert "mission_id" not in local["properties"]
    assert local["properties"]["subgraph"]["properties"]["nodes"]


def test_checkpoint_only_subgoal_is_compiled_without_policy_proposals() -> None:
    topology = MissionTopology(
        mission_id="mission",
        task="check scene",
        subgoals=(
            TopologySubgoal(
                "scene",
                "scene",
                "check scene tracks",
                success_predicates=("track_exists:bowl",),
                execution_kind="checkpoint_only",
            ),
        ),
        edges=(),
        entry_subgraph="scene",
        success_subgraphs=("scene",),
        failure_subgraphs=("scene",),
        parent_scene_version=4,
    )

    class Manager:
        name = "manager"

        def propose_topology(self, _task: str, _scene: SceneSnapshot) -> MissionTopology:
            return topology

    def must_not_be_called(*_args: object, **_kwargs: object) -> SubgraphSpec:
        raise AssertionError("checkpoint-only subgoal must not call a Policy Agent")

    policy = CallableGraphPolicyAgent(must_not_be_called)
    policy.name = "policy-should-not-run"  # type: ignore[attr-defined]
    result = LLMGraphScheduler(
        Manager(),
        {"*": (policy,)},
        max_workers=1,
    ).compile_staged("check scene", _scene())

    subgraph = result.graph.subgraph("scene")
    assert subgraph.nodes[0].node_type == "checkpoint"
    assert subgraph.nodes[0].skill_calls == ()
    assert result.arbitrations["scene"].selection_basis == "deterministic_checkpoint"


def test_ready_wave_checkpoint_only_subgoal_needs_no_policy_registration() -> None:
    topology = MissionTopology(
        mission_id="mission",
        task="check scene",
        subgoals=(
            TopologySubgoal(
                "scene",
                "scene",
                "check scene tracks",
                success_predicates=("track_exists:bowl",),
                execution_kind="checkpoint_only",
            ),
        ),
        edges=(),
        entry_subgraph="scene",
        success_subgraphs=("scene",),
        failure_subgraphs=("scene",),
        parent_scene_version=4,
    )

    class Manager:
        name = "manager"

        def propose_topology(self, _task: str, _scene: SceneSnapshot) -> MissionTopology:
            return topology

    result = LLMGraphScheduler(
        Manager(),
        {},
        max_workers=1,
        proposal_mode="ready_wave",
    ).compile_staged("check scene", _scene())

    assert result.proposal_waves == (("scene",),)
    assert result.arbitrations["scene"].selection_basis == "deterministic_checkpoint"


def test_topology_rejects_checkpoint_only_subgoal_without_success_predicates() -> None:
    topology = MissionTopology(
        mission_id="mission",
        task="check scene",
        subgoals=(
            TopologySubgoal(
                "failure",
                "failure",
                "record failure",
                execution_kind="checkpoint_only",
            ),
        ),
        edges=(),
        entry_subgraph="failure",
        success_subgraphs=("failure",),
        failure_subgraphs=("failure",),
        parent_scene_version=4,
    )

    result = TopologyValidator().validate(topology)

    assert result.valid is False
    assert any(
        item.code == "CHECKPOINT_SUCCESS_PREDICATE_MISSING"
        for item in result.errors
    )


def test_topology_rejects_unreachable_subgoal_before_policy_fanout() -> None:
    topology = MissionTopology(
        mission_id="mission",
        task="check scene",
        subgoals=(
            TopologySubgoal(
                "entry",
                "entry",
                "entry",
                success_predicates=("track_exists:bowl",),
                execution_kind="checkpoint_only",
            ),
            TopologySubgoal(
                "orphan",
                "orphan",
                "orphan",
                success_predicates=("track_exists:plate",),
                execution_kind="checkpoint_only",
            ),
        ),
        edges=(),
        entry_subgraph="entry",
        success_subgraphs=("entry",),
        failure_subgraphs=("entry",),
        parent_scene_version=4,
    )

    result = TopologyValidator().validate(topology)

    assert result.valid is False
    assert any(item.code == "UNREACHABLE_SUBGRAPH" for item in result.errors)


def test_topology_rejects_success_back_edge_for_rolling_execution() -> None:
    topology = MissionTopology(
        mission_id="mission",
        task="place object",
        subgoals=(
            TopologySubgoal("observe", "observe", "observe scene", success_predicates=("done",)),
            TopologySubgoal(
                "act",
                "act",
                "execute action",
                depends_on=("observe",),
                success_predicates=("done",),
            ),
            TopologySubgoal(
                "verify",
                "verify",
                "verify result",
                depends_on=("act",),
                success_predicates=("done",),
            ),
        ),
        edges=(
            MissionEdge("observe", "act", "success"),
            MissionEdge("act", "verify", "success"),
            MissionEdge("verify", "observe", "success"),
        ),
        entry_subgraph="observe",
        success_subgraphs=("verify",),
        failure_subgraphs=("observe", "act", "verify"),
        parent_scene_version=4,
    )

    result = TopologyValidator().validate(topology)

    assert result.valid is False
    assert any(item.code == "SUCCESS_BACK_EDGE" for item in result.errors)


def test_topology_round_trip_and_decoder_rejects_stale_scene() -> None:
    topology = _topology()
    assert topology_from_dict(topology_to_dict(topology)) == topology
    request = LLMRequest("topology", "manager", ())
    accepted = MissionTopologyDecoder().decode(
        LLMResponse("topology", "", structured=topology_to_dict(topology)),
        _scene(),
        request=request,
    )
    stale = MissionTopologyDecoder().decode(
        LLMResponse("topology", "", structured=topology_to_dict(_topology(3))),
        _scene(),
        request=request,
    )

    assert accepted.accepted
    assert not stale.accepted
    assert any(item.code == "STALE_SCENE" for item in stale.rejections)


def test_topology_assembler_normalizes_dependency_failure_label_to_success() -> None:
    topology = replace(_topology(), edges=(MissionEdge("first", "second", "failure"),))

    graph = topology.assemble((_subgraph("first"), _subgraph("second")))

    assert graph.edges == (MissionEdge("first", "second", "success"),)


def test_topology_assembler_preserves_failure_dependency_edges() -> None:
    topology = MissionTopology(
        mission_id="mission",
        task="test task",
        subgoals=(
            TopologySubgoal("first", "first", "first step"),
            TopologySubgoal(
                "main",
                "main",
                "main step",
                depends_on=("first",),
            ),
            TopologySubgoal(
                "fallback",
                "fallback",
                "fallback step",
                depends_on=("first",),
            ),
        ),
        edges=(
            MissionEdge("first", "main", "success"),
            MissionEdge("first", "fallback", "failure"),
        ),
        entry_subgraph="first",
        success_subgraphs=("main",),
        failure_subgraphs=("fallback",),
        parent_scene_version=4,
    )

    graph = topology.assemble(
        (_subgraph("first"), _subgraph("main"), _subgraph("fallback"))
    )

    assert MissionEdge("first", "main", "success") in graph.edges
    assert MissionEdge("first", "fallback", "failure") in graph.edges


def test_topology_assembler_keeps_failure_branch_during_success_deduplication() -> None:
    topology = MissionTopology(
        mission_id="mission",
        task="test task",
        subgoals=(
            TopologySubgoal("first", "first", "first step"),
            TopologySubgoal(
                "main",
                "main",
                "main step",
                depends_on=("first",),
            ),
            TopologySubgoal(
                "fallback",
                "fallback",
                "fallback step",
                depends_on=("first",),
            ),
        ),
        edges=(
            MissionEdge("first", "main", "success"),
            MissionEdge("first", "fallback", "failure"),
        ),
        entry_subgraph="first",
        success_subgraphs=("main", "fallback"),
        failure_subgraphs=("fallback",),
        parent_scene_version=4,
    )

    graph = topology.assemble(
        (_subgraph("first"), _subgraph("main"), _subgraph("fallback"))
    )

    assert MissionEdge("first", "main", "success") in graph.edges
    assert MissionEdge("first", "fallback", "failure") in graph.edges


def test_topology_decoder_rejects_ambiguous_outcome_transitions() -> None:
    topology = replace(
        _topology(),
        edges=(
            MissionEdge("first", "second", "success"),
            MissionEdge("first", "second", "success"),
        ),
    )

    with pytest.raises(ValueError, match="AMBIGUOUS_TRANSITION"):
        topology_from_dict(topology_to_dict(topology))


def test_topology_decoder_accepts_nullable_duplicate_success_transitions() -> None:
    topology = replace(
        _topology(),
        edges=(
            MissionEdge("first", "second", None),
            MissionEdge("first", "second", None),
        ),
    )

    decoded = topology_from_dict(topology_to_dict(topology))

    assert decoded.normalized_edges() == (MissionEdge("first", "second", "success"),)


def test_topology_decoder_resolves_semantic_success_transitions_before_rejecting() -> None:
    topology = replace(
        _topology(),
        edges=(
            MissionEdge("first", "second", "continue"),
            MissionEdge("first", "second", None),
        ),
    )

    decoded = topology_from_dict(topology_to_dict(topology))

    assert decoded.normalized_edges() == (MissionEdge("first", "second", "success"),)


def test_topology_decoder_rejects_unresolved_failure_transitions() -> None:
    topology = MissionTopology(
        mission_id="mission",
        task="test task",
        subgoals=(
            TopologySubgoal("first", "first", "first step"),
            TopologySubgoal("main", "main", "main step"),
            TopologySubgoal("fallback", "fallback", "fallback step"),
        ),
        edges=(
            MissionEdge("first", "main", "failure"),
            MissionEdge("first", "fallback", "failure"),
        ),
        entry_subgraph="first",
        success_subgraphs=("main",),
        failure_subgraphs=("fallback",),
        parent_scene_version=4,
    )

    with pytest.raises(ValueError, match="AMBIGUOUS_TRANSITION"):
        topology_from_dict(topology_to_dict(topology))


def test_topology_validator_rejects_recovery_dependency_on_failed_source() -> None:
    topology = MissionTopology(
        mission_id="mission",
        task="test task",
        subgoals=(
            TopologySubgoal("main", "main", "main step"),
            TopologySubgoal(
                "recover",
                "recover",
                "recover stale scene",
                depends_on=("main",),
                success_predicates=("scene_fresh(1000)",),
                execution_kind="checkpoint_only",
            ),
        ),
        edges=(MissionEdge("main", "recover", "failure"),),
        entry_subgraph="main",
        success_subgraphs=("main",),
        failure_subgraphs=("main", "recover"),
        parent_scene_version=4,
    )

    result = TopologyValidator().validate(topology)

    assert result.valid is False
    assert any(
        item.code == "RECOVERY_DEPENDS_ON_UNCOMMITTED"
        for item in result.errors
    )


def test_topology_validator_rejects_nonpositive_scene_fresh_threshold() -> None:
    topology = replace(
        _topology(),
        subgoals=(
            replace(_topology().subgoals[0], success_predicates=("scene_fresh(0)",)),
            _topology().subgoals[1],
        ),
    )

    result = TopologyValidator().validate(topology)

    assert result.valid is False
    assert any(item.code == "INVALID_SCENE_FRESH_THRESHOLD" for item in result.errors)


def test_local_decoder_normalizes_terminal_predicate_edges_to_typed_outcomes() -> None:
    action = SubgraphNodeSpec(
        node_id="action",
        description="perform action",
        skill_calls=(SkillCall(SkillRef("noop", "1.0.0"), {}),),
        postconditions=("object_in_gripper(bowl)", "gripper_closed()"),
    )
    success = SubgraphNodeSpec(
        node_id="success",
        description="verify success",
        node_type="checkpoint",
        postconditions=("object_in_gripper(bowl)", "gripper_closed()"),
    )
    failure = SubgraphNodeSpec(
        node_id="failure",
        description="verify failure",
        node_type="checkpoint",
        postconditions=("gripper_open()",),
    )
    subgraph = SubgraphSpec(
        subgraph_id="first",
        subgoal_id="first",
        description="first",
        nodes=(action, success, failure),
        edges=(
            GraphEdge(
                "action",
                "success",
                "object_in_gripper(bowl) && gripper_closed()",
            ),
            GraphEdge("action", "failure", "gripper_open()"),
        ),
        entry_node="action",
        success_nodes=("success",),
        failure_nodes=("failure",),
        checkpoints=(CheckpointSpec("verify", ("object_in_gripper(bowl)",)),),
    )

    decoded = LocalSubgraphDecoder().decode(
        LLMResponse("policy", "", structured=local_subgraph_to_dict(subgraph)),
        _scene(),
        request=LLMRequest("policy", "policy", ()),
        expected_subgraph_id="first",
        expected_subgoal_id="first",
    )

    assert decoded.accepted
    assert decoded.subgraph is not None
    assert decoded.subgraph.edges == (
        GraphEdge("action", "success", "success"),
        GraphEdge("action", "failure", "failure"),
    )


def test_topology_assembly_deduplicates_identical_success_edges() -> None:
    topology = replace(
        _topology(),
        edges=(
            MissionEdge("first", "second", "success"),
            MissionEdge("first", "second", "success"),
        ),
    )

    graph = topology.assemble((_subgraph("first"), _subgraph("second")))

    assert graph.edges == (MissionEdge("first", "second", "success"),)


def test_topology_assembly_prefers_explicit_dependency_success_edge() -> None:
    topology = replace(
        _topology(),
        edges=(
            MissionEdge("first", "second", "success"),
            MissionEdge("first", "first", "success"),
        ),
    )

    graph = topology.assemble((_subgraph("first"), _subgraph("second")))

    assert graph.edges == (MissionEdge("first", "second", "success"),)


def test_topology_assembly_uses_stable_forward_order_without_dependency_hint() -> None:
    topology = replace(
        _topology(),
        edges=(
            MissionEdge("first", "second", "success"),
            MissionEdge("first", "first", "success"),
        ),
    )

    graph = topology.assemble((_subgraph("first"), _subgraph("second")))

    assert graph.edges == (MissionEdge("first", "second", "success"),)


def test_local_decoder_requires_direct_versioned_subgraph_envelope() -> None:
    subgraph = _subgraph("first")
    request = LLMRequest("policy", "policy", ())
    accepted = LocalSubgraphDecoder().decode(
        LLMResponse("policy", "", structured=local_subgraph_to_dict(subgraph)),
        _scene(),
        request=request,
        expected_subgraph_id="first",
        expected_subgoal_id="first",
    )
    wrapper = {"schema_version": 1, "mission_id": "wrong", "subgraphs": []}
    rejected = LocalSubgraphDecoder().decode(
        LLMResponse("policy", json.dumps(wrapper)),
        _scene(),
        request=request,
        expected_subgraph_id="first",
        expected_subgoal_id="first",
    )

    assert accepted.accepted
    assert accepted.subgraph == subgraph
    assert not rejected.accepted
    assert rejected.rejections[0].code == "SUBGRAPH_SCHEMA_INVALID"


def test_staged_scheduler_assembles_only_matching_local_candidates() -> None:
    class Manager:
        name = "manager"

        def propose_topology(self, _task: str, _scene: SceneSnapshot) -> MissionTopology:
            return _topology()

    def proposal(subgoal: AgentArtifact, _scene: SceneSnapshot, _context: AgentContext) -> SubgraphSpec:
        return _subgraph(str(subgoal.payload["subgraph_id"]), str(subgoal.payload["subgoal_id"]))

    policy = CallableGraphPolicyAgent(proposal)
    policy.name = "policy-a"  # type: ignore[attr-defined]
    scheduler = LLMGraphScheduler(Manager(), {"*": (policy,)}, max_workers=1)

    result = scheduler.compile_staged("test task", _scene())

    assert result.graph.entry_subgraph == "first"
    assert [item.subgraph_id for item in result.graph.subgraphs] == ["first", "second"]
    assert set(result.arbitrations) == {"first", "second"}


def test_staged_policy_artifact_carries_topology_predicates() -> None:
    class Manager:
        name = "manager"

        def propose_topology(self, _task: str, _scene: SceneSnapshot) -> MissionTopology:
            return _topology()

    received: list[AgentArtifact] = []

    def proposal(subgoal: AgentArtifact, _scene: SceneSnapshot, _context: AgentContext) -> SubgraphSpec:
        received.append(subgoal)
        return _subgraph(str(subgoal.payload["subgraph_id"]), str(subgoal.payload["subgoal_id"]))

    policy = CallableGraphPolicyAgent(proposal)
    policy.name = "policy-a"  # type: ignore[attr-defined]
    LLMGraphScheduler(Manager(), {"*": (policy,)}, max_workers=1).compile_staged(
        "test task", _scene()
    )

    assert received[0].payload["success_predicates"] == ["done"]
    assert received[0].payload["failure_predicates"] == []


def test_staged_scheduler_ready_wave_fans_out_dependency_ready_subgoals() -> None:
    topology = MissionTopology(
        mission_id="mission",
        task="test task",
        subgoals=(
            TopologySubgoal("root", "root", "root", success_predicates=("done",)),
            TopologySubgoal("fallback", "fallback", "fallback", success_predicates=("done",)),
            TopologySubgoal(
                "main",
                "main",
                "main",
                depends_on=("root",),
                success_predicates=("done",),
            ),
        ),
        edges=(
            MissionEdge("root", "main", "success"),
            MissionEdge("root", "fallback", "failure"),
        ),
        entry_subgraph="root",
        success_subgraphs=("main",),
        failure_subgraphs=("fallback",),
        parent_scene_version=4,
    )

    class Manager:
        name = "manager"

        def propose_topology(self, _task: str, _scene: SceneSnapshot) -> MissionTopology:
            return topology

    ready_barrier = Barrier(2)

    def proposal(subgoal: AgentArtifact, _scene: SceneSnapshot, _context: AgentContext) -> SubgraphSpec:
        subgraph_id = str(subgoal.payload["subgraph_id"])
        if subgraph_id in {"root", "fallback"}:
            ready_barrier.wait(timeout=1.0)
        return _subgraph(subgraph_id, str(subgoal.payload["subgoal_id"]))

    policy = CallableGraphPolicyAgent(proposal)
    policy.name = "policy-a"  # type: ignore[attr-defined]
    scheduler = LLMGraphScheduler(
        Manager(),
        {"*": (policy,)},
        max_workers=2,
        proposal_mode="ready_wave",
    )

    result = scheduler.compile_staged("test task", _scene())

    assert result.proposal_mode == "ready_wave"
    assert result.proposal_waves == (("root", "fallback"), ("main",))
    assert result.graph.subgraph("main").description == "subgoal main"


def test_staged_assembly_infers_one_unique_predecessor_binding() -> None:
    first = replace(_subgraph("first"), outputs=(PortSpec("track_id", "track_id"),))
    second = replace(_subgraph("second"), inputs=(PortSpec("track_id", "track_id"),))
    assembled = _topology().assemble((first, second))

    assert assembled.bindings == (
        MissionBinding("first", "track_id", "second", "track_id"),
    )


def test_staged_assembly_discards_invalid_manager_binding_hint() -> None:
    first = replace(_subgraph("first"), outputs=(PortSpec("track_id", "track_id"),))
    second = replace(_subgraph("second"), inputs=(PortSpec("track_id", "track_id"),))
    topology = replace(
        _topology(),
        bindings=(MissionBinding("first", "wrong_port", "second", "track_id"),),
    )

    assembled = topology.assemble((first, second))

    assert assembled.bindings == (
        MissionBinding("first", "track_id", "second", "track_id"),
    )


def test_staged_scheduler_rejects_missing_topology_postcondition() -> None:
    class Manager:
        name = "manager"

        def propose_topology(self, _task: str, _scene: SceneSnapshot) -> MissionTopology:
            return _topology()

    policy = CallableGraphPolicyAgent(
        lambda subgoal, _scene, _context: _subgraph(
            str(subgoal.payload["subgraph_id"]),
            str(subgoal.payload["subgoal_id"]),
            predicate="different",
        )
    )
    policy.name = "policy-a"  # type: ignore[attr-defined]
    scheduler = LLMGraphScheduler(Manager(), {"*": (policy,)}, max_workers=1)

    with pytest.raises(LLMGraphScheduleError, match="success predicates"):
        scheduler.compile_staged("test task", _scene())


def test_staged_scheduler_failure_retains_candidate_context() -> None:
    class Manager:
        name = "manager"

        def propose_topology(self, _task: str, _scene: SceneSnapshot) -> MissionTopology:
            return _topology()

    policy = CallableGraphPolicyAgent(
        lambda subgoal, _scene, _context: _subgraph(
            str(subgoal.payload["subgraph_id"]),
            str(subgoal.payload["subgoal_id"]),
        )
    )
    policy.name = "policy-a"  # type: ignore[attr-defined]

    scheduler = LLMGraphScheduler(
        Manager(),
        {"*": (policy,)},
        max_workers=1,
        skill_validator=lambda _graph, _context: (_ for _ in ()).throw(
            ValueError("invalid test skill")
        ),
    )

    with pytest.raises(LLMGraphScheduleError) as error:
        scheduler.compile_staged("test task", _scene())

    failure = error.value.proposal_failures[0]
    assert failure.candidate_id == "first:policy-a:0"
    assert failure.diagnostics["node_id"] == "first_action"
    assert failure.diagnostics["skill_id"] == "noop"


def test_topology_manager_retries_with_rejection_feedback() -> None:
    class FakeLLM:
        def __init__(self) -> None:
            self.requests: list[LLMRequest] = []
            self.responses = [
                LLMResponse(
                    "request-0",
                    "",
                    structured=topology_to_dict(
                        MissionTopology(
                            mission_id="mission",
                            task="test task",
                            subgoals=(),
                            edges=(),
                            entry_subgraph="missing",
                            success_subgraphs=("missing",),
                            failure_subgraphs=("missing",),
                            parent_scene_version=4,
                        )
                    ),
                ),
                LLMResponse("request-1", "", structured=topology_to_dict(_topology())),
            ]

        def complete(self, request: LLMRequest) -> LLMResponse:
            self.requests.append(request)
            response = self.responses.pop(0)
            return LLMResponse(request.request_id, response.content, structured=response.structured)

    llm = FakeLLM()
    manager = LLMTopologyManager(
        llm,
        lambda _task, _scene: LLMRequest("request-0", "manager", ()),
        proposal_retries=1,
        repair_request_builder=lambda _task, _scene, feedback: LLMRequest(
            "request-1", "manager", ({"repair": feedback},)
        ),
    )

    result = manager.propose_topology("test task", _scene())

    assert result == _topology()
    assert len(llm.requests) == 2
    assert "UNKNOWN_ENTRY" in str(llm.requests[1].messages[0])


def test_staged_prompts_use_smaller_role_specific_outputs() -> None:
    scene = _scene()
    topology_request = build_topology_request("place object", scene, request_id="topology")
    policy_request = build_staged_policy_request(
        AgentArtifact(
            "artifact",
            "topology_subgoal",
            {"subgraph_id": "first", "subgoal_id": "first", "description": "step"},
            "manager",
        ),
        scene,
        AgentContext("task", "episode", 1, scene),
        request_id="policy",
        skill_metadata=("noop@1.0.0",),
    )

    assert topology_request.response_schema == mission_topology_response_schema()
    assert policy_request.response_schema == subgraph_response_schema()
    assert "subgraph_id is the only control-flow ID" in topology_request.messages[0]["content"]
    assert "smallest linear topology" in topology_request.messages[0]["content"]
    assert "Do not automatically create subgoals or IDs containing retry" in topology_request.messages[0]["content"]
    assert "never use subgraph_artifact" in policy_request.messages[0]["content"]
    assert "failure terminal has no success_predicates" in policy_request.messages[0]["content"]
    assert len(str(topology_request.response_schema)) < len(str(mission_graph_response_schema()))
