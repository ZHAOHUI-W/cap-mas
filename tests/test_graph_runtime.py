from __future__ import annotations

from dataclasses import replace

import pytest

from capmas.agents.arbiter import CandidateArbiter
from capmas.contracts.action import SkillCall, SkillOutputRef
from capmas.contracts.candidates import CandidateEvidence, GraphCandidate, subgraph_fingerprint
from capmas.contracts.core import SkillRef
from capmas.contracts.graph import (
    CheckpointSpec,
    GraphEdge,
    LoopSpec,
    MissionBinding,
    MissionEdge,
    MissionGraph,
    PortBinding,
    PortSpec,
    ResourceRequirement,
    SubgraphNodeSpec,
    SubgraphOutputBinding,
    SubgraphSpec,
)
from capmas.contracts.scene import SceneSnapshot
from capmas.graph.serialization import (
    GraphSchemaError,
    mission_graph_from_dict,
    mission_graph_to_dict,
)
from capmas.graph.validator import GraphValidator
from capmas.runtime.artifact_bus import ArtifactEnvelope, ArtifactStore, EventBus, RuntimeEvent
from capmas.runtime.graph_interpreter import FixedGraphInterpreter
from capmas.runtime.recovery import MappingRecoverySelector

NOOP = SkillRef("noop", "1.0.0")


def _node(node_id: str, *, resources: tuple[ResourceRequirement, ...] = ()) -> SubgraphNodeSpec:
    return SubgraphNodeSpec(
        node_id=node_id,
        description=node_id,
        skill_calls=(SkillCall(NOOP, {}),),
        postconditions=("step_done",),
        resources=resources,
        proposed_by="test-agent",
    )


def _subgraph(
    subgraph_id: str,
    *,
    nodes: tuple[SubgraphNodeSpec, ...],
    edges: tuple[GraphEdge, ...] = (),
    loops: tuple[LoopSpec, ...] = (),
) -> SubgraphSpec:
    return SubgraphSpec(
        subgraph_id=subgraph_id,
        subgoal_id=subgraph_id,
        description=subgraph_id,
        nodes=nodes,
        edges=edges,
        entry_node=nodes[0].node_id,
        success_nodes=(nodes[-1].node_id,),
        failure_nodes=(nodes[-1].node_id,),
        checkpoints=(CheckpointSpec(f"{subgraph_id}-checkpoint", ("step_done",)),),
        loops=loops,
    )


def _mission(
    *subgraphs: SubgraphSpec,
    edges: tuple[MissionEdge, ...] = (),
    loops=(),
) -> MissionGraph:
    return MissionGraph(
        mission_id="mission",
        task="test mission",
        subgraphs=subgraphs,
        edges=edges,
        bindings=(),
        entry_subgraph=subgraphs[0].subgraph_id,
        success_subgraphs=(subgraphs[-1].subgraph_id,),
        failure_subgraphs=(subgraphs[0].subgraph_id,),
        loops=loops,
    )


def test_graph_round_trip_is_versioned_and_rejects_unknown_fields() -> None:
    graph = _mission(
        _subgraph("pick", nodes=(_node("grasp"),)),
    )

    encoded = mission_graph_to_dict(graph)
    assert encoded["schema_version"] == 1
    assert mission_graph_from_dict(encoded) == graph

    encoded["unexpected"] = True
    with pytest.raises(GraphSchemaError, match="unknown fields"):
        mission_graph_from_dict(encoded)


def test_graph_deserializer_restores_legacy_call_index_skill_output_refs() -> None:
    graph = _mission(
        _subgraph(
            "pick",
            nodes=(
                SubgraphNodeSpec(
                    node_id="grasp",
                    description="grasp",
                    skill_calls=(
                        SkillCall(
                            SkillRef("goto_pose", "1.0.0"),
                            {
                                "position": SkillOutputRef(0, ("result", 0)),
                            },
                        ),
                    ),
                    postconditions=("step_done",),
                    proposed_by="test-agent",
                ),
            ),
        ),
    )

    decoded = mission_graph_from_dict(mission_graph_to_dict(graph))
    decoded_args = decoded.subgraphs[0].nodes[0].skill_calls[0].args

    assert decoded_args["position"] == SkillOutputRef(0, ("result", 0))

    legacy = mission_graph_to_dict(graph)
    legacy["subgraphs"][0]["nodes"][0]["skill_calls"][0]["args"]["position"] = {
        "call_index": 0,
        "path": ["result", 0],
    }
    legacy_decoded = mission_graph_from_dict(legacy)

    assert legacy_decoded.subgraphs[0].nodes[0].skill_calls[0].args["position"] == SkillOutputRef(
        0,
        ("result", 0),
    )


def test_validator_rejects_unbounded_cycle_but_accepts_explicit_loop() -> None:
    cyclic = _subgraph(
        "pick",
        nodes=(_node("grasp"), _node("retry")),
        edges=(GraphEdge("grasp", "retry"), GraphEdge("retry", "grasp", "retry")),
    )
    graph = _mission(cyclic)
    result = GraphValidator().validate(graph)
    assert any(d.code == "UNBOUNDED_CYCLE" for d in result.diagnostics)

    bounded = replace(
        cyclic,
        loops=(LoopSpec("grasp", max_visits=2, exit_conditions=("success", "failure")),),
    )
    result = GraphValidator().validate(_mission(bounded))
    assert result.valid is True


def test_fixed_graph_interpreter_dispatches_action_nodes_in_order() -> None:
    first = _subgraph("pick", nodes=(_node("grasp"),))
    second = _subgraph("place", nodes=(_node("place"),))
    graph = _mission(first, second, edges=(MissionEdge("pick", "place", "success"),))
    scene = SceneSnapshot("episode", 1, 0, 1, 2, {})

    class Scheduler:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def dispatch(self, contract, current_scene):
            from capmas.contracts.trace import ExecutionTrace
            from capmas.contracts.verification import PredicateReport, VerificationResult
            from capmas.runtime.orchestrator import CycleResult

            self.calls.append(contract.subgoal_id)
            after = replace(current_scene, scene_version=current_scene.scene_version + 1)
            verification = VerificationResult(
                contract.contract_id,
                "commit",
                after.scene_version,
                (PredicateReport("step_done", True),),
            )
            trace = ExecutionTrace(
                trace_id=f"trace-{len(self.calls)}",
                episode_id=current_scene.episode_id,
                episode_epoch=current_scene.episode_epoch,
                contract_id=contract.contract_id,
                lease_id=f"lease-{len(self.calls)}",
                parent_scene_version=current_scene.scene_version,
                start_scene_version=current_scene.scene_version,
                end_scene_version=after.scene_version,
                started_at_ns=1,
                finished_at_ns=2,
                status="completed",
            )
            return CycleResult(True, current_scene, after, trace, verification)

    scheduler = Scheduler()
    result = FixedGraphInterpreter(scheduler).run(
        graph,
        scene,
        episode_id="episode",
        episode_epoch=1,
    )

    assert result.completed is True
    assert result.terminal_subgraph == "place"
    assert scheduler.calls == ["pick", "place"]
    assert result.scene.scene_version == 2
    assert result.traces[0].metadata["subgraph_id"] == "pick"
    assert result.traces[0].metadata["node_id"] == "grasp"
    assert result.traces[0].metadata["candidate_fingerprint"] == subgraph_fingerprint(first)


def test_fixed_graph_interpreter_can_stop_after_one_verified_subgraph() -> None:
    first = _subgraph("pick", nodes=(_node("grasp"),))
    second = _subgraph("place", nodes=(_node("place"),))
    graph = _mission(first, second, edges=(MissionEdge("pick", "place", "success"),))
    scene = SceneSnapshot("episode", 1, 0, 1, 2, {})

    class Scheduler:
        def dispatch(self, contract, current_scene):
            from capmas.contracts.trace import ExecutionTrace
            from capmas.contracts.verification import PredicateReport, VerificationResult
            from capmas.runtime.orchestrator import CycleResult

            after = replace(current_scene, scene_version=current_scene.scene_version + 1)
            verification = VerificationResult(
                contract.contract_id,
                "commit",
                after.scene_version,
                (PredicateReport("step_done", True),),
            )
            trace = ExecutionTrace(
                trace_id="trace-one",
                episode_id=current_scene.episode_id,
                episode_epoch=current_scene.episode_epoch,
                contract_id=contract.contract_id,
                lease_id="lease-one",
                parent_scene_version=current_scene.scene_version,
                start_scene_version=current_scene.scene_version,
                end_scene_version=after.scene_version,
                started_at_ns=1,
                finished_at_ns=2,
                status="completed",
            )
            return CycleResult(True, current_scene, after, trace, verification)

    result = FixedGraphInterpreter(Scheduler()).run(
        graph,
        scene,
        stop_after_subgraph=True,
    )

    assert result.completed is False
    assert result.failure is None
    assert result.terminal_subgraph == "pick"
    assert result.next_subgraph == "place"
    assert result.scene.scene_version == 1


def test_fixed_graph_interpreter_normalizes_completed_local_transition() -> None:
    action = _node("place")
    checkpoint = SubgraphNodeSpec(
        node_id="validate",
        description="validate",
        node_type="checkpoint",
        postconditions=("step_done",),
    )
    subgraph = replace(
        _subgraph("place", nodes=(action, checkpoint)),
        edges=(GraphEdge("place", "validate", "completed"),),
        entry_node="place",
        success_nodes=("validate",),
        failure_nodes=("place",),
    )
    graph = _mission(subgraph)
    scene = SceneSnapshot("episode", 1, 0, 1, 2, {})

    class Scheduler:
        def dispatch(self, contract, current_scene):
            from capmas.contracts.trace import ExecutionTrace
            from capmas.contracts.verification import PredicateReport, VerificationResult
            from capmas.runtime.orchestrator import CycleResult

            after = replace(current_scene, scene_version=1)
            verification = VerificationResult(
                contract.contract_id,
                "commit",
                1,
                (PredicateReport("step_done", True),),
            )
            trace = ExecutionTrace(
                "trace", "episode", 1, contract.contract_id, "lease", 0, 0, 1, 1, 2, "completed"
            )
            return CycleResult(True, current_scene, after, trace, verification)

    result = FixedGraphInterpreter(Scheduler()).run(graph, scene)

    assert result.completed is True
    assert result.terminal_subgraph == "place"


def test_fixed_graph_interpreter_normalizes_action_complete_local_transition() -> None:
    action = _node("place")
    checkpoint = SubgraphNodeSpec(
        node_id="validate",
        description="validate",
        node_type="checkpoint",
        postconditions=("step_done",),
    )
    subgraph = replace(
        _subgraph("place", nodes=(action, checkpoint)),
        edges=(GraphEdge("place", "validate", "action_complete"),),
        entry_node="place",
        success_nodes=("validate",),
        failure_nodes=("place",),
    )
    graph = _mission(subgraph)
    scene = SceneSnapshot("episode", 1, 0, 1, 2, {})

    class Scheduler:
        def dispatch(self, contract, current_scene):
            from capmas.contracts.trace import ExecutionTrace
            from capmas.contracts.verification import PredicateReport, VerificationResult
            from capmas.runtime.orchestrator import CycleResult

            after = replace(current_scene, scene_version=1)
            verification = VerificationResult(
                contract.contract_id,
                "commit",
                1,
                (PredicateReport("step_done", True),),
            )
            trace = ExecutionTrace(
                "trace", "episode", 1, contract.contract_id, "lease", 0, 0, 1, 1, 2, "completed"
            )
            return CycleResult(True, current_scene, after, trace, verification)

    result = FixedGraphInterpreter(Scheduler()).run(graph, scene)

    assert result.completed is True
    assert result.terminal_subgraph == "place"


def test_fixed_graph_interpreter_evaluates_checkpoint_predicates() -> None:
    checkpoint = SubgraphNodeSpec(
        node_id="scene-check",
        description="check scene",
        node_type="checkpoint",
        postconditions=("track_exists:bowl",),
    )
    failure = SubgraphNodeSpec(
        node_id="scene-failure",
        description="record scene check failure",
        node_type="checkpoint",
    )
    subgraph = SubgraphSpec(
        subgraph_id="scene",
        subgoal_id="scene",
        description="scene check",
        nodes=(checkpoint, failure),
        edges=(GraphEdge("scene-check", "scene-failure", "failure"),),
        entry_node="scene-check",
        success_nodes=("scene-check",),
        failure_nodes=("scene-failure",),
        checkpoints=(CheckpointSpec("scene-check", ("track_exists:bowl",)),),
    )
    graph = _mission(subgraph)

    class Scheduler:
        def dispatch(self, *_args: object, **_kwargs: object) -> object:
            raise AssertionError("checkpoint test must not dispatch a robot action")

    result = FixedGraphInterpreter(
        Scheduler(),
        checkpoint_evaluator=lambda _subgraph, _node, _context: "failure",
    ).run(graph, SceneSnapshot("episode", 1, 0, 1, 1, {}))

    assert result.completed is False
    assert result.failure is not None
    assert result.failure.subgraph_id == "scene"


def test_fixed_graph_interpreter_resolves_local_and_mission_port_bindings() -> None:
    pick_node = SubgraphNodeSpec(
        node_id="observe",
        description="observe",
        skill_calls=(SkillCall(NOOP, {"emit": "pose"}),),
        outputs=(PortSpec("pose", "Pose"),),
        postconditions=("step_done",),
        proposed_by="test-agent",
    )
    pick = replace(
        _subgraph("pick", nodes=(pick_node,)),
        outputs=(PortSpec("pose", "Pose"),),
        output_bindings=(SubgraphOutputBinding("observe", "pose", "pose"),),
    )

    prepare = SubgraphNodeSpec(
        node_id="prepare",
        description="prepare",
        skill_calls=(SkillCall(NOOP, {"emit": "local_pose"}),),
        inputs=(PortSpec("mission_pose", "Pose"),),
        outputs=(PortSpec("pose", "Pose"),),
        postconditions=("step_done",),
        proposed_by="test-agent",
    )
    use_pose = SubgraphNodeSpec(
        node_id="use_pose",
        description="use pose",
        skill_calls=(SkillCall(NOOP, {}),),
        inputs=(PortSpec("pose", "Pose"),),
        postconditions=("step_done",),
        proposed_by="test-agent",
    )
    place = replace(
        _subgraph(
            "place",
            nodes=(prepare, use_pose),
            edges=(GraphEdge("prepare", "use_pose"),),
        ),
        inputs=(PortSpec("mission_pose", "Pose"),),
        bindings=(PortBinding("prepare", "pose", "use_pose", "pose"),),
        output_bindings=(SubgraphOutputBinding("prepare", "pose", "placed_pose"),),
        outputs=(PortSpec("placed_pose", "Pose"),),
    )
    graph = replace(
        _mission(
            pick,
            place,
            edges=(MissionEdge("pick", "place", "success"),),
        ),
        bindings=(MissionBinding("pick", "pose", "place", "mission_pose"),),
    )
    scene = SceneSnapshot("episode", 1, 0, 1, 2, {})

    class Scheduler:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        def dispatch(self, contract, current_scene):
            from capmas.contracts.trace import ExecutionTrace, SkillTrace
            from capmas.contracts.verification import PredicateReport, VerificationResult
            from capmas.runtime.orchestrator import CycleResult

            args = contract.skills[0].args
            self.calls.append((contract.subgoal_id, args))
            emitted = args.get("emit")
            output = {"pose": f"{emitted}-value"} if emitted else {}
            after = replace(current_scene, scene_version=current_scene.scene_version + 1)
            verification = VerificationResult(
                contract.contract_id,
                "commit",
                after.scene_version,
                (PredicateReport("step_done", True),),
            )
            trace = ExecutionTrace(
                trace_id=f"trace-{len(self.calls)}",
                episode_id=current_scene.episode_id,
                episode_epoch=current_scene.episode_epoch,
                contract_id=contract.contract_id,
                lease_id=f"lease-{len(self.calls)}",
                parent_scene_version=current_scene.scene_version,
                start_scene_version=current_scene.scene_version,
                end_scene_version=after.scene_version,
                started_at_ns=1,
                finished_at_ns=2,
                status="completed",
                skill_traces=(
                    SkillTrace(
                        invocation_id=f"invocation-{len(self.calls)}",
                        skill_id="noop",
                        skill_version="1.0.0",
                        args=args,
                        started_at_ns=1,
                        finished_at_ns=2,
                        status="completed",
                        output=output,
                    ),
                ),
            )
            return CycleResult(True, current_scene, after, trace, verification)

    scheduler = Scheduler()
    result = FixedGraphInterpreter(scheduler).run(graph, scene)

    assert result.completed is True
    assert scheduler.calls[-2] == (
        "place",
        {"emit": "local_pose", "mission_pose": "pose-value"},
    )
    assert scheduler.calls[-1] == ("place", {"pose": "local_pose-value"})
    assert result.outputs["pick"]["pose"] == "pose-value"
    assert result.outputs["place"]["placed_pose"] == "local_pose-value"


def test_arbiter_selects_valid_candidate_for_current_scene() -> None:
    scene = SceneSnapshot("episode", 1, 4, 1, 2, {})
    invalid = GraphCandidate(
        "invalid",
        _subgraph("bad", nodes=(_node("a"), _node("b")), edges=(GraphEdge("a", "missing"),)),
        4,
        "policy-a",
        confidence=0.99,
    )
    valid = GraphCandidate(
        "valid",
        _subgraph("good", nodes=(_node("a"),)),
        4,
        "policy-b",
        confidence=0.8,
    )

    result = CandidateArbiter().select((invalid, valid), scene)

    assert result.selected == valid
    assert any(rejection.candidate_id == "invalid" for rejection in result.rejections)


def test_arbiter_uses_rehearsal_ood_and_latency_evidence_for_ranking() -> None:
    scene = SceneSnapshot("episode", 1, 4, 1, 2, {})
    nominal = GraphCandidate(
        "nominal",
        _subgraph("pick", nodes=(_node("a"),)),
        4,
        "policy-a",
        confidence=0.95,
        evidence=CandidateEvidence(
            verifier_pass_rate=0.60,
            rehearsal_success_rate=0.55,
            ood_success_rate=0.20,
            expected_latency_ms=900,
            recovery_cost=0.8,
            evidence_refs=("trace://nominal",),
        ),
    )
    robust = GraphCandidate(
        "robust",
        _subgraph("pick", nodes=(_node("b"),)),
        4,
        "policy-b",
        confidence=0.80,
        evidence=CandidateEvidence(
            verifier_pass_rate=0.90,
            rehearsal_success_rate=0.88,
            ood_success_rate=0.85,
            expected_latency_ms=500,
            recovery_cost=0.2,
            evidence_refs=("trace://robust",),
        ),
    )

    arbiter = CandidateArbiter()
    result = arbiter.select((nominal, robust), scene)

    assert result.selected == robust
    assert arbiter.score(robust) > arbiter.score(nominal)


def test_arbiter_marks_selection_as_tie_break_when_candidates_have_no_evidence() -> None:
    scene = SceneSnapshot("episode", 1, 4, 1, 2, {})
    first = GraphCandidate(
        "policy-a",
        _subgraph("pick", nodes=(_node("a"),)),
        4,
        "policy-a",
        confidence=0.5,
    )
    second = GraphCandidate(
        "policy-b",
        _subgraph("pick", nodes=(_node("b"),)),
        4,
        "policy-b",
        confidence=0.5,
    )

    result = CandidateArbiter().select((first, second), scene)

    assert result.selected is not None
    assert result.selection_basis == "confidence_fallback"
    assert result.tie_broken is True


def test_artifact_store_and_event_bus_are_typed_and_isolated() -> None:
    store = ArtifactStore()
    envelope = ArtifactEnvelope("candidate-1", "graph_candidate", {"ok": True}, "policy-a", 3)
    store.put(envelope)
    assert store.get("candidate-1") == envelope
    with pytest.raises(ValueError, match="already exists"):
        store.put(envelope)

    bus = EventBus()
    received: list[str] = []
    unsubscribe = bus.subscribe("candidate", lambda event: received.append(event.event_id))
    bus.publish(RuntimeEvent("event-1", "candidate", 3, envelope))
    unsubscribe()
    bus.publish(RuntimeEvent("event-2", "candidate", 3, envelope))
    assert received == ["event-1"]


def test_failure_artifact_routes_through_store_bus_and_recovery_selector() -> None:
    failure = __import__("capmas.contracts.failures", fromlist=["FailureArtifact"]).FailureArtifact(
        "failure-1",
        "POSTCONDITION_FAILED",
        "object was not placed",
        5,
        node_id="place",
        subgraph_id="place",
        recovery_policy="reacquire_and_retry",
        evidence_refs=("artifact://rgbd/5",),
    )
    store = ArtifactStore()
    bus = EventBus()
    seen: list[str] = []
    bus.subscribe("failure", lambda event: seen.append(event.artifact.artifact_id))

    envelope = ArtifactEnvelope(
        failure.failure_id,
        "failure",
        {"failure": failure},
        "runtime",
        failure.scene_version,
    )
    store.put(envelope)
    bus.publish(RuntimeEvent("event-failure-1", "failure", 5, envelope))

    decision = MappingRecoverySelector(
        {"POSTCONDITION_FAILED": "reacquire", "*": "abort"}
    ).select(failure, None)

    assert seen == ["failure-1"]
    assert decision is not None
    assert decision.target_subgraph == "reacquire"


def test_interpreter_publishes_failure_once_and_uses_declared_recovery_edge() -> None:
    failing_node = _node("place")
    failing = _subgraph("place", nodes=(failing_node,))
    recovery = _subgraph("reacquire", nodes=(_node("reacquire"),))
    graph = _mission(
        failing,
        recovery,
        edges=(
            MissionEdge("place", "reacquire", "POSTCONDITION_FAILED"),
        ),
    )
    scene = SceneSnapshot("episode", 1, 0, 1, 2, {})
    events: list[str] = []
    store = ArtifactStore()
    bus = EventBus()
    bus.subscribe("failure", lambda event: events.append(event.artifact.artifact_id))

    class Scheduler:
        def __init__(self) -> None:
            self.calls = 0

        def dispatch(self, contract, current_scene):
            from capmas.contracts.trace import ExecutionTrace
            from capmas.contracts.verification import VerificationResult
            from capmas.runtime.orchestrator import CycleResult

            self.calls += 1
            after = replace(current_scene, scene_version=current_scene.scene_version + 1)
            verification = VerificationResult(
                contract.contract_id,
                "recover" if self.calls == 1 else "commit",
                after.scene_version,
                failure_class="POSTCONDITION_FAILED" if self.calls == 1 else None,
            )
            trace = ExecutionTrace(
                trace_id=f"trace-{self.calls}",
                episode_id=current_scene.episode_id,
                episode_epoch=current_scene.episode_epoch,
                contract_id=contract.contract_id,
                lease_id=f"lease-{self.calls}",
                parent_scene_version=current_scene.scene_version,
                start_scene_version=current_scene.scene_version,
                end_scene_version=after.scene_version,
                started_at_ns=1,
                finished_at_ns=2,
                status="failed" if self.calls == 1 else "completed",
            )
            return CycleResult(self.calls > 1, current_scene, after, trace, verification)

    result = FixedGraphInterpreter(
        Scheduler(),
        artifact_store=store,
        event_bus=bus,
        recovery_selector=MappingRecoverySelector(
            {"POSTCONDITION_FAILED": "reacquire"}
        ),
    ).run(graph, scene)

    assert result.completed is True
    assert len(store.snapshot()) == 1
    assert events == [store.snapshot()[0].artifact_id]


def test_interpreter_routes_structured_precondition_rejection_to_recovery() -> None:
    failing = _subgraph("place", nodes=(_node("place"),))
    recovery = _subgraph("reacquire", nodes=(_node("reacquire"),))
    graph = _mission(
        failing,
        recovery,
        edges=(MissionEdge("place", "reacquire", "PRECONDITION_FAILED"),),
    )
    scene = SceneSnapshot("episode", 1, 0, 1, 2, {})
    store = ArtifactStore()

    class Scheduler:
        def __init__(self) -> None:
            self.calls = 0

        def dispatch(self, contract, current_scene):
            from capmas.contracts.trace import ExecutionTrace
            from capmas.contracts.verification import PredicateReport, VerificationResult
            from capmas.runtime.orchestrator import CycleResult

            self.calls += 1
            if self.calls == 1:
                verification = VerificationResult(
                    contract.contract_id,
                    "reject",
                    current_scene.scene_version,
                    (
                        PredicateReport(
                            "object_in_gripper(bowl)",
                            False,
                            confidence=0.0,
                            reason="gripper is not closed",
                        ),
                    ),
                    failure_class="PRECONDITION_FAILED",
                )
                trace = ExecutionTrace(
                    "rejected-trace",
                    "episode",
                    1,
                    contract.contract_id,
                    "not_acquired",
                    0,
                    0,
                    0,
                    1,
                    2,
                    "rejected",
                    precondition_result=verification,
                    failure_class="PRECONDITION_FAILED",
                )
                return CycleResult(
                    False,
                    current_scene,
                    current_scene,
                    trace,
                    verification,
                    rejected=True,
                    reason="object_in_gripper(bowl): gripper is not closed",
                )

            after = replace(current_scene, scene_version=current_scene.scene_version + 1)
            verification = VerificationResult(
                contract.contract_id,
                "commit",
                after.scene_version,
                (PredicateReport("step_done", True),),
            )
            trace = ExecutionTrace(
                "recovery-trace",
                "episode",
                1,
                contract.contract_id,
                "recovery-lease",
                current_scene.scene_version,
                current_scene.scene_version,
                after.scene_version,
                1,
                2,
                "completed",
            )
            return CycleResult(True, current_scene, after, trace, verification)

    result = FixedGraphInterpreter(
        Scheduler(),
        artifact_store=store,
        recovery_selector=MappingRecoverySelector(
            {"PRECONDITION_FAILED": "reacquire"}
        ),
    ).run(graph, scene)

    assert result.completed is True
    assert result.terminal_subgraph == "reacquire"
    assert len(store.snapshot()) == 1
    failure = store.snapshot()[0].payload["failure"]
    assert failure.failure_class == "PRECONDITION_FAILED"
    assert failure.metadata["rejected"] is True
    assert failure.metadata["predicate_results"][0]["name"] == "object_in_gripper(bowl)"


def test_interpreter_preserves_failure_class_through_failure_checkpoint() -> None:
    action = _node("place-action")
    checkpoint = SubgraphNodeSpec(
        node_id="place-failure-checkpoint",
        description="record failed placement",
        node_type="checkpoint",
        postconditions=("scene_fresh(1000)",),
    )
    failing = replace(
        _subgraph("place", nodes=(action, checkpoint)),
        edges=(GraphEdge("place-action", "place-failure-checkpoint", "verification_failed_or_action_failed"),),
        entry_node="place-action",
        success_nodes=("place-action",),
        failure_nodes=("place-failure-checkpoint",),
    )
    recovery = _subgraph("reacquire", nodes=(_node("reacquire"),))
    graph = _mission(
        failing,
        recovery,
        edges=(MissionEdge("place", "reacquire", "PRECONDITION_FAILED"),),
    )
    scene = SceneSnapshot("episode", 1, 0, 1, 2, {})

    class Scheduler:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def dispatch(self, contract, current_scene):
            from capmas.contracts.trace import ExecutionTrace
            from capmas.contracts.verification import PredicateReport, VerificationResult
            from capmas.runtime.orchestrator import CycleResult

            self.calls.append(contract.subgoal_id)
            after = replace(current_scene, scene_version=current_scene.scene_version + 1)
            verification = VerificationResult(
                contract.contract_id,
                "commit",
                after.scene_version,
                (PredicateReport("step_done", True),),
            )
            trace = ExecutionTrace(
                trace_id=f"trace-{len(self.calls)}",
                episode_id="episode",
                episode_epoch=1,
                contract_id=contract.contract_id,
                lease_id="lease",
                parent_scene_version=current_scene.scene_version,
                start_scene_version=current_scene.scene_version,
                end_scene_version=after.scene_version,
                started_at_ns=1,
                finished_at_ns=2,
                status="completed",
            )
            return CycleResult(True, current_scene, after, trace, verification)

    class RejectingScheduler(Scheduler):
        def dispatch(self, contract, current_scene):
            from capmas.contracts.trace import ExecutionTrace
            from capmas.contracts.verification import PredicateReport, VerificationResult
            from capmas.runtime.orchestrator import CycleResult

            if not self.calls:
                verification = VerificationResult(
                    contract.contract_id,
                    "reject",
                    current_scene.scene_version,
                    (PredicateReport("gripper_closed()", False, reason="open"),),
                    failure_class="PRECONDITION_FAILED",
                )
                trace = ExecutionTrace(
                    "rejected",
                    "episode",
                    1,
                    contract.contract_id,
                    "not_acquired",
                    0,
                    0,
                    0,
                    1,
                    2,
                    "rejected",
                    precondition_result=verification,
                    failure_class="PRECONDITION_FAILED",
                )
                self.calls.append(contract.subgoal_id)
                return CycleResult(
                    False,
                    current_scene,
                    current_scene,
                    trace,
                    verification,
                    rejected=True,
                    reason="gripper_closed(): open",
                )
            return super().dispatch(contract, current_scene)

    scheduler = RejectingScheduler()
    result = FixedGraphInterpreter(
        scheduler,
        recovery_selector=MappingRecoverySelector(
            {"PRECONDITION_FAILED": "reacquire"}
        ),
    ).run(graph, scene)

    assert result.completed is True
    assert scheduler.calls == ["place", "reacquire"]
