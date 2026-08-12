from __future__ import annotations

from dataclasses import replace

from capmas.contracts.action import SkillCall
from capmas.contracts.calibration import horizon_bucket
from capmas.contracts.core import SkillRef
from capmas.contracts.graph import (
    CheckpointSpec,
    GraphEdge,
    LoopSpec,
    MissionEdge,
    MissionGraph,
    SubgraphNodeSpec,
    SubgraphSpec,
)
from capmas.contracts.scene import SceneSnapshot
from capmas.contracts.trace import ExecutionTrace, GraphExecutionEvent
from capmas.contracts.verification import PredicateReport, VerificationResult
from capmas.evaluation.labels import extract_horizon, planned_horizon
from capmas.runtime.graph_interpreter import FixedGraphInterpreter
from capmas.runtime.orchestrator import CycleResult
from capmas.runtime.recovery import MappingRecoverySelector

NOOP = SkillRef("noop", "1.0.0")


def _scene() -> SceneSnapshot:
    return SceneSnapshot("episode", 1, 0, 1, 2, {})


def _action(node_id: str) -> SubgraphNodeSpec:
    return SubgraphNodeSpec(
        node_id=node_id,
        description=node_id,
        skill_calls=(SkillCall(NOOP, {}),),
        postconditions=("step_done",),
        proposed_by="test-agent",
    )


def _checkpoint(node_id: str) -> SubgraphNodeSpec:
    return SubgraphNodeSpec(
        node_id=node_id,
        description=node_id,
        node_type="checkpoint",
        postconditions=("step_done",),
    )


def _subgraph(
    subgraph_id: str,
    nodes: tuple[SubgraphNodeSpec, ...],
    *,
    edges: tuple[GraphEdge, ...] = (),
) -> SubgraphSpec:
    return SubgraphSpec(
        subgraph_id=subgraph_id,
        subgoal_id=subgraph_id,
        description=subgraph_id,
        nodes=nodes,
        edges=edges,
        entry_node=nodes[0].node_id,
        success_nodes=(nodes[-1].node_id,),
        failure_nodes=(nodes[0].node_id,),
        checkpoints=(CheckpointSpec(f"{subgraph_id}-check", ("step_done",)),),
    )


def _mission(
    *subgraphs: SubgraphSpec,
    edges: tuple[MissionEdge, ...] = (),
    loops: tuple[LoopSpec, ...] = (),
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


def _action_then_checkpoint_mission() -> MissionGraph:
    return _mission(
        _subgraph(
            "pick",
            (_action("grasp"), _checkpoint("verify")),
            edges=(GraphEdge("grasp", "verify", "success"),),
        )
    )


def _two_action_subgraphs_then_checkpoint_only_mission() -> MissionGraph:
    pick = _subgraph("pick", (_action("grasp"),))
    place = _subgraph("place", (_action("place"),))
    verify = _subgraph("verify", (_checkpoint("verify"),))
    return _mission(
        pick,
        place,
        verify,
        edges=(
            MissionEdge("pick", "place", "success"),
            MissionEdge("place", "verify", "completed"),
        ),
    )


def _single_action_loop_mission(max_visits: int) -> MissionGraph:
    pick = _subgraph("pick", (_action("grasp"),))
    return _mission(
        pick,
        edges=(MissionEdge("pick", "pick", "success"),),
        loops=(LoopSpec("pick", max_visits=max_visits),),
    )


class _SuccessfulScheduler:
    def dispatch(self, contract, current_scene):
        after = replace(current_scene, scene_version=current_scene.scene_version + 1)
        verification = VerificationResult(
            contract.contract_id,
            "commit",
            after.scene_version,
            (PredicateReport("step_done", True),),
        )
        trace = ExecutionTrace(
            "trace",
            current_scene.episode_id,
            current_scene.episode_epoch,
            contract.contract_id,
            "lease",
            current_scene.scene_version,
            current_scene.scene_version,
            after.scene_version,
            1,
            2,
            "completed",
        )
        return CycleResult(True, current_scene, after, trace, verification)


def _event(
    sequence: int,
    kind: str,
    subgraph_id: str,
    node_id: str | None = None,
    node_type: str | None = None,
    *,
    attempt: int,
) -> GraphExecutionEvent:
    return GraphExecutionEvent(
        sequence=sequence,
        kind=kind,  # type: ignore[arg-type]
        subgraph_id=subgraph_id,
        node_id=node_id,
        node_type=node_type,  # type: ignore[arg-type]
        attempt=attempt,
        outcome=None,
        occurred_at_ns=sequence,
    )


def test_interpreter_records_action_and_checkpoint_events_in_order() -> None:
    graph = _action_then_checkpoint_mission()
    result = FixedGraphInterpreter(
        _SuccessfulScheduler(), clock=iter(range(100)).__next__
    ).run(graph, _scene())

    assert [(event.kind, event.node_type) for event in result.events] == [
        ("subgraph_started", None),
        ("node_started", "action"),
        ("node_completed", "action"),
        ("node_started", "checkpoint"),
        ("node_completed", "checkpoint"),
        ("subgraph_completed", None),
    ]
    assert [event.sequence for event in result.events] == list(range(6))


def test_checkpoint_only_subgraph_does_not_inflate_planned_bucket() -> None:
    graph = _two_action_subgraphs_then_checkpoint_only_mission()
    label = planned_horizon(graph)

    assert label.planned_critical_path_actions == 2
    assert label.planned_critical_path_subgoals == 2
    assert label.planned_checkpoint_subgraphs == 1
    assert horizon_bucket(label) == "H2-3"


def test_realized_horizon_counts_reentry_as_an_attempt() -> None:
    graph = _single_action_loop_mission(max_visits=2)
    events = (
        _event(0, "subgraph_started", "pick", attempt=1),
        _event(1, "node_started", "pick", "grasp", "action", attempt=1),
        _event(2, "node_failed", "pick", "grasp", "action", attempt=1),
        _event(3, "subgraph_failed", "pick", attempt=1),
        _event(4, "subgraph_started", "pick", attempt=2),
        _event(5, "node_started", "pick", "grasp", "action", attempt=2),
        _event(6, "node_completed", "pick", "grasp", "action", attempt=2),
        _event(7, "subgraph_completed", "pick", attempt=2),
    )

    label = extract_horizon(graph, events)

    assert label.attempted_actions == 2
    assert label.completed_actions == 1
    assert label.attempted_subgoals == 2
    assert label.completed_subgoals == 1


def test_interpreter_records_terminal_subgraph_attempts_after_recovery_reentry() -> None:
    pick = _subgraph("pick", (_action("grasp"),))
    done = _subgraph("done", (_checkpoint("verify"),))
    graph = _mission(
        pick,
        done,
        edges=(
            MissionEdge("pick", "pick", "POSTCONDITION_FAILED"),
            MissionEdge("pick", "done", "success"),
        ),
        loops=(LoopSpec("pick", max_visits=2),),
    )

    class _FailOnceScheduler:
        def __init__(self) -> None:
            self.calls = 0

        def dispatch(self, contract, current_scene):
            self.calls += 1
            committed = self.calls > 1
            after = replace(current_scene, scene_version=current_scene.scene_version + 1)
            verification = VerificationResult(
                contract.contract_id,
                "commit" if committed else "recover",
                after.scene_version,
                failure_class=None if committed else "POSTCONDITION_FAILED",
            )
            trace = ExecutionTrace(
                f"trace-{self.calls}",
                current_scene.episode_id,
                current_scene.episode_epoch,
                contract.contract_id,
                "lease",
                current_scene.scene_version,
                current_scene.scene_version,
                after.scene_version,
                1,
                2,
                "completed" if committed else "failed",
            )
            return CycleResult(committed, current_scene, after, trace, verification)

    result = FixedGraphInterpreter(
        _FailOnceScheduler(),
        recovery_selector=MappingRecoverySelector({"POSTCONDITION_FAILED": "pick"}),
    ).run(graph, _scene())

    assert [
        (event.kind, event.node_type, event.attempt)
        for event in result.events
    ] == [
        ("subgraph_started", None, 1),
        ("node_started", "action", 1),
        ("node_failed", "action", 1),
        ("subgraph_failed", None, 1),
        ("subgraph_started", None, 2),
        ("node_started", "action", 2),
        ("node_completed", "action", 2),
        ("subgraph_completed", None, 2),
        ("subgraph_started", None, 1),
        ("node_started", "checkpoint", 1),
        ("node_completed", "checkpoint", 1),
        ("subgraph_completed", None, 1),
    ]
    assert [event.sequence for event in result.events] == list(range(12))


def test_max_steps_is_not_used_as_horizon() -> None:
    graph = _two_action_subgraphs_then_checkpoint_only_mission()
    result = FixedGraphInterpreter(
        _SuccessfulScheduler(), max_steps=32
    ).run(graph, _scene())

    assert extract_horizon(graph, result.events).planned_critical_path_subgoals == 2
