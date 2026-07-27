from __future__ import annotations

from dataclasses import replace

import pytest

from capmas.agents.policy import CallableGraphPolicyAgent
from capmas.contracts.action import SkillCall
from capmas.contracts.agent import AgentArtifact, AgentContext
from capmas.contracts.graph import CheckpointSpec, MissionEdge, MissionGraph, SubgraphNodeSpec, SubgraphSpec
from capmas.contracts.core import SkillRef
from capmas.contracts.scene import SceneSnapshot
from capmas.contracts.staged import MissionTopology, TopologySubgoal
from capmas.graph.validator import GraphValidator
from capmas.runtime.graph_interpreter import FixedGraphInterpreter
from capmas.runtime.llm_scheduler import LLMGraphCompileResult, LLMGraphScheduleError
from capmas.runtime.llm_scheduler import LLMGraphScheduler
from capmas.runtime.rolling import RollingGraphError, RollingGraphRunner


def _subgraph(subgraph_id: str) -> SubgraphSpec:
    node = SubgraphNodeSpec(
        node_id=f"{subgraph_id}-action",
        description=subgraph_id,
        skill_calls=(SkillCall(SkillRef("noop", "1.0.0"), {}),),
        postconditions=("step_done",),
        proposed_by="test",
    )
    return SubgraphSpec(
        subgraph_id=subgraph_id,
        subgoal_id=subgraph_id,
        description=subgraph_id,
        nodes=(node,),
        edges=(),
        entry_node=node.node_id,
        success_nodes=(node.node_id,),
        failure_nodes=(node.node_id,),
        checkpoints=(CheckpointSpec(f"{subgraph_id}-checkpoint", ("step_done",)),),
    )


def _graph() -> MissionGraph:
    return MissionGraph(
        mission_id="mission",
        task="test task",
        subgraphs=(_subgraph("pick"), _subgraph("place")),
        edges=(MissionEdge("pick", "place", "success"),),
        bindings=(),
        entry_subgraph="pick",
        success_subgraphs=("place",),
        failure_subgraphs=("pick", "place"),
    )


class _Planner:
    def __init__(self, graph: MissionGraph) -> None:
        self.graph = graph
        self.scenes: list[int] = []

    def compile(self, _task: str, scene: SceneSnapshot, *, context, protocol: str):
        self.scenes.append(scene.scene_version)
        return LLMGraphCompileResult(self.graph, {})


class _RobotScheduler:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def dispatch(self, contract, scene):
        from capmas.contracts.trace import ExecutionTrace
        from capmas.contracts.verification import PredicateReport, VerificationResult
        from capmas.runtime.orchestrator import CycleResult

        self.calls.append(contract.subgoal_id)
        after = replace(scene, scene_version=scene.scene_version + 1)
        verification = VerificationResult(
            contract.contract_id,
            "commit",
            after.scene_version,
            (PredicateReport("step_done", True),),
        )
        trace = ExecutionTrace(
            trace_id=f"trace-{len(self.calls)}",
            episode_id=scene.episode_id,
            episode_epoch=scene.episode_epoch,
            contract_id=contract.contract_id,
            lease_id=f"lease-{len(self.calls)}",
            parent_scene_version=scene.scene_version,
            start_scene_version=scene.scene_version,
            end_scene_version=after.scene_version,
            started_at_ns=1,
            finished_at_ns=2,
            status="completed",
        )
        return CycleResult(True, scene, after, trace, verification)


def test_rolling_runner_replans_from_each_committed_scene() -> None:
    scene = SceneSnapshot("episode", 1, 0, 1, 2, {})
    planner = _Planner(_graph())
    robot = _RobotScheduler()
    interpreter = FixedGraphInterpreter(robot)

    result = RollingGraphRunner().run(
        "test task",
        scene,
        planner,
        interpreter,
        protocol="staged",
        max_cycles=4,
    )

    assert result.completed is True
    assert result.scene.scene_version == 2
    assert result.replan_count == 1
    assert planner.scenes == [0, 1]
    assert robot.calls == ["pick", "place"]
    assert len(result.traces) == 2


def test_rolling_runner_fails_closed_when_replanned_suffix_drops_next_subgraph() -> None:
    class DroppingPlanner(_Planner):
        def compile(self, task, scene, *, context, protocol):
            self.scenes.append(scene.scene_version)
            graph = self.graph if len(self.scenes) == 1 else replace(
                self.graph,
                subgraphs=(self.graph.subgraphs[0],),
                edges=(),
                entry_subgraph="pick",
                success_subgraphs=("pick",),
                failure_subgraphs=("pick",),
            )
            return LLMGraphCompileResult(graph, {})

    with pytest.raises(RollingGraphError, match="dropped required next subgraph"):
        RollingGraphRunner().run(
            "test task",
            SceneSnapshot("episode", 1, 0, 1, 2, {}),
            DroppingPlanner(_graph()),
            FixedGraphInterpreter(_RobotScheduler()),
            max_cycles=3,
        )


def test_staged_rolling_reuses_topology_and_proposes_only_current_frontier() -> None:
    topology = MissionTopology(
        mission_id="mission",
        task="test task",
        subgoals=(
            TopologySubgoal("first", "first", "first step", success_predicates=("step_done",)),
            TopologySubgoal(
                "second",
                "second",
                "second step",
                depends_on=("first",),
                success_predicates=("step_done",),
            ),
        ),
        edges=(MissionEdge("first", "second", "success"),),
        entry_subgraph="first",
        success_subgraphs=("second",),
        failure_subgraphs=("first", "second"),
        parent_scene_version=0,
    )

    class Manager:
        name = "manager"

        def __init__(self) -> None:
            self.calls = 0

        def propose_topology(self, _task: str, _scene: SceneSnapshot) -> MissionTopology:
            self.calls += 1
            return topology

    policy_calls: list[str] = []

    def propose(subgoal: AgentArtifact, _scene: SceneSnapshot, _context: AgentContext) -> SubgraphSpec:
        policy_calls.append(str(subgoal.payload["subgraph_id"]))
        return _subgraph(str(subgoal.payload["subgraph_id"]))

    policy = CallableGraphPolicyAgent(propose)
    policy.name = "policy-a"  # type: ignore[attr-defined]
    manager = Manager()
    planner = LLMGraphScheduler(manager, {"*": (policy,)}, max_workers=1)

    result = RollingGraphRunner().run(
        "test task",
        SceneSnapshot("episode", 1, 0, 1, 2, {}),
        planner,
        FixedGraphInterpreter(_RobotScheduler()),
        protocol="staged",
        max_cycles=4,
    )

    assert result.completed is True
    assert manager.calls == 1
    assert policy_calls == ["first", "second"]
    assert result.frontier_subgraphs == (("first",), ("second",))
    assert result.planning_mode == "ready_frontier"
    assert [item.manager_topology_calls for item in result.compilations] == [1, 0]
    assert [item.planning_scope for item in result.compilations] == [
        "ready_frontier",
        "ready_frontier",
    ]


def test_staged_rolling_routes_failure_to_fixed_topology_recovery() -> None:
    topology = MissionTopology(
        mission_id="mission",
        task="test task",
        subgoals=(
            TopologySubgoal("main", "main", "main", success_predicates=("step_done",)),
            TopologySubgoal("recover", "recover", "recover", success_predicates=("step_done",)),
        ),
        edges=(MissionEdge("main", "recover", "failure"),),
        entry_subgraph="main",
        success_subgraphs=("recover",),
        failure_subgraphs=("main", "recover"),
        parent_scene_version=0,
    )

    class Manager:
        name = "manager"

        def propose_topology(self, _task: str, _scene: SceneSnapshot) -> MissionTopology:
            return topology

    def propose(subgoal: AgentArtifact, _scene: SceneSnapshot, _context: AgentContext) -> SubgraphSpec:
        return _subgraph(str(subgoal.payload["subgraph_id"]))

    class FailingScheduler(_RobotScheduler):
        def dispatch(self, contract, scene):
            result = super().dispatch(contract, scene)
            if contract.subgoal_id == "main":
                return replace(result, committed=False)
            return result

    planner = LLMGraphScheduler(
        Manager(),
        {"*": (CallableGraphPolicyAgent(propose),)},
        max_workers=1,
    )
    robot = FailingScheduler()
    result = RollingGraphRunner().run(
        "test task",
        SceneSnapshot("episode", 1, 0, 1, 2, {}),
        planner,
        FixedGraphInterpreter(robot),
        protocol="staged",
        max_cycles=3,
    )

    assert result.completed is True
    assert robot.calls == ["main", "recover"]
    assert result.frontier_subgraphs == (("main",), ("recover",))


def test_staged_rolling_stops_when_success_has_no_transition() -> None:
    topology = MissionTopology(
        mission_id="mission",
        task="test task",
        subgoals=(
            TopologySubgoal("only", "only", "only", success_predicates=("step_done",)),
            TopologySubgoal("recover", "recover", "recover", success_predicates=("step_done",)),
        ),
        edges=(MissionEdge("only", "recover", "failure"),),
        entry_subgraph="only",
        success_subgraphs=("recover",),
        failure_subgraphs=("only", "recover"),
        parent_scene_version=0,
    )

    class Manager:
        name = "manager"

        def propose_topology(self, _task: str, _scene: SceneSnapshot) -> MissionTopology:
            return topology

    planner = LLMGraphScheduler(
        Manager(),
        {"*": (CallableGraphPolicyAgent(
            lambda artifact, scene, context: _subgraph(str(artifact.payload["subgraph_id"]))
        ),)},
        max_workers=1,
    )
    result = RollingGraphRunner().run(
        "test task",
        SceneSnapshot("episode", 1, 0, 1, 2, {}),
        planner,
        FixedGraphInterpreter(_RobotScheduler()),
        protocol="staged",
        max_cycles=2,
    )

    assert result.completed is False
    assert result.stop_reason == "no_next_subgraph"


def test_staged_rolling_rebases_frontier_after_scene_refresh() -> None:
    topology = MissionTopology(
        mission_id="mission",
        task="test task",
        subgoals=(
            TopologySubgoal("first", "first", "first", success_predicates=("step_done",)),
            TopologySubgoal(
                "second",
                "second",
                "second",
                depends_on=("first",),
                success_predicates=("step_done",),
            ),
        ),
        edges=(MissionEdge("first", "second", "success"),),
        entry_subgraph="first",
        success_subgraphs=("second",),
        failure_subgraphs=("first", "second"),
        parent_scene_version=0,
    )

    class Manager:
        name = "manager"
        calls = 0

        def propose_topology(self, _task: str, _scene: SceneSnapshot) -> MissionTopology:
            self.calls += 1
            return topology

    policy_scenes: list[int] = []

    def propose(subgoal: AgentArtifact, scene: SceneSnapshot, _context: AgentContext) -> SubgraphSpec:
        policy_scenes.append(scene.scene_version)
        return _subgraph(str(subgoal.payload["subgraph_id"]))

    manager = Manager()
    planner = LLMGraphScheduler(
        manager,
        {"*": (CallableGraphPolicyAgent(propose),)},
        max_workers=1,
    )

    def refresh(scene: SceneSnapshot) -> SceneSnapshot:
        return replace(scene, scene_version=scene.scene_version + 1)

    result = RollingGraphRunner().run(
        "test task",
        SceneSnapshot("episode", 1, 0, 1, 2, {}),
        planner,
        FixedGraphInterpreter(_RobotScheduler()),
        protocol="staged",
        max_cycles=4,
        scene_refresh=refresh,
    )

    assert result.completed is True
    assert manager.calls == 1
    assert policy_scenes == [0, 2]
    assert [item.topology.parent_scene_version for item in result.compilations] == [1, 3]
    assert [item.graph.parent_scene_version for item in result.compilations] == [1, 3]


def test_ready_frontier_unknown_subgraph_is_a_typed_schedule_error() -> None:
    topology = MissionTopology(
        mission_id="mission",
        task="test task",
        subgoals=(TopologySubgoal("first", "first", "first", success_predicates=("step_done",)),),
        edges=(),
        entry_subgraph="first",
        success_subgraphs=("first",),
        failure_subgraphs=("first",),
        parent_scene_version=0,
    )

    class Manager:
        name = "manager"

        def propose_topology(self, _task: str, _scene: SceneSnapshot) -> MissionTopology:
            return topology

    planner = LLMGraphScheduler(
        Manager(),
        {"*": (CallableGraphPolicyAgent(lambda artifact, scene, context: _subgraph("first")),)},
        max_workers=1,
    )

    with pytest.raises(LLMGraphScheduleError, match="unknown ready-frontier subgraph"):
        planner.compile_ready_frontier(
            "test task",
            SceneSnapshot("episode", 1, 0, 1, 2, {}),
            topology=topology,
            subgraph_id="missing",
        )


@pytest.mark.parametrize(
    ("refresh", "message"),
    (
        (
            lambda scene: replace(scene, scene_version=scene.scene_version - 1),
            "older scene version",
        ),
        (
            lambda scene: replace(scene, episode_id="other", scene_version=scene.scene_version + 1),
            "different episode",
        ),
    ),
)
def test_rolling_rejects_invalid_scene_refresh(refresh, message: str) -> None:
    topology = MissionTopology(
        mission_id="mission",
        task="test task",
        subgoals=(TopologySubgoal("first", "first", "first", success_predicates=("step_done",)),),
        edges=(),
        entry_subgraph="first",
        success_subgraphs=("first",),
        failure_subgraphs=("first",),
        parent_scene_version=0,
    )

    class Manager:
        name = "manager"

        def propose_topology(self, _task: str, _scene: SceneSnapshot) -> MissionTopology:
            return topology

    planner = LLMGraphScheduler(
        Manager(),
        {"*": (CallableGraphPolicyAgent(lambda artifact, scene, context: _subgraph("first")),)},
        max_workers=1,
    )

    with pytest.raises(RollingGraphError, match=message):
        RollingGraphRunner().run(
            "test task",
            SceneSnapshot("episode", 1, 0, 1, 2, {}),
            planner,
            FixedGraphInterpreter(_RobotScheduler()),
            protocol="staged",
            max_cycles=1,
            scene_refresh=refresh,
        )


def test_rolling_normalizes_nullable_topology_success_edge() -> None:
    topology = MissionTopology(
        mission_id="mission",
        task="test task",
        subgoals=(
            TopologySubgoal("first", "first", "first", success_predicates=("step_done",)),
            TopologySubgoal(
                "second",
                "second",
                "second",
                depends_on=("first",),
                success_predicates=("step_done",),
            ),
        ),
        edges=(MissionEdge("first", "second", None),),
        entry_subgraph="first",
        success_subgraphs=("second",),
        failure_subgraphs=("first", "second"),
        parent_scene_version=0,
    )

    class Manager:
        name = "manager"

        def propose_topology(self, _task: str, _scene: SceneSnapshot) -> MissionTopology:
            return topology

    policy = CallableGraphPolicyAgent(
        lambda artifact, scene, context: _subgraph(str(artifact.payload["subgraph_id"]))
    )
    result = RollingGraphRunner().run(
        "test task",
        SceneSnapshot("episode", 1, 0, 1, 2, {}),
        LLMGraphScheduler(Manager(), {"*": (policy,)}, max_workers=1),
        FixedGraphInterpreter(_RobotScheduler()),
        protocol="staged",
        max_cycles=2,
    )

    assert result.completed is True
