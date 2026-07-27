"""Closed-loop rolling execution for staged MissionGraphs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Callable
from typing import Mapping, Protocol

from capmas.contracts.agent import AgentContext
from capmas.contracts.graph import MissionGraph
from capmas.contracts.scene import SceneSnapshot
from capmas.contracts.staged import MissionTopology
from capmas.contracts.trace import ExecutionTrace
from capmas.runtime.graph_interpreter import FixedGraphInterpreter, GraphExecutionResult
from capmas.runtime.llm_scheduler import LLMGraphCompileResult


class RollingGraphError(RuntimeError):
    """Raised when a rolling plan cannot be safely aligned with execution."""


class RollingGraphPlanner(Protocol):
    def compile(
        self,
        task: str,
        scene: SceneSnapshot,
        *,
        context: AgentContext,
        protocol: str,
    ) -> LLMGraphCompileResult: ...

    def rebase_graph(
        self,
        graph: MissionGraph,
        scene: SceneSnapshot,
        *,
        context: AgentContext,
    ) -> MissionGraph: ...

    def compile_ready_frontier(
        self,
        task: str,
        scene: SceneSnapshot,
        *,
        topology: MissionTopology | None,
        subgraph_id: str | None,
        completed_subgraphs: tuple[str, ...],
        context: AgentContext,
    ) -> LLMGraphCompileResult: ...


@dataclass(frozen=True)
class RollingGraphRunResult:
    completed: bool
    scene: SceneSnapshot
    context: AgentContext
    traces: tuple[ExecutionTrace, ...]
    compilations: tuple[LLMGraphCompileResult, ...]
    replan_count: int
    stop_reason: str
    failure: object | None = None
    last_execution: GraphExecutionResult | None = None
    planning_mode: str = "full_recompile"
    frontier_subgraphs: tuple[tuple[str, ...], ...] = ()


class RollingGraphRunner:
    """Recompile and execute one verified subgraph per rolling cycle.

    The planner receives the latest committed scene on every iteration. A
    previously selected next subgraph must still exist in the newly compiled
    graph; otherwise the runner fails closed instead of silently executing a
    different suffix. The interpreter remains the sole physical executor.
    """

    def run(
        self,
        task: str,
        scene: SceneSnapshot,
        planner: RollingGraphPlanner,
        interpreter: FixedGraphInterpreter,
        *,
        context: AgentContext | None = None,
        protocol: str = "staged",
        max_cycles: int = 32,
        scene_refresh: Callable[[SceneSnapshot], SceneSnapshot] | None = None,
    ) -> RollingGraphRunResult:
        if max_cycles <= 0:
            raise ValueError("max_cycles must be positive")
        context = context or AgentContext(
            task_id=task,
            episode_id=scene.episode_id,
            episode_epoch=scene.episode_epoch,
            scene=scene,
        )
        current_scene = scene
        next_subgraph: str | None = None
        topology: MissionTopology | None = None
        completed_subgraphs: set[str] = set()
        frontier_subgraphs: list[tuple[str, ...]] = []
        frontier_planner = getattr(planner, "compile_ready_frontier", None)
        planning_mode = (
            "ready_frontier"
            if protocol == "staged" and callable(frontier_planner)
            else "full_recompile"
        )
        mission_outputs: dict[str, dict[str, object]] = {}
        traces: list[ExecutionTrace] = []
        compilations: list[LLMGraphCompileResult] = []
        last_execution: GraphExecutionResult | None = None

        for cycle_index in range(max_cycles):
            if planning_mode == "ready_frontier":
                compiled = frontier_planner(
                    task,
                    current_scene,
                    topology=topology,
                    subgraph_id=next_subgraph,
                    completed_subgraphs=tuple(sorted(completed_subgraphs)),
                    context=context,
                )
                if compiled.topology is None:
                    raise RollingGraphError(
                        "ready-frontier planner did not return the fixed topology"
                    )
                topology = compiled.topology
                frontier_subgraphs.append(compiled.proposal_waves[0])
            else:
                compiled = planner.compile(
                    task,
                    current_scene,
                    context=context,
                    protocol=protocol,
                )
            compilations.append(compiled)
            graph = compiled.graph
            if scene_refresh is not None:
                refreshed_scene = scene_refresh(current_scene)
                _validate_scene_transition(current_scene, refreshed_scene, "scene refresh")
                if refreshed_scene.scene_version != current_scene.scene_version:
                    rebase = getattr(planner, "rebase_graph", None)
                    if not callable(rebase):
                        raise RollingGraphError(
                            "scene_refresh requires planner.rebase_graph for safe dispatch"
                        )
                    graph = rebase(graph, refreshed_scene, context=context)
                    if compiled.topology is not None:
                        topology = replace(
                            compiled.topology,
                            parent_scene_version=refreshed_scene.scene_version,
                        )
                    compiled = replace(
                        compiled,
                        graph=graph,
                        topology=(
                            topology
                            if planning_mode == "ready_frontier"
                            else compiled.topology
                        ),
                    )
                    compilations[-1] = compiled
                    current_scene = refreshed_scene
            if graph.parent_scene_version is not None and graph.parent_scene_version != current_scene.scene_version:
                raise RollingGraphError(
                    f"rolling planner returned scene {graph.parent_scene_version}, "
                    f"current scene is {current_scene.scene_version}"
                )
            if next_subgraph is not None:
                declared = {subgraph.subgraph_id for subgraph in graph.subgraphs}
                if next_subgraph not in declared:
                    raise RollingGraphError(
                        f"rolling planner dropped required next subgraph {next_subgraph!r}"
                    )
                graph = _suffix_graph(graph, next_subgraph)

            dispatch_scene = current_scene
            execution = interpreter.run(
                graph,
                dispatch_scene,
                context=context,
                stop_after_subgraph=True,
                initial_outputs=mission_outputs,
            )
            _validate_scene_transition(dispatch_scene, execution.scene, "executor result")
            last_execution = execution
            traces.extend(execution.traces)
            mission_outputs = dict(execution.outputs or mission_outputs)
            context = execution.context
            current_scene = execution.scene

            if planning_mode == "ready_frontier":
                assert topology is not None
                current_id = graph.entry_subgraph
                if execution.failure is not None:
                    recovery_target = _topology_transition(
                        topology,
                        current_id,
                        "failure",
                    )
                    if recovery_target is None:
                        return RollingGraphRunResult(
                            False,
                            current_scene,
                            context,
                            tuple(traces),
                            tuple(compilations),
                            replan_count=cycle_index,
                            stop_reason="execution_failure",
                            failure=execution.failure,
                            last_execution=execution,
                            planning_mode=planning_mode,
                            frontier_subgraphs=tuple(frontier_subgraphs),
                        )
                    next_subgraph = recovery_target
                    continue
                if not execution.completed:
                    return RollingGraphRunResult(
                        False,
                        current_scene,
                        context,
                        tuple(traces),
                        tuple(compilations),
                        replan_count=cycle_index,
                        stop_reason="frontier_not_committed",
                        failure=execution.failure,
                        last_execution=execution,
                        planning_mode=planning_mode,
                        frontier_subgraphs=tuple(frontier_subgraphs),
                    )
                completed_subgraphs.add(current_id)
                if current_id in topology.success_subgraphs:
                    return RollingGraphRunResult(
                        True,
                        current_scene,
                        context,
                        tuple(traces),
                        tuple(compilations),
                        replan_count=cycle_index,
                        stop_reason="task_completed",
                        last_execution=execution,
                        planning_mode=planning_mode,
                        frontier_subgraphs=tuple(frontier_subgraphs),
                    )
                next_subgraph = _topology_transition(topology, current_id, "success")
                if next_subgraph is None:
                    return RollingGraphRunResult(
                        False,
                        current_scene,
                        context,
                        tuple(traces),
                        tuple(compilations),
                        replan_count=cycle_index,
                        stop_reason="no_next_subgraph",
                        failure=execution.failure,
                        last_execution=execution,
                        planning_mode=planning_mode,
                        frontier_subgraphs=tuple(frontier_subgraphs),
                    )
                continue

            if execution.failure is not None:
                return RollingGraphRunResult(
                    False,
                    current_scene,
                    context,
                    tuple(traces),
                    tuple(compilations),
                    replan_count=cycle_index,
                    stop_reason="execution_failure",
                    failure=execution.failure,
                    last_execution=execution,
                )
            if execution.completed:
                return RollingGraphRunResult(
                    True,
                    current_scene,
                    context,
                    tuple(traces),
                    tuple(compilations),
                    replan_count=cycle_index,
                    stop_reason="task_completed",
                    last_execution=execution,
                )
            if execution.next_subgraph is None:
                return RollingGraphRunResult(
                    False,
                    current_scene,
                    context,
                    tuple(traces),
                    tuple(compilations),
                    replan_count=cycle_index,
                    stop_reason="no_next_subgraph",
                    last_execution=execution,
                )
            next_subgraph = execution.next_subgraph

        return RollingGraphRunResult(
            False,
            current_scene,
            context,
            tuple(traces),
            tuple(compilations),
            replan_count=max_cycles - 1,
            stop_reason="max_cycles",
            last_execution=last_execution,
        )


def _suffix_graph(graph: MissionGraph, entry_subgraph: str) -> MissionGraph:
    """Remove completed prefix nodes so the static validator sees a reachable graph."""
    by_id = {subgraph.subgraph_id: subgraph for subgraph in graph.subgraphs}
    if entry_subgraph not in by_id:
        raise RollingGraphError(f"unknown rolling entry subgraph {entry_subgraph!r}")
    reachable = {entry_subgraph}
    frontier = [entry_subgraph]
    while frontier:
        source = frontier.pop()
        for edge in graph.edges:
            if edge.source == source and edge.target in by_id and edge.target not in reachable:
                reachable.add(edge.target)
                frontier.append(edge.target)
    subgraphs = tuple(subgraph for subgraph in graph.subgraphs if subgraph.subgraph_id in reachable)
    edges = tuple(
        edge for edge in graph.edges if edge.source in reachable and edge.target in reachable
    )
    bindings = tuple(
        binding
        for binding in graph.bindings
        if binding.source_subgraph in reachable and binding.target_subgraph in reachable
    )
    success = tuple(item for item in graph.success_subgraphs if item in reachable)
    failure = tuple(item for item in graph.failure_subgraphs if item in reachable)
    if not success:
        success = (entry_subgraph,)
    if not failure:
        failure = (entry_subgraph,)
    return replace(
        graph,
        subgraphs=subgraphs,
        edges=edges,
        bindings=bindings,
        entry_subgraph=entry_subgraph,
        success_subgraphs=success,
        failure_subgraphs=failure,
    )


def _topology_transition(
    topology: MissionTopology,
    source_subgraph: str,
    outcome: str,
) -> str | None:
    """Return the unique fixed-topology edge for an execution outcome.

    Ready-frontier compilation intentionally executes a one-subgraph graph,
    so mission-level control flow cannot be delegated to the interpreter in
    that mode. The topology remains authoritative for the next frontier. A
    missing transition is a valid terminal condition for the requested
    outcome; ambiguous transitions fail closed instead of guessing.
    """
    if source_subgraph not in {item.subgraph_id for item in topology.subgoals}:
        raise RollingGraphError(
            f"topology transition source is unknown: {source_subgraph!r}"
        )
    if outcome not in {"success", "failure"}:
        raise RollingGraphError(f"unsupported topology transition outcome: {outcome!r}")
    matches = tuple(
        edge
        for edge in topology.normalized_edges()
        if edge.source == source_subgraph and edge.condition == outcome
    )
    if len(matches) > 1:
        raise RollingGraphError(
            f"multiple {outcome} transitions from {source_subgraph!r}"
        )
    if not matches:
        return None
    known_subgraphs = {item.subgraph_id for item in topology.subgoals}
    if matches[0].target not in known_subgraphs:
        raise RollingGraphError(
            f"topology transition target is unknown: {matches[0].target!r}"
        )
    return matches[0].target


def _validate_scene_transition(
    before: SceneSnapshot,
    after: SceneSnapshot,
    source: str,
) -> None:
    if (
        after.episode_id != before.episode_id
        or after.episode_epoch != before.episode_epoch
    ):
        raise RollingGraphError(
            f"{source} returned a different episode: "
            f"expected {before.episode_id}/{before.episode_epoch}, "
            f"got {after.episode_id}/{after.episode_epoch}"
        )
    if after.scene_version < before.scene_version:
        raise RollingGraphError(
            f"{source} returned an older scene version: "
            f"{after.scene_version} < {before.scene_version}"
        )
