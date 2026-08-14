"""Verified horizon labels derived from mission graphs and graph events."""

from __future__ import annotations

from collections.abc import Iterable

from capmas.contracts.calibration import HorizonLabel
from capmas.contracts.graph import MissionGraph, SubgraphNodeSpec, SubgraphSpec
from capmas.contracts.trace import GraphExecutionEvent

_PLANNED_EDGE_CONDITIONS = {
    None,
    "success",
    "completed",
    "action_complete",
    "action_completed",
}


def planned_horizon(graph: MissionGraph) -> HorizonLabel:
    """Return the lexicographically preferred successful graph-path horizon."""
    paths = tuple(_successful_paths(graph))
    if not paths:
        return _unknown_horizon()

    def score(path: tuple[str, ...]) -> tuple[int, int, int, tuple[str, ...]]:
        subgraphs = tuple(graph.subgraph(subgraph_id) for subgraph_id in path)
        return (
            sum(_is_action_bearing(subgraph) for subgraph in subgraphs),
            sum(_action_count(subgraph) for subgraph in subgraphs),
            sum(_is_checkpoint_only(subgraph) for subgraph in subgraphs),
            path,
        )

    selected = max(paths, key=score)
    subgraphs = tuple(graph.subgraph(subgraph_id) for subgraph_id in selected)
    return HorizonLabel(
        planned_critical_path_actions=sum(_action_count(subgraph) for subgraph in subgraphs),
        planned_critical_path_subgoals=sum(_is_action_bearing(subgraph) for subgraph in subgraphs),
        planned_checkpoint_subgraphs=sum(_is_checkpoint_only(subgraph) for subgraph in subgraphs),
        attempted_actions=None,
        completed_actions=None,
        attempted_subgoals=None,
        completed_subgoals=None,
        attempted_checkpoints=None,
        completed_checkpoints=None,
        planned_source="mission_graph",
        realized_source="unknown",
        planned_valid=True,
        realized_valid=False,
    )


def realized_horizon(
    graph: MissionGraph, events: Iterable[GraphExecutionEvent]
) -> HorizonLabel:
    """Count retries and recovery re-entry directly from graph telemetry."""
    attempted_actions = completed_actions = 0
    attempted_subgoals = completed_subgoals = 0
    attempted_checkpoints = completed_checkpoints = 0

    for event in events:
        subgraph = _subgraph_or_none(graph, event.subgraph_id)
        if subgraph is None:
            continue
        if event.kind == "subgraph_started" and _is_action_bearing(subgraph):
            attempted_subgoals += 1
        elif event.kind == "subgraph_completed" and _is_action_bearing(subgraph):
            completed_subgoals += 1
        elif event.node_id is not None:
            node = _node_or_none(subgraph, event.node_id)
            if node is None:
                continue
            if node.node_type == "action":
                if event.kind == "node_started":
                    attempted_actions += 1
                elif event.kind == "node_completed":
                    completed_actions += 1
            elif node.node_type == "checkpoint":
                if event.kind == "node_started":
                    attempted_checkpoints += 1
                elif event.kind == "node_completed":
                    completed_checkpoints += 1

    return HorizonLabel(
        planned_critical_path_actions=None,
        planned_critical_path_subgoals=None,
        planned_checkpoint_subgraphs=None,
        attempted_actions=attempted_actions,
        completed_actions=completed_actions,
        attempted_subgoals=attempted_subgoals,
        completed_subgoals=completed_subgoals,
        attempted_checkpoints=attempted_checkpoints,
        completed_checkpoints=completed_checkpoints,
        planned_source="unknown",
        realized_source="execution_trace",
        planned_valid=False,
        realized_valid=True,
    )


def extract_horizon(
    graph: MissionGraph, events: Iterable[GraphExecutionEvent]
) -> HorizonLabel:
    """Merge static planned fields with event-derived realized fields."""
    planned = planned_horizon(graph)
    realized = realized_horizon(graph, events)
    return HorizonLabel(
        planned_critical_path_actions=planned.planned_critical_path_actions,
        planned_critical_path_subgoals=planned.planned_critical_path_subgoals,
        planned_checkpoint_subgraphs=planned.planned_checkpoint_subgraphs,
        attempted_actions=realized.attempted_actions,
        completed_actions=realized.completed_actions,
        attempted_subgoals=realized.attempted_subgoals,
        completed_subgoals=realized.completed_subgoals,
        attempted_checkpoints=realized.attempted_checkpoints,
        completed_checkpoints=realized.completed_checkpoints,
        planned_source=planned.planned_source,
        realized_source=realized.realized_source,
        planned_valid=planned.planned_valid,
        realized_valid=realized.realized_valid,
    )


def _successful_paths(graph: MissionGraph) -> Iterable[tuple[str, ...]]:
    success = set(graph.success_subgraphs)

    def visit(current: str, path: tuple[str, ...]) -> Iterable[tuple[str, ...]]:
        if current in success:
            yield path
            return
        for edge in graph.edges:
            if edge.source != current or edge.condition not in _PLANNED_EDGE_CONDITIONS:
                continue
            if edge.target not in path:
                yield from visit(edge.target, path + (edge.target,))

    yield from visit(graph.entry_subgraph, (graph.entry_subgraph,))


def _is_action_bearing(subgraph: SubgraphSpec) -> bool:
    return any(node.node_type == "action" for node in subgraph.nodes)


def _is_checkpoint_only(subgraph: SubgraphSpec) -> bool:
    return not _is_action_bearing(subgraph) and any(
        node.node_type == "checkpoint" for node in subgraph.nodes
    )


def _action_count(subgraph: SubgraphSpec) -> int:
    return sum(node.node_type == "action" for node in subgraph.nodes)


def _subgraph_or_none(graph: MissionGraph, subgraph_id: str) -> SubgraphSpec | None:
    return next(
        (subgraph for subgraph in graph.subgraphs if subgraph.subgraph_id == subgraph_id),
        None,
    )


def _node_or_none(
    subgraph: SubgraphSpec, node_id: str
) -> SubgraphNodeSpec | None:
    return next((node for node in subgraph.nodes if node.node_id == node_id), None)


def _unknown_horizon() -> HorizonLabel:
    return HorizonLabel(
        planned_critical_path_actions=None,
        planned_critical_path_subgoals=None,
        planned_checkpoint_subgraphs=None,
        attempted_actions=None,
        completed_actions=None,
        attempted_subgoals=None,
        completed_subgoals=None,
        attempted_checkpoints=None,
        completed_checkpoints=None,
        planned_source="unknown",
        realized_source="unknown",
        planned_valid=False,
        realized_valid=False,
    )
