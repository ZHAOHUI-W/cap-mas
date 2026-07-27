"""Contracts for the staged graph protocol.

The staged protocol separates global topology from local executable graph
generation.  The Manager never has to emit skill calls, node ports, or the
full local graph schema; Policy Agents receive one bounded subgoal and return
one ``SubgraphSpec``.  The server-side assembler lowers both stages into the
existing ``MissionGraph`` contract before validation or execution.
"""

from __future__ import annotations

from dataclasses import dataclass

from capmas.contracts.graph import (
    LoopSpec,
    MissionBinding,
    MissionEdge,
    MissionGraph,
    PortSpec,
    SubgraphSpec,
)


STAGED_TOPOLOGY_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class TopologySubgoal:
    """A global subgoal without executable skill details."""

    subgraph_id: str
    subgoal_id: str
    description: str
    depends_on: tuple[str, ...] = ()
    success_predicates: tuple[str, ...] = ()
    failure_predicates: tuple[str, ...] = ()
    required_agent_role: str = "local_policy_agent"
    execution_kind: str = "physical_action"

    def __post_init__(self) -> None:
        if not self.subgraph_id or not self.subgoal_id:
            raise ValueError("topology subgraph and subgoal ids must not be empty")
        if not self.description:
            raise ValueError("topology subgoal description must not be empty")
        if not self.required_agent_role:
            raise ValueError("topology agent role must not be empty")
        if self.execution_kind not in {"physical_action", "checkpoint_only"}:
            raise ValueError(
                "topology execution_kind must be physical_action or checkpoint_only"
            )
        if self.subgraph_id in self.depends_on:
            raise ValueError("topology subgoal cannot depend on itself")


@dataclass(frozen=True)
class MissionTopology:
    """Compact Manager output used by the staged graph protocol."""

    mission_id: str
    task: str
    subgoals: tuple[TopologySubgoal, ...]
    edges: tuple[MissionEdge, ...]
    entry_subgraph: str
    success_subgraphs: tuple[str, ...]
    failure_subgraphs: tuple[str, ...]
    bindings: tuple[MissionBinding, ...] = ()
    parent_scene_version: int | None = None
    graph_version: int = 1
    loops: tuple[LoopSpec, ...] = ()

    def subgoal(self, subgraph_id: str) -> TopologySubgoal:
        for subgoal in self.subgoals:
            if subgoal.subgraph_id == subgraph_id:
                return subgoal
        raise KeyError(f"unknown topology subgoal: {subgraph_id}")

    def normalized_edges(self) -> tuple[MissionEdge, ...]:
        """Return execution outcomes using the staged graph conventions.

        Rolling execution consumes the topology before local subgraphs are
        assembled, so it must use the same nullable/semantic edge handling as
        ``assemble``. This keeps a dependency edge with ``condition=None`` on
        the normal success path while preserving explicit recovery branches.
        """
        dependency_edges = {
            (dependency, subgoal.subgraph_id)
            for subgoal in self.subgoals
            for dependency in subgoal.depends_on
        }
        normalized_edges = tuple(self.normalized_edge(edge) for edge in self.edges)
        explicit_failure_edges = {
            (edge.source, edge.target)
            for edge in self.edges
            if _is_failure_label(edge.condition)
        }
        return _prefer_dependency_success_edges(
            normalized_edges,
            dependency_edges,
            tuple(subgoal.subgraph_id for subgoal in self.subgoals),
            set(self.failure_subgraphs),
            explicit_failure_edges,
        )

    def normalized_edge(self, edge: MissionEdge) -> MissionEdge:
        """Normalize one topology edge without resolving sibling ambiguity."""
        dependency_edges = {
            (dependency, subgoal.subgraph_id)
            for subgoal in self.subgoals
            for dependency in subgoal.depends_on
        }
        return _normalize_edge(
            edge,
            set(self.failure_subgraphs),
            set(self.success_subgraphs),
            dependency_edges,
        )

    def assemble(self, subgraphs: tuple[SubgraphSpec, ...]) -> MissionGraph:
        """Join local policy outputs into the existing executable graph type."""
        expected = {subgoal.subgraph_id: subgoal for subgoal in self.subgoals}
        provided = {subgraph.subgraph_id: subgraph for subgraph in subgraphs}
        if len(provided) != len(subgraphs):
            raise ValueError("staged subgraph set contains duplicate subgraph ids")
        if set(provided) != set(expected):
            missing = sorted(set(expected) - set(provided))
            extra = sorted(set(provided) - set(expected))
            raise ValueError(f"staged subgraph set mismatch: missing={missing}, extra={extra}")
        for subgraph_id, subgraph in provided.items():
            topology_subgoal = expected[subgraph_id]
            if subgraph.subgoal_id != topology_subgoal.subgoal_id:
                raise ValueError(
                    f"staged subgoal mismatch for {subgraph_id!r}: "
                    f"{subgraph.subgoal_id!r} != {topology_subgoal.subgoal_id!r}"
                )
        bindings = self._assemble_bindings(tuple(provided[subgoal.subgraph_id] for subgoal in self.subgoals))
        edges = self.normalized_edges()
        return MissionGraph(
            mission_id=self.mission_id,
            task=self.task,
            subgraphs=tuple(provided[subgoal.subgraph_id] for subgoal in self.subgoals),
            edges=edges,
            bindings=bindings,
            entry_subgraph=self.entry_subgraph,
            success_subgraphs=self.success_subgraphs,
            failure_subgraphs=self.failure_subgraphs,
            parent_scene_version=self.parent_scene_version,
            graph_version=self.graph_version,
            loops=self.loops,
        )

    def _assemble_bindings(
        self,
        subgraphs: tuple[SubgraphSpec, ...],
    ) -> tuple[MissionBinding, ...]:
        """Add only unambiguous predecessor bindings after local graphs exist.

        The Manager cannot know Policy-created port names in advance. Explicit
        bindings remain authoritative; missing bindings are inferred only from
        one direct predecessor with an exact name/type match, or a unique type
        match. Ambiguous dataflow is left unbound so GraphValidator fails closed.
        """
        by_id = {subgraph.subgraph_id: subgraph for subgraph in subgraphs}
        bindings = [
            binding
            for binding in self.bindings
            if _binding_matches_local_ports(binding, by_id)
        ]
        bound_targets = {(item.target_subgraph, item.target_port) for item in bindings}
        predecessors: dict[str, tuple[str, ...]] = {
            subgraph_id: tuple(
                edge.source for edge in self.edges if edge.target == subgraph_id
            )
            for subgraph_id in by_id
        }
        for target_id, target in by_id.items():
            for target_port in target.inputs:
                if not target_port.required or (target_id, target_port.name) in bound_targets:
                    continue
                candidates = _binding_candidates(
                    target_port,
                    predecessors.get(target_id, ()),
                    by_id,
                )
                if len(candidates) == 1:
                    source_id, source_port = candidates[0]
                    bindings.append(
                        MissionBinding(source_id, source_port, target_id, target_port.name)
                    )
                    bound_targets.add((target_id, target_port.name))
        return tuple(bindings)


def _binding_candidates(
    target_port: PortSpec,
    predecessor_ids: tuple[str, ...],
    by_id: dict[str, SubgraphSpec],
) -> list[tuple[str, str]]:
    exact: list[tuple[str, str]] = []
    typed: list[tuple[str, str]] = []
    for source_id in predecessor_ids:
        source = by_id.get(source_id)
        if source is None:
            continue
        for source_port in source.outputs:
            if source_port.type_name != target_port.type_name:
                continue
            candidate = (source_id, source_port.name)
            typed.append(candidate)
            if source_port.name == target_port.name:
                exact.append(candidate)
    if exact:
        return exact
    return typed


def _binding_matches_local_ports(
    binding: MissionBinding,
    by_id: dict[str, SubgraphSpec],
) -> bool:
    source = by_id.get(binding.source_subgraph)
    target = by_id.get(binding.target_subgraph)
    if source is None or target is None:
        return False
    source_port = next(
        (port for port in source.outputs if port.name == binding.source_port),
        None,
    )
    target_port = next(
        (port for port in target.inputs if port.name == binding.target_port),
        None,
    )
    return source_port is not None and target_port is not None and source_port.type_name == target_port.type_name


def _normalize_edge(
    edge: MissionEdge,
    failure_subgraphs: set[str],
    success_subgraphs: set[str],
    dependency_edges: set[tuple[str, str]],
) -> MissionEdge:
    """Lower semantic LLM edge labels to the interpreter's binary outcomes."""
    label = edge.condition.lower() if edge.condition else ""
    failure_words = ("fail", "abort", "unbound", "unavailable", "unsafe", "error")
    # A failure-only terminal may also be listed in ``depends_on`` so the
    # Manager can describe the state dependency that leads to its recovery
    # branch. Keep an explicitly labelled failure transition in that case.
    # A subgraph listed as both success and failure is a legacy/provider
    # convention for a normal dependency, so the dependency normalization
    # below remains authoritative for it.
    if edge.target in failure_subgraphs and (
        edge.condition == "failure" or any(word in label for word in failure_words)
    ) and edge.target not in success_subgraphs:
        return MissionEdge(edge.source, edge.target, "failure")
    # Topology dependencies represent the normal forward path. Some providers
    # emit ``failure`` for those edges despite the target not being a failure
    # terminal; preserving that label would strand a successful checkpoint.
    if (edge.source, edge.target) in dependency_edges:
        return MissionEdge(edge.source, edge.target, "success")
    if edge.condition == "failure":
        return edge
    if edge.condition == "success":
        return edge
    if edge.condition is None:
        return MissionEdge(edge.source, edge.target, "failure")
    outcome = "failure" if edge.target in failure_subgraphs or any(
        word in label for word in failure_words
    ) else "success"
    return MissionEdge(edge.source, edge.target, outcome)


def _is_failure_label(condition: str | None) -> bool:
    label = condition.lower() if condition else ""
    return condition == "failure" or any(
        word in label
        for word in ("fail", "abort", "unbound", "unavailable", "unsafe", "error")
    )


def _prefer_dependency_success_edges(
    edges: tuple[MissionEdge, ...],
    dependency_edges: set[tuple[str, str]],
    subgraph_order: tuple[str, ...],
    failure_subgraphs: set[str],
    explicit_failure_edges: set[tuple[str, str]],
) -> tuple[MissionEdge, ...]:
    """Resolve duplicate success branches without losing failure routing."""
    by_source: dict[str, tuple[MissionEdge, ...]] = {}
    for edge in edges:
        by_source[edge.source] = (*by_source.get(edge.source, ()), edge)
    selected: list[MissionEdge] = []
    for source_edges in by_source.values():
        success_indices = [
            index
            for index, edge in enumerate(source_edges)
            if edge.condition == "success"
        ]
        preferred_indices = [
            index
            for index in success_indices
            if (source_edges[index].source, source_edges[index].target)
            in dependency_edges
        ]
        if len(success_indices) > 1 and preferred_indices:
            chosen_index = preferred_indices[0]
            for index, edge in enumerate(source_edges):
                if edge.condition != "success" or index == chosen_index:
                    selected.append(edge)
                elif (
                    (edge.source, edge.target) in explicit_failure_edges
                    and edge.target in failure_subgraphs
                ):
                    selected.append(MissionEdge(edge.source, edge.target, "failure"))
        elif len(success_indices) > 1:
            order = {subgraph_id: index for index, subgraph_id in enumerate(subgraph_order)}
            source_index = order.get(source_edges[success_indices[0]].source, -1)
            forward_indices = [
                index
                for index in success_indices
                if order.get(source_edges[index].target, -1) > source_index
            ]
            chosen_index = min(
                forward_indices or success_indices,
                key=lambda index: order.get(source_edges[index].target, len(order)),
            )
            for index, edge in enumerate(source_edges):
                if edge.condition != "success" or index == chosen_index:
                    selected.append(edge)
                elif (
                    (edge.source, edge.target) in explicit_failure_edges
                    and edge.target in failure_subgraphs
                ):
                    selected.append(MissionEdge(edge.source, edge.target, "failure"))
        else:
            selected.extend(source_edges)
    return tuple(selected)


__all__ = [
    "MissionTopology",
    "STAGED_TOPOLOGY_SCHEMA_VERSION",
    "TopologySubgoal",
]
