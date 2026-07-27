from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from capmas.contracts.graph import (
    LoopSpec,
    MissionGraph,
    SubgraphSpec,
)


@dataclass(frozen=True)
class GraphDiagnostic:
    code: str
    message: str
    path: str = ""
    severity: str = "error"


@dataclass(frozen=True)
class GraphValidationResult:
    valid: bool
    diagnostics: tuple[GraphDiagnostic, ...] = ()

    @property
    def errors(self) -> tuple[GraphDiagnostic, ...]:
        return tuple(
            diagnostic
            for diagnostic in self.diagnostics
            if diagnostic.severity == "error"
        )


class GraphValidator:
    """Validate graph structure before any node reaches the robot runtime."""

    def validate(
        self,
        graph: MissionGraph,
        *,
        initial_facts: Iterable[str] = (),
    ) -> GraphValidationResult:
        diagnostics: list[GraphDiagnostic] = []
        subgraphs = {subgraph.subgraph_id: subgraph for subgraph in graph.subgraphs}
        if len(subgraphs) != len(graph.subgraphs):
            diagnostics.append(GraphDiagnostic("DUPLICATE_SUBGRAPH", "subgraph ids must be unique"))
        if not graph.subgraphs:
            diagnostics.append(GraphDiagnostic("EMPTY_MISSION", "mission graph has no subgraphs"))
        if graph.entry_subgraph not in subgraphs:
            diagnostics.append(
                GraphDiagnostic("UNKNOWN_ENTRY", f"unknown entry subgraph: {graph.entry_subgraph}")
            )

        self._validate_mission_edges(graph, subgraphs, diagnostics)
        _validate_loops(graph.loops, subgraphs, "mission", diagnostics)
        self._validate_mission_terminals(graph, subgraphs, diagnostics)
        self._validate_mission_reachability(graph, subgraphs, diagnostics)
        for subgraph in graph.subgraphs:
            self._validate_subgraph(subgraph, diagnostics)
        self._validate_mission_bindings(graph, subgraphs, diagnostics)
        self._validate_parallel_resources(graph, subgraphs, diagnostics)
        self._validate_state_flow(graph, initial_facts, diagnostics)
        return GraphValidationResult(
            not any(d.severity == "error" for d in diagnostics),
            tuple(diagnostics),
        )

    def raise_if_invalid(
        self,
        graph: MissionGraph,
        *,
        initial_facts: Iterable[str] = (),
    ) -> None:
        result = self.validate(graph, initial_facts=initial_facts)
        if not result.valid:
            summary = "; ".join(f"{d.code}: {d.message}" for d in result.errors)
            raise ValueError(f"invalid mission graph: {summary}")

    def validate_subgraph(self, subgraph: SubgraphSpec) -> GraphValidationResult:
        """Validate one local graph without inventing a mission wrapper."""
        diagnostics: list[GraphDiagnostic] = []
        self._validate_subgraph(subgraph, diagnostics)
        return GraphValidationResult(
            not any(d.severity == "error" for d in diagnostics),
            tuple(diagnostics),
        )

    def _validate_mission_edges(
        self,
        graph: MissionGraph,
        subgraphs: dict[str, SubgraphSpec],
        diagnostics: list[GraphDiagnostic],
    ) -> None:
        for edge in graph.edges:
            if edge.source not in subgraphs or edge.target not in subgraphs:
                diagnostics.append(
                    GraphDiagnostic(
                        "DANGLING_MISSION_EDGE",
                        f"mission edge references {edge.source}->{edge.target}",
                        "mission.edges",
                    )
                )
        for source in subgraphs:
            for condition in ("success", "failure"):
                matches = [
                    edge for edge in graph.edges
                    if edge.source == source and edge.condition == condition
                ]
                if len(matches) > 1:
                    diagnostics.append(
                        GraphDiagnostic(
                            "AMBIGUOUS_MISSION_TRANSITION",
                            f"{source} has multiple {condition} transitions",
                            "mission.edges",
                        )
                    )
        diagnostics.extend(
            _cycle_diagnostics(
                graph.entry_subgraph,
                graph.edges,
                subgraphs,
                loops=graph.loops,
                path="mission",
            )
        )

    def _validate_mission_terminals(
        self,
        graph: MissionGraph,
        subgraphs: dict[str, SubgraphSpec],
        diagnostics: list[GraphDiagnostic],
    ) -> None:
        for field_name, values in (
            ("success_subgraphs", graph.success_subgraphs),
            ("failure_subgraphs", graph.failure_subgraphs),
        ):
            if not values:
                diagnostics.append(
                    GraphDiagnostic("MISSING_MISSION_TERMINAL", f"{field_name} must not be empty")
                )
            for subgraph_id in values:
                if subgraph_id not in subgraphs:
                    diagnostics.append(
                        GraphDiagnostic(
                            "UNKNOWN_MISSION_TERMINAL",
                            f"unknown mission terminal: {subgraph_id}",
                            field_name,
                        )
                    )

    def _validate_mission_reachability(
        self,
        graph: MissionGraph,
        subgraphs: dict[str, SubgraphSpec],
        diagnostics: list[GraphDiagnostic],
    ) -> None:
        if graph.entry_subgraph not in subgraphs:
            return
        reachable = _reachable(graph.entry_subgraph, graph.edges, subgraphs)
        for subgraph_id in subgraphs:
            if subgraph_id not in reachable:
                diagnostics.append(
                    GraphDiagnostic(
                        "UNREACHABLE_SUBGRAPH",
                        f"subgraph is unreachable: {subgraph_id}",
                        f"subgraphs.{subgraph_id}",
                    )
                )

    def _validate_subgraph(
        self,
        subgraph: SubgraphSpec,
        diagnostics: list[GraphDiagnostic],
    ) -> None:
        prefix = f"subgraphs.{subgraph.subgraph_id}"
        nodes = {node.node_id: node for node in subgraph.nodes}
        if len(nodes) != len(subgraph.nodes):
            diagnostics.append(GraphDiagnostic("DUPLICATE_NODE", "node ids must be unique", prefix))
        if not nodes:
            diagnostics.append(GraphDiagnostic("EMPTY_SUBGRAPH", "subgraph has no nodes", prefix))
        if subgraph.entry_node not in nodes:
            diagnostics.append(
                GraphDiagnostic(
                    "UNKNOWN_ENTRY_NODE",
                    f"unknown entry node: {subgraph.entry_node}",
                    prefix,
                )
            )
        for edge in subgraph.edges:
            if edge.source not in nodes or edge.target not in nodes:
                diagnostics.append(
                    GraphDiagnostic(
                        "DANGLING_EDGE",
                        f"subgraph edge references {edge.source}->{edge.target}",
                        prefix,
                    )
                )
        diagnostics.extend(
            _cycle_diagnostics(
                subgraph.entry_node,
                subgraph.edges,
                nodes,
                loops=subgraph.loops,
                path=prefix,
            )
        )
        _validate_loops(subgraph.loops, nodes, prefix, diagnostics)
        if subgraph.entry_node in nodes:
            reachable = _reachable(subgraph.entry_node, subgraph.edges, nodes)
            for node_id in nodes:
                if node_id not in reachable:
                    diagnostics.append(
                        GraphDiagnostic(
                            "UNREACHABLE_NODE",
                            f"node is unreachable: {node_id}",
                            f"{prefix}.nodes.{node_id}",
                        )
                    )

        if not subgraph.success_nodes:
            diagnostics.append(
                GraphDiagnostic(
                    "MISSING_SUCCESS_NODE",
                    "success_nodes must not be empty",
                    prefix,
                )
            )
        if not subgraph.failure_nodes:
            diagnostics.append(
                GraphDiagnostic(
                    "MISSING_FAILURE_NODE",
                    "failure_nodes must not be empty",
                    prefix,
                )
            )
        for node_id in (*subgraph.success_nodes, *subgraph.failure_nodes):
            if node_id not in nodes:
                diagnostics.append(
                    GraphDiagnostic(
                        "UNKNOWN_TERMINAL_NODE",
                        f"unknown terminal node: {node_id}",
                        prefix,
                    )
                )
        if not any(checkpoint.validate for checkpoint in subgraph.checkpoints):
            diagnostics.append(
                GraphDiagnostic(
                    "MISSING_VALID_CHECKPOINT",
                    "subgraph needs a validating checkpoint",
                    prefix,
                )
            )

        for node in subgraph.nodes:
            _validate_ports(node.inputs, f"{prefix}.nodes.{node.node_id}.inputs", diagnostics)
            _validate_ports(node.outputs, f"{prefix}.nodes.{node.node_id}.outputs", diagnostics)
            if node.node_type == "action" and not node.skill_calls:
                diagnostics.append(
                    GraphDiagnostic(
                        "ACTION_WITHOUT_SKILL",
                        "action node must declare at least one skill call",
                        f"{prefix}.nodes.{node.node_id}",
                    )
                )
            if node.node_type == "action" and not node.postconditions:
                diagnostics.append(
                    GraphDiagnostic(
                        "ACTION_WITHOUT_POSTCONDITION",
                        "action node must declare an observable postcondition",
                        f"{prefix}.nodes.{node.node_id}",
                    )
                )
            resource_ids = [resource.resource_id for resource in node.resources]
            if len(resource_ids) != len(set(resource_ids)):
                diagnostics.append(
                    GraphDiagnostic(
                        "DUPLICATE_RESOURCE",
                        "node declares the same resource more than once",
                        f"{prefix}.nodes.{node.node_id}",
                    )
                )

        self._validate_local_bindings(subgraph, nodes, diagnostics)

    def _validate_local_bindings(
        self,
        subgraph: SubgraphSpec,
        nodes: dict[str, object],
        diagnostics: list[GraphDiagnostic],
    ) -> None:
        prefix = f"subgraphs.{subgraph.subgraph_id}"
        node_map = {node.node_id: node for node in subgraph.nodes}
        external_inputs = {port.name: port.type_name for port in subgraph.inputs}
        for binding in subgraph.bindings:
            source = node_map.get(binding.source_node)
            target = node_map.get(binding.target_node)
            if source is None or target is None:
                diagnostics.append(
                    GraphDiagnostic(
                        "DANGLING_BINDING",
                        "binding references an unknown node",
                        prefix,
                    )
                )
                continue
            source_port = _port(source.outputs, binding.source_port)
            target_port = _port(target.inputs, binding.target_port)
            if source_port is None or target_port is None:
                diagnostics.append(
                    GraphDiagnostic("UNKNOWN_PORT", "binding references an unknown port", prefix)
                )
                continue
            if source_port.type_name != target_port.type_name:
                diagnostics.append(
                    GraphDiagnostic(
                        "PORT_TYPE_MISMATCH",
                        f"{binding.source_node}.{binding.source_port} ({source_port.type_name}) "
                        f"cannot feed {binding.target_node}.{binding.target_port} "
                        f"({target_port.type_name})",
                        prefix,
                    )
                )

        bound_inputs = {(binding.target_node, binding.target_port) for binding in subgraph.bindings}
        for node in subgraph.nodes:
            for port in node.inputs:
                if (node.node_id, port.name) in bound_inputs:
                    continue
                if port.required and port.name not in external_inputs:
                    diagnostics.append(
                        GraphDiagnostic(
                            "UNBOUND_INPUT",
                            f"required input is not bound: {node.node_id}.{port.name}",
                            prefix,
                        )
                    )

        output_map = {port.name: port.type_name for port in subgraph.outputs}
        for binding in subgraph.output_bindings:
            source = node_map.get(binding.source_node)
            if source is None:
                diagnostics.append(
                    GraphDiagnostic(
                        "DANGLING_OUTPUT_BINDING",
                        "output binding references an unknown node",
                        prefix,
                    )
                )
                continue
            source_port = _port(source.outputs, binding.source_port)
            output_type = output_map.get(binding.output_port)
            if source_port is None or output_type is None:
                diagnostics.append(
                    GraphDiagnostic(
                        "UNKNOWN_OUTPUT_PORT",
                        "output binding references an unknown port",
                        prefix,
                    )
                )
                continue
            if source_port.type_name != output_type:
                diagnostics.append(
                    GraphDiagnostic(
                        "PORT_TYPE_MISMATCH",
                        f"{binding.source_node}.{binding.source_port} ({source_port.type_name}) "
                        f"cannot expose {binding.output_port} ({output_type})",
                        prefix,
                    )
                )
        bound_outputs = {binding.output_port for binding in subgraph.output_bindings}
        for port in subgraph.outputs:
            if port.required and port.name not in bound_outputs:
                diagnostics.append(
                    GraphDiagnostic(
                        "UNBOUND_OUTPUT",
                        f"required output is not bound: {port.name}",
                        prefix,
                    )
                )

    def _validate_mission_bindings(
        self,
        graph: MissionGraph,
        subgraphs: dict[str, SubgraphSpec],
        diagnostics: list[GraphDiagnostic],
    ) -> None:
        reachable = _reachable(graph.entry_subgraph, graph.edges, subgraphs)
        bindings_by_target: dict[tuple[str, str], list[object]] = {}
        for binding in graph.bindings:
            bindings_by_target.setdefault(
                (binding.target_subgraph, binding.target_port), []
            ).append(binding)
            source = subgraphs.get(binding.source_subgraph)
            target = subgraphs.get(binding.target_subgraph)
            if source is None or target is None:
                diagnostics.append(
                    GraphDiagnostic(
                        "DANGLING_MISSION_BINDING",
                        "mission binding references an unknown subgraph",
                    )
                )
                continue
            if binding.source_subgraph not in reachable:
                diagnostics.append(
                    GraphDiagnostic(
                        "UNREACHABLE_BINDING_SOURCE",
                        f"mission binding source is unreachable: "
                        f"{binding.source_subgraph}",
                        "mission.bindings",
                    )
                )
            if binding.source_subgraph == binding.target_subgraph:
                diagnostics.append(
                    GraphDiagnostic(
                        "INVALID_MISSION_BINDING_ORDER",
                        "a mission binding cannot source and target the same subgraph",
                        "mission.bindings",
                    )
                )
            elif binding.target_subgraph not in _reachable(
                binding.source_subgraph, graph.edges, subgraphs
            ):
                diagnostics.append(
                    GraphDiagnostic(
                        "MISSION_BINDING_SOURCE_NOT_PREDECESSOR",
                        f"{binding.source_subgraph} cannot reach "
                        f"{binding.target_subgraph} through mission edges",
                        "mission.bindings",
                    )
                )
            source_port = _port(source.outputs, binding.source_port)
            target_port = _port(target.inputs, binding.target_port)
            if source_port is None or target_port is None:
                diagnostics.append(
                    GraphDiagnostic(
                        "UNKNOWN_MISSION_PORT",
                        "mission binding references an unknown port",
                    )
                )
                continue
            if source_port.type_name != target_port.type_name:
                diagnostics.append(
                    GraphDiagnostic(
                        "PORT_TYPE_MISMATCH",
                        f"{binding.source_subgraph}.{binding.source_port} "
                        f"({source_port.type_name}) "
                        f"cannot feed {binding.target_subgraph}.{binding.target_port} "
                        f"({target_port.type_name})",
                        "mission.bindings",
                    )
                )

        for subgraph in graph.subgraphs:
            for port in subgraph.inputs:
                if not port.required:
                    continue
                target = (subgraph.subgraph_id, port.name)
                bindings = bindings_by_target.get(target, ())
                if not bindings:
                    diagnostics.append(
                        GraphDiagnostic(
                            "UNBOUND_MISSION_INPUT",
                            f"required mission input is not bound: "
                            f"{subgraph.subgraph_id}.{port.name}",
                            f"subgraphs.{subgraph.subgraph_id}.inputs",
                        )
                    )
                elif len(bindings) > 1:
                    diagnostics.append(
                        GraphDiagnostic(
                            "MULTIPLE_MISSION_INPUT_BINDINGS",
                            f"required mission input has multiple bindings: "
                            f"{subgraph.subgraph_id}.{port.name}",
                            "mission.bindings",
                        )
                    )

    def _validate_parallel_resources(
        self,
        graph: MissionGraph,
        subgraphs: dict[str, SubgraphSpec],
        diagnostics: list[GraphDiagnostic],
    ) -> None:
        for source in subgraphs:
            targets = [
                edge.target
                for edge in graph.edges
                if edge.source == source and edge.condition is None
            ]
            for index, left_id in enumerate(targets):
                for right_id in targets[index + 1 :]:
                    left_resources = _exclusive_resources(subgraphs.get(left_id))
                    right_resources = _exclusive_resources(subgraphs.get(right_id))
                    conflict = left_resources & right_resources
                    if conflict:
                        diagnostics.append(
                            GraphDiagnostic(
                                "PARALLEL_RESOURCE_CONFLICT",
                                f"parallel branches {left_id} and {right_id} share "
                                f"exclusive resources {sorted(conflict)}",
                                f"mission.edges.{source}",
                            )
                        )

    def _validate_state_flow(
        self,
        graph: MissionGraph,
        initial_facts: Iterable[str],
        diagnostics: list[GraphDiagnostic],
    ) -> None:
        """Check that every node precondition has a guaranteed fact source.

        This is a conservative must-analysis over normal success paths. A
        fact is available at a join only when every reachable predecessor
        establishes it. Failure/recovery edges do not establish facts. Truth
        in the current SceneSnapshot is evaluated by the runtime verifier;
        this method only checks provenance.
        """
        initial = {_fact_key(fact) for fact in initial_facts}
        subgraphs = {subgraph.subgraph_id: subgraph for subgraph in graph.subgraphs}
        predecessors: dict[str, tuple[str, ...]] = {
            subgraph_id: tuple(
                edge.source
                for edge in graph.edges
                if edge.target == subgraph_id
                and edge.source in subgraphs
                and not _is_failure_transition(edge.condition)
            )
            for subgraph_id in subgraphs
        }

        summaries: dict[str, set[str]] = {subgraph_id: set() for subgraph_id in subgraphs}
        available: dict[str, set[str]] = {}
        for _ in range(max(1, len(subgraphs) * 2 + 1)):
            next_available: dict[str, set[str]] = {}
            for subgraph_id in subgraphs:
                if subgraph_id == graph.entry_subgraph:
                    next_available[subgraph_id] = set(initial)
                    continue
                upstream = predecessors[subgraph_id]
                next_available[subgraph_id] = (
                    _intersection(summaries[source] for source in upstream)
                    if upstream
                    else set()
                )
            next_summaries = {
                subgraph_id: _local_exit_facts(subgraph, next_available[subgraph_id])
                for subgraph_id, subgraph in subgraphs.items()
            }
            available = next_available
            if next_summaries == summaries:
                summaries = next_summaries
                break
            summaries = next_summaries

        for subgraph_id, subgraph in subgraphs.items():
            if subgraph_id not in available:
                continue
            missing = _local_precondition_gaps(subgraph, available[subgraph_id])
            for node_id, predicates in missing.items():
                for predicate in predicates:
                    diagnostics.append(
                        GraphDiagnostic(
                            "UNESTABLISHED_PRECONDITION",
                            f"node {node_id!r} requires {predicate!r}, but no initial "
                            "fact or mandatory predecessor postcondition establishes it",
                            f"subgraphs.{subgraph_id}.nodes.{node_id}.preconditions",
                        )
                    )


def _validate_ports(
    ports: tuple[object, ...],
    path: str,
    diagnostics: list[GraphDiagnostic],
) -> None:
    names = [port.name for port in ports]
    if len(names) != len(set(names)):
        diagnostics.append(GraphDiagnostic("DUPLICATE_PORT", "port names must be unique", path))


def _port(ports: Iterable[object], name: str):
    return next((port for port in ports if port.name == name), None)


def _exclusive_resources(subgraph: SubgraphSpec | None) -> set[str]:
    if subgraph is None:
        return set()
    return {
        resource.resource_id
        for node in subgraph.nodes
        for resource in node.resources
        if resource.mode == "exclusive"
    }


def scene_initial_facts(graph: MissionGraph, scene: object) -> tuple[str, ...]:
    """Return graph preconditions that are true in the supplied scene.

    The import is lazy so the graph contract layer remains independent from
    the verifier implementation. Unknown predicates are deliberately omitted;
    they cannot be used as an initial fact source.
    """
    from capmas.verification.predicates import PredicateBasedVerifier

    predicates = tuple(
        predicate
        for subgraph in graph.subgraphs
        for node in subgraph.nodes
        for predicate in node.preconditions
    )
    verifier = PredicateBasedVerifier()
    reports = verifier.evaluate_predicates(predicates, scene)  # type: ignore[arg-type]
    return tuple(
        report.name for report in reports if report.passed
    )


def _local_precondition_gaps(
    subgraph: SubgraphSpec,
    entry_facts: set[str],
) -> dict[str, tuple[str, ...]]:
    nodes = {node.node_id: node for node in subgraph.nodes}
    reachable = _reachable(subgraph.entry_node, subgraph.edges, nodes)
    predecessors: dict[str, tuple[str, ...]] = {
        node_id: tuple(
            edge.source
            for edge in subgraph.edges
            if edge.target == node_id
            and edge.source in nodes
            and not _is_failure_transition(edge.condition)
        )
        for node_id in nodes
    }
    available: dict[str, set[str]] = {node_id: set() for node_id in nodes}
    for _ in range(max(1, len(nodes) * 2 + 1)):
        next_available: dict[str, set[str]] = {}
        for node_id in nodes:
            if node_id == subgraph.entry_node:
                next_available[node_id] = set(entry_facts)
                continue
            upstream = predecessors[node_id]
            next_available[node_id] = (
                _intersection(
                    available[source] | _node_effects(nodes[source], subgraph)
                    for source in upstream
                )
                if upstream
                else set()
            )
        if next_available == available:
            available = next_available
            break
        available = next_available

    gaps: dict[str, tuple[str, ...]] = {}
    for node_id in reachable:
        missing = tuple(
            predicate
            for predicate in nodes[node_id].preconditions
            if not _is_runtime_deferred_predicate(predicate)
            and _fact_key(predicate) not in available[node_id]
        )
        if missing:
            gaps[node_id] = missing
    return gaps


def _local_exit_facts(subgraph: SubgraphSpec, entry_facts: set[str]) -> set[str]:
    nodes = {node.node_id: node for node in subgraph.nodes}
    gaps = _local_precondition_gaps(subgraph, entry_facts)
    del gaps  # Effects are still summarized; invalidity is reported separately.
    available: dict[str, set[str]] = {node_id: set() for node_id in nodes}
    predecessors: dict[str, tuple[str, ...]] = {
        node_id: tuple(
            edge.source
            for edge in subgraph.edges
            if edge.target == node_id
            and edge.source in nodes
            and not _is_failure_transition(edge.condition)
        )
        for node_id in nodes
    }
    for _ in range(max(1, len(nodes) * 2 + 1)):
        next_available: dict[str, set[str]] = {}
        for node_id in nodes:
            if node_id == subgraph.entry_node:
                next_available[node_id] = set(entry_facts)
                continue
            upstream = predecessors[node_id]
            next_available[node_id] = (
                _intersection(
                    available[source] | _node_effects(nodes[source], subgraph)
                    for source in upstream
                )
                if upstream
                else set()
            )
        if next_available == available:
            available = next_available
            break
        available = next_available
    exits = [
        available[node_id] | _node_effects(nodes[node_id], subgraph)
        for node_id in subgraph.success_nodes
        if node_id in nodes
    ]
    return _intersection(exits) if exits else set()


def _node_effects(node: object, subgraph: SubgraphSpec) -> set[str]:
    effects = {
        _fact_key(predicate)
        for predicate in node.postconditions
        if not _is_runtime_deferred_predicate(predicate)
    }
    if getattr(node, "node_type", None) == "checkpoint":
        checkpoint = next(
            (
                item
                for item in subgraph.checkpoints
                if item.validate and item.name == node.node_id
            ),
            None,
        )
        if checkpoint is not None:
            effects.update(
                _fact_key(predicate)
                for predicate in checkpoint.predicates
                if not _is_runtime_deferred_predicate(predicate)
            )
    return effects


def _intersection(values: Iterable[set[str]]) -> set[str]:
    items = [set(value) for value in values]
    if not items:
        return set()
    result = items[0]
    for item in items[1:]:
        result.intersection_update(item)
    return result


def _fact_key(predicate: str) -> str:
    value = re.sub(r"\s+", " ", predicate.strip().lower())
    value = re.sub(r"\s*,\s*", ",", value)
    if value.endswith("()"):
        value = value[:-2]
    return value


def _is_runtime_deferred_predicate(predicate: str) -> bool:
    """Identify predicates whose truth must be checked at dispatch time.

    Scene freshness is a property of the observation-to-dispatch interval. A
    long LLM compile can make an initially fresh snapshot stale, so it must not
    be treated as a persistent static graph fact or effect.
    """
    value = _fact_key(predicate)
    return value == "scene_fresh" or value.startswith("scene_fresh(")


def _is_failure_transition(condition: str | None) -> bool:
    if condition is None:
        return False
    value = condition.strip().lower()
    return any(
        marker in value
        for marker in ("fail", "error", "abort", "timeout", "unsafe", "recover", "not(")
    )


def _reachable(start: str, edges: Iterable[object], nodes: dict[str, object]) -> set[str]:
    reachable = {start} if start in nodes else set()
    changed = True
    while changed:
        changed = False
        for edge in edges:
            if edge.source in reachable and edge.target in nodes and edge.target not in reachable:
                reachable.add(edge.target)
                changed = True
    return reachable


def _cycle_diagnostics(
    start: str,
    edges: Iterable[object],
    nodes: dict[str, object],
    loops: tuple[LoopSpec, ...] = (),
    path: str = "graph",
) -> list[GraphDiagnostic]:
    adjacency: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    for edge in edges:
        if edge.source in adjacency and edge.target in nodes:
            adjacency[edge.source].append(edge.target)
    visiting: set[str] = set()
    visited: set[str] = set()
    stack: list[str] = []
    cycles: list[set[str]] = []

    def visit(node_id: str) -> None:
        if node_id in visiting:
            try:
                start_index = stack.index(node_id)
            except ValueError:  # pragma: no cover - defensive
                start_index = 0
            cycles.append(set(stack[start_index:]))
            return
        if node_id in visited:
            return
        visiting.add(node_id)
        stack.append(node_id)
        for child in adjacency[node_id]:
            visit(child)
        stack.pop()
        visiting.remove(node_id)
        visited.add(node_id)

    if start in adjacency:
        visit(start)

    loop_entries = {loop.entry_node for loop in loops}
    diagnostics: list[GraphDiagnostic] = []
    for cycle in cycles:
        if not cycle & loop_entries:
            diagnostics.append(
                GraphDiagnostic(
                    "UNBOUNDED_CYCLE",
                    "mission/subgraph cycle requires an explicit LoopSpec budget",
                    path,
                )
            )
    return diagnostics


def _validate_loops(
    loops: tuple[LoopSpec, ...],
    nodes: dict[str, object],
    path: str,
    diagnostics: list[GraphDiagnostic],
) -> None:
    entries: set[str] = set()
    for loop in loops:
        loop_path = f"{path}.loops.{loop.entry_node}"
        if loop.entry_node in entries:
            diagnostics.append(
                GraphDiagnostic("DUPLICATE_LOOP_ENTRY", "loop entries must be unique", loop_path)
            )
        entries.add(loop.entry_node)
        if loop.entry_node not in nodes:
            diagnostics.append(
                GraphDiagnostic(
                    "UNKNOWN_LOOP_ENTRY",
                    f"loop entry is not declared: {loop.entry_node}",
                    loop_path,
                )
            )
        if loop.max_visits <= 0:
            diagnostics.append(
                GraphDiagnostic(
                    "INVALID_LOOP_BUDGET",
                    "loop max_visits must be positive",
                    loop_path,
                )
            )
        if loop.max_duration_ms < 0:
            diagnostics.append(
                GraphDiagnostic(
                    "INVALID_LOOP_BUDGET",
                    "loop max_duration_ms must not be negative",
                    loop_path,
                )
            )
        if not loop.exit_conditions:
            diagnostics.append(
                GraphDiagnostic(
                    "MISSING_LOOP_EXIT",
                    "loop must declare at least one exit condition",
                    loop_path,
                )
            )
