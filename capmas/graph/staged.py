"""Strict serialization and static validation for staged topology artifacts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import math
import re
from typing import Any

from capmas.contracts.graph import LoopSpec, MissionBinding, MissionEdge
from capmas.contracts.staged import (
    MissionTopology,
    STAGED_TOPOLOGY_SCHEMA_VERSION,
    TopologySubgoal,
)


class TopologySchemaError(ValueError):
    """Raised when staged topology data is not versioned strict JSON."""


@dataclass(frozen=True)
class TopologyDiagnostic:
    code: str
    message: str
    path: str = ""


@dataclass(frozen=True)
class TopologyValidationResult:
    valid: bool
    diagnostics: tuple[TopologyDiagnostic, ...] = ()

    @property
    def errors(self) -> tuple[TopologyDiagnostic, ...]:
        return self.diagnostics


class TopologyValidator:
    """Validate topology before local Policy Agents are fanned out."""

    def validate(self, topology: MissionTopology) -> TopologyValidationResult:
        diagnostics: list[TopologyDiagnostic] = []
        by_id = {item.subgraph_id: item for item in topology.subgoals}
        by_subgoal_id = {item.subgoal_id: item for item in topology.subgoals}
        if len(by_id) != len(topology.subgoals):
            diagnostics.append(TopologyDiagnostic("DUPLICATE_SUBGRAPH", "topology subgraph ids must be unique"))
        if len(by_subgoal_id) != len(topology.subgoals):
            diagnostics.append(TopologyDiagnostic("DUPLICATE_SUBGOAL", "topology subgoal ids must be unique"))
        if not topology.subgoals:
            diagnostics.append(TopologyDiagnostic("EMPTY_TOPOLOGY", "topology must contain at least one subgoal"))
        if topology.entry_subgraph not in by_id:
            diagnostics.append(TopologyDiagnostic("UNKNOWN_ENTRY", f"unknown entry subgraph: {topology.entry_subgraph}"))
        for field_name, values in (
            ("success_subgraphs", topology.success_subgraphs),
            ("failure_subgraphs", topology.failure_subgraphs),
        ):
            if not values:
                diagnostics.append(TopologyDiagnostic("MISSING_TERMINAL", f"{field_name} must not be empty"))
            for value in values:
                if value not in by_id:
                    diagnostics.append(TopologyDiagnostic("UNKNOWN_TERMINAL", f"unknown terminal: {value}", field_name))

        dependencies: dict[str, set[str]] = {item.subgraph_id: set(item.depends_on) for item in topology.subgoals}
        for item in topology.subgoals:
            if item.execution_kind == "checkpoint_only" and not item.success_predicates:
                diagnostics.append(
                    TopologyDiagnostic(
                        "CHECKPOINT_SUCCESS_PREDICATE_MISSING",
                        f"checkpoint-only subgoal {item.subgraph_id} must declare at least one success predicate",
                        f"subgoals.{item.subgraph_id}",
                    )
                )
            diagnostics.extend(
                _scene_fresh_diagnostics(
                    (*item.success_predicates, *item.failure_predicates),
                    f"subgoals.{item.subgraph_id}",
                )
            )
            for dependency in item.depends_on:
                if dependency not in by_id:
                    diagnostics.append(
                        TopologyDiagnostic(
                            "UNKNOWN_DEPENDENCY",
                            f"{item.subgraph_id} depends on unknown subgraph {dependency}",
                            f"subgoals.{item.subgraph_id}",
                        )
                    )
        for edge in topology.edges:
            if edge.source not in by_id or edge.target not in by_id:
                diagnostics.append(
                    TopologyDiagnostic("DANGLING_EDGE", f"topology edge references {edge.source}->{edge.target}", "edges")
                )
        if topology.entry_subgraph in by_id:
            reachable = {topology.entry_subgraph}
            changed = True
            while changed:
                changed = False
                for edge in topology.edges:
                    if edge.source in reachable and edge.target in by_id and edge.target not in reachable:
                        reachable.add(edge.target)
                        changed = True
            for subgraph_id in by_id:
                if subgraph_id not in reachable:
                    diagnostics.append(
                        TopologyDiagnostic(
                            "UNREACHABLE_SUBGRAPH",
                            f"subgraph is unreachable: {subgraph_id}",
                            f"subgoals.{subgraph_id}",
                        )
                    )
        for source in by_id:
            raw_outgoing = tuple(edge for edge in topology.edges if edge.source == source)
            # Exact canonical duplicates are a schema error.  This remains a
            # strict wire-level check even though the assembler can defensively
            # deduplicate them when called directly.
            for condition in ("success", "failure"):
                exact_counts = Counter(
                    (edge.source, edge.target, edge.condition)
                    for edge in raw_outgoing
                    if edge.condition == condition
                )
                if any(count > 1 for count in exact_counts.values()):
                    diagnostics.append(
                        TopologyDiagnostic(
                            "AMBIGUOUS_TRANSITION",
                            f"{source} has duplicate canonical {condition} transitions",
                            "edges",
                        )
                    )
            # Nullable and semantic edges use the same resolution policy as
            # MissionTopology.assemble(): dependency edges take precedence,
            # otherwise stable forward subgoal order selects a success edge.
            # Failure edges have no equivalent safe default and remain
            # ambiguous unless exactly one survives normalization.
            normalized_outgoing = tuple(
                edge
                for edge in topology.normalized_edges()
                if edge.source == source
            )
            for condition in ("success", "failure"):
                matches = [edge for edge in normalized_outgoing if edge.condition == condition]
                if len(matches) > 1:
                    diagnostics.append(
                        TopologyDiagnostic(
                            "AMBIGUOUS_TRANSITION",
                            f"{source} has multiple {condition} transitions after normalization",
                            "edges",
                        )
                    )
        normalized_edges = topology.normalized_edges()
        normal_success_edges = tuple(
            edge for edge in normalized_edges if edge.condition == "success"
        )
        for edge in normal_success_edges:
            if edge.target in _normal_success_ancestors(edge.source, normal_success_edges):
                diagnostics.append(
                    TopologyDiagnostic(
                        "SUCCESS_BACK_EDGE",
                        f"success transition {edge.source}->{edge.target} re-enters a "
                        "previously committed subgraph in rolling execution",
                        "edges",
                    )
                )
        for edge in normalized_edges:
            if edge.condition != "failure":
                continue
            target = by_id[edge.target]
            committed_before_source = _normal_success_ancestors(
                edge.source,
                normal_success_edges,
            )
            uncommitted_dependencies = set(target.depends_on) - committed_before_source
            if uncommitted_dependencies:
                diagnostics.append(
                    TopologyDiagnostic(
                        "RECOVERY_DEPENDS_ON_UNCOMMITTED",
                        f"failure transition {edge.source}->{edge.target} requires "
                        f"uncommitted dependencies: {sorted(uncommitted_dependencies)}",
                        f"subgoals.{target.subgraph_id}.depends_on",
                    )
                )
        diagnostics.extend(_cycle_diagnostics(dependencies))

        # The edge list is authoritative for runtime control flow.  A
        # dependency is allowed only when the corresponding success edge is
        # present, preventing an LLM from silently changing execution order.
        edge_pairs = {(edge.source, edge.target) for edge in topology.edges}
        for item in topology.subgoals:
            for dependency in item.depends_on:
                if (dependency, item.subgraph_id) not in edge_pairs:
                    diagnostics.append(
                        TopologyDiagnostic(
                            "DEPENDENCY_EDGE_MISSING",
                            f"dependency {dependency}->{item.subgraph_id} has no matching edge",
                            f"subgoals.{item.subgraph_id}",
                        )
                    )
        return TopologyValidationResult(not diagnostics, tuple(diagnostics))

    def raise_if_invalid(self, topology: MissionTopology) -> None:
        result = self.validate(topology)
        if not result.valid:
            details = "; ".join(f"{item.code}: {item.message}" for item in result.errors)
            raise ValueError(f"invalid mission topology: {details}")


def topology_to_dict(topology: MissionTopology) -> dict[str, Any]:
    return {
        "schema_version": STAGED_TOPOLOGY_SCHEMA_VERSION,
        "mission_id": topology.mission_id,
        "task": topology.task,
        "graph_version": topology.graph_version,
        "parent_scene_version": topology.parent_scene_version,
        "entry_subgraph": topology.entry_subgraph,
        "success_subgraphs": list(topology.success_subgraphs),
        "failure_subgraphs": list(topology.failure_subgraphs),
        "subgoals": [
            {
                "subgraph_id": item.subgraph_id,
                "subgoal_id": item.subgoal_id,
                "description": item.description,
                "depends_on": list(item.depends_on),
                "success_predicates": list(item.success_predicates),
                "failure_predicates": list(item.failure_predicates),
                "required_agent_role": item.required_agent_role,
                "execution_kind": item.execution_kind,
            }
            for item in topology.subgoals
        ],
        "edges": [_edge_to_dict(edge) for edge in topology.edges],
        "bindings": [_binding_to_dict(binding) for binding in topology.bindings],
        "loops": [_loop_to_dict(loop) for loop in topology.loops],
    }


def _normal_success_ancestors(
    source: str,
    edges: tuple[MissionEdge, ...],
) -> set[str]:
    """Return subgraphs that may be committed before a source succeeds."""
    predecessors: dict[str, set[str]] = {}
    for edge in edges:
        predecessors.setdefault(edge.target, set()).add(edge.source)
    ancestors: set[str] = set()
    frontier = list(predecessors.get(source, ()))
    while frontier:
        candidate = frontier.pop()
        if candidate in ancestors:
            continue
        ancestors.add(candidate)
        frontier.extend(predecessors.get(candidate, ()))
    return ancestors


def topology_from_dict(raw: Mapping[str, Any]) -> MissionTopology:
    data = _object(raw, _ROOT_KEYS, "topology")
    _require_version(data, "topology")
    subgoals = tuple(
        _subgoal_from_dict(item, f"topology.subgoals[{index}]")
        for index, item in enumerate(_list(data, "subgoals", "topology"))
    )
    topology = MissionTopology(
        mission_id=_string(data, "mission_id", "topology"),
        task=_string(data, "task", "topology"),
        graph_version=_integer(data, "graph_version", "topology"),
        parent_scene_version=_optional_integer(data, "parent_scene_version", "topology"),
        entry_subgraph=_string(data, "entry_subgraph", "topology"),
        success_subgraphs=_string_tuple(data, "success_subgraphs", "topology"),
        failure_subgraphs=_string_tuple(data, "failure_subgraphs", "topology"),
        subgoals=subgoals,
        edges=tuple(_edge_from_dict(item, f"topology.edges[{index}]") for index, item in enumerate(_list(data, "edges", "topology"))),
        bindings=tuple(_binding_from_dict(item, f"topology.bindings[{index}]") for index, item in enumerate(_list(data, "bindings", "topology"))),
        loops=tuple(_loop_from_dict(item, f"topology.loops[{index}]") for index, item in enumerate(_list(data, "loops", "topology"))),
    )
    validation = TopologyValidator().validate(topology)
    if not validation.valid:
        details = "; ".join(f"{item.code}: {item.message}" for item in validation.errors)
        raise TopologySchemaError(details)
    return topology


def _subgoal_from_dict(raw: Any, path: str) -> TopologySubgoal:
    data = _object(raw, _SUBGOAL_KEYS, path)
    return TopologySubgoal(
        subgraph_id=_string(data, "subgraph_id", path),
        subgoal_id=_string(data, "subgoal_id", path),
        description=_string(data, "description", path),
        depends_on=_string_tuple(data, "depends_on", path),
        success_predicates=_string_tuple(data, "success_predicates", path),
        failure_predicates=_string_tuple(data, "failure_predicates", path),
        required_agent_role=_string(data, "required_agent_role", path),
        execution_kind=_string(data, "execution_kind", path),
    )


def _edge_to_dict(edge: MissionEdge) -> dict[str, Any]:
    return {"source": edge.source, "target": edge.target, "condition": edge.condition}


def _edge_from_dict(raw: Any, path: str) -> MissionEdge:
    data = _object(raw, _EDGE_KEYS, path)
    return MissionEdge(_string(data, "source", path), _string(data, "target", path), _optional_string(data, "condition", path))


def _binding_to_dict(binding: MissionBinding) -> dict[str, Any]:
    return {
        "source_subgraph": binding.source_subgraph,
        "source_port": binding.source_port,
        "target_subgraph": binding.target_subgraph,
        "target_port": binding.target_port,
    }


def _binding_from_dict(raw: Any, path: str) -> MissionBinding:
    data = _object(raw, _BINDING_KEYS, path)
    return MissionBinding(
        _string(data, "source_subgraph", path),
        _string(data, "source_port", path),
        _string(data, "target_subgraph", path),
        _string(data, "target_port", path),
    )


def _loop_to_dict(loop: LoopSpec) -> dict[str, Any]:
    return {
        "entry_node": loop.entry_node,
        "max_visits": loop.max_visits,
        "max_duration_ms": loop.max_duration_ms,
        "exit_conditions": list(loop.exit_conditions),
    }


def _loop_from_dict(raw: Any, path: str) -> LoopSpec:
    data = _object(raw, _LOOP_KEYS, path)
    return LoopSpec(
        entry_node=_string(data, "entry_node", path),
        max_visits=_integer(data, "max_visits", path),
        max_duration_ms=_integer(data, "max_duration_ms", path),
        exit_conditions=_string_tuple(data, "exit_conditions", path),
    )


def _cycle_diagnostics(dependencies: Mapping[str, set[str]]) -> list[TopologyDiagnostic]:
    visiting: set[str] = set()
    visited: set[str] = set()
    diagnostics: list[TopologyDiagnostic] = []

    def visit(node: str) -> None:
        if node in visiting:
            diagnostics.append(TopologyDiagnostic("UNBOUNDED_TOPOLOGY_CYCLE", f"topology dependency cycle includes {node}"))
            return
        if node in visited:
            return
        visiting.add(node)
        for dependency in dependencies.get(node, ()):
            if dependency in dependencies:
                visit(dependency)
        visiting.remove(node)
        visited.add(node)

    for node in dependencies:
        visit(node)
    return diagnostics


_SCENE_FRESH_RE = re.compile(r"^scene_fresh\(([^()]*)\)$")


def _scene_fresh_diagnostics(
    predicates: tuple[str, ...],
    path: str,
) -> list[TopologyDiagnostic]:
    diagnostics: list[TopologyDiagnostic] = []
    for index, predicate in enumerate(predicates):
        match = _SCENE_FRESH_RE.fullmatch(predicate.strip())
        if match is None:
            continue
        try:
            threshold_ms = float(match.group(1).strip())
        except ValueError:
            threshold_ms = float("nan")
        if not math.isfinite(threshold_ms) or threshold_ms <= 0.0:
            diagnostics.append(
                TopologyDiagnostic(
                    "INVALID_SCENE_FRESH_THRESHOLD",
                    f"{predicate!r} must use a finite threshold_ms > 0; use scene_fresh(1000) "
                    "for a normal observation checkpoint",
                    f"{path}.predicates[{index}]",
                )
            )
    return diagnostics


_ROOT_KEYS = frozenset({
    "schema_version", "mission_id", "task", "graph_version", "parent_scene_version",
    "entry_subgraph", "success_subgraphs", "failure_subgraphs", "subgoals", "edges",
    "bindings", "loops",
})
_SUBGOAL_KEYS = frozenset({
    "subgraph_id", "subgoal_id", "description", "depends_on", "success_predicates",
    "failure_predicates", "required_agent_role", "execution_kind",
})
_EDGE_KEYS = frozenset({"source", "target", "condition"})
_BINDING_KEYS = frozenset({"source_subgraph", "source_port", "target_subgraph", "target_port"})
_LOOP_KEYS = frozenset({"entry_node", "max_visits", "max_duration_ms", "exit_conditions"})


def _object(value: Any, allowed: frozenset[str], path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TopologySchemaError(f"{path} must be an object")
    data = dict(value)
    unknown = set(data) - allowed
    if unknown:
        raise TopologySchemaError(f"{path} has unknown fields: {sorted(unknown)}")
    return data


def _require_version(data: Mapping[str, Any], path: str) -> None:
    if data.get("schema_version") != STAGED_TOPOLOGY_SCHEMA_VERSION:
        raise TopologySchemaError(
            f"{path}.schema_version must be {STAGED_TOPOLOGY_SCHEMA_VERSION}, "
            f"got {data.get('schema_version')!r}"
        )


def _list(data: Mapping[str, Any], key: str, path: str) -> list[Any]:
    value = data.get(key)
    if not isinstance(value, list):
        raise TopologySchemaError(f"{path}.{key} must be an array")
    return value


def _string(data: Mapping[str, Any], key: str, path: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise TopologySchemaError(f"{path}.{key} must be a non-empty string")
    return value


def _optional_string(data: Mapping[str, Any], key: str, path: str) -> str | None:
    value = data.get(key)
    if value is not None and not isinstance(value, str):
        raise TopologySchemaError(f"{path}.{key} must be a string or null")
    return value


def _integer(data: Mapping[str, Any], key: str, path: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TopologySchemaError(f"{path}.{key} must be an integer")
    if key in {"graph_version", "max_visits"} and value <= 0:
        raise TopologySchemaError(f"{path}.{key} must be positive")
    if key == "max_duration_ms" and value < 0:
        raise TopologySchemaError(f"{path}.{key} must not be negative")
    return value


def _optional_integer(data: Mapping[str, Any], key: str, path: str) -> int | None:
    value = data.get(key)
    if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
        raise TopologySchemaError(f"{path}.{key} must be an integer or null")
    return value


def _string_tuple(data: Mapping[str, Any], key: str, path: str) -> tuple[str, ...]:
    values = _list(data, key, path)
    if any(not isinstance(value, str) or not value for value in values):
        raise TopologySchemaError(f"{path}.{key} must contain non-empty strings")
    return tuple(values)


__all__ = [
    "TopologyDiagnostic",
    "TopologySchemaError",
    "TopologyValidationResult",
    "TopologyValidator",
    "topology_from_dict",
    "topology_to_dict",
]
