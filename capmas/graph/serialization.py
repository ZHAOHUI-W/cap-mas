"""Strict, versioned serialization for CAP-MAS mission graphs.

The wire format is intentionally separate from the frozen runtime contracts.
LLM-produced JSON must pass this parser before it can reach
``GraphValidator`` or the robot runtime.  Every object has an explicit field
allowlist so a misspelled safety or resource field cannot silently disappear.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from capmas.contracts.action import SkillCall, SkillOutputRef
from capmas.contracts.core import SkillRef
from capmas.contracts.graph import (
    CheckpointSpec,
    GraphEdge,
    LoopSpec,
    MissionBinding,
    MissionEdge,
    MissionGraph,
    MotionIntent,
    PortBinding,
    PortSpec,
    ResourceRequirement,
    SubgraphNodeSpec,
    SubgraphOutputBinding,
    SubgraphSpec,
)


GRAPH_SCHEMA_VERSION = 1


class GraphSchemaError(ValueError):
    """Raised when a graph artifact is not valid schema-versioned data."""


def mission_graph_to_dict(graph: MissionGraph) -> dict[str, Any]:
    return {
        "schema_version": GRAPH_SCHEMA_VERSION,
        "mission_id": graph.mission_id,
        "task": graph.task,
        "graph_version": graph.graph_version,
        "parent_scene_version": graph.parent_scene_version,
        "entry_subgraph": graph.entry_subgraph,
        "success_subgraphs": list(graph.success_subgraphs),
        "failure_subgraphs": list(graph.failure_subgraphs),
        "subgraphs": [_subgraph_to_dict(subgraph) for subgraph in graph.subgraphs],
        "edges": [_mission_edge_to_dict(edge) for edge in graph.edges],
        "bindings": [_mission_binding_to_dict(binding) for binding in graph.bindings],
        "loops": [_loop_to_dict(loop) for loop in graph.loops],
    }


def mission_graph_from_dict(raw: Mapping[str, Any]) -> MissionGraph:
    data = _object(raw, _ROOT_KEYS, "graph")
    _require_version(data, "graph")
    return MissionGraph(
        mission_id=_string(data, "mission_id", "graph"),
        task=_string(data, "task", "graph"),
        graph_version=_integer(data, "graph_version", "graph"),
        parent_scene_version=_optional_integer(data, "parent_scene_version", "graph"),
        entry_subgraph=_string(data, "entry_subgraph", "graph"),
        success_subgraphs=_string_tuple(data, "success_subgraphs", "graph"),
        failure_subgraphs=_string_tuple(data, "failure_subgraphs", "graph"),
        subgraphs=tuple(
            _subgraph_from_dict(item, f"graph.subgraphs[{index}]")
            for index, item in enumerate(_list(data, "subgraphs", "graph"))
        ),
        edges=tuple(
            _mission_edge_from_dict(item, f"graph.edges[{index}]")
            for index, item in enumerate(_list(data, "edges", "graph"))
        ),
        bindings=tuple(
            _mission_binding_from_dict(item, f"graph.bindings[{index}]")
            for index, item in enumerate(_list(data, "bindings", "graph"))
        ),
        loops=tuple(
            _loop_from_dict(item, f"graph.loops[{index}]")
            for index, item in enumerate(_list(data, "loops", "graph"))
        ),
    )


def local_subgraph_to_dict(subgraph: SubgraphSpec) -> dict[str, Any]:
    """Serialize one local graph for the staged Policy protocol."""
    return {
        "schema_version": GRAPH_SCHEMA_VERSION,
        "subgraph": _subgraph_to_dict(subgraph),
    }


def local_subgraph_from_dict(raw: Mapping[str, Any]) -> SubgraphSpec:
    """Load the versioned local graph envelope emitted by a Policy Agent."""
    data = _object(raw, {"schema_version", "subgraph"}, "subgraph_artifact")
    _require_version(data, "subgraph_artifact")
    return _subgraph_from_dict(data.get("subgraph"), "subgraph_artifact.subgraph")


def _subgraph_to_dict(subgraph: SubgraphSpec) -> dict[str, Any]:
    return {
        "subgraph_id": subgraph.subgraph_id,
        "subgoal_id": subgraph.subgoal_id,
        "description": subgraph.description,
        "nodes": [_node_to_dict(node) for node in subgraph.nodes],
        "edges": [_edge_to_dict(edge) for edge in subgraph.edges],
        "entry_node": subgraph.entry_node,
        "success_nodes": list(subgraph.success_nodes),
        "failure_nodes": list(subgraph.failure_nodes),
        "inputs": [_port_to_dict(port) for port in subgraph.inputs],
        "outputs": [_port_to_dict(port) for port in subgraph.outputs],
        "bindings": [_binding_to_dict(binding) for binding in subgraph.bindings],
        "output_bindings": [
            _output_binding_to_dict(binding) for binding in subgraph.output_bindings
        ],
        "checkpoints": [_checkpoint_to_dict(checkpoint) for checkpoint in subgraph.checkpoints],
        "assigned_agent": subgraph.assigned_agent,
        "loops": [_loop_to_dict(loop) for loop in subgraph.loops],
    }


def _subgraph_from_dict(raw: Any, path: str) -> SubgraphSpec:
    data = _object(raw, _SUBGRAPH_KEYS, path)
    return SubgraphSpec(
        subgraph_id=_string(data, "subgraph_id", path),
        subgoal_id=_string(data, "subgoal_id", path),
        description=_string(data, "description", path),
        nodes=tuple(
            _node_from_dict(item, f"{path}.nodes[{index}]")
            for index, item in enumerate(_list(data, "nodes", path))
        ),
        edges=tuple(
            _edge_from_dict(item, f"{path}.edges[{index}]")
            for index, item in enumerate(_list(data, "edges", path))
        ),
        entry_node=_string(data, "entry_node", path),
        success_nodes=_string_tuple(data, "success_nodes", path),
        failure_nodes=_string_tuple(data, "failure_nodes", path),
        inputs=tuple(
            _port_from_dict(item, f"{path}.inputs[{index}]")
            for index, item in enumerate(_list(data, "inputs", path))
        ),
        outputs=tuple(
            _port_from_dict(item, f"{path}.outputs[{index}]")
            for index, item in enumerate(_list(data, "outputs", path))
        ),
        bindings=tuple(
            _binding_from_dict(item, f"{path}.bindings[{index}]")
            for index, item in enumerate(_list(data, "bindings", path))
        ),
        output_bindings=tuple(
            _output_binding_from_dict(item, f"{path}.output_bindings[{index}]")
            for index, item in enumerate(_list(data, "output_bindings", path))
        ),
        checkpoints=tuple(
            _checkpoint_from_dict(item, f"{path}.checkpoints[{index}]")
            for index, item in enumerate(_list(data, "checkpoints", path))
        ),
        assigned_agent=_string(data, "assigned_agent", path),
        loops=tuple(
            _loop_from_dict(item, f"{path}.loops[{index}]")
            for index, item in enumerate(_list(data, "loops", path))
        ),
    )


def _node_to_dict(node: SubgraphNodeSpec) -> dict[str, Any]:
    return {
        "node_id": node.node_id,
        "description": node.description,
        "skill_calls": [_skill_call_to_dict(call) for call in node.skill_calls],
        "inputs": [_port_to_dict(port) for port in node.inputs],
        "outputs": [_port_to_dict(port) for port in node.outputs],
        "preconditions": list(node.preconditions),
        "postconditions": list(node.postconditions),
        "resources": [_resource_to_dict(resource) for resource in node.resources],
        "max_duration_ms": node.max_duration_ms,
        "max_sim_steps": node.max_sim_steps,
        "proposed_by": node.proposed_by,
        "recovery_policy": node.recovery_policy,
        "node_type": node.node_type,
        "motion_intent": _motion_intent_to_dict(node.motion_intent),
    }


def _node_from_dict(raw: Any, path: str) -> SubgraphNodeSpec:
    data = _object(raw, _NODE_KEYS, path)
    return SubgraphNodeSpec(
        node_id=_string(data, "node_id", path),
        description=_string(data, "description", path),
        skill_calls=tuple(
            _skill_call_from_dict(item, f"{path}.skill_calls[{index}]")
            for index, item in enumerate(_list(data, "skill_calls", path))
        ),
        inputs=tuple(
            _port_from_dict(item, f"{path}.inputs[{index}]")
            for index, item in enumerate(_list(data, "inputs", path))
        ),
        outputs=tuple(
            _port_from_dict(item, f"{path}.outputs[{index}]")
            for index, item in enumerate(_list(data, "outputs", path))
        ),
        preconditions=_string_tuple(data, "preconditions", path),
        postconditions=_string_tuple(data, "postconditions", path),
        resources=tuple(
            _resource_from_dict(item, f"{path}.resources[{index}]")
            for index, item in enumerate(_list(data, "resources", path))
        ),
        max_duration_ms=_integer(data, "max_duration_ms", path),
        max_sim_steps=_integer(data, "max_sim_steps", path),
        proposed_by=_string(data, "proposed_by", path),
        recovery_policy=_string(data, "recovery_policy", path),
        node_type=_string(data, "node_type", path),
        motion_intent=_motion_intent_from_dict(
            data.get("motion_intent"), f"{path}.motion_intent"
        ),
    )


def _motion_intent_to_dict(intent: MotionIntent | None) -> dict[str, Any] | None:
    if intent is None:
        return None
    return {
        "kind": intent.kind,
        "object_track_id": intent.object_track_id,
        "target_track_id": intent.target_track_id,
        "approach_vector_xyz": (
            list(intent.approach_vector_xyz) if intent.approach_vector_xyz is not None else None
        ),
        "standoff_m": intent.standoff_m,
        "target_pose_wxyz_xyz": (
            list(intent.target_pose_wxyz_xyz) if intent.target_pose_wxyz_xyz is not None else None
        ),
    }


def _motion_intent_from_dict(raw: Any, path: str) -> MotionIntent | None:
    if raw is None:
        return None
    data = _object(
        raw,
        {
            "kind",
            "object_track_id",
            "target_track_id",
            "approach_vector_xyz",
            "standoff_m",
            "target_pose_wxyz_xyz",
        },
        path,
    )
    return MotionIntent(
        kind=_string(data, "kind", path),
        object_track_id=_optional_string(data, "object_track_id", path),
        target_track_id=_optional_string(data, "target_track_id", path),
        approach_vector_xyz=_optional_number_tuple(
            data.get("approach_vector_xyz"), f"{path}.approach_vector_xyz", 3
        ),
        standoff_m=_optional_number(data.get("standoff_m"), f"{path}.standoff_m"),
        target_pose_wxyz_xyz=_optional_number_tuple(
            data.get("target_pose_wxyz_xyz"), f"{path}.target_pose_wxyz_xyz", 7
        ),
    )


def _skill_call_to_dict(call: SkillCall) -> dict[str, Any]:
    return {
        "skill": {"skill_id": call.skill.skill_id, "version": call.skill.version},
        "args": {key: _skill_arg_to_dict(value) for key, value in call.args.items()},
    }


def _skill_call_from_dict(raw: Any, path: str) -> SkillCall:
    data = _object(raw, {"skill", "args"}, path)
    skill_data = _object(data.get("skill"), {"skill_id", "version"}, f"{path}.skill")
    args = data.get("args")
    if not isinstance(args, dict):
        raise GraphSchemaError(f"{path}.args must be an object")
    # Strict provider schemas encode the union of registered skill parameters
    # and use null for keys that do not apply to this particular skill. Keep
    # the runtime contract compact and pass only actual arguments to CAP-X.
    args = {
        key: _skill_arg_from_dict(value)
        for key, value in args.items()
        if value is not None
    }
    return SkillCall(
        SkillRef(
            _string(skill_data, "skill_id", f"{path}.skill"),
            _string(skill_data, "version", f"{path}.skill"),
        ),
        dict(args),
    )


def _skill_arg_to_dict(value: object) -> object:
    if isinstance(value, SkillOutputRef):
        return {
            "$skill_output": value.call_index,
            "path": list(value.path),
        }
    if isinstance(value, dict):
        return {key: _skill_arg_to_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_skill_arg_to_dict(item) for item in value]
    if isinstance(value, tuple):
        return [_skill_arg_to_dict(item) for item in value]
    return value


def _skill_arg_from_dict(value: object) -> object:
    if isinstance(value, Mapping):
        reference_key: str | None = None
        if set(value) == {"$skill_output", "path"}:
            reference_key = "$skill_output"
        elif set(value) == {"call_index", "path"}:
            # Compatibility for rehearsal artifacts emitted before the
            # canonical graph serializer was used at this boundary.
            reference_key = "call_index"
        if reference_key is not None:
            call_index = value[reference_key]
            path = value["path"]
            if (
                isinstance(call_index, int)
                and not isinstance(call_index, bool)
                and isinstance(path, list)
                and all(isinstance(item, (str, int)) and not isinstance(item, bool) for item in path)
            ):
                return SkillOutputRef(call_index, tuple(path))
        return {key: _skill_arg_from_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_skill_arg_from_dict(item) for item in value]
    return value


def _port_to_dict(port: PortSpec) -> dict[str, Any]:
    return {"name": port.name, "type_name": port.type_name, "required": port.required}


def _port_from_dict(raw: Any, path: str) -> PortSpec:
    data = _object(raw, {"name", "type_name", "required"}, path)
    required = data.get("required")
    if not isinstance(required, bool):
        raise GraphSchemaError(f"{path}.required must be a boolean")
    return PortSpec(_string(data, "name", path), _string(data, "type_name", path), required)


def _resource_to_dict(resource: ResourceRequirement) -> dict[str, Any]:
    return {"resource_id": resource.resource_id, "mode": resource.mode}


def _resource_from_dict(raw: Any, path: str) -> ResourceRequirement:
    data = _object(raw, {"resource_id", "mode"}, path)
    return ResourceRequirement(_string(data, "resource_id", path), _string(data, "mode", path))


def _checkpoint_to_dict(checkpoint: CheckpointSpec) -> dict[str, Any]:
    return {
        "name": checkpoint.name,
        "predicates": list(checkpoint.predicates),
        "validate": checkpoint.validate,
        "weight": checkpoint.weight,
    }


def _checkpoint_from_dict(raw: Any, path: str) -> CheckpointSpec:
    data = _object(raw, {"name", "predicates", "validate", "weight"}, path)
    validate = data.get("validate")
    weight = data.get("weight")
    if not isinstance(validate, bool):
        raise GraphSchemaError(f"{path}.validate must be a boolean")
    if not isinstance(weight, (int, float)) or isinstance(weight, bool):
        raise GraphSchemaError(f"{path}.weight must be numeric")
    return CheckpointSpec(
        _string(data, "name", path),
        _string_tuple(data, "predicates", path),
        validate,
        float(weight),
    )


def _edge_to_dict(edge: GraphEdge) -> dict[str, Any]:
    return {"source": edge.source, "target": edge.target, "condition": edge.condition}


def _edge_from_dict(raw: Any, path: str) -> GraphEdge:
    data = _object(raw, {"source", "target", "condition"}, path)
    return GraphEdge(
        _string(data, "source", path),
        _string(data, "target", path),
        _optional_string(data, "condition", path),
    )


def _binding_to_dict(binding: PortBinding) -> dict[str, Any]:
    return {
        "source_node": binding.source_node,
        "source_port": binding.source_port,
        "target_node": binding.target_node,
        "target_port": binding.target_port,
    }


def _binding_from_dict(raw: Any, path: str) -> PortBinding:
    data = _object(raw, {"source_node", "source_port", "target_node", "target_port"}, path)
    return PortBinding(
        _string(data, "source_node", path),
        _string(data, "source_port", path),
        _string(data, "target_node", path),
        _string(data, "target_port", path),
    )


def _output_binding_to_dict(binding: SubgraphOutputBinding) -> dict[str, Any]:
    return {
        "source_node": binding.source_node,
        "source_port": binding.source_port,
        "output_port": binding.output_port,
    }


def _output_binding_from_dict(raw: Any, path: str) -> SubgraphOutputBinding:
    data = _object(raw, {"source_node", "source_port", "output_port"}, path)
    return SubgraphOutputBinding(
        _string(data, "source_node", path),
        _string(data, "source_port", path),
        _string(data, "output_port", path),
    )


def _mission_edge_to_dict(edge: MissionEdge) -> dict[str, Any]:
    return {"source": edge.source, "target": edge.target, "condition": edge.condition}


def _mission_edge_from_dict(raw: Any, path: str) -> MissionEdge:
    data = _object(raw, {"source", "target", "condition"}, path)
    return MissionEdge(
        _string(data, "source", path),
        _string(data, "target", path),
        _optional_string(data, "condition", path),
    )


def _mission_binding_to_dict(binding: MissionBinding) -> dict[str, Any]:
    return {
        "source_subgraph": binding.source_subgraph,
        "source_port": binding.source_port,
        "target_subgraph": binding.target_subgraph,
        "target_port": binding.target_port,
    }


def _mission_binding_from_dict(raw: Any, path: str) -> MissionBinding:
    data = _object(raw, {"source_subgraph", "source_port", "target_subgraph", "target_port"}, path)
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
    data = _object(raw, {"entry_node", "max_visits", "max_duration_ms", "exit_conditions"}, path)
    return LoopSpec(
        entry_node=_string(data, "entry_node", path),
        max_visits=_integer(data, "max_visits", path),
        max_duration_ms=_integer(data, "max_duration_ms", path),
        exit_conditions=_string_tuple(data, "exit_conditions", path),
    )


_ROOT_KEYS = frozenset({
    "schema_version", "mission_id", "task", "graph_version", "parent_scene_version",
    "entry_subgraph", "success_subgraphs", "failure_subgraphs", "subgraphs", "edges",
    "bindings", "loops",
})
_SUBGRAPH_KEYS = frozenset({
    "subgraph_id", "subgoal_id", "description", "nodes", "edges", "entry_node",
    "success_nodes", "failure_nodes", "inputs", "outputs", "bindings", "output_bindings",
    "checkpoints", "assigned_agent", "loops",
})
_NODE_KEYS = frozenset({
    "node_id", "description", "skill_calls", "inputs", "outputs", "preconditions",
    "postconditions", "resources", "max_duration_ms", "max_sim_steps", "proposed_by",
    "recovery_policy", "node_type", "motion_intent",
})


def _object(value: Any, allowed: frozenset[str] | set[str], path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GraphSchemaError(f"{path} must be an object")
    extra = set(value) - set(allowed)
    if extra:
        raise GraphSchemaError(f"{path} has unknown fields: {sorted(extra)}")
    return value


def _require_version(data: dict[str, Any], path: str) -> None:
    if data.get("schema_version") != GRAPH_SCHEMA_VERSION:
        raise GraphSchemaError(
            f"{path}.schema_version must be {GRAPH_SCHEMA_VERSION}, "
            f"got {data.get('schema_version')!r}"
        )


def _list(data: dict[str, Any], key: str, path: str) -> list[Any]:
    value = data.get(key)
    if not isinstance(value, list):
        raise GraphSchemaError(f"{path}.{key} must be a list")
    return value


def _string(data: dict[str, Any], key: str, path: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise GraphSchemaError(f"{path}.{key} must be a string")
    return value


def _optional_string(data: dict[str, Any], key: str, path: str) -> str | None:
    value = data.get(key)
    if value is not None and not isinstance(value, str):
        raise GraphSchemaError(f"{path}.{key} must be a string or null")
    return value


def _integer(data: dict[str, Any], key: str, path: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise GraphSchemaError(f"{path}.{key} must be an integer")
    return value


def _optional_integer(data: dict[str, Any], key: str, path: str) -> int | None:
    value = data.get(key)
    if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
        raise GraphSchemaError(f"{path}.{key} must be an integer or null")
    return value


def _string_tuple(data: dict[str, Any], key: str, path: str) -> tuple[str, ...]:
    values = _list(data, key, path)
    if any(not isinstance(value, str) for value in values):
        raise GraphSchemaError(f"{path}.{key} must contain only strings")
    return tuple(values)


def _optional_number(value: Any, path: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GraphSchemaError(f"{path} must be a number or null")
    return float(value)


def _optional_number_tuple(value: Any, path: str, size: int) -> tuple[float, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise GraphSchemaError(f"{path} must be a list or null")
    if len(value) != size:
        raise GraphSchemaError(f"{path} must contain {size} numbers")
    numbers = tuple(_optional_number(item, path) for item in value)
    if any(item is None for item in numbers):
        raise GraphSchemaError(f"{path} must contain only numbers")
    return tuple(item for item in numbers if item is not None)
