"""Stable prompt builders for the Manager and local Policy Agents."""

from __future__ import annotations

import json
from copy import deepcopy
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any
from uuid import uuid4

from capmas.contracts.agent import AgentArtifact, AgentContext
from capmas.contracts.core import ArtifactRef
from capmas.contracts.scene import SceneSnapshot
from capmas.contracts.staged import STAGED_TOPOLOGY_SCHEMA_VERSION
from capmas.contracts.strategy import StrategyProfile
from capmas.llm.protocol import LLMRequest


def mission_graph_response_schema(
    *,
    skill_arg_names: Sequence[str] = (),
    skill_arg_schemas: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return the strict JSON schema accepted by ``MissionGraphDecoder``."""

    def obj(
        properties: dict[str, Any],
        required: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": properties,
            "required": list(required or properties),
            "additionalProperties": False,
        }

    string_list = {"type": "array", "items": {"type": "string"}}
    port = obj(
        {
            "name": {"type": "string"},
            "type_name": {"type": "string"},
            "required": {"type": "boolean"},
        }
    )
    resource = obj(
        {
            "resource_id": {"type": "string"},
            "mode": {"type": "string", "enum": ["exclusive", "shared"]},
        }
    )
    loop = obj(
        {
            "entry_node": {"type": "string"},
            "max_visits": {"type": "integer", "minimum": 1},
            "max_duration_ms": {"type": "integer", "minimum": 0},
            "exit_conditions": string_list,
        }
    )
    edge = obj(
        {
            "source": {"type": "string"},
            "target": {"type": "string"},
            "condition": {"type": ["string", "null"]},
        }
    )
    # Strict providers reject open-ended objects. Keep the runtime argument
    # contract as a dict, but represent its registered keys explicitly on the
    # wire and use null for parameters unused by a particular skill call.
    argument_names = tuple(sorted({str(name) for name in skill_arg_names if name}))
    argument_schemas = skill_arg_schemas or {}
    skill_args = obj(
        {
            name: _nullable_argument_schema(name, argument_schemas.get(name))
            for name in argument_names
        },
        required=argument_names,
    )
    skill_call = obj(
        {
            "skill": obj(
                {"skill_id": {"type": "string"}, "version": {"type": "string"}}
            ),
            "args": skill_args,
        }
    )
    node = obj(
        {
            "node_id": {"type": "string"},
            "description": {"type": "string"},
            "skill_calls": {"type": "array", "items": skill_call},
            "inputs": {"type": "array", "items": port},
            "outputs": {"type": "array", "items": port},
            "preconditions": string_list,
            "postconditions": string_list,
            "resources": {"type": "array", "items": resource},
            "max_duration_ms": {"type": "integer", "minimum": 1},
            "max_sim_steps": {"type": "integer", "minimum": 1},
            "proposed_by": {"type": "string"},
            "recovery_policy": {"type": "string"},
            "node_type": {"type": "string", "enum": ["action", "router", "checkpoint"]},
        }
    )
    binding = obj(
        {
            "source_node": {"type": "string"},
            "source_port": {"type": "string"},
            "target_node": {"type": "string"},
            "target_port": {"type": "string"},
        }
    )
    output_binding = obj(
        {
            "source_node": {"type": "string"},
            "source_port": {"type": "string"},
            "output_port": {"type": "string"},
        }
    )
    checkpoint = obj(
        {
            "name": {"type": "string"},
            "predicates": string_list,
            "validate": {"type": "boolean"},
            "weight": {"type": "number", "exclusiveMinimum": 0},
        }
    )
    subgraph = obj(
        {
            "subgraph_id": {"type": "string"},
            "subgoal_id": {"type": "string"},
            "description": {"type": "string"},
            "nodes": {"type": "array", "items": node},
            "edges": {"type": "array", "items": edge},
            "entry_node": {"type": "string"},
            "success_nodes": string_list,
            "failure_nodes": string_list,
            "inputs": {"type": "array", "items": port},
            "outputs": {"type": "array", "items": port},
            "bindings": {"type": "array", "items": binding},
            "output_bindings": {"type": "array", "items": output_binding},
            "checkpoints": {"type": "array", "items": checkpoint},
            "assigned_agent": {"type": "string"},
            "loops": {"type": "array", "items": loop},
        }
    )
    mission_edge = obj(
        {
            "source": {"type": "string"},
            "target": {"type": "string"},
            "condition": {"type": ["string", "null"]},
        }
    )
    mission_binding = obj(
        {
            "source_subgraph": {"type": "string"},
            "source_port": {"type": "string"},
            "target_subgraph": {"type": "string"},
            "target_port": {"type": "string"},
        }
    )
    return obj(
        {
            "schema_version": {"type": "integer", "const": 1},
            "mission_id": {"type": "string"},
            "task": {"type": "string"},
            "graph_version": {"type": "integer", "minimum": 1},
            "parent_scene_version": {"type": ["integer", "null"]},
            "entry_subgraph": {"type": "string"},
            "success_subgraphs": string_list,
            "failure_subgraphs": string_list,
            "subgraphs": {"type": "array", "items": subgraph},
            "edges": {"type": "array", "items": mission_edge},
            "bindings": {"type": "array", "items": mission_binding},
            "loops": {"type": "array", "items": loop},
        }
    )


def _nullable_argument_schema(
    name: str,
    declared: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build a provider-safe nullable schema for one skill parameter."""
    if declared is None:
        declared = {
            "object_name": {"type": "string"},
            "target": {"type": "string"},
            "emit": {"type": "string"},
            "use_multiview": {"type": "boolean"},
            "position": {"type": "array", "items": {"type": "number"}},
            "quaternion_wxyz": {"type": "array", "items": {"type": "number"}},
            "z_approach": {"type": "number"},
            "arm": {"type": "integer"},
        }.get(name, {"type": "string"})
    schema = deepcopy(dict(declared))
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        schema["type"] = [schema_type, "null"]
    elif isinstance(schema_type, list) and "null" not in schema_type:
        schema["type"] = [*schema_type, "null"]
    return schema


def subgraph_response_schema(
    *,
    skill_arg_names: Sequence[str] = (),
    skill_arg_schemas: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return the smaller versioned envelope for one local Policy graph."""
    local_schema = deepcopy(
        mission_graph_response_schema(
            skill_arg_names=skill_arg_names,
            skill_arg_schemas=skill_arg_schemas,
        )["properties"]["subgraphs"]["items"]
    )
    return {
        "type": "object",
        "properties": {
            "schema_version": {"type": "integer", "const": 1},
            "subgraph": local_schema,
        },
        "required": ["schema_version", "subgraph"],
        "additionalProperties": False,
    }


def mission_topology_response_schema() -> dict[str, Any]:
    """Return the compact Manager schema used by the staged protocol."""

    def obj(properties: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }

    strings = {"type": "array", "items": {"type": "string"}}
    terminals = {"type": "array", "items": {"type": "string"}, "minItems": 1}
    subgoal = obj(
        {
            "subgraph_id": {"type": "string"},
            "subgoal_id": {"type": "string"},
            "description": {"type": "string"},
            "depends_on": strings,
            "success_predicates": strings,
            "failure_predicates": strings,
            "required_agent_role": {"type": "string"},
            "execution_kind": {
                "type": "string",
                "enum": ["physical_action", "checkpoint_only"],
            },
        }
    )
    edge = obj(
        {
            "source": {"type": "string"},
            "target": {"type": "string"},
            "condition": {"type": ["string", "null"]},
        }
    )
    binding = obj(
        {
            "source_subgraph": {"type": "string"},
            "source_port": {"type": "string"},
            "target_subgraph": {"type": "string"},
            "target_port": {"type": "string"},
        }
    )
    loop = obj(
        {
            "entry_node": {"type": "string"},
            "max_visits": {"type": "integer", "minimum": 1},
            "max_duration_ms": {"type": "integer", "minimum": 0},
            "exit_conditions": strings,
        }
    )
    return obj(
        {
            "schema_version": {
                "type": "integer",
                "const": STAGED_TOPOLOGY_SCHEMA_VERSION,
            },
            "mission_id": {"type": "string"},
            "task": {"type": "string"},
            "graph_version": {"type": "integer", "minimum": 1},
            "parent_scene_version": {"type": ["integer", "null"]},
            "entry_subgraph": {"type": "string"},
            "success_subgraphs": terminals,
            "failure_subgraphs": terminals,
            "subgoals": {"type": "array", "items": subgoal},
            "edges": {"type": "array", "items": edge},
            "bindings": {"type": "array", "items": binding},
            "loops": {"type": "array", "items": loop},
        }
    )


def build_manager_request(
    task: str,
    scene: SceneSnapshot,
    *,
    skill_metadata: Sequence[str] = (),
    skill_arg_names: Sequence[str] = (),
    skill_arg_schemas: Mapping[str, Mapping[str, Any]] | None = None,
    request_id: str | None = None,
    agent_name: str = "mission_manager",
    include_schema_in_prompt: bool = False,
    deadline_ms: int = 30_000,
    max_output_tokens: int = 4096,
) -> LLMRequest:
    payload = {
        "task": task,
        "scene": _scene_payload(scene),
        "available_skills": list(skill_metadata),
        "skill_argument_names": list(skill_arg_names),
        "constraints": _graph_constraints(),
    }
    if include_schema_in_prompt:
        payload["output_schema"] = mission_graph_response_schema(
            skill_arg_names=skill_arg_names,
            skill_arg_schemas=skill_arg_schemas,
        )
    return LLMRequest(
        request_id=request_id or str(uuid4()),
        agent_name=agent_name,
        messages=(
            {"role": "system", "content": _manager_system_prompt()},
            {"role": "user", "content": _json_text(payload)},
        ),
        response_schema=mission_graph_response_schema(
            skill_arg_names=skill_arg_names,
            skill_arg_schemas=skill_arg_schemas,
        ),
        deadline_ms=deadline_ms,
        max_output_tokens=max_output_tokens,
    )


def build_topology_request(
    task: str,
    scene: SceneSnapshot,
    *,
    request_id: str | None = None,
    agent_name: str = "mission_manager",
    include_schema_in_prompt: bool = False,
    deadline_ms: int = 30_000,
    max_output_tokens: int = 1536,
    repair_feedback: str | None = None,
) -> LLMRequest:
    """Build the compact Manager request for stage one."""
    payload: dict[str, Any] = {
        "task": task,
        "scene": _scene_payload(scene),
        "constraints": _topology_constraints(),
    }
    if include_schema_in_prompt:
        payload["output_schema"] = mission_topology_response_schema()
    if repair_feedback:
        payload["previous_rejection"] = repair_feedback[:2000]
    return LLMRequest(
        request_id=request_id or str(uuid4()),
        agent_name=agent_name,
        messages=(
            {"role": "system", "content": _topology_system_prompt()},
            {"role": "user", "content": _json_text(payload)},
        ),
        response_schema=mission_topology_response_schema(),
        deadline_ms=deadline_ms,
        max_output_tokens=max_output_tokens,
    )


def build_policy_request(
    subgoal: AgentArtifact,
    scene: SceneSnapshot,
    context: AgentContext,
    *,
    skill_metadata: Sequence[str] = (),
    skill_arg_names: Sequence[str] = (),
    skill_arg_schemas: Mapping[str, Mapping[str, Any]] | None = None,
    request_id: str | None = None,
    agent_name: str = "local_policy_agent",
    policy_strategy: str = "balanced",
    include_schema_in_prompt: bool = False,
    deadline_ms: int = 30_000,
    max_output_tokens: int = 4096,
) -> LLMRequest:
    payload = {
        "subgoal": dict(subgoal.payload),
        "scene": _scene_payload(scene),
        "history": {
            "current_subgoal": context.history.current_subgoal,
            "recovery_count": context.history.recovery_count,
            "last_verification": (
                context.history.last_verification.decision
                if context.history.last_verification is not None
                else None
            ),
        },
        "available_skills": list(skill_metadata),
        "skill_argument_names": list(skill_arg_names),
        "policy_strategy": policy_strategy,
        "strategy_profile": _strategy_profile_payload(policy_strategy),
        "constraints": _graph_constraints(),
    }
    if include_schema_in_prompt:
        payload["output_schema"] = mission_graph_response_schema(
            skill_arg_names=skill_arg_names,
            skill_arg_schemas=skill_arg_schemas,
        )
    return LLMRequest(
        request_id=request_id or str(uuid4()),
        agent_name=agent_name,
        messages=(
            {"role": "system", "content": _policy_system_prompt(policy_strategy)},
            {"role": "user", "content": _json_text(payload)},
        ),
        response_schema=mission_graph_response_schema(
            skill_arg_names=skill_arg_names,
            skill_arg_schemas=skill_arg_schemas,
        ),
        deadline_ms=deadline_ms,
        max_output_tokens=max_output_tokens,
    )


def build_staged_policy_request(
    subgoal: AgentArtifact,
    scene: SceneSnapshot,
    context: AgentContext,
    *,
    skill_metadata: Sequence[str] = (),
    skill_arg_names: Sequence[str] = (),
    skill_arg_schemas: Mapping[str, Mapping[str, Any]] | None = None,
    request_id: str | None = None,
    agent_name: str = "local_policy_agent",
    policy_strategy: str = "balanced",
    include_schema_in_prompt: bool = False,
    deadline_ms: int = 30_000,
    max_output_tokens: int = 1536,
    repair_feedback: str | None = None,
) -> LLMRequest:
    """Build the compact local graph request for stage two."""
    payload: dict[str, Any] = {
        "subgoal": dict(subgoal.payload),
        "scene": _scene_payload(scene),
        "history": {
            "current_subgoal": context.history.current_subgoal,
            "recovery_count": context.history.recovery_count,
            "last_verification": (
                context.history.last_verification.decision
                if context.history.last_verification is not None
                else None
            ),
        },
        "available_skills": list(skill_metadata),
        "skill_argument_names": list(skill_arg_names),
        "policy_strategy": policy_strategy,
        "strategy_profile": _strategy_profile_payload(policy_strategy),
        "constraints": _local_graph_constraints(),
    }
    if include_schema_in_prompt:
        payload["output_schema"] = subgraph_response_schema(
            skill_arg_names=skill_arg_names,
            skill_arg_schemas=skill_arg_schemas,
        )
    if repair_feedback:
        payload["previous_rejection"] = repair_feedback[:2000]
    return LLMRequest(
        request_id=request_id or str(uuid4()),
        agent_name=agent_name,
        messages=(
            {"role": "system", "content": _staged_policy_system_prompt(policy_strategy)},
            {"role": "user", "content": _json_text(payload)},
        ),
        response_schema=subgraph_response_schema(
            skill_arg_names=skill_arg_names,
            skill_arg_schemas=skill_arg_schemas,
        ),
        deadline_ms=deadline_ms,
        max_output_tokens=max_output_tokens,
    )


def _scene_payload(scene: SceneSnapshot) -> dict[str, Any]:
    return {
        "episode_id": scene.episode_id,
        "episode_epoch": scene.episode_epoch,
        "scene_version": scene.scene_version,
        "sensor_timestamp_ns": scene.sensor_timestamp_ns,
        "freshness_ms": scene.freshness_ms,
        "robot": _prompt_value(scene.robot),
        "objects": [
            {
                "track_id": obj.track_id,
                "label": obj.label,
                "pose_wxyz_xyz": list(obj.pose_wxyz_xyz),
                "confidence": obj.confidence,
                "last_seen_ns": obj.last_seen_ns,
            }
            for obj in scene.objects
        ],
        "spatial_relations": [
            {
                "subject_track_id": relation.subject_track_id,
                "object_track_id": relation.object_track_id,
                "relation": relation.relation,
                "confidence": relation.confidence,
            }
            for relation in scene.spatial_relations
        ],
        "uncertainty": {
            "scene_confidence": scene.uncertainty.scene_confidence,
            "ambiguous_track_ids": list(scene.uncertainty.ambiguous_track_ids),
            "stale_track_ids": list(scene.uncertainty.stale_track_ids),
        },
    }


def _json_text(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _strategy_profile_payload(policy_strategy: str) -> dict[str, object]:
    """Expose typed strategy constraints to the Policy without free-form state."""
    return asdict(StrategyProfile.for_name(policy_strategy))


def _prompt_value(value: object) -> object:
    """Convert typed observation values to JSON without exposing raw handles."""
    if isinstance(value, ArtifactRef):
        return {
            "uri": value.uri,
            "media_type": value.media_type,
            "sha256": value.sha256,
            "byte_size": value.byte_size,
        }
    if isinstance(value, Mapping):
        return {str(key): _prompt_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_prompt_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _graph_constraints() -> dict[str, object]:
    return {
        "schema_version": 1,
        "one_physical_executor": True,
        "actions_must_use_registered_typed_skills": True,
        "checkpoint_only_subgoals_must_not_use_policy_skills": True,
        "checkpoint_only_subgoals_use_checkpoint_node_with_empty_skill_calls": True,
        "observation_only_skills_are_not_executable": True,
        "do_not_use_get_observation_or_get_object_pose_in_skill_calls": True,
        "skill_args_must_match_registered_function_signature": True,
        "strict_args_wire_format": (
            "args contains every registered parameter name; use null for an "
            "unused parameter"
        ),
        "never_put_episode_or_scene_contract_fields_in_skill_args": True,
        "every_executable_subgraph_needs_validating_checkpoint": True,
        "cycles_require_finite_loop_spec": True,
        "parent_scene_version_must_match_current_scene": True,
        "return_json_only": True,
    }


def _topology_constraints() -> dict[str, object]:
    return {
        "schema_version": STAGED_TOPOLOGY_SCHEMA_VERSION,
        "topology_only": True,
        "do_not_emit_skill_calls_or_local_nodes": True,
        "subgraph_id_is_the_only_control_flow_identifier": True,
        "all_edges_dependencies_and_terminals_must_reference_declared_subgraph_ids": True,
        "depends_on_must_contain_subgraph_ids_not_subgoal_ids": True,
        "for_every_dependency_emit_matching_success_edge": True,
        "do_not_invent_undeclared_failure_subgraphs": True,
        "success_and_failure_terminals_must_be_nonempty": True,
        "execution_kind_must_be_explicit": True,
        "dependencies_must_have_matching_edges": True,
        "failure_recovery_dependencies_must_be_committed_before_failed_source": True,
        "one_success_and_one_failure_edge_max_per_source": True,
        "rolling_success_edges_must_not_reenter_committed_subgraphs": True,
        "cycles_require_finite_loop_spec": True,
        "parent_scene_version_must_match_current_scene": True,
        "predicate_grammar": [
            "object_in_gripper(obj_id)",
            "object_at_target(obj_id,target_id)",
            "gripper_open()",
            "gripper_closed()",
            "scene_fresh(threshold_ms)",
            "track_exists:track_id",
            "object_visible:label",
        ],
        "no_natural_language_predicates": True,
        "return_json_only": True,
    }


def _local_graph_constraints() -> dict[str, object]:
    return {
        "schema_version": 1,
        "one_subgraph_only": True,
        "exact_top_level_keys": ["schema_version", "subgraph"],
        "never_use_subgraph_artifact_as_top_level_key": True,
        "preserve_requested_subgraph_and_subgoal_ids": True,
        "actions_must_use_registered_typed_skills": True,
        "checkpoint_nodes_may_have_empty_skill_calls": True,
        "checkpoint_only_subgoals_must_not_invent_physical_actions": True,
        "strict_args_wire_format": (
            "args contains every registered parameter name; use null for an "
            "unused parameter"
        ),
        "prefer_one_minimal_action_node": True,
        "omit_ports_and_bindings_when_skill_args_are_literal": True,
        "every_local_binding_must_reference_existing_node_and_port": True,
        "required_node_inputs_must_have_a_local_or_external_binding": True,
        "copy_all_topology_success_predicates_verbatim": True,
        "preconditions_and_postconditions_must_use_supported_predicate_grammar": True,
        "omit_preconditions_when_no_supported_predicate_applies": True,
        "failure_terminal_checkpoint_fallback": "scene_fresh(1000)",
        "every_executable_subgraph_needs_validating_checkpoint": True,
        "success_nodes_and_failure_nodes_must_both_be_nonempty": True,
        "single_action_may_be_used_for_both_success_and_failure": True,
        "cycles_require_finite_loop_spec": True,
        "return_versioned_subgraph_envelope_only": True,
    }


def _manager_system_prompt() -> str:
    return (
        "You are the CAP-MAS Mission Manager. Build a bounded MissionGraph for the "
        "robot task. You own global topology and subgoal dependencies. Do not write "
        "Python, call tools, access env handles, or use privileged completion signals. "
        "Return exactly one JSON object matching the supplied schema. Every action must "
        "use an available typed skill and every executable subgraph must contain an "
        "observable validating checkpoint."
    )


def _policy_system_prompt(policy_strategy: str = "balanced") -> str:
    return (
        "You are a CAP-MAS local Policy Agent. Propose the bounded SubgraphSpec for "
        "the requested subgoal inside a MissionGraph JSON wrapper. Preserve the "
        "requested subgraph_id and subgoal_id. Use only registered typed skills, keep "
        "physical resources explicit, and express postconditions as observable "
        "predicates. Do not access env handles, simulator state, arbitrary imports, "
        "or privileged evaluator signals. "
        + _policy_strategy_guidance(policy_strategy)
        + " Return JSON only."
    )


def _topology_system_prompt() -> str:
    return (
        "You are the CAP-MAS Mission Manager in topology stage. Emit only a compact "
        "MissionTopology: mission identity, bounded subgoals, dependencies, terminal "
        "subgraphs, and finite loops. Do not emit skill calls, executable nodes, Python, "
        "tool calls, env handles, or privileged completion signals. Use a unique "
        "subgraph_id for every subgoal. subgraph_id is the only control-flow ID: "
        "entry_subgraph, success_subgraphs, failure_subgraphs, edge endpoints, and "
        "depends_on must reference exactly those declared subgraph_id strings; never "
        "use subgoal_id for control flow and never invent an undeclared failure node. "
        "Every subgoal must set execution_kind to physical_action when it needs a "
        "registered robot skill, or checkpoint_only when it only evaluates observable "
        "scene predicates. A checkpoint_only subgoal must not require a Policy skill "
        "proposal and must declare at least one success_predicate. Do not create a "
        "standalone empty-predicate failure_terminal subgoal; represent failure with "
        "the declared failure edge or a local failure checkpoint. Every declared "
        "subgraph must be reachable from entry_subgraph through an edge. "
        "Both success_subgraphs and failure_subgraphs must be non-empty and contain "
        "only declared subgraph_id strings. Every dependency must have one matching edge. "
        "Construct edges mechanically: for every item in every subgoal's depends_on, "
        "emit exactly one edge {source: dependency, target: subgraph_id, condition: success}; "
        "never return a depends_on entry without that exact edge. "
        "A failure edge target is a recovery branch: its depends_on may contain only "
        "subgraphs on the source's normal success ancestry that could already be committed; "
        "never make a failure target depend on the failed source or on an uncommitted branch. "
        "For rolling execution, normal success edges must follow the forward committed order "
        "and must not re-enter a previously committed subgraph; do not emit success back-edges "
        "or success cycles. "
        "All terminal predicates must use only these exact observable forms: "
        "object_in_gripper(obj_id), object_at_target(obj_id,target_id), gripper_open(), "
        "gripper_closed(), scene_fresh(threshold_ms), track_exists:track_id, or "
        "object_visible:label; never write prose predicates. "
        "If a previous rejection is supplied, correct it before returning. Return exactly one JSON object "
        "matching the schema."
    )


def _staged_policy_system_prompt(policy_strategy: str = "balanced") -> str:
    return (
        "You are a CAP-MAS local Policy Agent in local-graph stage. Emit exactly one "
        "versioned subgraph envelope for the requested subgraph. The top-level JSON "
        "shape is exactly {\"schema_version\":1,\"subgraph\":{...}}; never use "
        "subgraph_artifact, mission_graph, or any other top-level wrapper key. Preserve the requested "
        "subgraph_id and subgoal_id. Use only registered typed skills, explicit physical "
        "resources, bounded nodes, and observable postconditions. Prefer one minimal "
        "Do not use observation-only skills such as get_observation or get_object_pose "
        "inside skill_calls; perception is already represented by SceneSnapshot. Skill "
        "args must contain only the registered API parameters, never parent_scene_version, "
        "episode_id, or other runtime contract fields. For LIBERO, goto_pose requires "
        "position and quaternion_wxyz, sample_grasp_pose requires object_name, and "
        "open_gripper/close_gripper take no required args. Prefer one minimal "
        "The strict wire schema requires every registered argument key in args; emit null "
        "for keys unused by the selected skill. CAP-MAS removes those null placeholders "
        "before execution. "
        "For any subgoal whose success predicate includes object_in_gripper, the action "
        "sequence is mandatory: sample_grasp_pose, then goto_pose, then close_gripper; "
        "never replace this with a direct goto_pose followed by close_gripper. "
        "If the requested failure terminal has no success_predicates, use "
        "scene_fresh(1000) as its validating checkpoint predicate; never emit an empty "
        "checkpoint predicates list. "
        "If the requested subgoal is checkpoint_only or requires only scene predicates, "
        "emit a checkpoint node with node_type=checkpoint and skill_calls=[]; do not "
        "invent a physical action or an unregistered perception skill. "
        "action node and omit inputs, outputs, bindings, and output_bindings when skill "
        "arguments are literal. If ports are needed, every binding must reference an "
        "existing node and port, and every required node input must be locally or "
        "externally bound. Both success_nodes and "
        "failure_nodes must be non-empty node-id lists; for a single action node, use "
        "that same node id in both lists. Copy every string in the requested subgoal's "
        "success_predicates verbatim into the action node postconditions or a validating "
        "checkpoint; do not paraphrase or omit them. Do not emit a mission "
        "Use only supported observable predicate forms for preconditions and postconditions; "
        "if no supported precondition applies, use an empty list instead of prose. Do not emit a mission "
        "graph wrapper, topology, Python, env handles, arbitrary imports, or privileged "
        "evaluator signals. If a previous rejection is supplied, correct that exact issue. "
        + _policy_strategy_guidance(policy_strategy)
        + " Return JSON only."
    )


def _policy_strategy_guidance(policy_strategy: str) -> str:
    guidance = {
        "balanced": "Use a balanced tradeoff between feasibility, safety, and action count.",
        "safety": "Prioritize conservative geometry, explicit safety margins, and recoverability over speed.",
        "robust": "Prioritize observable verification and OOD robustness, even if the graph is longer.",
        "efficient": "Prioritize the shortest verified action sequence and low latency without weakening constraints.",
    }
    if policy_strategy not in guidance:
        raise ValueError(
            "policy_strategy must be one of balanced, safety, robust, or efficient"
        )
    return guidance[policy_strategy]


__all__ = [
    "build_manager_request",
    "build_policy_request",
    "build_staged_policy_request",
    "build_topology_request",
    "mission_graph_response_schema",
    "mission_topology_response_schema",
    "subgraph_response_schema",
]
