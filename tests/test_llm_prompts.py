from __future__ import annotations

import json

from capmas.contracts.agent import AgentArtifact, AgentContext
from capmas.contracts.core import ArtifactRef
from capmas.contracts.scene import ObjectTrack, SceneSnapshot
from capmas.llm.prompts import (
    build_manager_request,
    build_policy_request,
    build_staged_policy_request,
    build_topology_request,
    mission_graph_response_schema,
)


def _scene() -> SceneSnapshot:
    return SceneSnapshot(
        "episode",
        1,
        4,
        10,
        11,
        {
            "gripper_opening": 0.02,
            "joint_position": ArtifactRef("artifact://joints", "array/joint-position"),
        },
        objects=(ObjectTrack("bowl-1", "bowl", (1, 2, 3, 0, 0, 0, 1), 0.9, 10),),
        freshness_ms=12.0,
    )


def test_graph_response_schema_matches_versioned_wire_boundary() -> None:
    schema = mission_graph_response_schema()

    assert schema["type"] == "object"
    assert schema["properties"]["schema_version"]["const"] == 1
    assert "subgraphs" in schema["required"]
    assert schema["additionalProperties"] is False


def test_strict_graph_schema_declares_skill_argument_keys() -> None:
    schema = mission_graph_response_schema(
        skill_arg_names=("position", "object_name")
    )
    args_schema = (
        schema["properties"]["subgraphs"]["items"]["properties"]["nodes"]
        ["items"]["properties"]["skill_calls"]["items"]["properties"]["args"]
    )

    assert args_schema["additionalProperties"] is False
    assert args_schema["required"] == ["object_name", "position"]
    assert set(args_schema["properties"]) == {"object_name", "position"}


def test_strict_graph_schema_uses_narrow_nullable_argument_types() -> None:
    schema = mission_graph_response_schema(
        skill_arg_names=("object_name", "position"),
        skill_arg_schemas={
            "object_name": {"type": "string"},
            "position": {"type": "array", "items": {"type": "number"}},
        },
    )
    args_schema = (
        schema["properties"]["subgraphs"]["items"]["properties"]["nodes"]
        ["items"]["properties"]["skill_calls"]["items"]["properties"]["args"]
    )

    assert args_schema["properties"]["object_name"]["type"] == ["string", "null"]
    assert args_schema["properties"]["position"]["type"] == ["array", "null"]
    assert args_schema["properties"]["position"]["items"] == {"type": "number"}


def test_manager_prompt_contains_scene_version_and_skill_allowlist() -> None:
    request = build_manager_request(
        "place the bowl on the plate",
        _scene(),
        skill_metadata=("open_gripper@capx-compat-1", "goto_pose@capx-compat-1"),
        request_id="manager-1",
    )

    assert request.request_id == "manager-1"
    assert request.response_schema == mission_graph_response_schema()
    text = "\n".join(str(message["content"]) for message in request.messages)
    assert "scene_version" in text
    assert "4" in text
    assert "goto_pose@capx-compat-1" in text


def test_topology_prompt_forbids_rolling_success_back_edges() -> None:
    request = build_topology_request(
        "place the bowl on the plate",
        _scene(),
        request_id="topology-rolling-1",
    )

    text = "\n".join(str(message["content"]) for message in request.messages)

    assert "rolling_success_edges_must_not_reenter_committed_subgraphs" in text
    assert "must not re-enter a previously committed subgraph" in text


def test_policy_prompt_contains_target_subgraph_without_raw_runtime_handles() -> None:
    scene = _scene()
    context = AgentContext("task", "episode", 1, scene)
    artifact = AgentArtifact(
        "subgoal-1",
        "subgoal",
        {"subgraph_id": "grasp", "subgoal_id": "grasp", "task": "grasp bowl"},
        "manager",
    )
    request = build_policy_request(
        artifact,
        scene,
        context,
        skill_metadata=("sample_grasp_pose@capx-compat-1",),
        request_id="policy-1",
        policy_strategy="safety",
    )

    text = "\n".join(str(message["content"]) for message in request.messages)
    assert request.request_id == "policy-1"
    assert "grasp" in text
    assert "parent_scene_version" in text
    assert "safety" in text
    payload = json.loads(request.messages[1]["content"])
    assert "env" not in payload
    assert "simulator" not in payload
    assert payload["policy_strategy"] == "safety"
    assert payload["strategy_profile"]["min_scene_confidence"] == 0.7


def test_staged_policy_strategy_changes_system_prompt_and_payload() -> None:
    scene = _scene()
    context = AgentContext("task", "episode", 1, scene)
    artifact = AgentArtifact(
        "subgoal-1",
        "topology_subgoal",
        {"subgraph_id": "grasp", "subgoal_id": "grasp", "task": "grasp bowl"},
        "manager",
    )
    balanced = build_staged_policy_request(
        artifact,
        scene,
        context,
        agent_name="policy-0",
        policy_strategy="balanced",
    )
    safety = build_staged_policy_request(
        artifact,
        scene,
        context,
        agent_name="policy-1:safety",
        policy_strategy="safety",
    )

    assert balanced.messages[0]["content"] != safety.messages[0]["content"]
    assert "balanced" in str(balanced.messages[1]["content"])
    assert "safety" in str(safety.messages[1]["content"])
    balanced_payload = json.loads(balanced.messages[1]["content"])
    safety_payload = json.loads(safety.messages[1]["content"])
    assert balanced_payload["strategy_profile"] != safety_payload["strategy_profile"]
    assert safety_payload["strategy_profile"]["perception_weight"] > balanced_payload[
        "strategy_profile"
    ]["perception_weight"]
    assert balanced.agent_name == "policy-0"
    assert safety.agent_name == "policy-1:safety"
