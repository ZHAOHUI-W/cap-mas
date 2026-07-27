from __future__ import annotations

import json

from capmas.contracts.action import SkillCall
from capmas.contracts.agent import AgentArtifact, AgentContext
from capmas.contracts.core import SkillRef
from capmas.contracts.graph import CheckpointSpec, MissionGraph, SubgraphNodeSpec, SubgraphSpec
from capmas.contracts.scene import SceneSnapshot
from capmas.graph.serialization import (
    local_subgraph_from_dict,
    local_subgraph_to_dict,
    mission_graph_to_dict,
)
from capmas.llm.graph_decoder import MissionGraphDecoder
from capmas.llm.graph_decoder import GraphProposalError
from capmas.llm.protocol import LLMRequest, LLMResponse
from capmas.agents.manager import LLMMissionManager
from capmas.agents.policy import LLMGraphPolicyAgent


def _graph(parent_scene_version: int | None = 4) -> MissionGraph:
    node = SubgraphNodeSpec(
        node_id="step_action",
        description="one safe step",
        skill_calls=(SkillCall(SkillRef("noop", "1.0.0"), {}),),
        postconditions=("step_done",),
        proposed_by="llm-policy",
    )
    subgraph = SubgraphSpec(
        subgraph_id="step",
        subgoal_id="step",
        description="one step",
        nodes=(node,),
        edges=(),
        entry_node="step_action",
        success_nodes=("step_action",),
        failure_nodes=("step_action",),
        checkpoints=(CheckpointSpec("step-check", ("step_done",)),),
    )
    return MissionGraph(
        mission_id="mission",
        task="test",
        subgraphs=(subgraph,),
        edges=(),
        bindings=(),
        entry_subgraph="step",
        success_subgraphs=("step",),
        failure_subgraphs=("step",),
        parent_scene_version=parent_scene_version,
    )


def _scene(version: int = 4) -> SceneSnapshot:
    return SceneSnapshot("episode", 1, version, 1, 1, {})


def test_decoder_accepts_strict_structured_graph_at_current_scene() -> None:
    graph = _graph()
    request = LLMRequest("request-1", "policy", (), response_schema={"type": "object"})
    response = LLMResponse("request-1", "ignored", structured=mission_graph_to_dict(graph))

    result = MissionGraphDecoder().decode(response, _scene(), request=request)

    assert result.accepted is True
    assert result.graph == graph


def test_local_subgraph_decoder_removes_strict_schema_null_argument_placeholders() -> None:
    raw = local_subgraph_to_dict(_graph().subgraphs[0])
    raw["subgraph"]["nodes"][0]["skill_calls"][0]["args"] = {
        "position": [0.1, 0.2, 0.3],
        "object_name": None,
    }

    subgraph = local_subgraph_from_dict(raw)

    assert subgraph.nodes[0].skill_calls[0].args == {
        "position": [0.1, 0.2, 0.3]
    }


def test_decoder_parses_json_content_but_rejects_markdown_or_invalid_json() -> None:
    graph = _graph()
    encoded = json.dumps(mission_graph_to_dict(graph))
    decoder = MissionGraphDecoder()

    accepted = decoder.decode(LLMResponse("r", encoded), _scene())
    rejected = decoder.decode(LLMResponse("r", "```json\n{}\n```"), _scene())

    assert accepted.accepted is True
    assert rejected.accepted is False
    assert rejected.rejections[0].code == "JSON_INVALID"


def test_decoder_rejects_request_mismatch_stale_scene_and_invalid_graph() -> None:
    request = LLMRequest("expected", "policy", ())
    stale = LLMResponse("other", "{}")
    stale_result = MissionGraphDecoder().decode(stale, _scene(), request=request)
    assert {item.code for item in stale_result.rejections} == {"REQUEST_ID_MISMATCH"}

    graph = _graph(parent_scene_version=3)
    response = LLMResponse("expected", "", structured=mission_graph_to_dict(graph))
    result = MissionGraphDecoder().decode(response, _scene(), request=request)
    assert any(item.code == "STALE_SCENE" for item in result.rejections)


def test_decoder_never_returns_a_default_graph_for_empty_or_invalid_response() -> None:
    decoder = MissionGraphDecoder()

    for response in (
        LLMResponse("r", ""),
        LLMResponse("r", "not-json"),
        LLMResponse("r", json.dumps({"schema_version": 1})),
    ):
        result = decoder.decode(response, _scene())
        assert result.accepted is False
        assert result.graph is None


class _FakeLLM:
    def __init__(self, response: LLMResponse) -> None:
        self.response = response
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return self.response


def test_llm_manager_and_policy_return_only_validated_typed_artifacts() -> None:
    graph = _graph()
    response = LLMResponse("request", "", structured=mission_graph_to_dict(graph))
    llm = _FakeLLM(response)
    manager = LLMMissionManager(
        llm,
        lambda task, scene: LLMRequest(
            "request", "manager", ({"task": task, "scene_version": scene.scene_version},)
        ),
    )

    proposed = manager.propose_graph("test", _scene())
    artifact = manager.propose_subgoal("test", _scene())

    assert proposed == graph
    assert artifact.kind == "mission_graph"
    assert len(llm.requests) == 2

    policy = LLMGraphPolicyAgent(
        llm,
        lambda subgoal, scene, context: LLMRequest(
            "request",
            "policy",
            ({"subgraph_id": subgoal.payload["subgraph_id"], "scene": scene.scene_version},),
        ),
    )
    context = AgentContext("task", "episode", 1, _scene())
    subgraph = policy.propose_subgraph(
        AgentArtifact("subgoal", "subgoal", {"subgraph_id": "step"}, "manager"),
        _scene(),
        context,
    )

    assert subgraph.subgraph_id == "step"


def test_llm_policy_propagates_rejection_without_default_action() -> None:
    llm = _FakeLLM(LLMResponse("request", "not-json"))
    policy = LLMGraphPolicyAgent(
        llm,
        lambda _subgoal, _scene, _context: LLMRequest("request", "policy", ()),
    )

    try:
        policy.propose_subgraph(
            AgentArtifact("subgoal", "subgoal", {"subgraph_id": "step"}, "manager"),
            _scene(),
            AgentContext("task", "episode", 1, _scene()),
        )
    except GraphProposalError as exc:
        assert any(item.code == "JSON_INVALID" for item in exc.result.rejections)
    else:  # pragma: no cover - assertion keeps the no-fallback contract explicit
        raise AssertionError("invalid LLM output must be rejected")
