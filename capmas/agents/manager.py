from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from uuid import uuid4

from capmas.contracts.agent import AgentArtifact, MissionManager
from capmas.contracts.graph import MissionGraph
from capmas.contracts.scene import SceneSnapshot
from capmas.contracts.staged import MissionTopology
from capmas.graph.serialization import mission_graph_to_dict
from capmas.llm.graph_decoder import GraphProposalError, MissionGraphDecoder
from capmas.llm.staged_decoder import MissionTopologyDecoder, StagedProposalError
from capmas.llm.protocol import LLMClient, LLMRequest


@dataclass
class SimpleMissionManager(MissionManager):
    agent_name: str = "mission_manager"

    def propose_subgoal(self, task: str, scene: SceneSnapshot) -> AgentArtifact:
        return AgentArtifact(
            artifact_id=str(uuid4()),
            kind="subgoal",
            payload={"task": task, "scene_version": scene.scene_version},
            source_agent=self.agent_name,
        )


class LLMMissionManager:
    """LLM adapter that emits only a schema- and scene-validated MissionGraph."""

    def __init__(
        self,
        llm: LLMClient,
        request_builder: Callable[[str, SceneSnapshot], LLMRequest],
        *,
        decoder: MissionGraphDecoder | None = None,
        agent_name: str = "llm_mission_manager",
    ) -> None:
        self.llm = llm
        self.request_builder = request_builder
        self.decoder = decoder or MissionGraphDecoder()
        self.name = agent_name

    def propose_graph(self, task: str, scene: SceneSnapshot) -> MissionGraph:
        request = self.request_builder(task, scene)
        response = self.llm.complete(request)
        result = self.decoder.decode(response, scene, request=request)
        if not result.accepted:
            raise GraphProposalError(result)
        assert result.graph is not None
        return result.graph

    def propose_subgoal(self, task: str, scene: SceneSnapshot) -> AgentArtifact:
        graph = self.propose_graph(task, scene)
        return AgentArtifact(
            artifact_id=str(uuid4()),
            kind="mission_graph",
            payload={
                "graph": mission_graph_to_dict(graph),
                "scene_version": scene.scene_version,
            },
            source_agent=self.name,
        )


class LLMTopologyManager:
    """Stage-one Manager that emits topology without executable details."""

    def __init__(
        self,
        llm: LLMClient,
        request_builder: Callable[[str, SceneSnapshot], LLMRequest],
        *,
        decoder: MissionTopologyDecoder | None = None,
        agent_name: str = "llm_topology_manager",
        proposal_retries: int = 0,
        repair_request_builder: Callable[[str, SceneSnapshot, str], LLMRequest] | None = None,
    ) -> None:
        if proposal_retries < 0:
            raise ValueError("proposal_retries must not be negative")
        self.llm = llm
        self.request_builder = request_builder
        self.decoder = decoder or MissionTopologyDecoder()
        self.name = agent_name
        self.proposal_retries = proposal_retries
        self.repair_request_builder = repair_request_builder

    def propose_topology(self, task: str, scene: SceneSnapshot) -> MissionTopology:
        last_error: Exception | None = None
        feedback: str | None = None
        for attempt in range(self.proposal_retries + 1):
            if attempt and self.repair_request_builder is not None and feedback:
                request = self.repair_request_builder(task, scene, feedback)
            else:
                request = self.request_builder(task, scene)
            try:
                response = self.llm.complete(request)
                result = self.decoder.decode(response, scene, request=request)
                if not result.accepted:
                    raise StagedProposalError(result)
                assert result.topology is not None
                return result.topology
            except Exception as exc:
                last_error = exc
                feedback = str(exc)
                if attempt >= self.proposal_retries:
                    raise
        assert last_error is not None
        raise last_error
