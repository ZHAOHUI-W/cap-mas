from __future__ import annotations

from typing import Callable

from capmas.contracts.action import ActionContract
from capmas.contracts.agent import (
    AgentContext,
    AgentArtifact,
    GraphPolicyAgent,
    GroundedPolicyAgent,
    PolicyDecision,
    PolicyAgent,
)
from capmas.contracts.graph import SubgraphSpec
from capmas.contracts.scene import SceneSnapshot
from capmas.llm.graph_decoder import (
    GraphDecodeRejection,
    GraphDecodeResult,
    GraphProposalError,
    MissionGraphDecoder,
)
from capmas.llm.protocol import LLMClient, LLMRequest
from capmas.llm.staged_decoder import (
    LocalSubgraphDecoder,
    StagedDecodeRejection,
    StagedProposalError,
    SubgraphDecodeResult,
)


class CallablePolicyAgent(PolicyAgent):
    """Adapter for an LLM or deterministic planner that returns contracts."""

    def __init__(
        self,
        proposer: Callable[[AgentArtifact, SceneSnapshot, AgentContext], ActionContract],
    ) -> None:
        self._proposer = proposer

    def propose_action(
        self,
        subgoal: AgentArtifact,
        scene: SceneSnapshot,
        context: AgentContext,
    ) -> ActionContract:
        return self._proposer(subgoal, scene, context)


class CallableGroundedPolicyAgent(GroundedPolicyAgent):
    """Adapter for a policy that can ask for evidence before proposing action."""

    def __init__(
        self,
        decider: Callable[[AgentArtifact, SceneSnapshot, AgentContext], PolicyDecision],
    ) -> None:
        self._decider = decider

    def decide(
        self,
        subgoal: AgentArtifact,
        scene: SceneSnapshot,
        context: AgentContext,
    ) -> PolicyDecision:
        return self._decider(subgoal, scene, context)


class CallableGraphPolicyAgent(GraphPolicyAgent):
    """Deterministic adapter for graph proposals during the P3 foundation."""

    name = "callable_graph_policy"

    def __init__(
        self,
        proposer: Callable[[AgentArtifact, SceneSnapshot, AgentContext], SubgraphSpec],
    ) -> None:
        self._proposer = proposer

    def propose_subgraph(
        self,
        subgoal: AgentArtifact,
        scene: SceneSnapshot,
        context: AgentContext,
    ) -> SubgraphSpec:
        return self._proposer(subgoal, scene, context)


class LLMGraphPolicyAgent(GraphPolicyAgent):
    """Generate a local subgraph through the strict MissionGraph boundary.

    The provider response uses the versioned MissionGraph wire format so the
    same schema and validator protect Manager and Policy outputs. The target
    subgraph is selected by ``subgraph_id`` or ``subgoal_id`` in the incoming
    subgoal artifact; no arbitrary model-selected subgraph is accepted.
    """

    name = "llm_graph_policy"

    def __init__(
        self,
        llm: LLMClient,
        request_builder: Callable[[AgentArtifact, SceneSnapshot, AgentContext], LLMRequest],
        *,
        decoder: MissionGraphDecoder | None = None,
        agent_name: str = "llm_graph_policy",
    ) -> None:
        self.llm = llm
        self.request_builder = request_builder
        self.decoder = decoder or MissionGraphDecoder()
        self.name = agent_name

    def propose_subgraph(
        self,
        subgoal: AgentArtifact,
        scene: SceneSnapshot,
        context: AgentContext,
    ) -> SubgraphSpec:
        request = self.request_builder(subgoal, scene, context)
        response = self.llm.complete(request)
        result = self.decoder.decode(response, scene, request=request)
        if not result.accepted:
            raise GraphProposalError(result)
        assert result.graph is not None
        requested_id = subgoal.payload.get("subgraph_id", subgoal.payload.get("subgoal_id"))
        if not isinstance(requested_id, str) or not requested_id:
            raise GraphProposalError(
                GraphDecodeResult(
                    rejections=(
                        GraphDecodeRejection(
                            "SUBGOAL_ID_MISSING",
                            "subgoal artifact must declare subgraph_id or subgoal_id",
                        ),
                    )
                )
            )
        matches = tuple(
            candidate
            for candidate in result.graph.subgraphs
            if candidate.subgraph_id == requested_id or candidate.subgoal_id == requested_id
        )
        if len(matches) != 1:
            raise GraphProposalError(
                GraphDecodeResult(
                    rejections=(
                        GraphDecodeRejection(
                            "SUBGOAL_MISMATCH",
                            f"expected one subgraph for {requested_id!r}, found {len(matches)}",
                        ),
                    )
                )
            )
        return matches[0]


class LLMStagedGraphPolicyAgent(GraphPolicyAgent):
    """Stage-two Policy Agent that returns a direct local graph envelope."""

    name = "llm_staged_graph_policy"

    def __init__(
        self,
        llm: LLMClient,
        request_builder: Callable[[AgentArtifact, SceneSnapshot, AgentContext], LLMRequest],
        *,
        decoder: LocalSubgraphDecoder | None = None,
        agent_name: str = "llm_staged_graph_policy",
        policy_strategy: str = "balanced",
        proposal_retries: int = 0,
        repair_request_builder: Callable[[AgentArtifact, SceneSnapshot, AgentContext, str], LLMRequest] | None = None,
        condition_enricher: Callable[[SubgraphSpec, SceneSnapshot, str], SubgraphSpec]
        | None = None,
    ) -> None:
        if proposal_retries < 0:
            raise ValueError("proposal_retries must not be negative")
        from capmas.contracts.strategy import StrategyProfile

        StrategyProfile.for_name(policy_strategy)
        self.llm = llm
        self.request_builder = request_builder
        if decoder is not None:
            self.decoder = decoder
        elif condition_enricher is None:
            self.decoder = LocalSubgraphDecoder()
        else:
            self.decoder = LocalSubgraphDecoder(
                condition_enricher=lambda subgraph, scene: condition_enricher(
                    subgraph, scene, policy_strategy
                )
            )
        self.name = agent_name
        self.policy_strategy = policy_strategy
        self.proposal_retries = proposal_retries
        self.repair_request_builder = repair_request_builder

    def propose_subgraph(
        self,
        subgoal: AgentArtifact,
        scene: SceneSnapshot,
        context: AgentContext,
    ) -> SubgraphSpec:
        requested_subgraph_id = subgoal.payload.get("subgraph_id")
        requested_subgoal_id = subgoal.payload.get("subgoal_id")
        if not isinstance(requested_subgraph_id, str) or not requested_subgraph_id:
            raise StagedProposalError(
                SubgraphDecodeResult(
                    rejections=(
                        StagedDecodeRejection(
                            "SUBGRAPH_ID_MISSING",
                            "topology subgoal must declare subgraph_id",
                        ),
                    )
                )
            )
        if not isinstance(requested_subgoal_id, str) or not requested_subgoal_id:
            raise StagedProposalError(
                SubgraphDecodeResult(
                    rejections=(
                        StagedDecodeRejection(
                            "SUBGOAL_ID_MISSING",
                            "topology subgoal must declare subgoal_id",
                        ),
                    )
                )
            )
        feedback: str | None = None
        for attempt in range(self.proposal_retries + 1):
            if attempt and self.repair_request_builder is not None and feedback:
                request = self.repair_request_builder(subgoal, scene, context, feedback)
            else:
                request = self.request_builder(subgoal, scene, context)
            try:
                response = self.llm.complete(request)
                result = self.decoder.decode(
                    response,
                    scene,
                    request=request,
                    expected_subgraph_id=requested_subgraph_id,
                    expected_subgoal_id=requested_subgoal_id,
                )
                if not result.accepted:
                    raise StagedProposalError(result)
                assert result.subgraph is not None
                return result.subgraph
            except Exception as exc:
                feedback = str(exc)
                if attempt >= self.proposal_retries:
                    raise
        raise AssertionError("staged policy proposal loop exited unexpectedly")
