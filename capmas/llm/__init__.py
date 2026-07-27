"""LLM boundary; the runtime can also use mock or replay clients."""

from capmas.llm.protocol import (
    LLMCallTrace,
    LLMClient,
    LLMRequest,
    LLMResponse,
    LLMTraceCollector,
    LLMTraceSink,
)
from capmas.llm.graph_decoder import (
    GraphDecodeRejection,
    GraphDecodeResult,
    GraphProposalError,
    MissionGraphDecoder,
)
from capmas.llm.staged_decoder import (
    LocalSubgraphDecoder,
    MissionTopologyDecoder,
    StagedDecodeRejection,
    StagedProposalError,
)
from capmas.llm.capx_compatible import (
    CAPXCompatibleConfig,
    CAPXCompatibleLLMClient,
    LLMTransportError,
)
from capmas.llm.prompts import (
    build_manager_request,
    build_policy_request,
    build_staged_policy_request,
    build_topology_request,
    mission_graph_response_schema,
    mission_topology_response_schema,
    subgraph_response_schema,
)

__all__ = [
    "LLMClient",
    "LLMCallTrace",
    "LLMRequest",
    "LLMResponse",
    "LLMTraceCollector",
    "LLMTraceSink",
    "GraphDecodeRejection",
    "GraphDecodeResult",
    "GraphProposalError",
    "MissionGraphDecoder",
    "LocalSubgraphDecoder",
    "MissionTopologyDecoder",
    "StagedDecodeRejection",
    "StagedProposalError",
    "CAPXCompatibleConfig",
    "CAPXCompatibleLLMClient",
    "LLMTransportError",
    "build_manager_request",
    "build_policy_request",
    "build_staged_policy_request",
    "build_topology_request",
    "mission_graph_response_schema",
    "mission_topology_response_schema",
    "subgraph_response_schema",
]
