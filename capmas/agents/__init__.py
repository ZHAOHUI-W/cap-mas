"""Agent role interfaces and deterministic/LLM proposal implementations."""

from capmas.agents.arbiter import CandidateArbiter
from capmas.agents.manager import LLMTopologyManager, LLMMissionManager, SimpleMissionManager
from capmas.agents.policy import (
    CallableGraphPolicyAgent,
    CallablePolicyAgent,
    LLMGraphPolicyAgent,
    LLMStagedGraphPolicyAgent,
)

__all__ = [
    "CandidateArbiter",
    "CallableGraphPolicyAgent",
    "CallablePolicyAgent",
    "LLMGraphPolicyAgent",
    "LLMStagedGraphPolicyAgent",
    "LLMMissionManager",
    "LLMTopologyManager",
    "SimpleMissionManager",
]
