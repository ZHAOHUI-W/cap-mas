from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence

from capmas.contracts.action import ActionContract
from capmas.contracts.graph import MissionGraph, SubgraphSpec
from capmas.contracts.memory import MemoryContext
from capmas.contracts.scene import SceneSnapshot
from capmas.contracts.staged import MissionTopology
from capmas.contracts.trace import ExecutionTrace
from capmas.contracts.verification import VerificationResult
from capmas.perception.protocol import PerceptionRequest


@dataclass(frozen=True)
class CycleHistory:
    traces: tuple[ExecutionTrace, ...] = ()
    last_verification: VerificationResult | None = None
    current_subgoal: str | None = None
    recovery_count: int = 0


@dataclass(frozen=True)
class AgentContext:
    task_id: str
    episode_id: str
    episode_epoch: int
    scene: SceneSnapshot
    memories: MemoryContext | None = None
    budget: Mapping[str, int] = field(default_factory=dict)
    history: CycleHistory = field(default_factory=CycleHistory)


@dataclass(frozen=True)
class AgentArtifact:
    artifact_id: str
    kind: str
    payload: Mapping[str, object]
    source_agent: str
    parent_artifact_ids: Sequence[str] = ()


@dataclass(frozen=True)
class PolicyDecision:
    """A grounded policy either acts or requests targeted visual evidence."""

    action: ActionContract | None = None
    perception_request: PerceptionRequest | None = None
    grounded_track_ids: tuple[str, ...] = ()
    rationale: str = ""

    def __post_init__(self) -> None:
        if (self.action is None) == (self.perception_request is None):
            raise ValueError(
                "policy decision must contain exactly one action or perception request"
            )


class Agent(Protocol):
    name: str

    def handle(self, artifact: AgentArtifact, context: AgentContext) -> list[AgentArtifact]: ...


class MissionManager(Protocol):
    def propose_subgoal(self, task: str, scene: SceneSnapshot) -> AgentArtifact: ...


class MissionGraphManager(Protocol):
    """Manager boundary that returns the validated global typed graph."""

    name: str

    def propose_graph(self, task: str, scene: SceneSnapshot) -> MissionGraph: ...


class MissionTopologyManager(Protocol):
    """Stage-one Manager boundary for compact global planning."""

    name: str

    def propose_topology(self, task: str, scene: SceneSnapshot) -> MissionTopology: ...


class PolicyAgent(Protocol):
    def propose_action(
        self,
        subgoal: AgentArtifact,
        scene: SceneSnapshot,
        context: AgentContext,
    ) -> ActionContract: ...


class GraphPolicyAgent(Protocol):
    """A local Policy Agent that proposes a typed subgraph."""

    name: str

    def propose_subgraph(
        self,
        subgoal: AgentArtifact,
        scene: SceneSnapshot,
        context: AgentContext,
    ) -> SubgraphSpec: ...


class GroundedPolicyAgent(Protocol):
    """Policy boundary for action proposals and targeted evidence requests."""

    def decide(
        self,
        subgoal: AgentArtifact,
        scene: SceneSnapshot,
        context: AgentContext,
    ) -> PolicyDecision: ...


class RecoveryAgent(Protocol):
    def recover(
        self,
        trace: object,
        verification: VerificationResult,
        context: AgentContext,
    ) -> ActionContract | None: ...
