from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from capmas.contracts.core import ArtifactRef
from capmas.contracts.verification import VerificationResult

GraphEventKind = Literal[
    "subgraph_started",
    "subgraph_completed",
    "subgraph_failed",
    "node_started",
    "node_completed",
    "node_failed",
]


@dataclass(frozen=True)
class GraphExecutionEvent:
    sequence: int
    kind: GraphEventKind
    subgraph_id: str
    node_id: str | None
    node_type: Literal["action", "checkpoint", "router"] | None
    attempt: int
    outcome: str | None
    occurred_at_ns: int


@dataclass(frozen=True)
class SkillTrace:
    invocation_id: str
    skill_id: str
    skill_version: str
    args: dict[str, object]
    started_at_ns: int
    finished_at_ns: int
    status: str
    error_type: str | None = None
    error_message: str | None = None
    output: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionTrace:
    trace_id: str
    episode_id: str
    episode_epoch: int
    contract_id: str
    lease_id: str
    parent_scene_version: int
    start_scene_version: int
    end_scene_version: int | None
    started_at_ns: int
    finished_at_ns: int
    status: str
    skill_traces: tuple[SkillTrace, ...] = ()
    precondition_result: VerificationResult | None = None
    postcondition_result: VerificationResult | None = None
    failure_class: str | None = None
    observation_before: ArtifactRef | None = None
    observation_after: ArtifactRef | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeTrace:
    episode_id: str
    episode_epoch: int
    traces: tuple[ExecutionTrace, ...] = ()
