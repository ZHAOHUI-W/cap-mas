from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from capmas.contracts.core import ArtifactRef, SkillRef


class MemoryLayer:
    EPISODE = "episode"
    EXPERIENCE = "experience"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class MemoryOperation:
    ADD = "add"
    UPSERT = "upsert"
    INVALIDATE = "invalidate"
    RETIRE = "retire"
    CONSOLIDATE = "consolidate"
    NOOP = "noop"


@dataclass(frozen=True)
class MemoryBudget:
    max_items: int = 3
    max_latency_ms: int = 100
    max_tokens: int = 2048


@dataclass(frozen=True)
class TraceSpan:
    trace_ids: tuple[str, ...]
    start_ns: int
    end_ns: int
    summary: str = ""


@dataclass(frozen=True)
class MemoryItem:
    memory_id: str
    memory_version: str
    kind: str
    content: Mapping[str, object]
    applicability: Mapping[str, object]
    confidence: float
    evidence_count: int
    source_episode_ids: tuple[str, ...]
    source_trace_ids: tuple[str, ...]
    status: str = "candidate"
    contradiction_set: tuple[str, ...] = ()
    created_at_ns: int = 0
    last_validated_at_ns: int = 0
    ttl_seconds: int | None = None


@dataclass(frozen=True)
class MemoryItemRef:
    memory_id: str
    memory_version: str
    kind: str
    summary: str
    confidence: float
    evidence_count: int
    applicability: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class MemorySkillRef:
    skill_id: str
    version: str
    summary: str = ""


@dataclass(frozen=True)
class MemoryContext:
    context_id: str
    episode_id: str
    task_id: str
    task_family: str
    scene_version: int
    current_subgoal: str
    trace_span: TraceSpan
    retrieved_memories: tuple[MemoryItemRef, ...] = ()
    hard_cases: tuple[MemoryItemRef, ...] = ()
    memory_skill_candidates: tuple[MemorySkillRef, ...] = ()
    active_memory_bank_version: str = "0"
    active_robot_registry_version: str = "0"
    budget: MemoryBudget = field(default_factory=MemoryBudget)
    novelty: float = 0.0
    uncertainty: float = 0.0
    current_failure: str | None = None
    recent_recovery: str | None = None


@dataclass(frozen=True)
class MemorySelection:
    selection_id: str
    selected_skills: tuple[MemorySkillRef, ...]
    skipped: bool
    rationale: str


@dataclass(frozen=True)
class MemoryUpdate:
    update_id: str
    episode_id: str
    task_id: str
    base_memory_version: str
    target_layer: str
    operation: str
    items: tuple[MemoryItem, ...]
    invalidated_memory_ids: tuple[str, ...] = ()
    retired_memory_ids: tuple[str, ...] = ()
    source_trace_ids: tuple[str, ...] = ()
    evidence_refs: tuple[ArtifactRef, ...] = ()
    produced_by_skill: SkillRef | None = None
    controller_selection_id: str | None = None
    confidence: float = 0.0
    applicability: Mapping[str, object] = field(default_factory=dict)
    ttl_seconds: int | None = None
    idempotency_key: str = ""
    status: str = "proposed"
