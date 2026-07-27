from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


class FailureClass:
    STALE_STATE = "STALE_STATE"
    PRECONDITION_FAILED = "PRECONDITION_FAILED"
    EXECUTION_ERROR = "EXECUTION_ERROR"
    MOTION_TIMEOUT = "MOTION_TIMEOUT"
    POSTCONDITION_FAILED = "POSTCONDITION_FAILED"
    PERCEPTION_UNCERTAIN = "PERCEPTION_UNCERTAIN"
    COLLISION_RISK = "COLLISION_RISK"
    EPISODE_INVALIDATED = "EPISODE_INVALIDATED"


@dataclass(frozen=True)
class FailureArtifact:
    """Structured failure output used by recovery and memory agents."""

    failure_id: str
    failure_class: str
    message: str
    scene_version: int
    source_agent: str = "runtime"
    node_id: str | None = None
    subgraph_id: str | None = None
    recoverable: bool = True
    retry_count: int = 0
    recovery_policy: str = "replan"
    evidence_refs: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.failure_id or not self.failure_class or not self.message:
            raise ValueError("failure artifact identifiers and message must be non-empty")
        if self.scene_version < 0 or self.retry_count < 0:
            raise ValueError("failure scene version and retry count must not be negative")
        if not self.recovery_policy:
            raise ValueError("failure recovery policy must not be empty")
