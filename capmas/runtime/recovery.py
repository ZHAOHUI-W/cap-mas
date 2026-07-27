from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from capmas.contracts.agent import AgentContext
from capmas.contracts.failures import FailureArtifact


@dataclass(frozen=True)
class RecoveryDecision:
    """A recovery proposal; it never directly executes a robot action."""

    target_subgraph: str
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.target_subgraph:
            raise ValueError("recovery target subgraph must not be empty")


class RecoverySelector(Protocol):
    def select(
        self,
        failure: FailureArtifact,
        context: AgentContext,
    ) -> RecoveryDecision | None: ...


class MappingRecoverySelector:
    """Deterministic failure-class-to-subgraph recovery policy."""

    def __init__(self, routes: dict[str, str]) -> None:
        self._routes = dict(routes)

    def select(
        self,
        failure: FailureArtifact,
        context: AgentContext,
    ) -> RecoveryDecision | None:
        target = self._routes.get(failure.failure_class)
        if target is None:
            target = self._routes.get("*")
        return RecoveryDecision(target, f"route for {failure.failure_class}") if target else None
