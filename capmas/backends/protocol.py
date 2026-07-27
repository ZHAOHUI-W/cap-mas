from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol

from capmas.contracts.action import ExecutionBudget
from capmas.contracts.scene import EpisodeStart, SceneSnapshot


@dataclass(frozen=True)
class SkillExecutionResult:
    ok: bool
    output: dict[str, object] = field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None


class RobotBackend(Protocol):
    def reset(
        self,
        seed: int | None = None,
        options: Mapping[str, object] | None = None,
    ) -> EpisodeStart: ...

    def observe(self) -> SceneSnapshot: ...

    def execute_skill(
        self,
        skill: object,
        args: dict[str, object],
        budget: ExecutionBudget,
    ) -> SkillExecutionResult: ...

    def stop(self, lease: object) -> None: ...

    def evaluator_success(self) -> bool: ...
