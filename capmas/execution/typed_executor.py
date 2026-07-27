from __future__ import annotations

from typing import Protocol

from capmas.backends.protocol import RobotBackend, SkillExecutionResult
from capmas.contracts.action import ExecutionBudget
from capmas.skills.protocol import TypedSkill


class TypedExecutor(Protocol):
    def execute(
        self,
        backend: RobotBackend,
        skill: TypedSkill,
        args: dict[str, object],
        budget: ExecutionBudget,
    ) -> SkillExecutionResult: ...


class BackendTypedExecutor:
    def execute(self, backend: RobotBackend, skill: TypedSkill, args: dict[str, object], budget: ExecutionBudget) -> SkillExecutionResult:
        return backend.execute_skill(skill, args, budget)
