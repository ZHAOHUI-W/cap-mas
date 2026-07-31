from __future__ import annotations

from typing import Protocol

from capmas.backends.protocol import SkillExecutionResult
from capmas.contracts.action import ExecutionBudget
from capmas.contracts.core import SkillRef


class TypedSkill(Protocol):
    skill_id: str
    version: str
    default_postconditions: tuple[str, ...]

    def validate_args(self, args: dict[str, object]) -> None: ...

    def execute(
        self,
        args: dict[str, object],
        budget: ExecutionBudget,
    ) -> SkillExecutionResult: ...


def skill_ref(skill: TypedSkill) -> SkillRef:
    return SkillRef(skill_id=skill.skill_id, version=skill.version)
