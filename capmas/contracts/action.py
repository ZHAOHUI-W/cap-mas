from __future__ import annotations

from dataclasses import dataclass

from capmas.contracts.core import SkillRef


@dataclass(frozen=True)
class ExecutionBudget:
    max_duration_ms: int
    max_sim_steps: int


@dataclass(frozen=True)
class SkillOutputRef:
    """Reference to a previous skill call's output in one action graph."""

    call_index: int
    path: tuple[str | int, ...] = ()


@dataclass(frozen=True)
class SkillCall:
    skill: SkillRef
    args: dict[str, object]


@dataclass(frozen=True)
class ActionContract:
    contract_id: str
    episode_id: str
    episode_epoch: int
    parent_scene_version: int
    subgoal_id: str
    skills: tuple[SkillCall, ...]
    expected_postconditions: tuple[str, ...]
    max_duration_ms: int
    max_sim_steps: int
    proposed_by: str
    preconditions: tuple[str, ...] = ()
    safety_invariants: tuple[str, ...] = ()
    recovery_policy: str = "replan"

    @property
    def budget(self) -> ExecutionBudget:
        return ExecutionBudget(
            max_duration_ms=self.max_duration_ms,
            max_sim_steps=self.max_sim_steps,
        )
