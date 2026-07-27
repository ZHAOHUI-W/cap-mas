from __future__ import annotations

from uuid import uuid4

from capmas.contracts.memory import MemoryContext, MemorySelection


class RuleBasedMemoryController:
    def select(self, context: MemoryContext) -> MemorySelection:
        candidates = context.memory_skill_candidates[: context.budget.max_items]
        if not candidates:
            return MemorySelection(str(uuid4()), (), True, "no applicable Memory Skills")
        return MemorySelection(
            selection_id=str(uuid4()),
            selected_skills=tuple(candidates),
            skipped=False,
            rationale="selected active candidates within budget",
        )
