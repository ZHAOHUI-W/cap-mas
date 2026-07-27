from __future__ import annotations

from typing import Protocol

from capmas.contracts.memory import (
    MemoryContext,
    MemorySelection,
    MemoryUpdate,
    TraceSpan,
)


class MemoryController(Protocol):
    def select(self, context: MemoryContext) -> MemorySelection: ...


class MemoryExecutor(Protocol):
    def apply(self, selection: MemorySelection, trace_span: TraceSpan) -> MemoryUpdate: ...
