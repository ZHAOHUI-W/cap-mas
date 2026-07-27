from __future__ import annotations

from dataclasses import dataclass

from capmas.contracts.memory import MemoryItem, MemoryUpdate


@dataclass(frozen=True)
class MemorySnapshot:
    version: str
    items: tuple[MemoryItem, ...]


class InMemoryMemoryStore:
    def __init__(self) -> None:
        self._version = "0"
        self._items: dict[str, MemoryItem] = {}
        self._committed_updates: set[str] = set()

    def snapshot(self) -> MemorySnapshot:
        return MemorySnapshot(self._version, tuple(self._items.values()))

    def commit(self, update: MemoryUpdate) -> MemorySnapshot:
        if not update.idempotency_key:
            raise ValueError("memory update requires idempotency_key")
        if update.idempotency_key in self._committed_updates:
            return self.snapshot()
        if not update.source_trace_ids and update.operation != "noop":
            raise ValueError("memory update requires trace provenance")
        if update.base_memory_version != self._version:
            raise ValueError("memory base version conflict")
        for item in update.items:
            if item.memory_id in self._items and update.operation == "add":
                raise ValueError(f"memory item already exists: {item.memory_id}")
            self._items[item.memory_id] = item
        for memory_id in update.invalidated_memory_ids:
            if memory_id in self._items:
                item = self._items[memory_id]
                self._items[memory_id] = MemoryItem(**{**item.__dict__, "status": "contradicted"})
        for memory_id in update.retired_memory_ids:
            if memory_id in self._items:
                item = self._items[memory_id]
                self._items[memory_id] = MemoryItem(**{**item.__dict__, "status": "retired"})
        self._committed_updates.add(update.idempotency_key)
        self._version = str(int(self._version) + 1)
        return self.snapshot()
