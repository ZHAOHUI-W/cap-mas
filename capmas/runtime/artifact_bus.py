from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from threading import RLock
from typing import Any


@dataclass(frozen=True)
class ArtifactEnvelope:
    """Immutable, version-grounded payload exchanged by agents."""

    artifact_id: str
    kind: str
    payload: Mapping[str, Any]
    producer_agent: str
    parent_scene_version: int
    parent_artifact_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.artifact_id or not self.kind or not self.producer_agent:
            raise ValueError("artifact id, kind, and producer must be non-empty")
        if self.parent_scene_version < 0:
            raise ValueError("parent scene version must not be negative")


class ArtifactStore:
    """Small in-memory artifact store used by the fixed scheduler seam.

    A future durable implementation can replace this class without changing
    the Agent/Arbiter contracts.  Values are write-once to prevent one agent
    from silently mutating another agent's proposal.
    """

    def __init__(self) -> None:
        self._items: dict[str, ArtifactEnvelope] = {}
        self._lock = RLock()

    def put(self, artifact: ArtifactEnvelope) -> None:
        with self._lock:
            if artifact.artifact_id in self._items:
                raise ValueError(f"artifact {artifact.artifact_id!r} already exists")
            self._items[artifact.artifact_id] = artifact

    def get(self, artifact_id: str) -> ArtifactEnvelope:
        with self._lock:
            try:
                return self._items[artifact_id]
            except KeyError as exc:
                raise KeyError(f"unknown artifact: {artifact_id}") from exc

    def snapshot(self) -> tuple[ArtifactEnvelope, ...]:
        with self._lock:
            return tuple(self._items.values())


@dataclass(frozen=True)
class RuntimeEvent:
    event_id: str
    topic: str
    scene_version: int
    artifact: ArtifactEnvelope


EventHandler = Callable[[RuntimeEvent], None]


class EventBus:
    """Synchronous typed event fan-out for agent-plane coordination.

    Handlers are isolated: one failing subscriber is reported in the return
    value and cannot prevent other subscribers from observing the event.
    Robot execution is intentionally not performed by this class.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, dict[int, EventHandler]] = defaultdict(dict)
        self._counter = 0
        self._lock = RLock()

    def subscribe(self, topic: str, handler: EventHandler) -> Callable[[], None]:
        if not topic:
            raise ValueError("event topic must not be empty")
        with self._lock:
            self._counter += 1
            token = self._counter
            self._handlers[topic][token] = handler

        def unsubscribe() -> None:
            with self._lock:
                self._handlers.get(topic, {}).pop(token, None)

        return unsubscribe

    def publish(self, event: RuntimeEvent) -> tuple[Exception, ...]:
        with self._lock:
            handlers = tuple(self._handlers.get(event.topic, {}).values())
        failures: list[Exception] = []
        for handler in handlers:
            try:
                handler(event)
            except Exception as exc:  # subscriber isolation is intentional
                failures.append(exc)
        return tuple(failures)

