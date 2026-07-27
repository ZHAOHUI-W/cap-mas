from __future__ import annotations

from dataclasses import dataclass
import heapq
import time
from typing import Callable, Protocol

from capmas.contracts.scene import SceneSnapshot
from capmas.perception.protocol import PerceptionRequest, PerceptionResult


@dataclass(frozen=True)
class SemanticQueueMetrics:
    submitted: int = 0
    deduplicated: int = 0
    dropped: int = 0
    completed: int = 0
    timed_out: int = 0
    queued: int = 0


class SemanticRequestQueue(Protocol):
    def submit(self, request: PerceptionRequest) -> bool: ...

    def poll(self, max_items: int = 1) -> tuple[PerceptionRequest, ...]: ...

    def complete(self, request_id: str, result: PerceptionResult) -> None: ...

    def cancel_episode(self, episode_id: str, episode_epoch: int) -> int: ...

    def metrics(self) -> SemanticQueueMetrics: ...


@dataclass
class _QueueEntry:
    request: PerceptionRequest
    submitted_at_ns: int
    key: tuple[object, ...]


class DeterministicSemanticRequestQueue:
    def __init__(
        self,
        *,
        capacity: int = 32,
        max_latency_ms: int = 500,
        clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if max_latency_ms < 0:
            raise ValueError("max_latency_ms must be non-negative")
        self.capacity = capacity
        self.max_latency_ms = max_latency_ms
        self.clock = clock
        self._entries: dict[str, _QueueEntry] = {}
        self._in_flight: dict[str, _QueueEntry] = {}
        self._dedup_keys: dict[tuple[object, ...], str] = {}
        self._heap: list[tuple[int, int, str]] = []
        self._counter = 0
        self._submitted = 0
        self._deduplicated = 0
        self._dropped = 0
        self._completed = 0
        self._timed_out = 0

    def submit(self, request: PerceptionRequest) -> bool:
        self._expire()
        key = self._dedup_key(request)
        if key in self._dedup_keys:
            self._deduplicated += 1
            return False
        if len(self._entries) >= self.capacity:
            self._dropped += 1
            return False
        submitted_at_ns = self.clock()
        entry = _QueueEntry(request, submitted_at_ns, key)
        self._entries[request.request_id] = entry
        self._dedup_keys[key] = request.request_id
        heapq.heappush(self._heap, (-request.priority, self._counter, request.request_id))
        self._counter += 1
        self._submitted += 1
        return True

    def poll(self, max_items: int = 1) -> tuple[PerceptionRequest, ...]:
        if max_items <= 0:
            return ()
        self._expire()
        result: list[PerceptionRequest] = []
        while self._heap and len(result) < max_items:
            _, _, request_id = heapq.heappop(self._heap)
            entry = self._entries.pop(request_id, None)
            if entry is None:
                continue
            self._dedup_keys.pop(entry.key, None)
            self._in_flight[request_id] = entry
            result.append(entry.request)
        return tuple(result)

    def complete(self, request_id: str, result: PerceptionResult) -> None:
        del result
        if self._in_flight.pop(request_id, None) is not None:
            self._completed += 1

    def cancel_episode(self, episode_id: str, episode_epoch: int) -> int:
        cancelled = 0
        for request_id, entry in tuple(self._entries.items()):
            request = entry.request
            if request.episode_id == episode_id and request.episode_epoch == episode_epoch:
                self._entries.pop(request_id)
                self._dedup_keys.pop(entry.key, None)
                cancelled += 1
        for request_id, entry in tuple(self._in_flight.items()):
            request = entry.request
            if request.episode_id == episode_id and request.episode_epoch == episode_epoch:
                self._in_flight.pop(request_id)
                cancelled += 1
        return cancelled

    def metrics(self) -> SemanticQueueMetrics:
        return SemanticQueueMetrics(
            submitted=self._submitted,
            deduplicated=self._deduplicated,
            dropped=self._dropped,
            completed=self._completed,
            timed_out=self._timed_out,
            queued=len(self._entries),
        )

    def _expire(self) -> None:
        now = self.clock()
        for request_id, entry in tuple(self._entries.items()):
            budget_ms = min(self.max_latency_ms, max(0, entry.request.max_latency_ms))
            if now - entry.submitted_at_ns <= budget_ms * 1_000_000:
                continue
            self._entries.pop(request_id)
            self._dedup_keys.pop(entry.key, None)
            self._timed_out += 1

    @staticmethod
    def _dedup_key(request: PerceptionRequest) -> tuple[object, ...]:
        return (
            request.episode_id,
            request.episode_epoch,
            request.scene_version,
            request.target_track_ids,
            request.evidence_types,
        )


class DeterministicSemanticTrigger:
    def __init__(
        self,
        queue: SemanticRequestQueue,
        *,
        confidence_threshold: float = 0.5,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        self.queue = queue
        self.confidence_threshold = confidence_threshold

    def inspect(self, scene: SceneSnapshot) -> tuple[PerceptionRequest, ...]:
        requests: list[PerceptionRequest] = []
        for track in scene.objects:
            if track.confidence >= self.confidence_threshold and track.track_status not in {"stale", "lost"}:
                continue
            priority = 100 if track.track_status == "lost" else 50
            request = PerceptionRequest(
                request_id=(
                    f"semantic:{scene.episode_id}:{scene.episode_epoch}:"
                    f"{scene.scene_version}:{track.track_id}:identity"
                ),
                query=f"resolve identity and pose for {track.track_id}",
                target_track_ids=(track.track_id,),
                evidence_types=("rgb_crop", "depth_crop", "mask"),
                purpose="identity_disambiguation",
                max_latency_ms=500,
                priority=priority,
                episode_id=scene.episode_id,
                episode_epoch=scene.episode_epoch,
                scene_version=scene.scene_version,
            )
            if self.queue.submit(request):
                requests.append(request)
        return tuple(requests)
