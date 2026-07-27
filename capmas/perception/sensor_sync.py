from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

from capmas.perception.protocol import ObservationBundle
from capmas.perception.serialization import observation_from_json, observation_to_json


class RecordingObservationSource(Protocol):
    def iter_records(self) -> Iterator[ObservationBundle]: ...


class ReplayObservationSource(Protocol):
    def capture(self) -> ObservationBundle: ...

    def exhausted(self) -> bool: ...


class SensorSynchronizer(Protocol):
    def push(self, observation: ObservationBundle) -> None: ...

    def pop_ready(self) -> ObservationBundle | None: ...

    def metrics(self) -> "SynchronizerMetrics": ...


class JsonlObservationRecorder:
    """Append-only metadata recorder; frame bytes remain in artifact storage."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, observation: ObservationBundle) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(observation_to_json(observation))
            handle.write("\n")
            handle.flush()

    def iter_records(self) -> Iterator[ObservationBundle]:
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield observation_from_json(line)


class JsonlReplaySource:
    """Replay one JSONL record at a time with a one-line lookahead only."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._handle = None
        self._next_line: str | None = None
        self._exhausted = False

    def capture(self) -> ObservationBundle:
        if self._exhausted:
            raise StopIteration
        if self._handle is None:
            self._handle = self.path.open("r", encoding="utf-8")
        line = self._next_line
        self._next_line = None
        if line is None:
            line = self._read_nonempty_line()
        if line is None:
            self._mark_exhausted()
            raise StopIteration
        observation = observation_from_json(line)
        self._next_line = self._read_nonempty_line()
        if self._next_line is None:
            self._mark_exhausted()
        return observation

    def exhausted(self) -> bool:
        return self._exhausted

    def iter_records(self) -> Iterator[ObservationBundle]:
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield observation_from_json(line)

    def _read_nonempty_line(self) -> str | None:
        assert self._handle is not None
        for line in self._handle:
            if line.strip():
                return line
        return None

    def _mark_exhausted(self) -> None:
        self._exhausted = True
        if self._handle is not None:
            self._handle.close()
            self._handle = None


@dataclass(frozen=True)
class SynchronizerMetrics:
    accepted: int = 0
    rejected_out_of_order: int = 0
    rejected_episode: int = 0
    dropped_oldest: int = 0
    queued: int = 0


class BoundedSensorSynchronizer:
    def __init__(
        self,
        capacity: int,
        *,
        episode_id: str | None = None,
        episode_epoch: int | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.episode_id = episode_id
        self.episode_epoch = episode_epoch
        self._queue: deque[ObservationBundle] = deque(maxlen=capacity)
        self._last_sequence: int | None = None
        self._last_timestamp_ns: int | None = None
        self._accepted = 0
        self._rejected_out_of_order = 0
        self._rejected_episode = 0
        self._dropped_oldest = 0

    def push(self, observation: ObservationBundle) -> None:
        if self._wrong_episode(observation):
            self._rejected_episode += 1
            raise ValueError("observation belongs to another episode")
        if (
            self._last_sequence is not None
            and observation.sequence > 0
            and observation.sequence <= self._last_sequence
        ) or (
            self._last_timestamp_ns is not None
            and observation.timestamp_ns < self._last_timestamp_ns
        ):
            self._rejected_out_of_order += 1
            raise ValueError("observation sequence/timestamp must be monotonic")
        if len(self._queue) == self.capacity:
            self._queue.popleft()
            self._dropped_oldest += 1
        self._queue.append(observation)
        self._accepted += 1
        if observation.sequence > 0:
            self._last_sequence = observation.sequence
        self._last_timestamp_ns = observation.timestamp_ns

    def pop_ready(self) -> ObservationBundle | None:
        if not self._queue:
            return None
        return self._queue.popleft()

    def metrics(self) -> SynchronizerMetrics:
        return SynchronizerMetrics(
            accepted=self._accepted,
            rejected_out_of_order=self._rejected_out_of_order,
            rejected_episode=self._rejected_episode,
            dropped_oldest=self._dropped_oldest,
            queued=len(self._queue),
        )

    def _wrong_episode(self, observation: ObservationBundle) -> bool:
        if self.episode_id is not None and observation.episode_id is not None:
            if observation.episode_id != self.episode_id:
                return True
        if self.episode_epoch is not None and observation.episode_epoch is not None:
            if observation.episode_epoch != self.episode_epoch:
                return True
        return False
