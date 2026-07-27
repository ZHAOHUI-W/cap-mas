from __future__ import annotations

from typing import Protocol, Sequence

from capmas.contracts.trace import EpisodeTrace


class Evaluator(Protocol):
    def benchmark_success(self, episode: EpisodeTrace) -> bool: ...


class TraceSink(Protocol):
    def append(self, trace: EpisodeTrace) -> None: ...


class MetricsSink(Protocol):
    def record(self, name: str, value: float, tags: dict[str, str] | None = None) -> None: ...
