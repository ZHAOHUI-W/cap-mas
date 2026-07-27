from __future__ import annotations

from dataclasses import dataclass
from statistics import median


@dataclass(frozen=True)
class RealTimeMetricsSummary:
    target_hz: float
    achieved_hz: float
    observations: int
    dropped_frames: int
    queue_delay_p50_ms: float
    queue_delay_p95_ms: float
    processing_latency_p50_ms: float
    processing_latency_p95_ms: float
    snapshot_age_p50_ms: float
    snapshot_age_p95_ms: float


class RealTimeMetrics:
    def __init__(self, target_hz: float) -> None:
        if target_hz <= 0.0:
            raise ValueError("target_hz must be positive")
        self.target_hz = target_hz
        self._observations: list[int] = []
        self._dropped_frames = 0
        self._queue_delays: list[float] = []
        self._processing_latencies: list[float] = []
        self._snapshot_ages: list[float] = []

    def record_observation(self, timestamp_ns: int) -> None:
        self._observations.append(timestamp_ns)

    def record_drop(self) -> None:
        self._dropped_frames += 1

    def record_queue_delay(self, delay_ms: float) -> None:
        self._queue_delays.append(float(delay_ms))

    def record_processing_latency(self, latency_ms: float) -> None:
        self._processing_latencies.append(float(latency_ms))

    def record_snapshot_age(self, age_ms: float) -> None:
        self._snapshot_ages.append(float(age_ms))

    def summary(self, now_ns: int | None = None) -> RealTimeMetricsSummary:
        del now_ns
        achieved_hz = 0.0
        if len(self._observations) >= 2:
            elapsed_s = (self._observations[-1] - self._observations[0]) / 1_000_000_000
            if elapsed_s > 0.0:
                achieved_hz = (len(self._observations) - 1) / elapsed_s
        return RealTimeMetricsSummary(
            target_hz=self.target_hz,
            achieved_hz=achieved_hz,
            observations=len(self._observations),
            dropped_frames=self._dropped_frames,
            queue_delay_p50_ms=_percentile(self._queue_delays, 0.50),
            queue_delay_p95_ms=_percentile(self._queue_delays, 0.95),
            processing_latency_p50_ms=_percentile(self._processing_latencies, 0.50),
            processing_latency_p95_ms=_percentile(self._processing_latencies, 0.95),
            snapshot_age_p50_ms=_percentile(self._snapshot_ages, 0.50),
            snapshot_age_p95_ms=_percentile(self._snapshot_ages, 0.95),
        )


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return ordered[index]
