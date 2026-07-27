"""Isolated process-level graph rehearsal boundary.

Rehearsal is deliberately separate from live robot execution. A worker must
receive serializable job data and return a serializable result; it cannot own
the live CAP-X backend or an ActionLease.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeout, as_completed
from dataclasses import dataclass
import multiprocessing
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class RehearsalJob:
    candidate_id: str
    seed: int
    payload: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("rehearsal candidate id must not be empty")


@dataclass(frozen=True)
class RehearsalResult:
    candidate_id: str
    seed: int
    success: bool
    latency_ms: float
    failure_class: str | None = None
    worker_pid: int | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("rehearsal result candidate id must not be empty")
        if self.latency_ms < 0:
            raise ValueError("rehearsal latency must not be negative")


RehearsalWorker = Callable[[RehearsalJob], RehearsalResult]


class RehearsalTimeout(RuntimeError):
    """Raised when isolated rehearsal exceeds its wall-clock budget."""


class ProcessRehearsalPool:
    """Run serializable candidate rehearsals in isolated spawned processes."""

    def __init__(
        self,
        *,
        max_workers: int = 1,
        timeout_s: float = 30.0,
    ) -> None:
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if timeout_s <= 0:
            raise ValueError("rehearsal timeout must be positive")
        self.max_workers = max_workers
        self.timeout_s = timeout_s

    def run(
        self,
        jobs: Sequence[RehearsalJob],
        worker: RehearsalWorker,
    ) -> tuple[RehearsalResult, ...]:
        if not jobs:
            return ()
        context = multiprocessing.get_context("spawn")
        pool = ProcessPoolExecutor(
            max_workers=min(self.max_workers, len(jobs)),
            mp_context=context,
        )
        futures = {
            pool.submit(worker, job): job
            for job in jobs
        }
        results: list[RehearsalResult] = []
        timed_out = False
        try:
            for future in as_completed(futures, timeout=self.timeout_s):
                result = future.result()
                if not isinstance(result, RehearsalResult):
                    raise TypeError("rehearsal worker must return RehearsalResult")
                results.append(result)
        except FutureTimeout as exc:
            timed_out = True
            for future in futures:
                future.cancel()
            raise RehearsalTimeout(
                f"rehearsal exceeded timeout_s={self.timeout_s}"
            ) from exc
        finally:
            pool.shutdown(wait=not timed_out, cancel_futures=True)
        return tuple(sorted(results, key=lambda item: (item.candidate_id, item.seed)))
