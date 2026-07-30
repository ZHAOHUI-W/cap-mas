"""Isolated process-level graph rehearsal boundary.

Rehearsal is deliberately separate from live robot execution. A worker must
receive serializable job data and return a serializable result; it cannot own
the live CAP-X backend or an ActionLease.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeout, as_completed
from dataclasses import dataclass
from enum import StrEnum
import multiprocessing
from typing import Any, Callable, Mapping, Sequence


class RehearsalFailureClass(StrEnum):
    INVALID_GRAPH = "invalid_graph"
    RESET_FAILURE = "reset_failure"
    SKILL_FAILURE = "skill_failure"
    POSTCONDITION_FAILURE = "postcondition_failure"
    TIMEOUT = "timeout"
    WORKER_CRASH = "worker_crash"


@dataclass(frozen=True)
class RehearsalJob:
    candidate_id: str
    seed: int
    payload: dict[str, Any]
    task_id: str = ""
    scene_version: int = 0
    candidate_fingerprint: str = ""
    checkpoint_budget: int = 0

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("rehearsal candidate id must not be empty")
        if self.scene_version < 0:
            raise ValueError("rehearsal scene version must not be negative")
        if self.checkpoint_budget < 0:
            raise ValueError("rehearsal checkpoint budget must not be negative")
        if not self.candidate_fingerprint:
            object.__setattr__(self, "candidate_fingerprint", self.candidate_id)


@dataclass(frozen=True)
class RehearsalResult:
    candidate_id: str
    seed: int
    success: bool
    latency_ms: float
    failure_class: str | None = None
    worker_pid: int | None = None
    checkpoint_results: tuple[Mapping[str, Any], ...] = ()
    failure_step: int | None = None
    failure_reason: str | None = None
    scene_version: int | None = None
    candidate_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("rehearsal result candidate id must not be empty")
        if self.latency_ms < 0:
            raise ValueError("rehearsal latency must not be negative")
        if self.failure_step is not None and self.failure_step < 0:
            raise ValueError("rehearsal failure step must not be negative")
        if self.scene_version is not None and self.scene_version < 0:
            raise ValueError("rehearsal scene version must not be negative")


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
