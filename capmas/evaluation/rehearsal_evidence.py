"""Version-bound rehearsal evidence and bounded worker respawn."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Mapping, Sequence

from capmas.evaluation.evidence_contracts import (
    EvidenceRequestContext,
    assert_evidence_compatible,
)
from capmas.evaluation.rehearsal import (
    ProcessRehearsalPool,
    RehearsalFailureClass,
    RehearsalJob,
    RehearsalResult,
    RehearsalTimeout,
)


@dataclass(frozen=True)
class RehearsalEvidence:
    candidate_id: str
    candidate_fingerprint: str
    seed: int
    scene_version: int
    success: bool
    score: float
    latency_ms: float
    failure_class: str | None = None
    failure_reason: str | None = None
    checkpoint_results: tuple[Mapping[str, object], ...] = ()
    fingerprint_scope: str = "subgraph"
    arbiter_subgraph_id: str | None = None
    arbiter_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.candidate_fingerprint:
            raise ValueError("rehearsal evidence identities must not be empty")
        if self.scene_version < 0 or self.latency_ms < 0:
            raise ValueError("rehearsal evidence versions and latency must be non-negative")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("rehearsal evidence score must be in [0, 1]")
        if self.fingerprint_scope not in {"graph", "subgraph"}:
            raise ValueError("rehearsal fingerprint scope must be graph or subgraph")
        if self.fingerprint_scope == "graph" and (
            not self.arbiter_subgraph_id or not self.arbiter_fingerprint
        ):
            raise ValueError(
                "graph-scoped rehearsal evidence requires arbiter identity mapping"
            )


@dataclass(frozen=True)
class RehearsalPoolConfig:
    max_workers: int = 1
    timeout_s: float = 30.0
    max_restarts: int = 1

    def __post_init__(self) -> None:
        if self.max_workers <= 0:
            raise ValueError("rehearsal max_workers must be positive")
        if self.timeout_s <= 0:
            raise ValueError("rehearsal timeout_s must be positive")
        if self.max_restarts < 0:
            raise ValueError("rehearsal max_restarts must not be negative")


def rehearsal_result_to_evidence(
    result: RehearsalResult,
    context: EvidenceRequestContext,
) -> RehearsalEvidence:
    if result.scene_version is None or result.candidate_fingerprint is None:
        raise ValueError("rehearsal result must include scene version and candidate fingerprint")
    assert_evidence_compatible(
        context,
        candidate_fingerprint=result.candidate_fingerprint,
        scene_version=result.scene_version,
    )
    return RehearsalEvidence(
        candidate_id=result.candidate_id,
        candidate_fingerprint=result.candidate_fingerprint,
        seed=result.seed,
        scene_version=result.scene_version,
        success=result.success,
        score=1.0 if result.success else 0.0,
        latency_ms=result.latency_ms,
        failure_class=result.failure_class,
        failure_reason=result.failure_reason,
        checkpoint_results=tuple(result.checkpoint_results),
        fingerprint_scope=result.fingerprint_scope,
        arbiter_subgraph_id=result.arbiter_subgraph_id,
        arbiter_fingerprint=result.arbiter_fingerprint,
    )


def run_with_respawn(
    jobs: Sequence[RehearsalJob],
    worker_factory: Callable[[], object],
    pool_config: RehearsalPoolConfig,
) -> tuple[RehearsalResult, ...]:
    """Run jobs, replacing a failed pool a bounded number of times."""

    pending = tuple(jobs)
    final: dict[tuple[str, int], RehearsalResult] = {}
    for attempt in range(pool_config.max_restarts + 1):
        if not pending:
            break
        pool = ProcessRehearsalPool(
            max_workers=pool_config.max_workers,
            timeout_s=pool_config.timeout_s,
        )
        try:
            batch = pool.run(pending, worker_factory())
        except RehearsalTimeout as exc:
            batch = tuple(
                _failure_result(job, RehearsalFailureClass.TIMEOUT, str(exc), pool_config.timeout_s)
                for job in pending
            )
        except Exception as exc:
            batch = tuple(
                _failure_result(job, RehearsalFailureClass.WORKER_CRASH, str(exc), 0.0)
                for job in pending
            )

        result_by_key = {(result.candidate_id, result.seed): result for result in batch}
        retry: list[RehearsalJob] = []
        for job in pending:
            result = result_by_key.get((job.candidate_id, job.seed))
            if result is None:
                result = _failure_result(
                    job,
                    RehearsalFailureClass.WORKER_CRASH,
                    "worker returned no result for job",
                    0.0,
                )
            is_crash = result.failure_class == RehearsalFailureClass.WORKER_CRASH
            if is_crash and attempt < pool_config.max_restarts:
                retry.append(job)
            else:
                final[(job.candidate_id, job.seed)] = result
        pending = tuple(retry)
    return tuple(final[key] for key in sorted(final))


def _failure_result(
    job: RehearsalJob,
    failure_class: RehearsalFailureClass,
    reason: str,
    latency_s: float,
) -> RehearsalResult:
    return RehearsalResult(
        candidate_id=job.candidate_id,
        seed=job.seed,
        success=False,
        latency_ms=max(0.0, latency_s * 1000.0),
        failure_class=failure_class,
        failure_reason=reason,
        scene_version=job.scene_version,
        candidate_fingerprint=job.candidate_fingerprint,
    )
