from __future__ import annotations

from capmas.evaluation.rehearsal import ProcessRehearsalPool, RehearsalJob


def _worker(job: RehearsalJob):
    from capmas.evaluation.rehearsal import RehearsalResult

    return RehearsalResult(
        candidate_id=job.candidate_id,
        seed=job.seed,
        success=job.candidate_id != "bad",
        latency_ms=1.0,
        failure_class=None if job.candidate_id != "bad" else "sim_failure",
    )


def test_process_rehearsal_pool_returns_deterministic_candidate_results() -> None:
    jobs = (
        RehearsalJob("bad", 2, {"graph": "bad"}),
        RehearsalJob("good", 1, {"graph": "good"}),
    )

    results = ProcessRehearsalPool(max_workers=2, timeout_s=5.0).run(jobs, _worker)

    assert [result.candidate_id for result in results] == ["bad", "good"]
    assert results[0].success is False
    assert results[1].success is True
    assert results[0].failure_class == "sim_failure"


def test_process_rehearsal_pool_rejects_invalid_limits() -> None:
    try:
        ProcessRehearsalPool(max_workers=0)
    except ValueError as exc:
        assert "max_workers" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected max_workers validation")
