from __future__ import annotations

from capmas.evaluation.rehearsal import (
    ProcessRehearsalPool,
    RehearsalFailureClass,
    RehearsalJob,
    RehearsalResult,
)


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


def test_rehearsal_job_carries_scene_and_candidate_identity():
    job = RehearsalJob(
        candidate_id="candidate-a",
        seed=7,
        payload={"graph": "serialized"},
        task_id="libero_spatial_0",
        scene_version=12,
        candidate_fingerprint="fp-a",
        checkpoint_budget=4,
    )

    assert job.task_id == "libero_spatial_0"
    assert job.scene_version == 12
    assert job.candidate_fingerprint == "fp-a"
    assert job.checkpoint_budget == 4


def test_rehearsal_result_records_checkpoint_and_failure_details():
    result = RehearsalResult(
        candidate_id="candidate-a",
        seed=7,
        success=False,
        latency_ms=12.5,
        failure_class=RehearsalFailureClass.POSTCONDITION_FAILURE,
        checkpoint_results=({"node_id": "place", "passed": False},),
        failure_step=2,
        failure_reason="target predicate did not hold",
        scene_version=12,
        candidate_fingerprint="fp-a",
    )

    assert result.failure_class == RehearsalFailureClass.POSTCONDITION_FAILURE
    assert result.checkpoint_results[0]["passed"] is False
    assert result.failure_step == 2
    assert result.scene_version == 12
