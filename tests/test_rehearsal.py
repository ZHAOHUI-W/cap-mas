from __future__ import annotations

import os
from pathlib import Path
import time

import pytest

from capmas.evaluation.rehearsal import (
    ProcessRehearsalPool,
    RehearsalFailureClass,
    RehearsalJob,
    RehearsalResult,
    RehearsalTimeout,
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


def _blocking_worker(job: RehearsalJob):
    Path(job.payload["pid_path"]).write_text(str(os.getpid()), encoding="utf-8")
    time.sleep(5.0)
    return RehearsalResult(
        candidate_id=job.candidate_id,
        seed=job.seed,
        success=True,
        latency_ms=5000.0,
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


def test_process_rehearsal_pool_terminates_worker_after_timeout(tmp_path) -> None:
    pid_path = tmp_path / "worker.pid"
    jobs = (RehearsalJob("slow", 1, {"pid_path": str(pid_path)}),)

    with pytest.raises(RehearsalTimeout):
        ProcessRehearsalPool(max_workers=1, timeout_s=2.0).run(jobs, _blocking_worker)

    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and not pid_path.exists():
        time.sleep(0.01)
    assert pid_path.exists()
    pid = int(pid_path.read_text(encoding="utf-8"))
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.01)
    else:
        raise AssertionError(f"timed-out rehearsal worker {pid} is still alive")


def test_rehearsal_job_carries_scene_and_candidate_identity():
    job = RehearsalJob(
        candidate_id="candidate-a",
        seed=7,
        payload={"graph": "serialized"},
        task_id="libero_spatial_0",
        scene_version=12,
        candidate_fingerprint="fp-a",
        fingerprint_scope="graph",
        arbiter_subgraph_id="sg_pick",
        arbiter_fingerprint="sub-fp-a",
        checkpoint_budget=4,
    )

    assert job.task_id == "libero_spatial_0"
    assert job.scene_version == 12
    assert job.candidate_fingerprint == "fp-a"
    assert job.fingerprint_scope == "graph"
    assert job.arbiter_subgraph_id == "sg_pick"
    assert job.arbiter_fingerprint == "sub-fp-a"
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
        fingerprint_scope="graph",
        arbiter_subgraph_id="sg_pick",
        arbiter_fingerprint="sub-fp-a",
    )

    assert result.failure_class == RehearsalFailureClass.POSTCONDITION_FAILURE
    assert result.checkpoint_results[0]["passed"] is False
    assert result.failure_step == 2
    assert result.scene_version == 12
    assert result.fingerprint_scope == "graph"
    assert result.arbiter_fingerprint == "sub-fp-a"
