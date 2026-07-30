from dataclasses import dataclass

import pytest

from capmas.evaluation.evidence_contracts import EvidenceRequestContext
from capmas.evaluation.rehearsal import RehearsalFailureClass, RehearsalJob, RehearsalResult
from capmas.evaluation.rehearsal_evidence import (
    RehearsalEvidence,
    RehearsalPoolConfig,
    rehearsal_result_to_evidence,
    run_with_respawn,
)


def _result(**overrides):
    values = {
        "candidate_id": "candidate-a",
        "seed": 3,
        "success": True,
        "latency_ms": 12.0,
        "scene_version": 4,
        "candidate_fingerprint": "fp-a",
        "fingerprint_scope": "graph",
        "arbiter_subgraph_id": "sg_pick",
        "arbiter_fingerprint": "sub-fp-a",
    }
    values.update(overrides)
    return RehearsalResult(**values)


def test_rehearsal_result_becomes_version_bound_evidence():
    evidence = rehearsal_result_to_evidence(
        _result(),
        EvidenceRequestContext("fp-a", scene_version=4),
    )

    assert isinstance(evidence, RehearsalEvidence)
    assert evidence.success is True
    assert evidence.score == 1.0
    assert evidence.scene_version == 4
    assert evidence.fingerprint_scope == "graph"
    assert evidence.arbiter_subgraph_id == "sg_pick"
    assert evidence.arbiter_fingerprint == "sub-fp-a"


def test_stale_rehearsal_result_cannot_become_evidence():
    with pytest.raises(ValueError, match="scene version"):
        rehearsal_result_to_evidence(
            _result(scene_version=3),
            EvidenceRequestContext("fp-a", scene_version=4),
        )


@dataclass
class _FakePool:
    total_calls = 0

    def __init__(self, *, max_workers, timeout_s):
        del max_workers, timeout_s

    def run(self, jobs, worker):
        del worker
        type(self).total_calls += 1
        if type(self).total_calls == 1:
            raise RuntimeError("worker crashed")
        return tuple(
            RehearsalResult(
                candidate_id=job.candidate_id,
                seed=job.seed,
                success=True,
                latency_ms=1.0,
                candidate_fingerprint=job.candidate_fingerprint,
                scene_version=job.scene_version,
            )
            for job in jobs
        )


def test_respawn_retries_worker_crashes_with_a_bounded_new_pool(monkeypatch):
    import capmas.evaluation.rehearsal_evidence as module

    monkeypatch.setattr(module, "ProcessRehearsalPool", _FakePool)
    _FakePool.total_calls = 0
    jobs = (RehearsalJob("candidate-a", 3, {}, scene_version=4, candidate_fingerprint="fp-a"),)

    results = run_with_respawn(
        jobs,
        worker_factory=lambda: object(),
        pool_config=RehearsalPoolConfig(max_workers=1, timeout_s=1.0, max_restarts=1),
    )

    assert len(results) == 1
    assert results[0].success is True


class _OrderingPool:
    def __init__(self, *, max_workers, timeout_s):
        del max_workers, timeout_s

    def run(self, jobs, worker):
        del worker
        return tuple(
            RehearsalResult(
                candidate_id=job.candidate_id,
                seed=job.seed,
                success=True,
                latency_ms=1.0,
                scene_version=job.scene_version,
                candidate_fingerprint=job.candidate_fingerprint,
            )
            for job in sorted(jobs, key=lambda item: item.candidate_id)
        )


def test_respawn_binds_sorted_results_by_candidate_identity(monkeypatch):
    import capmas.evaluation.rehearsal_evidence as module

    monkeypatch.setattr(module, "ProcessRehearsalPool", _OrderingPool)
    jobs = (
        RehearsalJob("candidate-b", 3, {}, scene_version=8, candidate_fingerprint="fp-b"),
        RehearsalJob("candidate-a", 3, {}, scene_version=7, candidate_fingerprint="fp-a"),
    )

    results = run_with_respawn(
        jobs,
        worker_factory=lambda: object(),
        pool_config=RehearsalPoolConfig(max_workers=1, timeout_s=1.0),
    )

    assert [(result.candidate_id, result.scene_version) for result in results] == [
        ("candidate-a", 7),
        ("candidate-b", 8),
    ]
