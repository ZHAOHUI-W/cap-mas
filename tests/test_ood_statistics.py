from __future__ import annotations

import pytest

from capmas.evaluation.ood_statistics import (
    aggregate_ood_pairs,
    exact_mcnemar_pvalue,
    paired_success_delta,
    wilson_interval,
)
from capmas.evaluation.ood import OODReplayEvidence


def _evidence(
    *, case_id: str, split: str, success: bool | None, pair_id: str
) -> OODReplayEvidence:
    return OODReplayEvidence(
        case_id=case_id,
        pair_id=pair_id,
        condition="capmas",
        candidate_id=f"candidate-{case_id}",
        split=split,
        ood_type="none" if split == "id" else "layout",
        source_scene_version=1,
        candidate_fingerprint=f"fingerprint-{case_id}",
        evaluator_success=success,
        verifier_success=success,
        graph_completed=success is True,
        failure_class=None if success else "timeout" if success is None else "task_failure",
        recovery_count=0,
        human_intervention_count=0,
        latency_ms=10.0,
        provider_call_count=1,
        cache_hit_count=0,
        selection_basis="evidence_tie_break",
    )


def test_wilson_interval_handles_empty_and_all_success_cases() -> None:
    assert wilson_interval(0, 0).estimate == 0.0
    interval = wilson_interval(5, 5)
    assert interval.estimate == 1.0
    assert 0.0 <= interval.lower <= interval.upper <= 1.0


def test_wilson_interval_rejects_invalid_counts() -> None:
    with pytest.raises(ValueError, match="successes"):
        wilson_interval(2, 1)


def test_paired_delta_and_exact_mcnemar_are_deterministic() -> None:
    delta = paired_success_delta(
        (True, False, True, False), (False, False, True, True)
    )
    assert delta == (1, 1, 2)
    assert exact_mcnemar_pvalue(1, 1) == 1.0
    assert exact_mcnemar_pvalue(2, 0) == 0.5


def test_paired_delta_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="length"):
        paired_success_delta((True,), (True, False))


def test_aggregate_reports_id_ood_gap_and_unknowns() -> None:
    evidence = (
        _evidence(case_id="id-1", split="id", success=True, pair_id="pair-1"),
        _evidence(case_id="ood-1", split="ood", success=False, pair_id="pair-1"),
        _evidence(case_id="id-2", split="id", success=None, pair_id="pair-2"),
        _evidence(case_id="ood-2", split="ood", success=False, pair_id="pair-2"),
    )
    report = aggregate_ood_pairs(evidence)
    assert report.id_success_count == 1
    assert report.id_success_total == 1
    assert report.ood_success_count == 0
    assert report.ood_success_total == 2
    assert report.infrastructure_unknown_count == 1
    assert report.ood_gap.estimate == 1.0
    assert report.selection_bases["evidence_tie_break"] == 4


def test_aggregate_pairs_by_explicit_pair_id_not_case_id_suffix() -> None:
    evidence = (
        _evidence(case_id="baseline-run-alpha", split="id", success=True, pair_id="matched-A"),
        _evidence(case_id="layout-replay-zeta", split="ood", success=False, pair_id="matched-A"),
    )
    report = aggregate_ood_pairs(evidence)
    assert report.paired_id_only == 1
    assert report.paired_ood_only == 0
