from dataclasses import FrozenInstanceError

import pytest

from capmas.contracts.verification import PredicateReport
from capmas.verification.evidence import (
    VerifierEvidence,
    VerifierPredicateEvidence,
    predicate_report_to_evidence,
    summarize_verifier_results,
)


def test_typed_evidence_is_frozen_and_serializable() -> None:
    item = VerifierPredicateEvidence(
        "gripper_closed()",
        "dynamic",
        "pass",
        0.9,
        None,
        ("artifact://scene/4",),
    )
    evidence = VerifierEvidence(
        "fp",
        4,
        1.0,
        1.0,
        "test",
        10,
        dynamic_results=(item,),
        source_verification="contract-1",
    )

    assert evidence.to_dict()["dynamic_results"][0]["status"] == "pass"
    assert evidence.to_dict()["candidate_fingerprint"] == "fp"
    with pytest.raises(FrozenInstanceError):
        item.status = "fail"  # type: ignore[misc]


def test_summary_excludes_unknown_from_pass_rate_but_counts_it_in_coverage() -> None:
    results = (
        VerifierPredicateEvidence("a", "static", "pass", 1.0, None),
        VerifierPredicateEvidence("b", "static", "fail", 1.0, "measured failure"),
        VerifierPredicateEvidence("c", "static", "unknown", None, "track not found"),
    )

    assert summarize_verifier_results(results) == (0.5, 2 / 3)


def test_invalid_phase_confidence_and_duplicate_are_rejected() -> None:
    with pytest.raises(ValueError, match="phase"):
        VerifierEvidence(
            "fp",
            1,
            1.0,
            1.0,
            "test",
            1,
            static_results=(
                VerifierPredicateEvidence("a", "dynamic", "pass", 1.0, None),
            ),
        )
    with pytest.raises(ValueError, match="confidence"):
        VerifierPredicateEvidence("a", "static", "pass", None, "missing confidence")
    with pytest.raises(ValueError, match="duplicate"):
        VerifierEvidence(
            "fp",
            1,
            1.0,
            1.0,
            "test",
            1,
            static_results=(
                VerifierPredicateEvidence("a", "static", "pass", 1.0, None),
                VerifierPredicateEvidence("a", "static", "fail", 1.0, "failure"),
            ),
        )


def test_unavailable_false_report_maps_to_unknown() -> None:
    item = predicate_report_to_evidence(
        PredicateReport(
            "object_at_target(bowl,plate)",
            False,
            0.0,
            ("bowl",),
            "track not found",
        ),
        phase="dynamic",
    )

    assert item.status == "unknown"
    assert item.confidence is None
