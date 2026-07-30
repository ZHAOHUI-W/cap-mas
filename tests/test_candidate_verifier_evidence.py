import pytest

from capmas.contracts.candidates import CandidateEvidence
from capmas.verification.evidence import (
    VerifierEvidence,
    VerifierPredicateEvidence,
    attach_verifier_evidence,
)


def _typed(
    pass_rate: float = 1.0,
    scene_version: int = 3,
    fingerprint: str = "fp",
) -> VerifierEvidence:
    result = VerifierPredicateEvidence(
        "track_exists:bowl",
        "static",
        "pass",
        1.0,
        None,
    )
    return VerifierEvidence(
        fingerprint,
        scene_version,
        pass_rate,
        1.0,
        "test",
        10,
        static_results=(result,),
    )


def test_legacy_scalar_constructor_still_works() -> None:
    evidence = CandidateEvidence(
        verifier_pass_rate=0.5,
        available_metrics=("verifier",),
        scene_version=3,
    )

    assert evidence.verifier is None


def test_attach_projects_typed_evidence_and_does_not_mutate_source() -> None:
    base = CandidateEvidence(available_metrics=("perception",), scene_version=3)
    typed = _typed()

    attached = attach_verifier_evidence(base, typed)

    assert base.verifier is None
    assert attached.verifier == typed
    assert attached.verifier_pass_rate == 1.0
    assert set(attached.available_metrics) == {"perception", "verifier"}
    assert attached.provider == "test"
    assert attached.captured_at_ns == 10


def test_inconsistent_typed_scalar_or_version_is_rejected() -> None:
    with pytest.raises(ValueError, match="verifier_pass_rate"):
        CandidateEvidence(
            verifier_pass_rate=0.2,
            verifier=_typed(1.0),
            available_metrics=("verifier",),
            scene_version=3,
            provider="test",
            captured_at_ns=10,
        )
    with pytest.raises(ValueError, match="scene"):
        CandidateEvidence(
            verifier_pass_rate=1.0,
            verifier=_typed(scene_version=4),
            available_metrics=("verifier",),
            scene_version=3,
            provider="test",
            captured_at_ns=10,
        )
