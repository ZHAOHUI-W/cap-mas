import pytest

from capmas.evaluation.evidence_contracts import (
    EvidenceCompatibilityError,
    EvidenceRequestContext,
    assert_evidence_compatible,
)


def test_matching_candidate_scene_and_map_metadata_is_accepted():
    context = EvidenceRequestContext("candidate-a", scene_version=7, map_version=3)

    assert_evidence_compatible(
        context,
        candidate_fingerprint="candidate-a",
        scene_version=7,
        map_version=3,
    ) is None


def test_stale_scene_evidence_is_rejected():
    context = EvidenceRequestContext("candidate-a", scene_version=7)

    with pytest.raises(EvidenceCompatibilityError, match="scene version"):
        assert_evidence_compatible(
            context,
            candidate_fingerprint="candidate-a",
            scene_version=6,
        )


def test_candidate_fingerprint_mismatch_is_rejected():
    context = EvidenceRequestContext("candidate-a", scene_version=7)

    with pytest.raises(EvidenceCompatibilityError, match="fingerprint"):
        assert_evidence_compatible(
            context,
            candidate_fingerprint="candidate-b",
            scene_version=7,
        )


def test_required_map_version_rejects_missing_or_stale_map_evidence():
    context = EvidenceRequestContext("candidate-a", scene_version=7, map_version=3)

    with pytest.raises(EvidenceCompatibilityError, match="map version"):
        assert_evidence_compatible(
            context,
            candidate_fingerprint="candidate-a",
            scene_version=7,
        )

    with pytest.raises(EvidenceCompatibilityError, match="map version"):
        assert_evidence_compatible(
            context,
            candidate_fingerprint="candidate-a",
            scene_version=7,
            map_version=2,
        )


def test_unrequested_map_version_is_not_required():
    context = EvidenceRequestContext("candidate-a", scene_version=7)

    assert_evidence_compatible(
        context,
        candidate_fingerprint="candidate-a",
        scene_version=7,
        map_version=99,
    ) is None
