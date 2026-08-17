from __future__ import annotations

import math

import pytest

from capmas.contracts.calibration import FEATURE_SCHEMA_VERSION, CandidateFeatureSnapshot
from capmas.evaluation.feature_snapshots import FEATURE_GROUPS_V1


def _snapshot(
    values: dict[str, float | None] | None = None,
    statuses: dict[str, str] | None = None,
) -> CandidateFeatureSnapshot:
    values = values or {}
    features = {name: values.get(name) for name in FEATURE_GROUPS_V1}
    feature_status = {
        name: statuses.get(name, "present" if features[name] is not None else "unknown")
        if statuses is not None
        else ("present" if features[name] is not None else "unknown")
        for name in FEATURE_GROUPS_V1
    }
    return CandidateFeatureSnapshot(
        episode_id="episode-1",
        episode_epoch=1,
        family_id="object-6",
        candidate_id="candidate-a",
        candidate_fingerprint="a" * 64,
        scene_version=4,
        map_version=2,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        captured_at_ns=100,
        collection_lane="physical",
        features=features,
        feature_status=feature_status,  # type: ignore[arg-type]
        correlation_groups=FEATURE_GROUPS_V1,
        memory_skill_version="memory-v1",
        robot_skill_version="robot-v1",
        evidence_refs=("artifact://decision",),
        evidence_providers={"provider": "unit"},
        rewrite_metadata={"changed": False},
    )


def test_reducer_prefers_static_verifier_over_rehearsal() -> None:
    from capmas.evaluation.correlation import reduce_feature_snapshot

    reduced = reduce_feature_snapshot(
        _snapshot(
            {
                "static_verifier_pass_rate": 0.25,
                "static_verifier_coverage": 1.0,
                "rehearsal_success_rate": 1.0,
            }
        )
    )

    feasibility = reduced.dimension("action_feasibility")

    assert feasibility.value == 0.25
    assert feasibility.sources == ("static_verifier_pass_rate",)
    assert feasibility.coverage == 1.0


def test_reducer_falls_back_to_rehearsal_when_static_coverage_is_zero() -> None:
    from capmas.evaluation.correlation import reduce_feature_snapshot

    reduced = reduce_feature_snapshot(
        _snapshot(
            {
                "static_verifier_pass_rate": 0.1,
                "static_verifier_coverage": 0.0,
                "rehearsal_success_rate": 0.8,
            }
        )
    )

    feasibility = reduced.dimension("action_feasibility")

    assert feasibility.value == 0.8
    assert feasibility.sources == ("rehearsal_success_rate",)


def test_reducer_aggregates_scene_support_and_preserves_unknown_groups() -> None:
    from capmas.evaluation.correlation import reduce_feature_snapshot

    reduced = reduce_feature_snapshot(
        _snapshot(
            {
                "scene_confidence": 0.2,
                "target_visibility": 0.8,
                "rehearsal_success_rate": 1.0,
            }
        )
    )

    scene = reduced.dimension("scene_grounding")
    collision = reduced.dimension("collision_risk")

    assert scene.value == 0.5
    assert scene.coverage == pytest.approx(2 / 9)
    assert scene.sources == ("scene_confidence", "target_visibility")
    assert collision.value is None
    assert collision.status == "unknown"
    assert collision.coverage == 0.0


def test_reducer_transforms_cost_dimensions_without_creating_ood_feature() -> None:
    from capmas.evaluation.correlation import reduce_feature_snapshot

    reduced = reduce_feature_snapshot(
        _snapshot(
            {
                "rehearsal_success_rate": 1.0,
                "collision_risk": 0.4,
                "expected_latency_ms": 1000.0,
                "recovery_cost": 2.0,
            }
        )
    )

    assert reduced.dimension("collision_risk").value == 0.4
    assert reduced.dimension("expected_latency_risk").value == pytest.approx(1.0 - math.exp(-1.0))
    assert reduced.dimension("recovery_cost_risk").value == pytest.approx(1.0 - math.exp(-2.0))
    assert "ood" not in {dimension.name for dimension in reduced.dimensions}


def test_reducer_keeps_missing_required_feasibility_explicit() -> None:
    from capmas.evaluation.correlation import reduce_feature_snapshot

    feasibility = reduce_feature_snapshot(_snapshot()).dimension("action_feasibility")

    assert feasibility.value is None
    assert feasibility.status == "unknown"
    assert feasibility.sources == ()


@pytest.mark.parametrize(
    ("values", "statuses", "message"),
    [
        (
            {"rehearsal_success_rate": 1.2},
            None,
            "rehearsal_success_rate",
        ),
        (
            {"rehearsal_success_rate": 1.0},
            {"rehearsal_success_rate": "invalid"},
            "invalid",
        ),
        (
            {"rehearsal_success_rate": 1.0, "expected_latency_ms": -1.0},
            None,
            "expected_latency_ms",
        ),
    ],
)
def test_reducer_rejects_invalid_decision_time_evidence(
    values: dict[str, float | None],
    statuses: dict[str, str] | None,
    message: str,
) -> None:
    from capmas.evaluation.correlation import reduce_feature_snapshot

    with pytest.raises(ValueError, match=message):
        reduce_feature_snapshot(_snapshot(values, statuses))


def test_reduction_serialization_is_sorted_and_json_safe() -> None:
    from capmas.evaluation.correlation import reduce_feature_snapshot

    payload = reduce_feature_snapshot(_snapshot({"rehearsal_success_rate": 1.0})).to_dict()

    assert list(payload) == sorted(payload)
    assert payload["dimensions"][0]["name"] == "scene_grounding"
