from __future__ import annotations

import json

import pytest

from capmas.contracts.calibration import FEATURE_SCHEMA_VERSION
from capmas.evaluation.baseline import (
    BASELINE_VERSION,
    FixedWeightBaseline,
    fit_fixed_weight_mapping,
)
from capmas.evaluation.correlation import ReducedDimension, ReducedFeatureVector


def _vector(action_feasibility: float, *, episode_id: str) -> ReducedFeatureVector:
    return ReducedFeatureVector(
        episode_id=episode_id,
        candidate_id="candidate",
        candidate_fingerprint=f"{int(action_feasibility) + 1:064x}",
        family_id="object-6",
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        dimensions=(
            ReducedDimension("scene_grounding", None, "unknown", (), 0.0),
            ReducedDimension("action_feasibility", action_feasibility, "present", ("rehearsal_success_rate",), 1.0),
            ReducedDimension("collision_risk", None, "unknown", (), 0.0),
            ReducedDimension("expected_latency_risk", None, "unknown", (), 0.0),
            ReducedDimension("recovery_cost_risk", None, "unknown", (), 0.0),
        ),
    )


def test_fixed_baseline_is_monotonic_and_json_safe() -> None:
    baseline = FixedWeightBaseline.object6_v1()
    low = baseline.score(_vector(0.0, episode_id="low"))
    high = baseline.score(_vector(1.0, episode_id="high"))

    assert high > low
    assert baseline.baseline_version == BASELINE_VERSION
    payload = baseline.to_dict()
    assert list(payload) == sorted(payload)
    json.dumps(payload)


def test_fixed_mapping_uses_only_supplied_train_and_calibration_rows() -> None:
    baseline = FixedWeightBaseline.object6_v1()
    mapping = fit_fixed_weight_mapping(
        baseline,
        (
            (_vector(0.0, episode_id="train-negative"), False),
            (_vector(0.0, episode_id="calibration-positive"), True),
            (_vector(1.0, episode_id="calibration-positive-2"), True),
        ),
    )

    assert mapping.calibration_version == "p56b.fixed_weight_pava.v1"
    assert len(mapping.blocks) == 2
    assert mapping.blocks[0].probability == pytest.approx(0.5)
    assert mapping.blocks[1].probability == pytest.approx(1.0)


def test_fixed_mapping_rejects_mixed_family_and_single_class() -> None:
    baseline = FixedWeightBaseline.object6_v1()
    vector = _vector(0.0, episode_id="episode")

    with pytest.raises(ValueError, match="both task_success classes"):
        fit_fixed_weight_mapping(baseline, ((vector, False),))

    other = ReducedFeatureVector(
        episode_id=vector.episode_id,
        candidate_id=vector.candidate_id,
        candidate_fingerprint=vector.candidate_fingerprint,
        family_id="goal-1",
        feature_schema_version=vector.feature_schema_version,
        dimensions=vector.dimensions,
    )
    with pytest.raises(ValueError, match="family"):
        fit_fixed_weight_mapping(baseline, ((vector, False), (other, True)))
