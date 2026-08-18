from __future__ import annotations

import hashlib
import json

import pytest

from capmas.contracts.calibration import FEATURE_SCHEMA_VERSION
from capmas.evaluation.calibration_diagnostics import analyze_train_design
from capmas.evaluation.correlation import ReducedDimension, ReducedFeatureVector


def _vector(
    candidate_id: str,
    *,
    feasibility: float,
    scene: float | None = None,
    collision: float | None = None,
    family_id: str = "object-6",
    feature_schema_version: str = FEATURE_SCHEMA_VERSION,
) -> ReducedFeatureVector:
    def dimension(
        name: str,
        value: float | None,
        source: str,
    ) -> ReducedDimension:
        return ReducedDimension(
            name=name,
            value=value,
            status="present" if value is not None else "unknown",
            sources=(source,) if value is not None else (),
            coverage=1.0 if value is not None else 0.0,
        )

    return ReducedFeatureVector(
        episode_id=f"episode-{candidate_id}",
        candidate_id=candidate_id,
        candidate_fingerprint=hashlib.sha256(candidate_id.encode()).hexdigest(),
        family_id=family_id,
        feature_schema_version=feature_schema_version,
        dimensions=(
            dimension("scene_grounding", scene, "scene_confidence"),
            dimension("action_feasibility", feasibility, "rehearsal_success_rate"),
            dimension("collision_risk", collision, "collision_risk"),
            dimension("expected_latency_risk", None, "expected_latency_ms"),
            dimension("recovery_cost_risk", None, "recovery_cost"),
        ),
    )


def _train_vectors() -> tuple[ReducedFeatureVector, ...]:
    return tuple(
        _vector(f"candidate-{index}", feasibility=value)
        for index, value in enumerate((0.0, 0.1, 0.2, 0.8, 0.9, 1.0))
    )


def test_design_diagnostics_freeze_constant_unknown_columns() -> None:
    diagnostics = analyze_train_design(_train_vectors())

    assert diagnostics.row_count == 6
    assert diagnostics.column_count == 10
    assert diagnostics.active_parameter_count == 2
    assert diagnostics.matrix_rank == 2
    assert diagnostics.availability_signature["scene_grounding"] == "all_unknown"
    assert diagnostics.availability_signature["action_feasibility"] == "all_present"
    frozen = set(diagnostics.frozen_parameters)
    assert "support.scene_grounding" in frozen
    assert "missing.scene_grounding" in frozen
    assert "risk.collision_risk" in frozen
    assert "missing.collision_risk" in frozen


def test_design_diagnostics_preserve_mixed_availability() -> None:
    diagnostics = analyze_train_design(
        (
            _vector("candidate-a", feasibility=0.1),
            _vector("candidate-b", feasibility=0.2, scene=0.3),
            _vector("candidate-c", feasibility=0.8),
            _vector("candidate-d", feasibility=0.9, scene=0.7),
        )
    )

    assert diagnostics.availability_signature["scene_grounding"] == "mixed"
    assert "support.scene_grounding" not in diagnostics.frozen_parameters
    assert "missing.scene_grounding" not in diagnostics.frozen_parameters


def test_design_diagnostics_are_deterministic_and_json_safe() -> None:
    first = analyze_train_design(_train_vectors())
    second = analyze_train_design(_train_vectors())

    assert first == second
    payload = first.to_dict()
    assert list(payload) == sorted(payload)
    assert json.loads(json.dumps(payload))["rank_tolerance"] == pytest.approx(1e-12)


def test_design_diagnostics_reject_empty_mixed_family_and_mixed_schema() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        analyze_train_design(())

    with pytest.raises(ValueError, match="family"):
        analyze_train_design(
            (
                _vector("candidate-a", feasibility=0.1),
                _vector("candidate-b", feasibility=0.9, family_id="goal-1"),
            )
        )

    with pytest.raises(ValueError, match="schema"):
        analyze_train_design(
            (
                _vector("candidate-a", feasibility=0.1),
                _vector("candidate-b", feasibility=0.9, feature_schema_version="other.v1"),
            )
        )
