from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import pytest

from capmas.contracts.calibration import (
    FEATURE_SCHEMA_VERSION,
    CalibrationOutcome,
    CandidateFeatureSnapshot,
    HorizonLabel,
)
from capmas.evaluation.calibration import (
    MAX_ITERATIONS,
    brier_score,
    expected_calibration_error,
    fit_constrained_logistic,
    fit_isotonic,
    predict_offline,
    wilson_interval_width,
)
from capmas.evaluation.correlation import ReducedDimension, ReducedFeatureVector
from capmas.evaluation.feature_snapshots import FEATURE_GROUPS_V1
from capmas.evaluation.offline import OfflineExample


def _horizon() -> HorizonLabel:
    return HorizonLabel(
        planned_critical_path_actions=1,
        planned_critical_path_subgoals=1,
        planned_checkpoint_subgraphs=0,
        attempted_actions=1,
        completed_actions=1,
        attempted_subgoals=1,
        completed_subgoals=1,
        attempted_checkpoints=0,
        completed_checkpoints=0,
        planned_source="mission_graph",
        realized_source="execution_trace",
        planned_valid=True,
        realized_valid=True,
    )


def _vector(
    candidate_id: str,
    *,
    feasibility: float | None,
    scene: float | None = None,
    collision: float | None = None,
) -> ReducedFeatureVector:
    def dimension(name: str, value: float | None, sources: tuple[str, ...]) -> ReducedDimension:
        return ReducedDimension(
            name=name,
            value=value,
            status="present" if value is not None else "unknown",
            sources=sources if value is not None else (),
            coverage=1.0 if value is not None else 0.0,
        )

    return ReducedFeatureVector(
        episode_id=f"episode-{candidate_id}",
        candidate_id=candidate_id,
        candidate_fingerprint=hashlib.sha256(candidate_id.encode()).hexdigest(),
        family_id="object-6",
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        dimensions=(
            dimension("scene_grounding", scene, ("scene_confidence",)),
            dimension("action_feasibility", feasibility, ("rehearsal_success_rate",)),
            dimension("collision_risk", collision, ("collision_risk",)),
            dimension("expected_latency_risk", None, ("expected_latency_ms",)),
            dimension("recovery_cost_risk", None, ("recovery_cost",)),
        ),
    )


def _example(
    split: str,
    label: bool,
    feasibility: float | None,
    *,
    candidate_id: str,
    scene: float | None = None,
    collision: float | None = None,
) -> OfflineExample:
    vector = _vector(candidate_id, feasibility=feasibility, scene=scene, collision=collision)
    snapshot = CandidateFeatureSnapshot(
        episode_id=vector.episode_id,
        episode_epoch=1,
        family_id="object-6",
        candidate_id=candidate_id,
        candidate_fingerprint=vector.candidate_fingerprint,
        scene_version=1,
        map_version=None,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        captured_at_ns=1,
        collection_lane="physical",
        features={
            name: feasibility if name == "rehearsal_success_rate" else None
            for name in FEATURE_GROUPS_V1
        },
        feature_status={
            name: "present" if name == "rehearsal_success_rate" and feasibility is not None else "unknown"
            for name in FEATURE_GROUPS_V1
        },
        correlation_groups=FEATURE_GROUPS_V1,
        memory_skill_version="memory-v1",
        robot_skill_version="robot-v1",
        evidence_refs=("artifact://unit",),
        evidence_providers={"unit": "test"},
        rewrite_metadata={"changed": False},
    )
    outcome = CalibrationOutcome(
        episode_id=vector.episode_id,
        family_id="object-6",
        candidate_id=candidate_id,
        candidate_fingerprint=vector.candidate_fingerprint,
        tier="A",
        execution_status="selected_executed",
        task_success=label,
        graph_completed=label,
        verifier_success=None,
        rehearsal_success=None,
        failure_class=None if label else "TASK_FAILURE",
        horizon=_horizon(),
        feature_snapshot=snapshot,
        dataset_split="train",
    )
    return OfflineExample(
        outcome=outcome,
        dataset_split=split,  # type: ignore[arg-type]
        lineage_group_id=f"lineage-{candidate_id}",
        reduced=vector,
    )


def _train_examples() -> tuple[OfflineExample, ...]:
    return tuple(
        _example("train", label, value, candidate_id=f"{index:x}")
        for index, (label, value) in enumerate(
            (
                (False, 0.0),
                (False, 0.1),
                (False, 0.2),
                (True, 0.8),
                (True, 0.9),
                (True, 1.0),
            )
        )
    )


def test_projected_fit_stops_at_the_configured_iteration_budget() -> None:
    fitted = fit_constrained_logistic(_train_examples())

    assert MAX_ITERATIONS == 5_000
    assert fitted.iterations == MAX_ITERATIONS
    assert fitted.converged is False


def test_projected_fit_is_deterministic_and_keeps_constrained_weights_nonnegative() -> None:
    first = fit_constrained_logistic(_train_examples())
    second = fit_constrained_logistic(_train_examples())

    assert first == second
    assert all(weight >= 0.0 for weight in first.support_weights.values())
    assert all(weight >= 0.0 for weight in first.risk_weights.values())
    assert all(weight >= 0.0 for weight in first.missing_penalties.values())
    assert first.raw_probability(_train_examples()[-1].reduced) > first.raw_probability(
        _train_examples()[0].reduced
    )
    assert "ood" not in set(first.support_weights) | set(first.risk_weights)


@pytest.mark.parametrize(
    "examples",
    [
        (_example("calibration", True, 1.0, candidate_id="a"),),
        (_example("train", True, 1.0, candidate_id="a"),) * 2,
        (_example("train", True, None, candidate_id="a"), _example("train", False, 0.0, candidate_id="b")),
    ],
)
def test_fit_rejects_wrong_split_class_support_and_missing_required_evidence(
    examples: tuple[OfflineExample, ...],
) -> None:
    with pytest.raises(ValueError):
        fit_constrained_logistic(examples)


@dataclass(frozen=True)
class _FixedScoreModel:
    scores: dict[str, float]
    model_version: str = "fixed-v1"
    family_id: str = "object-6"
    feature_schema_version: str = FEATURE_SCHEMA_VERSION

    def raw_probability(self, vector: ReducedFeatureVector) -> float:
        return self.scores[vector.candidate_id]


def test_pava_merges_descending_empirical_rates_and_reports_wilson_width() -> None:
    calibration_examples = (
        _example("calibration", True, 0.1, candidate_id="a"),
        _example("calibration", False, 0.2, candidate_id="b"),
        _example("calibration", True, 0.3, candidate_id="c"),
    )
    calibration = fit_isotonic(
        _FixedScoreModel({"a": 0.1, "b": 0.2, "c": 0.3}),  # type: ignore[arg-type]
        calibration_examples,
    )

    assert [(block.lower, block.upper, block.probability) for block in calibration.blocks] == [
        (0.1, 0.2, 0.5),
        (0.3, 0.3, 1.0),
    ]
    assert all(0.0 <= block.uncertainty <= 1.0 for block in calibration.blocks)


def test_prediction_abstains_for_missing_required_evidence_and_never_qualifies_runtime() -> None:
    model = fit_constrained_logistic(_train_examples())
    calibration = fit_isotonic(
        model,
        (
            _example("calibration", False, 0.2, candidate_id="d"),
            _example("calibration", True, 0.8, candidate_id="e"),
        ),
    )

    abstained = predict_offline(
        model,
        calibration,
        _example("test", False, None, candidate_id="f"),
    )
    predicted = predict_offline(
        model,
        calibration,
        _example("test", True, 0.9, candidate_id="g"),
    )

    assert abstained.abstained is True
    assert abstained.success_probability is None
    assert abstained.reason == "prediction_abstained_missing_required_evidence"
    assert predicted.abstained is False
    assert predicted.eligible_family is False
    assert predicted.snapshot_id is None


def test_metrics_and_contract_serialization_are_deterministic() -> None:
    assert brier_score(((0.1, False), (0.9, True))) == pytest.approx(0.01)
    assert expected_calibration_error(((0.1, False), (0.9, True))) == pytest.approx(0.1)
    assert wilson_interval_width(2, 4) == pytest.approx(wilson_interval_width(2, 4))

    model = fit_constrained_logistic(_train_examples())
    payload = model.to_dict()

    assert list(payload) == sorted(payload)
    assert json.loads(json.dumps(payload))["model_version"] == model.model_version
    assert math.isfinite(model.final_loss)
