import json
from dataclasses import replace
from types import MappingProxyType

import pytest

from capmas.contracts.calibration import (
    CalibrationCollectionContext,
    CalibrationOutcome,
    CalibrationPrediction,
    CandidateFeatureSnapshot,
    HorizonLabel,
    horizon_bucket,
)


def _horizon() -> HorizonLabel:
    return HorizonLabel(
        planned_critical_path_actions=2,
        planned_critical_path_subgoals=2,
        planned_checkpoint_subgraphs=1,
        attempted_actions=2,
        completed_actions=2,
        attempted_subgoals=2,
        completed_subgoals=2,
        attempted_checkpoints=3,
        completed_checkpoints=3,
        planned_source="mission_graph",
        realized_source="execution_trace",
        planned_valid=True,
        realized_valid=True,
    )


def _snapshot() -> CandidateFeatureSnapshot:
    return CandidateFeatureSnapshot(
        episode_id="object-6-seed11",
        episode_epoch=1,
        family_id="object-6",
        candidate_id="candidate-a",
        candidate_fingerprint="a" * 64,
        scene_version=1,
        map_version=None,
        feature_schema_version="p56.feature.v1",
        captured_at_ns=10,
        collection_lane="physical",
        features={"scene_freshness": None, "rehearsal_success_rate": 1.0},
        feature_status={"scene_freshness": "unknown", "rehearsal_success_rate": "present"},
        correlation_groups={
            "scene_freshness": "scene_grounding",
            "rehearsal_success_rate": "action_feasibility",
        },
        memory_skill_version="memory-frozen-v1",
        robot_skill_version="robot-frozen-v1",
        evidence_refs=("rehearsal://candidate-a/11",),
        evidence_providers={"rehearsal": "libero_process_rehearsal"},
        rewrite_metadata={"changed": False},
    )


def _outcome(**overrides: object) -> CalibrationOutcome:
    values: dict[str, object] = {
        "episode_id": "object-6-seed11",
        "family_id": "object-6",
        "candidate_id": "candidate-a",
        "candidate_fingerprint": "a" * 64,
        "tier": "A",
        "execution_status": "selected_executed",
        "task_success": True,
        "graph_completed": True,
        "verifier_success": True,
        "rehearsal_success": None,
        "failure_class": None,
        "horizon": _horizon(),
        "feature_snapshot": _snapshot(),
        "dataset_split": "unassigned",
    }
    values.update(overrides)
    return CalibrationOutcome(**values)  # type: ignore[arg-type]


def test_contracts_round_trip_without_converting_unknown_to_zero() -> None:
    outcome = _outcome()

    restored = CalibrationOutcome.from_dict(outcome.to_dict())

    assert restored == outcome
    assert restored.feature_snapshot.features["scene_freshness"] is None
    assert horizon_bucket(restored.horizon) == "H2-3"


def test_horizon_rejects_completed_count_above_attempted_count() -> None:
    with pytest.raises(ValueError, match="completed_actions"):
        replace(_horizon(), attempted_actions=1, completed_actions=2)


def test_tier_a_requires_selected_execution_and_physical_label() -> None:
    with pytest.raises(ValueError, match="Tier A"):
        _outcome(execution_status="not_selected", task_success=False)


def test_tier_b_uses_rehearsal_label_and_never_physical_label() -> None:
    with pytest.raises(ValueError, match="Tier B"):
        _outcome(
            tier="B",
            execution_status="not_selected",
            task_success=True,
            rehearsal_success=True,
            feature_snapshot=replace(_snapshot(), collection_lane="rehearsal"),
            dataset_split="shadow",
        )


def test_contracts_are_immutable_and_mappings_are_deeply_normalized() -> None:
    raw_features = {"nested": None}
    raw_status = {"nested": "unknown"}
    raw_groups = {"nested": "scene_grounding"}
    raw_metadata = {"nested": {"marker": "value"}}
    snapshot = replace(
        _snapshot(),
        features=raw_features,
        feature_status=raw_status,
        correlation_groups=raw_groups,
        rewrite_metadata=raw_metadata,
    )

    raw_features["new"] = 1.0
    raw_metadata["nested"]["new"] = True

    assert isinstance(snapshot.features, MappingProxyType)
    assert isinstance(snapshot.rewrite_metadata, MappingProxyType)
    assert "new" not in snapshot.features
    assert snapshot.rewrite_metadata["nested"] == {"marker": "value"}
    with pytest.raises(TypeError):
        snapshot.features["nested"] = 1.0  # type: ignore[index]
    with pytest.raises(AttributeError):
        snapshot.candidate_id = "other"  # type: ignore[misc]


def test_nested_metadata_sequences_are_immutable_and_json_safe() -> None:
    snapshot = replace(
        _snapshot(),
        rewrite_metadata={"refs": [{"name": "one"}, "two"]},
    )

    assert snapshot.rewrite_metadata["refs"] == (
        {"name": "one"},
        "two",
    )
    with pytest.raises(TypeError):
        snapshot.rewrite_metadata["refs"][0]["name"] = "other"  # type: ignore[index]
    with pytest.raises(ValueError, match="JSON-safe"):
        replace(_snapshot(), rewrite_metadata={"object": object()})
    with pytest.raises(ValueError, match="secret"):
        replace(_snapshot(), rewrite_metadata={"api_key": "must-not-be-recorded"})


def test_snapshot_evidence_providers_are_immutable_string_mappings_without_secrets() -> None:
    raw_providers = {"rehearsal": "libero_process_rehearsal"}
    snapshot = replace(_snapshot(), evidence_providers=raw_providers)

    raw_providers["physical"] = "robot_executor"

    assert isinstance(snapshot.evidence_providers, MappingProxyType)
    assert snapshot.evidence_providers == {"rehearsal": "libero_process_rehearsal"}
    assert json.dumps(snapshot.to_dict())
    with pytest.raises(TypeError):
        snapshot.evidence_providers["physical"] = "robot_executor"  # type: ignore[index]
    with pytest.raises(TypeError, match="evidence_providers"):
        replace(_snapshot(), evidence_providers={"rehearsal": 1})
    with pytest.raises(ValueError, match="secret"):
        replace(_snapshot(), evidence_providers={"api_key": "must-not-be-recorded"})


def test_public_contracts_are_json_safe_and_sorted() -> None:
    payloads = [
        _horizon().to_dict(),
        _snapshot().to_dict(),
        _outcome().to_dict(),
        CalibrationPrediction(
            candidate_id="candidate-a",
            rank_score=None,
            success_probability=None,
            uncertainty=1.0,
            abstained=True,
            reason="unknown",
            model_version="model-v1",
            feature_schema_version="p56.feature.v1",
            snapshot_id=None,
            eligible_family=False,
        ).to_dict(),
        CalibrationCollectionContext(
            episode_id="episode",
            episode_epoch=1,
            family_id="family",
            feature_schema_version="p56.feature.v1",
            memory_skill_version="memory-v1",
            robot_skill_version="robot-v1",
        ).to_dict(),
    ]
    for payload in payloads:
        encoded = json.dumps(payload)
        assert "MappingProxyType" not in encoded
        assert list(payload) == sorted(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("candidate_fingerprint", "not-a-sha256"),
        ("scene_version", -1),
        ("captured_at_ns", -1),
        ("feature_schema_version", ""),
        ("episode_id", ""),
    ],
)
def test_snapshot_rejects_invalid_identity_version_and_fingerprint(
    field: str, value: object
) -> None:
    with pytest.raises(ValueError):
        replace(_snapshot(), **{field: value})


def test_snapshot_rejects_nonfinite_features_and_mismatched_keys() -> None:
    with pytest.raises(ValueError, match="finite"):
        replace(
            _snapshot(), features={"scene_freshness": float("nan"), "rehearsal_success_rate": 1.0}
        )
    with pytest.raises(ValueError, match="keys"):
        replace(_snapshot(), feature_status={"scene_freshness": "unknown"})


def test_strict_decoders_reject_missing_and_extra_fields() -> None:
    encoded = _outcome().to_dict()
    missing = dict(encoded)
    del missing["tier"]
    extra = dict(encoded, unexpected=True)

    with pytest.raises(ValueError, match="tier"):
        CalibrationOutcome.from_dict(missing)
    with pytest.raises(ValueError, match="unexpected"):
        CalibrationOutcome.from_dict(extra)


def test_horizon_bucket_preserves_unknown_planned_label() -> None:
    assert horizon_bucket(replace(_horizon(), planned_critical_path_subgoals=None)) == "N/A"


def test_horizon_bucket_rejects_zero_action_bearing_subgoals_as_h1() -> None:
    assert horizon_bucket(replace(_horizon(), planned_critical_path_subgoals=0)) == "N/A"


def test_snapshot_present_feature_requires_a_finite_value() -> None:
    with pytest.raises(ValueError, match="present feature"):
        replace(
            _snapshot(),
            feature_status={
                "scene_freshness": "present",
                "rehearsal_success_rate": "present",
            },
        )


def test_prediction_uncertainty_is_normalized() -> None:
    with pytest.raises(ValueError, match="uncertainty"):
        CalibrationPrediction(
            candidate_id="candidate-a",
            rank_score=None,
            success_probability=None,
            uncertainty=1.1,
            abstained=True,
            reason="insufficient_data",
            model_version="model-v1",
            feature_schema_version="p56.feature.v1",
            snapshot_id=None,
            eligible_family=False,
        )


def test_abstained_prediction_cannot_publish_score_or_probability() -> None:
    with pytest.raises(ValueError, match="abstained"):
        CalibrationPrediction(
            candidate_id="candidate-a",
            rank_score=0.5,
            success_probability=0.5,
            uncertainty=0.5,
            abstained=True,
            reason="low_margin",
            model_version="model-v1",
            feature_schema_version="p56.feature.v1",
            snapshot_id="snapshot-v1",
            eligible_family=True,
        )


@pytest.mark.parametrize("snapshot_id", [{}, []])
def test_prediction_rejects_non_string_snapshot_id(snapshot_id: object) -> None:
    with pytest.raises(ValueError, match="snapshot_id"):
        CalibrationPrediction(
            candidate_id="candidate-a",
            rank_score=None,
            success_probability=None,
            uncertainty=1.0,
            abstained=True,
            reason="unknown",
            model_version="model-v1",
            feature_schema_version="p56.feature.v1",
            snapshot_id=snapshot_id,  # type: ignore[arg-type]
            eligible_family=False,
        )


@pytest.mark.parametrize("failure_class", [{}, []])
def test_outcome_rejects_non_string_failure_class(failure_class: object) -> None:
    with pytest.raises(ValueError, match="failure_class"):
        _outcome(failure_class=failure_class)
