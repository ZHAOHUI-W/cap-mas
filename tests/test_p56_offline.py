from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest

from capmas.contracts.calibration import (
    FEATURE_SCHEMA_VERSION,
    CalibrationDatasetManifest,
    CalibrationLineage,
    CalibrationOutcome,
    CandidateFeatureSnapshot,
    HorizonLabel,
)
from capmas.evaluation.dataset import assign_lineage_splits, build_calibration_dataset
from capmas.evaluation.feature_snapshots import FEATURE_GROUPS_V1
from capmas.evaluation.offline import ExactQuotaSplitConfig, partition_tier_a_outcomes

_POSITIVE_INDICES = frozenset({0, 2, 3, 6, 7, 9, 10, 11, 13, 18})


def _horizon() -> HorizonLabel:
    return HorizonLabel(
        planned_critical_path_actions=2,
        planned_critical_path_subgoals=2,
        planned_checkpoint_subgraphs=1,
        attempted_actions=2,
        completed_actions=2,
        attempted_subgoals=2,
        completed_subgoals=2,
        attempted_checkpoints=1,
        completed_checkpoints=1,
        planned_source="mission_graph",
        realized_source="execution_trace",
        planned_valid=True,
        realized_valid=True,
    )


def _snapshot(index: int, candidate_id: str, *, episode_id: str) -> CandidateFeatureSnapshot:
    return CandidateFeatureSnapshot(
        episode_id=episode_id,
        episode_epoch=1,
        family_id="object-6",
        candidate_id=candidate_id,
        candidate_fingerprint=f"{index + 1:064x}",
        scene_version=1,
        map_version=None,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        captured_at_ns=100 + index,
        collection_lane="physical",
        features={
            name: 1.0 if name == "rehearsal_success_rate" else None
            for name in FEATURE_GROUPS_V1
        },
        feature_status={
            name: "present" if name == "rehearsal_success_rate" else "unknown"
            for name in FEATURE_GROUPS_V1
        },
        correlation_groups=FEATURE_GROUPS_V1,
        memory_skill_version="memory-v1",
        robot_skill_version="robot-v1",
        evidence_refs=(f"rehearsal://{candidate_id}",),
        evidence_providers={"rehearsal": "unit"},
        rewrite_metadata={"changed": False},
    )


def _outcome(index: int, *, family_id: str = "object-6") -> CalibrationOutcome:
    episode_id = f"episode-{index:02d}"
    candidate_id = f"candidate-{index:02d}"
    snapshot = _snapshot(index, candidate_id, episode_id=episode_id)
    if family_id != "object-6":
        snapshot = replace(snapshot, family_id=family_id)
    return CalibrationOutcome(
        episode_id=episode_id,
        family_id=family_id,
        candidate_id=candidate_id,
        candidate_fingerprint=snapshot.candidate_fingerprint,
        tier="A",
        execution_status="selected_executed",
        task_success=index in _POSITIVE_INDICES,
        graph_completed=index in _POSITIVE_INDICES,
        verifier_success=None,
        rehearsal_success=None,
        failure_class=None if index in _POSITIVE_INDICES else "TASK_FAILURE",
        horizon=_horizon(),
        feature_snapshot=snapshot,
        dataset_split="unassigned",
    )


def _lineage(index: int) -> CalibrationLineage:
    episode_id = f"episode-{index:02d}"
    return CalibrationLineage(
        episode_id=episode_id,
        lineage_group_id=f"lineage-{index:02d}",
        seed=index,
        split_identity="native",
        layout_pair_id=None,
        retry_of_episode_id=None,
        candidate_artifact_sha256=f"{index + 1:064x}",
        decision_boundary_ns=200 + index,
        evaluator_observed_at_ns=300 + index,
    )


def _manifest(
    count: int = 20,
    *,
    family_id: str = "object-6",
) -> CalibrationDatasetManifest:
    outcomes = tuple(_outcome(index, family_id=family_id) for index in range(count))
    lineages = tuple(_lineage(index) for index in range(count))
    assignments = assign_lineage_splits(lineages, salt="source-audit-v1")
    return build_calibration_dataset(
        outcomes,
        lineages,
        split_assignments=assignments,
        memory_skill_version="memory-v1",
        robot_skill_version="robot-v1",
        prompt_version="prompt-v1",
        environment_version="libero-v1",
        code_revision="revision-v1",
        split_salt="source-audit-v1",
    )


def test_exact_quota_partition_is_stable_and_preserves_source_outcomes() -> None:
    manifest = _manifest()
    source_splits = tuple(outcome.dataset_split for outcome in manifest.outcomes)
    config = ExactQuotaSplitConfig.object6_v1()

    first = partition_tier_a_outcomes(manifest, config)
    second = partition_tier_a_outcomes(manifest, config)

    assert first == second
    assert Counter(example.dataset_split for example in first) == {
        "train": 12,
        "calibration": 4,
        "test": 4,
    }
    assert tuple(outcome.dataset_split for outcome in manifest.outcomes) == source_splits
    assert {example.lineage_group_id for example in first} == {
        f"lineage-{index:02d}" for index in range(20)
    }


def test_partition_preserves_both_classes_in_fitting_splits() -> None:
    examples = partition_tier_a_outcomes(_manifest(), ExactQuotaSplitConfig.object6_v1())

    for split in ("train", "calibration"):
        labels = {example.outcome.task_success for example in examples if example.dataset_split == split}
        assert labels == {False, True}


def test_partition_rejects_wrong_family_and_quota_mismatch() -> None:
    with pytest.raises(ValueError, match="object-6"):
        partition_tier_a_outcomes(_manifest(family_id="goal-1"), ExactQuotaSplitConfig.object6_v1())
    with pytest.raises(ValueError, match="exactly 20"):
        partition_tier_a_outcomes(_manifest(count=19), ExactQuotaSplitConfig.object6_v1())


def test_partition_rejects_duplicate_selected_tier_a_lineage() -> None:
    manifest = _manifest()
    duplicate_manifest = build_calibration_dataset(
        manifest.outcomes + (manifest.outcomes[0],),
        manifest.lineages,
        split_assignments=assign_lineage_splits(manifest.lineages, salt="source-audit-v1"),
        memory_skill_version="memory-v1",
        robot_skill_version="robot-v1",
        prompt_version="prompt-v1",
        environment_version="libero-v1",
        code_revision="revision-v1",
        split_salt="source-audit-v1",
    )

    with pytest.raises(ValueError, match="multiple Tier A outcomes"):
        partition_tier_a_outcomes(duplicate_manifest, ExactQuotaSplitConfig.object6_v1())


def test_partition_rejects_source_audit_failure_before_split() -> None:
    manifest = _manifest()
    object.__setattr__(manifest.outcomes[0].feature_snapshot, "captured_at_ns", 1000)

    with pytest.raises(ValueError, match="ineligible"):
        partition_tier_a_outcomes(manifest, ExactQuotaSplitConfig.object6_v1())


def test_partition_config_and_examples_are_json_safe() -> None:
    config = ExactQuotaSplitConfig.object6_v1()
    example = partition_tier_a_outcomes(_manifest(), config)[0]

    assert list(config.to_dict()) == sorted(config.to_dict())
    assert example.to_dict()["lineage_group_id"] == example.lineage_group_id
