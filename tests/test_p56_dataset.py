from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

import capmas.evaluation.dataset as dataset_module
from capmas.contracts.calibration import (
    DATASET_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION,
    CalibrationDatasetManifest,
    CalibrationLineage,
    CalibrationOutcome,
    CandidateFeatureSnapshot,
    HorizonLabel,
)
from capmas.evaluation.dataset import (
    DatasetAudit,
    LeakageFinding,
    assert_dataset_eligible,
    assign_lineage_splits,
    audit_calibration_dataset,
    build_calibration_dataset,
    normalize_physical_outcomes,
)
from capmas.evaluation.feature_snapshots import FEATURE_GROUPS_V1


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


def _snapshot(
    candidate_id: str = "candidate",
    *,
    episode_id: str = "episode",
    fingerprint: str | None = None,
    captured_at_ns: int = 100,
    lane: str = "physical",
    feature_name: str | None = None,
    feature_value: float | None = None,
    memory_skill_version: str = "memory-v1",
    robot_skill_version: str = "robot-v1",
) -> CandidateFeatureSnapshot:
    return CandidateFeatureSnapshot(
        episode_id=episode_id,
        episode_epoch=1,
        family_id="object-6",
        candidate_id=candidate_id,
        candidate_fingerprint=fingerprint or (candidate_id[0] * 64),
        scene_version=1,
        map_version=1,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        captured_at_ns=captured_at_ns,
        collection_lane=lane,  # type: ignore[arg-type]
        features={
            name: feature_value if name == feature_name else None
            for name in FEATURE_GROUPS_V1
        },
        feature_status={
            name: "present" if name == feature_name and feature_value is not None else "unknown"
            for name in FEATURE_GROUPS_V1
        },
        correlation_groups=FEATURE_GROUPS_V1,
        memory_skill_version=memory_skill_version,
        robot_skill_version=robot_skill_version,
        evidence_refs=("artifact://decision-evidence",),
        evidence_providers={"evidence": "provider-v1"},
        rewrite_metadata={"raw_fingerprint": fingerprint or (candidate_id[0] * 64)},
    )


def _lineage(
    episode_id: str,
    *,
    group: str | None = None,
    decision_boundary_ns: int = 100,
    fingerprint: str = "a" * 64,
) -> CalibrationLineage:
    return CalibrationLineage(
        episode_id=episode_id,
        lineage_group_id=group or episode_id,
        seed=1,
        split_identity="native",
        layout_pair_id=None,
        retry_of_episode_id=None,
        candidate_artifact_sha256=fingerprint,
        decision_boundary_ns=decision_boundary_ns,
        evaluator_observed_at_ns=200,
    )


def _outcome(
    *,
    episode_id: str = "episode",
    candidate_id: str = "candidate",
    fingerprint: str = "a" * 64,
    snapshot_captured_at_ns: int = 100,
    dataset_split: str = "train",
) -> CalibrationOutcome:
    snapshot = _snapshot(
        candidate_id,
        episode_id=episode_id,
        fingerprint=fingerprint,
        captured_at_ns=snapshot_captured_at_ns,
    )
    return CalibrationOutcome(
        episode_id=episode_id,
        family_id="object-6",
        candidate_id=candidate_id,
        candidate_fingerprint=fingerprint,
        tier="A",
        execution_status="selected_executed",
        task_success=True,
        graph_completed=True,
        verifier_success=True,
        rehearsal_success=None,
        failure_class=None,
        horizon=_horizon(),
        feature_snapshot=snapshot,
        dataset_split=dataset_split,  # type: ignore[arg-type]
    )


def _manifest(
    outcomes: tuple[CalibrationOutcome, ...],
    lineages: tuple[CalibrationLineage, ...],
) -> CalibrationDatasetManifest:
    return build_calibration_dataset(
        outcomes,
        lineages,
        split_assignments=assign_lineage_splits(lineages, salt="frozen-split-v1"),
        memory_skill_version="memory-v1",
        robot_skill_version="robot-v1",
        prompt_version="prompt-v1",
        environment_version="libero-v1",
        code_revision="abc123",
        split_salt="frozen-split-v1",
    )


def _codes(manifest: CalibrationDatasetManifest) -> set[str]:
    return {finding.code for finding in audit_calibration_dataset(manifest).findings}


def test_selected_candidate_is_tier_a_and_unselected_candidates_are_tier_c() -> None:
    outcomes = normalize_physical_outcomes(
        snapshots=(_snapshot("a"), _snapshot("b")),
        selected_candidate_id="a",
        execution_started=True,
        task_success=False,
        graph_completed=True,
        verifier_success=True,
        failure_class="task_failure",
        horizon=_horizon(),
    )

    assert [(row.candidate_id, row.tier, row.task_success) for row in outcomes] == [
        ("a", "A", False),
        ("b", "C", None),
    ]


def test_selected_not_started_has_no_physical_label() -> None:
    (outcome,) = normalize_physical_outcomes(
        snapshots=(_snapshot("a"),),
        selected_candidate_id="a",
        execution_started=False,
        task_success=False,
        graph_completed=False,
        verifier_success=False,
        failure_class="executor_not_started",
        horizon=_horizon(),
    )

    assert outcome.execution_status == "selected_not_started"
    assert outcome.tier == "C"
    assert (outcome.task_success, outcome.graph_completed, outcome.verifier_success) == (
        None,
        None,
        None,
    )


def test_inconclusive_executed_candidate_is_tier_c_without_physical_labels() -> None:
    (outcome,) = normalize_physical_outcomes(
        snapshots=(_snapshot("a"),),
        selected_candidate_id="a",
        execution_started=True,
        task_success=None,
        graph_completed=False,
        verifier_success=False,
        failure_class="evaluator_unknown",
        horizon=_horizon(),
    )

    assert outcome.execution_status == "selected_executed"
    assert outcome.tier == "C"
    assert (
        outcome.task_success,
        outcome.graph_completed,
        outcome.verifier_success,
        outcome.failure_class,
    ) == (None, None, None, None)


@pytest.mark.parametrize(
    ("code", "status"),
    [
        ("STALE_SCENE", "stale"),
        ("STALE_EVIDENCE", "stale"),
        ("GEOMETRY_GATE", "rejected_safety"),
        ("PERCEPTION_GATE", "rejected_safety"),
        ("GRAPH_SCHEMA_INVALID", "rejected_schema"),
        ("JSON_INVALID", "rejected_schema"),
        ("JSON_NOT_OBJECT", "rejected_schema"),
        ("STRUCTURED_PAYLOAD_INVALID", "rejected_schema"),
        ("REQUEST_ID_MISMATCH", "rejected_schema"),
        ("MISSION_ID_MISMATCH", "rejected_schema"),
        ("EMPTY_RESPONSE", "rejected_schema"),
        ("SUBGRAPH_ID_MISMATCH", "rejected_schema"),
        ("SUBGOAL_ID_MISMATCH", "rejected_schema"),
        ("TOPOLOGY_SCHEMA_INVALID", "rejected_schema"),
        ("SUBGRAPH_SCHEMA_INVALID", "rejected_schema"),
        ("UNREACHABLE_SUBGRAPH", "rejected_schema"),
        ("DUPLICATE_NODE", "rejected_schema"),
        ("DANGLING_EDGE", "rejected_schema"),
        ("PORT_TYPE_MISMATCH", "rejected_schema"),
        ("ACTION_WITHOUT_SKILL", "rejected_schema"),
        ("UNBOUND_INPUT", "rejected_schema"),
        ("PARALLEL_RESOURCE_CONFLICT", "rejected_schema"),
        ("UNESTABLISHED_PRECONDITION", "rejected_schema"),
        ("UNBOUNDED_CYCLE", "rejected_schema"),
        ("UNRECOGNIZED_REJECTION", "not_selected"),
        ("DANGLING_REVIEWER_UNKNOWN", "not_selected"),
        ("FUTURE_SCHEMA_INVALID", "not_selected"),
        ("FUTURE_GEOMETRY_GATE", "not_selected"),
        ("REVIEWER_SAFETY_GATE", "not_selected"),
    ],
)
def test_rejection_codes_map_to_unlabeled_statuses(code: str, status: str) -> None:
    (outcome,) = normalize_physical_outcomes(
        snapshots=(_snapshot("a"),),
        selected_candidate_id=None,
        execution_started=False,
        task_success=None,
        graph_completed=None,
        verifier_success=None,
        failure_class=None,
        horizon=_horizon(),
        rejection_codes={"a": code},
    )

    assert (outcome.execution_status, outcome.tier, outcome.task_success) == (status, "C", None)


def test_decoder_schema_rejection_code_set_covers_staged_decoder_contract() -> None:
    staged_source = Path("capmas/llm/staged_decoder.py").read_text(encoding="utf-8")
    staged_schema_codes = set(
        re.findall(r'StagedDecodeRejection\(\s*"([A-Z0-9_]*SCHEMA_INVALID)"', staged_source)
    )

    assert staged_schema_codes == {
        "TOPOLOGY_SCHEMA_INVALID",
        "SUBGRAPH_SCHEMA_INVALID",
    }
    assert staged_schema_codes <= dataset_module._DECODER_SCHEMA_REJECTION_CODES


def test_safety_rejection_code_set_is_exact_arbiter_contract() -> None:
    assert dataset_module._SAFETY_REJECTION_CODES == {
        "GEOMETRY_GATE",
        "PERCEPTION_GATE",
        "SAFETY_GATE",
        "MISSING_EVIDENCE",
    }


def test_graph_validator_rejection_code_set_covers_validator_contract() -> None:
    validator_source = Path("capmas/graph/validator.py").read_text(encoding="utf-8")
    validator_codes = set(
        re.findall(r'GraphDiagnostic\(\s*"([A-Z0-9_]+)"', validator_source)
    )

    assert validator_codes
    assert validator_codes == dataset_module._GRAPH_VALIDATION_REJECTION_CODES


def test_only_rehearsal_lane_with_conclusive_label_becomes_tier_b() -> None:
    outcomes = normalize_physical_outcomes(
        snapshots=(_snapshot("a", lane="rehearsal"), _snapshot("b", lane="shadow")),
        selected_candidate_id=None,
        execution_started=False,
        task_success=None,
        graph_completed=None,
        verifier_success=None,
        failure_class=None,
        horizon=_horizon(),
        rehearsal_labels={"a": False, "b": True},
    )

    assert [(row.candidate_id, row.tier, row.rehearsal_success) for row in outcomes] == [
        ("a", "B", False),
        ("b", "C", None),
    ]


def test_related_id_ood_and_retry_lineage_stays_in_one_split() -> None:
    lineages = (
        _lineage("id-1", group="pair-1"),
        _lineage("ood-1", group="pair-1"),
        _lineage("retry-1", group="pair-1"),
    )

    assignments = assign_lineage_splits(lineages, salt="frozen-split-v1")

    assert len(set(assignments.values())) == 1


@pytest.mark.parametrize(
    ("train_fraction", "calibration_fraction"),
    [(0.0, 0.2), (1.0, 0.2), (0.6, 0.0), (0.6, 1.0), (0.8, 0.2)],
)
def test_split_ratios_fail_closed(
    train_fraction: float, calibration_fraction: float
) -> None:
    with pytest.raises(ValueError, match="fractions"):
        assign_lineage_splits(
            (_lineage("episode"),),
            salt="frozen-split-v1",
            train_fraction=train_fraction,
            calibration_fraction=calibration_fraction,
        )


def test_build_rejects_split_assignments_that_differ_from_default_lineage_split() -> None:
    lineage = _lineage("episode")
    expected = assign_lineage_splits((lineage,), salt="frozen-split-v1")
    conflicting_split = next(split for split in ("train", "calibration", "test") if split != expected["episode"])

    with pytest.raises(ValueError, match="default lineage split"):
        build_calibration_dataset(
            (_outcome(dataset_split="unassigned"),),
            (lineage,),
            split_assignments={"episode": conflicting_split},
            memory_skill_version="memory-v1",
            robot_skill_version="robot-v1",
            prompt_version="prompt-v1",
            environment_version="libero-v1",
            code_revision="abc123",
            split_salt="frozen-split-v1",
        )


def test_audit_rejects_manifest_with_tampered_default_lineage_split() -> None:
    lineage = _lineage("episode")
    assignment = assign_lineage_splits((lineage,), salt="frozen-split-v1")
    manifest = build_calibration_dataset(
        (_outcome(dataset_split="unassigned"),),
        (lineage,),
        split_assignments=assignment,
        memory_skill_version="memory-v1",
        robot_skill_version="robot-v1",
        prompt_version="prompt-v1",
        environment_version="libero-v1",
        code_revision="abc123",
        split_salt="frozen-split-v1",
    )
    expected = assignment["episode"]
    tampered_split = next(split for split in ("train", "calibration", "test") if split != expected)
    object.__setattr__(manifest.outcomes[0], "dataset_split", tampered_split)

    assert "LINEAGE_SPLIT_ASSIGNMENT_MISMATCH" in _codes(manifest)


def test_dataset_rejects_future_state_feature_timestamp() -> None:
    outcome = _outcome(snapshot_captured_at_ns=200)
    lineage = _lineage("episode", decision_boundary_ns=100)
    manifest = _manifest((outcome,), (lineage,))

    audit = audit_calibration_dataset(manifest)

    assert audit.passed is False
    assert "FUTURE_STATE_FEATURE" in {finding.code for finding in audit.findings}


def test_dataset_rejects_tier_b_or_c_as_supervised_label() -> None:
    snapshot = _snapshot("candidate", lane="rehearsal")
    tier_b = CalibrationOutcome(
        episode_id="episode",
        family_id="object-6",
        candidate_id="candidate",
        candidate_fingerprint="c" * 64,
        tier="B",
        execution_status="not_selected",
        task_success=None,
        graph_completed=None,
        verifier_success=None,
        rehearsal_success=True,
        failure_class=None,
        horizon=_horizon(),
        feature_snapshot=snapshot,
        dataset_split="train",
    )
    manifest = _manifest((tier_b,), (_lineage("episode"),))
    object.__setattr__(manifest.outcomes[0], "dataset_split", "train")

    with pytest.raises(ValueError, match="Tier B"):
        assert_dataset_eligible(audit_calibration_dataset(manifest))


def test_dataset_rejects_unknown_tier_even_when_shadow_split_is_consistent() -> None:
    outcome = _outcome()
    manifest = _manifest((outcome,), (_lineage("episode"),))
    object.__setattr__(manifest.outcomes[0], "tier", "D")
    object.__setattr__(manifest.outcomes[0], "dataset_split", "shadow")

    audit = audit_calibration_dataset(manifest)

    assert audit.passed is False
    assert "INVALID_TIER" in {finding.code for finding in audit.findings}


@pytest.mark.parametrize(
    ("feature_name", "code"),
    [
        ("evaluator_success", "EVALUATOR_DERIVED_FEATURE"),
        ("dynamic_verifier_pass_rate", "FORBIDDEN_FEATURE_V1"),
        ("ood_success_rate", "FORBIDDEN_FEATURE_V1"),
    ],
)
def test_dataset_rejects_post_outcome_or_forbidden_v1_features(
    feature_name: str, code: str
) -> None:
    outcome = _outcome()
    snapshot = replace(
        outcome.feature_snapshot,
        features={**outcome.feature_snapshot.features, feature_name: 1.0},
        feature_status={**outcome.feature_snapshot.feature_status, feature_name: "present"},
        correlation_groups={
            **outcome.feature_snapshot.correlation_groups,
            feature_name: "cost_risk",
        },
    )
    object.__setattr__(outcome, "feature_snapshot", snapshot)

    assert code in _codes(_manifest((outcome,), (_lineage("episode"),)))


@pytest.mark.parametrize(
    ("features", "groups"),
    [
        ({"ood_pass_rate": None}, {"ood_pass_rate": "scene_grounding"}),
        ({"scene_freshness": None}, {"scene_freshness": "cost_risk"}),
        ({"dynamic_pass_rate": None}, {"dynamic_pass_rate": "cost_risk"}),
        ({"ground_truth_label": None}, {"ground_truth_label": "cost_risk"}),
    ],
)
def test_dataset_requires_exact_v1_feature_groups(
    features: dict[str, float | None], groups: dict[str, str]
) -> None:
    outcome = _outcome()
    snapshot = replace(
        outcome.feature_snapshot,
        features=features,
        feature_status={name: "unknown" for name in features},
        correlation_groups=groups,
    )
    object.__setattr__(outcome, "feature_snapshot", snapshot)

    assert "FEATURE_SCHEMA_MISMATCH" in _codes(_manifest((outcome,), (_lineage("episode"),)))


def test_dataset_rejects_conflicting_lineage_and_cross_split_groups() -> None:
    first = _lineage("episode", group="pair")
    conflicting = replace(first, seed=2)
    second_episode = _lineage("episode-2", group="pair")
    manifest = _manifest(
        (_outcome(), _outcome(episode_id="episode-2", fingerprint="b" * 64)),
        (first, conflicting, second_episode),
    )
    object.__setattr__(manifest.outcomes[1], "dataset_split", "test")

    codes = _codes(manifest)

    assert "CONFLICTING_EPISODE_LINEAGE" in codes
    assert "LINEAGE_GROUP_SPLIT" in codes


def test_dataset_rejects_same_episode_candidate_in_multiple_supervised_splits() -> None:
    first = _outcome()
    duplicate = replace(first, dataset_split="test")
    manifest = _manifest((first, duplicate), (_lineage("episode"),))
    object.__setattr__(manifest.outcomes[1], "dataset_split", "test")

    assert "SUPERVISED_SPLIT_LEAKAGE" in _codes(manifest)


def test_independent_episodes_may_reuse_candidate_fingerprint() -> None:
    outcomes = (
        _outcome(episode_id="episode-1", fingerprint="a" * 64),
        _outcome(episode_id="episode-2", fingerprint="a" * 64),
    )
    lineages = (_lineage("episode-1"), _lineage("episode-2"))

    audit = audit_calibration_dataset(_manifest(outcomes, lineages))

    assert "SUPERVISED_SPLIT_LEAKAGE" not in {finding.code for finding in audit.findings}


def test_dataset_rejects_missing_provenance_and_identity_mismatch() -> None:
    outcome = _outcome()
    manifest = _manifest((outcome,), (_lineage("episode"),))
    object.__setattr__(manifest.outcomes[0].feature_snapshot, "memory_skill_version", "")
    object.__setattr__(manifest.outcomes[0].feature_snapshot, "candidate_id", "other")
    object.__setattr__(manifest, "prompt_version", "")

    codes = _codes(manifest)

    assert "MISSING_PROVENANCE" in codes
    assert "OUTCOME_SNAPSHOT_IDENTITY" in codes


def test_dataset_rejects_tier_a_without_selected_physical_execution() -> None:
    outcome = _outcome()
    manifest = _manifest((outcome,), (_lineage("episode"),))
    object.__setattr__(manifest.outcomes[0], "execution_status", "not_selected")

    assert "INVALID_TIER_A" in _codes(manifest)


def test_build_assigns_supervised_and_shadow_splits_and_canonical_digest() -> None:
    tier_a = _outcome(dataset_split="unassigned")
    tier_c_snapshot = _snapshot("b", fingerprint="b" * 64)
    tier_c = CalibrationOutcome(
        episode_id="episode",
        family_id="object-6",
        candidate_id="b",
        candidate_fingerprint="b" * 64,
        tier="C",
        execution_status="not_selected",
        task_success=None,
        graph_completed=None,
        verifier_success=None,
        rehearsal_success=None,
        failure_class=None,
        horizon=_horizon(),
        feature_snapshot=tier_c_snapshot,
        dataset_split="unassigned",
    )
    kwargs = {
        "split_assignments": assign_lineage_splits(
            (_lineage("episode"),), salt="frozen-split-v1"
        ),
        "memory_skill_version": "memory-v1",
        "robot_skill_version": "robot-v1",
        "prompt_version": "prompt-v1",
        "environment_version": "libero-v1",
        "code_revision": "abc123",
        "split_salt": "frozen-split-v1",
    }

    first = build_calibration_dataset((tier_a, tier_c), (_lineage("episode"),), **kwargs)
    second = build_calibration_dataset((tier_a, tier_c), (_lineage("episode"),), **kwargs)

    assert [row.dataset_split for row in first.outcomes] == [
        kwargs["split_assignments"]["episode"],
        "shadow",
    ]
    assert first == second
    assert first.dataset_schema_version == DATASET_SCHEMA_VERSION
    assert first.dataset_id == f"sha256:{first.manifest_sha256}"
    assert len(first.manifest_sha256) == 64
    encoded = json.dumps(first.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    assert encoded.isascii()
    assert audit_calibration_dataset(first).passed is True


@pytest.mark.parametrize(
    ("passed", "findings"),
    [
        (True, (LeakageFinding("LEAK", (), "must fail"),)),
        (False, ()),
    ],
)
def test_dataset_audit_rejects_inconsistent_passed_flag(
    passed: bool, findings: tuple[LeakageFinding, ...]
) -> None:
    with pytest.raises(ValueError, match="passed must equal"):
        DatasetAudit(passed=passed, findings=findings, tier_counts={}, split_counts={})


def test_eligibility_gate_rejects_a_forged_passing_audit() -> None:
    audit = DatasetAudit(
        passed=False,
        findings=(LeakageFinding("LEAK", (), "must fail"),),
        tier_counts={},
        split_counts={},
    )
    object.__setattr__(audit, "passed", True)

    with pytest.raises(ValueError, match="inconsistent passed flag"):
        assert_dataset_eligible(audit)
