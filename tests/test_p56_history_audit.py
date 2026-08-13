from __future__ import annotations

import json
from pathlib import Path

import pytest

from capmas.evaluation.history_audit import audit_p55_history, run_history_audit

FINGERPRINT = "a" * 64
ARTIFACT_DIGEST = "b" * 64
SOURCE_MANIFEST_SHA256 = "c" * 64


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _historical_case(
    tmp_path: Path,
    *,
    include_selection: bool,
    include_evaluator: bool,
    include_feature_snapshot: bool,
    include_horizon: bool,
    include_after_scene: bool = False,
    include_graph_events: bool = True,
) -> Path:
    suite = tmp_path / "suite"
    case_dir = suite / "cases" / "case-1"
    case_id = "object-6-case-1"
    candidate_id = "candidate-a"
    _write_json(suite / "suite_manifest.json", {"manifest_sha256": SOURCE_MANIFEST_SHA256})
    _write_json(
        case_dir / "case.json",
        {
            "case_id": case_id,
            "task_family": "object-6",
            "split": "id",
            "seed": 1,
            "candidate_artifact_sha256": ARTIFACT_DIGEST,
        },
    )
    _write_json(
        case_dir / "summary.json",
        {
            "case_id": case_id,
            "status": "completed",
            "primary_winner": candidate_id if include_selection else None,
            "evaluator_success": True if include_evaluator else None,
            "verifier_success": True if include_evaluator else None,
        },
    )
    evidence = {
        "case_id": case_id,
        "split": "id",
        "candidate_id": candidate_id if include_selection else "unselected",
        "candidate_fingerprint": FINGERPRINT,
        "evaluator_success": True if include_evaluator else None,
        "graph_completed": True if include_evaluator else None,
        "verifier_success": True if include_evaluator else None,
        "failure_class": None,
    }
    if include_horizon:
        evidence["horizon"] = _horizon()
    if include_graph_events:
        evidence["graph_events"] = [_graph_event()]
    if include_after_scene:
        evidence["scene_diagnostics.after"] = {
            "candidate_id": candidate_id,
            "candidate_fingerprint": FINGERPRINT,
            "captured_at_ns": 1,
        }
        evidence["physical_after"] = {
            "features": {"must_not_be_used": 1.0},
            "captured_at_ns": 1,
        }
    _write_json(case_dir / "evidence" / "ood_replay.json", evidence)
    if include_feature_snapshot:
        _write_json(
            case_dir / "evidence" / "calibration_feature_snapshots.json",
            [
                _snapshot(
                    case_id=case_id,
                    candidate_id=candidate_id,
                    captured_at_ns=100,
                )
            ],
        )
    return suite


def _native_p56_case(
    tmp_path: Path,
    *,
    feature_captured_at_ns: int,
    execution_started_at_ns: int,
    evaluator_observed_at_ns: int,
) -> Path:
    suite = _historical_case(
        tmp_path,
        include_selection=True,
        include_evaluator=True,
        include_feature_snapshot=False,
        include_horizon=True,
    )
    case_dir = suite / "cases" / "case-1"
    case_id = "object-6-case-1"
    candidate_id = "candidate-a"
    _write_json(
        case_dir / "evidence" / "calibration_feature_snapshots.json",
        [
            _snapshot(
                case_id=case_id,
                candidate_id=candidate_id,
                captured_at_ns=feature_captured_at_ns,
            )
        ],
    )
    online = json.loads((case_dir / "evidence" / "ood_replay.json").read_text(encoding="utf-8"))
    online["source_scene_version"] = 1
    online["execution_started_at_ns"] = execution_started_at_ns
    online["evaluator_observed_at_ns"] = evaluator_observed_at_ns
    _write_json(case_dir / "evidence" / "ood_replay.json", online)
    return suite


def _load_case_evidence(suite: Path) -> dict[str, object]:
    return json.loads(
        (suite / "cases" / "case-1" / "evidence" / "ood_replay.json").read_text(
            encoding="utf-8"
        )
    )


def _write_case_evidence(suite: Path, evidence: dict[str, object]) -> None:
    _write_json(suite / "cases" / "case-1" / "evidence" / "ood_replay.json", evidence)


def _load_case_summary(suite: Path) -> dict[str, object]:
    return json.loads(
        (suite / "cases" / "case-1" / "summary.json").read_text(encoding="utf-8")
    )


def _write_case_summary(suite: Path, summary: dict[str, object]) -> None:
    _write_json(suite / "cases" / "case-1" / "summary.json", summary)


def _snapshot(*, case_id: str, candidate_id: str, captured_at_ns: int) -> dict[str, object]:
    return {
        "episode_id": case_id,
        "episode_epoch": 1,
        "family_id": "object-6",
        "candidate_id": candidate_id,
        "candidate_fingerprint": FINGERPRINT,
        "scene_version": 1,
        "map_version": 1,
        "feature_schema_version": "p56.feature.v1",
        "captured_at_ns": captured_at_ns,
        "collection_lane": "physical",
        "features": {"scene_freshness": 1.0},
        "feature_status": {"scene_freshness": "present"},
        "correlation_groups": {"scene_freshness": "scene_grounding"},
        "memory_skill_version": "memory-v1",
        "robot_skill_version": "robot-v1",
        "selection_probability": 1.0,
        "evidence_refs": ("evidence/pre_execution_scene.json",),
        "evidence_providers": {"scene": "pre_execution"},
        "rewrite_metadata": {},
    }


def _horizon() -> dict[str, object]:
    return {
        "planned_critical_path_actions": 1,
        "planned_critical_path_subgoals": 1,
        "planned_checkpoint_subgraphs": 0,
        "attempted_actions": 1,
        "completed_actions": 1,
        "attempted_subgoals": 1,
        "completed_subgoals": 1,
        "attempted_checkpoints": 0,
        "completed_checkpoints": 0,
        "planned_source": "mission_graph",
        "realized_source": "execution_trace",
        "planned_valid": True,
        "realized_valid": True,
    }


def _graph_event() -> dict[str, object]:
    return {
        "sequence": 1,
        "kind": "node_completed",
        "subgraph_id": "sg_pick_butter",
        "node_id": "pick",
        "node_type": "action",
        "attempt": 1,
        "outcome": "success",
        "occurred_at_ns": 200,
    }


def test_history_audit_rejects_missing_preexecution_snapshot(tmp_path: Path) -> None:
    suite = _historical_case(
        tmp_path,
        include_selection=True,
        include_evaluator=True,
        include_feature_snapshot=False,
        include_horizon=False,
    )

    audit = audit_p55_history(suite, family_id="object-6")

    assert audit.admissible_tier_a_count == 0
    assert audit.rejection_counts["MISSING_PREEXECUTION_FEATURE_SNAPSHOT"] == 1
    assert audit.rejection_counts["MISSING_HORIZON_LINEAGE"] == 1


def test_history_audit_never_uses_after_scene_as_features(tmp_path: Path) -> None:
    suite = _historical_case(
        tmp_path,
        include_selection=True,
        include_evaluator=True,
        include_feature_snapshot=False,
        include_horizon=True,
        include_after_scene=True,
    )

    decision = audit_p55_history(suite, family_id="object-6").rows[0]

    assert decision.admissible is False
    assert "MISSING_PREEXECUTION_FEATURE_SNAPSHOT" in decision.reasons
    assert not any("after" in ref.lower() for ref in decision.source_refs)


def test_history_audit_accepts_only_timestamp_ordered_native_record(tmp_path: Path) -> None:
    suite = _native_p56_case(
        tmp_path,
        feature_captured_at_ns=100,
        execution_started_at_ns=200,
        evaluator_observed_at_ns=300,
    )

    audit = audit_p55_history(suite, family_id="object-6")

    assert audit.admissible_tier_a_count == 1
    assert audit.rows[0].admissible is True


def test_history_audit_rejects_snapshot_from_different_source_scene(
    tmp_path: Path,
) -> None:
    suite = _native_p56_case(
        tmp_path,
        feature_captured_at_ns=100,
        execution_started_at_ns=200,
        evaluator_observed_at_ns=300,
    )
    evidence = _load_case_evidence(suite)
    evidence["source_scene_version"] = 2
    _write_case_evidence(suite, evidence)

    decision = audit_p55_history(suite, family_id="object-6").rows[0]

    assert decision.admissible is False
    assert "FEATURE_SNAPSHOT_IDENTITY_MISMATCH" in decision.reasons


@pytest.mark.parametrize("source_scene_version", [None, "1", True])
def test_history_audit_rejects_missing_or_malformed_source_scene_version(
    tmp_path: Path,
    source_scene_version: object,
) -> None:
    suite = _native_p56_case(
        tmp_path,
        feature_captured_at_ns=100,
        execution_started_at_ns=200,
        evaluator_observed_at_ns=300,
    )
    evidence = _load_case_evidence(suite)
    if source_scene_version is None:
        del evidence["source_scene_version"]
    else:
        evidence["source_scene_version"] = source_scene_version
    _write_case_evidence(suite, evidence)

    decision = audit_p55_history(suite, family_id="object-6").rows[0]

    assert decision.admissible is False
    assert "FEATURE_SNAPSHOT_IDENTITY_MISMATCH" in decision.reasons


def test_history_audit_rejects_evaluator_observation_before_execution_start(
    tmp_path: Path,
) -> None:
    suite = _native_p56_case(
        tmp_path,
        feature_captured_at_ns=100,
        execution_started_at_ns=200,
        evaluator_observed_at_ns=199,
    )

    decision = audit_p55_history(suite, family_id="object-6").rows[0]

    assert decision.admissible is False
    assert "EVALUATOR_OBSERVED_BEFORE_EXECUTION_START" in decision.reasons


def test_history_audit_rejects_missing_evaluator_observation_for_native_snapshot(
    tmp_path: Path,
) -> None:
    suite = _native_p56_case(
        tmp_path,
        feature_captured_at_ns=100,
        execution_started_at_ns=200,
        evaluator_observed_at_ns=300,
    )
    evidence = _load_case_evidence(suite)
    del evidence["evaluator_observed_at_ns"]
    _write_case_evidence(suite, evidence)

    decision = audit_p55_history(suite, family_id="object-6").rows[0]

    assert decision.admissible is False
    assert "MISSING_EVALUATOR_OBSERVATION_TIMESTAMP" in decision.reasons


def test_history_audit_requires_native_top_level_lineage_timing_and_evaluator(
    tmp_path: Path,
) -> None:
    suite = _native_p56_case(
        tmp_path,
        feature_captured_at_ns=100,
        execution_started_at_ns=200,
        evaluator_observed_at_ns=300,
    )
    evidence = _load_case_evidence(suite)
    physical_result = {
        "execution_started_at_ns": evidence.pop("execution_started_at_ns"),
        "evaluator_observed_at_ns": evidence.pop("evaluator_observed_at_ns"),
        "evaluator_success": evidence.pop("evaluator_success"),
        "graph_events": evidence.pop("graph_events"),
        "horizon": evidence.pop("horizon"),
    }
    evidence["physical_result"] = physical_result
    _write_case_evidence(suite, evidence)

    decision = audit_p55_history(suite, family_id="object-6").rows[0]

    assert decision.admissible is False
    assert {
        "MISSING_EXECUTION_START_TIMESTAMP",
        "MISSING_EVALUATOR_OBSERVATION_TIMESTAMP",
        "INCONCLUSIVE_EVALUATOR",
        "MISSING_GRAPH_EVENTS",
        "MISSING_HORIZON_LINEAGE",
    } <= set(decision.reasons)


def test_history_audit_rejects_inconsistent_retained_evaluator_outcome(
    tmp_path: Path,
) -> None:
    suite = _native_p56_case(
        tmp_path,
        feature_captured_at_ns=100,
        execution_started_at_ns=200,
        evaluator_observed_at_ns=300,
    )
    summary = _load_case_summary(suite)
    summary["evaluator_success"] = False
    _write_case_summary(suite, summary)

    decision = audit_p55_history(suite, family_id="object-6").rows[0]

    assert decision.admissible is False
    assert "EVALUATOR_OUTCOME_MISMATCH" in decision.reasons


def test_history_audit_rejects_missing_case_level_evaluator_outcome(
    tmp_path: Path,
) -> None:
    suite = _native_p56_case(
        tmp_path,
        feature_captured_at_ns=100,
        execution_started_at_ns=200,
        evaluator_observed_at_ns=300,
    )
    summary = _load_case_summary(suite)
    summary["evaluator_success"] = None
    _write_case_summary(suite, summary)

    decision = audit_p55_history(suite, family_id="object-6").rows[0]

    assert decision.admissible is False
    assert "INCONCLUSIVE_EVALUATOR" in decision.reasons


def test_history_audit_rejects_non_structural_graph_events(tmp_path: Path) -> None:
    suite = _native_p56_case(
        tmp_path,
        feature_captured_at_ns=100,
        execution_started_at_ns=200,
        evaluator_observed_at_ns=300,
    )
    evidence = _load_case_evidence(suite)
    evidence["graph_events"] = ["not a graph event"]
    _write_case_evidence(suite, evidence)

    decision = audit_p55_history(suite, family_id="object-6").rows[0]

    assert decision.admissible is False
    assert "INVALID_GRAPH_EVENTS" in decision.reasons


def test_history_audit_rejects_future_snapshot_without_using_dynamic_verifier(
    tmp_path: Path,
) -> None:
    suite = _native_p56_case(
        tmp_path,
        feature_captured_at_ns=300,
        execution_started_at_ns=200,
        evaluator_observed_at_ns=400,
    )
    dynamic_verifier_path = (
        suite / "cases" / "case-1" / "evidence" / "dynamic_verifier_features.json"
    )
    _write_json(dynamic_verifier_path, [_snapshot(case_id="object-6-case-1", candidate_id="candidate-a", captured_at_ns=100)])

    decision = audit_p55_history(suite, family_id="object-6").rows[0]

    assert decision.admissible is False
    assert "FEATURE_SNAPSHOT_AFTER_EXECUTION_START" in decision.reasons
    assert not any("dynamic_verifier" in ref for ref in decision.source_refs)


def test_history_audit_reports_all_applicable_rejection_reasons(tmp_path: Path) -> None:
    suite = _historical_case(
        tmp_path,
        include_selection=False,
        include_evaluator=False,
        include_feature_snapshot=False,
        include_horizon=False,
        include_graph_events=False,
    )

    decision = audit_p55_history(suite, family_id="object-6").rows[0]

    assert decision.admissible is False
    assert {
        "MISSING_SELECTION_EVENT",
        "MISSING_PREEXECUTION_FEATURE_SNAPSHOT",
        "MISSING_HORIZON_LINEAGE",
        "MISSING_GRAPH_EVENTS",
        "INCONCLUSIVE_EVALUATOR",
    } <= set(decision.reasons)


def test_history_audit_rejects_output_root_inside_suite_even_through_symlink(
    tmp_path: Path,
) -> None:
    suite = _native_p56_case(
        tmp_path,
        feature_captured_at_ns=100,
        execution_started_at_ns=200,
        evaluator_observed_at_ns=300,
    )
    linked = tmp_path / "linked_suite_child"
    linked.symlink_to(suite / "nested-output", target_is_directory=True)

    with pytest.raises(ValueError, match="output_root"):
        run_history_audit(suite_dir=suite, family_id="object-6", output_root=linked)


def test_history_audit_cli_outputs_isolated_manifest_verified_artifacts(
    tmp_path: Path,
) -> None:
    suite = _native_p56_case(
        tmp_path,
        feature_captured_at_ns=100,
        execution_started_at_ns=200,
        evaluator_observed_at_ns=300,
    )
    output_root = tmp_path / "outputs"

    result = run_history_audit(suite_dir=suite, family_id="object-6", output_root=output_root)

    assert result.run_dir.parent.name == "P5.6.2a_object6_history_audit"
    assert (result.run_dir / "results" / "history_audit.json").is_file()
    assert (result.run_dir / "results" / "admissible_rows.json").is_file()
    assert (result.run_dir / "summary.md").is_file()
    assert (result.run_dir / "logs" / "runner.log").is_file()
    manifest = json.loads((result.run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest_paths = {entry["path"] for entry in manifest["files"]}
    assert {
        "results/history_audit.json",
        "results/admissible_rows.json",
        "summary.md",
        "logs/runner.log",
        "run_config.json",
    } <= manifest_paths
