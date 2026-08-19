from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from capmas.contracts.calibration import (
    FEATURE_SCHEMA_VERSION,
    CalibrationLineage,
    CalibrationOutcome,
    CandidateFeatureSnapshot,
    HorizonLabel,
)
from capmas.evaluation.feature_snapshots import FEATURE_GROUPS_V1
from scripts import run_p56_offline
from scripts.verify_phase5_manifest import inspect_manifest

_POSITIVE_INDICES = frozenset({0, 2, 3, 6, 7, 9, 10, 11, 13, 18})
_MANIFEST_SHA256 = "a" * 64


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


def _outcome(index: int) -> CalibrationOutcome:
    episode_id = f"episode-{index:02d}"
    candidate_id = f"candidate-{index:02d}"
    fingerprint = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()
    success = index in _POSITIVE_INDICES
    snapshot = CandidateFeatureSnapshot(
        episode_id=episode_id,
        episode_epoch=1,
        family_id="object-6",
        candidate_id=candidate_id,
        candidate_fingerprint=fingerprint,
        scene_version=1,
        map_version=None,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        captured_at_ns=100 + index,
        collection_lane="physical",
        features={
            name: (0.9 if success else 0.1) if name == "rehearsal_success_rate" else None
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
    return CalibrationOutcome(
        episode_id=episode_id,
        family_id="object-6",
        candidate_id=candidate_id,
        candidate_fingerprint=fingerprint,
        tier="A",
        execution_status="selected_executed",
        task_success=success,
        graph_completed=success,
        verifier_success=None,
        rehearsal_success=None,
        failure_class=None if success else "TASK_FAILURE",
        horizon=_horizon(),
        feature_snapshot=snapshot,
        dataset_split="unassigned",
    )


def _lineage(index: int) -> CalibrationLineage:
    return CalibrationLineage(
        episode_id=f"episode-{index:02d}",
        lineage_group_id=f"lineage-{index:02d}",
        seed=index,
        split_identity="id",
        layout_pair_id=None,
        retry_of_episode_id=None,
        candidate_artifact_sha256=hashlib.sha256(f"candidate-{index:02d}".encode()).hexdigest(),
        decision_boundary_ns=200 + index,
        evaluator_observed_at_ns=300 + index,
    )


def _write_collection_suite(root: Path, name: str, indices: range) -> Path:
    suite = root / name
    results = suite / "results"
    results.mkdir(parents=True)
    outcomes = [_outcome(index) for index in indices]
    lineages = [_lineage(index) for index in indices]
    (results / "outcomes.json").write_text(
        json.dumps([outcome.to_dict() for outcome in outcomes]), encoding="utf-8"
    )
    (results / "lineages.json").write_text(
        json.dumps([lineage.to_dict() for lineage in lineages]), encoding="utf-8"
    )
    (suite / "run_config.json").write_text(
        json.dumps({"manifest_sha256": _MANIFEST_SHA256}), encoding="utf-8"
    )
    return suite


def _write_provenance(root: Path) -> None:
    directory = root / "configs" / "phase5"
    directory.mkdir(parents=True)
    (directory / "p56_unit.json").write_text(
        json.dumps(
            {
                "manifest_sha256": _MANIFEST_SHA256,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "memory_skill_version": "memory-v1",
                "robot_skill_version": "robot-v1",
                "prompt_version": "prompt-v1",
                "environment_version": "libero-v1",
                "code_revision": "revision-v1",
            }
        ),
        encoding="utf-8",
    )


def test_offline_cli_writes_a_complete_verified_phase5_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_provenance(tmp_path)
    first = _write_collection_suite(tmp_path, "first", range(10))
    second = _write_collection_suite(tmp_path, "second", range(10, 20))
    monkeypatch.setattr(run_p56_offline, "PROJECT_ROOT", tmp_path)

    assert (
        run_p56_offline.main(
            [
                "--collection-run",
                str(first),
                "--collection-run",
                str(second),
                "--output-root",
                str(tmp_path / "outputs"),
                "--run-id",
                "unit",
            ]
        )
        == 0
    )

    run_dir = Path(capsys.readouterr().out.strip())
    assert (run_dir / "logs/runner.log").is_file()
    assert (run_dir / "artifacts/source_dataset_manifest.json").is_file()
    assert (run_dir / "artifacts/exact_quota_split.json").is_file()
    assert (run_dir / "artifacts/reduced_features.json").is_file()
    assert (run_dir / "artifacts/constrained_logistic_model.json").is_file()
    assert (run_dir / "artifacts/isotonic_calibration.json").is_file()
    assert (run_dir / "results/predictions.json").is_file()
    assert (run_dir / "results/offline_calibration_report.json").is_file()
    assert inspect_manifest(run_dir)["verified"] is True


def test_load_collection_rows_rejects_duplicate_episode_ids_across_suites(tmp_path: Path) -> None:
    first = _write_collection_suite(tmp_path, "first", range(10))
    duplicate = _write_collection_suite(tmp_path, "duplicate", range(10))

    with pytest.raises(ValueError, match="duplicate episode"):
        run_p56_offline.load_collection_rows((first, duplicate))
