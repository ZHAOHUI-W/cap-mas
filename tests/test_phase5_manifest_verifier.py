from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from capmas.evaluation.phase5_artifacts import Phase5RunDirectory
from scripts.verify_phase5_manifest import inspect_manifest, verify_and_record_manifest

ROOT = Path(__file__).parents[1]


def test_verifier_records_then_rechecks_a_regenerated_manifest(tmp_path) -> None:
    run = Phase5RunDirectory.create(tmp_path, "p56", "manifest-check")
    run.write_json("results/outcome.json", {"success": True})
    run.finalize_manifest()

    report = verify_and_record_manifest(run.path)

    assert report["verified"] is True
    persisted = json.loads((run.path / "results" / "manifest_verification.json").read_text())
    assert persisted["initial_manifest"]["verified"] is True
    assert inspect_manifest(run.path)["verified"] is True


def test_manifest_inspection_reports_untracked_files(tmp_path) -> None:
    run = Phase5RunDirectory.create(tmp_path, "p56", "untracked")
    run.write_json("results/outcome.json", {"success": True})
    run.finalize_manifest()
    run.write_text("logs/late.log", "late write\n")

    report = inspect_manifest(run.path)

    assert report["verified"] is False
    assert report["untracked_files"] == ["logs/late.log"]


def test_verifier_rebuilds_a_manifest_after_recording_stale_entries(tmp_path) -> None:
    run = Phase5RunDirectory.create(tmp_path, "p56", "rebuild")
    run.write_json("results/outcome.json", {"success": True})
    run.finalize_manifest()
    run.write_text("logs/late.log", "late write\n")

    report = verify_and_record_manifest(run.path)

    assert report["verified"] is True
    persisted = json.loads((run.path / "results" / "manifest_verification.json").read_text())
    assert persisted["initial_manifest"]["verified"] is False
    assert persisted["initial_manifest"]["untracked_files"] == ["logs/late.log"]


def test_manifest_verifier_cli_runs_from_the_project_root(tmp_path) -> None:
    run = Phase5RunDirectory.create(tmp_path, "p56", "cli")
    run.write_json("results/outcome.json", {"success": True})
    run.finalize_manifest()

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "verify_phase5_manifest.py"),
            "--run-dir",
            str(run.path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
