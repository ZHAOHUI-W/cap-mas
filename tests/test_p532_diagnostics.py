from __future__ import annotations

import json

import pytest


def test_preview_diagnostic_is_ineligible_and_forbids_physical_submission(tmp_path) -> None:
    from scripts.run_libero_p532_diagnostics import DiagnosticRequest, run_diagnostic

    request = DiagnosticRequest(
        config_path="configs/object6.yaml",
        candidate_artifact="candidates/object6.json",
        seed=57,
        object_name="butter",
        target_name="basket",
    )

    outcome = run_diagnostic(
        request,
        mode="preview",
        output_root=tmp_path,
        mode_runner=lambda _request, _run_dir: {
            "physical_execution_count": 0,
            "candidate_count": 2,
            "segments": [],
        },
    )

    config = json.loads((outcome.run_dir.path / "run_config.json").read_text())
    diagnostic = json.loads((outcome.run_dir.path / "results" / "diagnostic.json").read_text())
    assert config["diagnostic_only"] is True
    assert config["eligible_for_evaluation"] is False
    assert config["physical_execution_limit"] == 0
    assert config["status"] == "completed"
    assert diagnostic["physical_execution_count"] == 0
    assert (outcome.run_dir.path / "manifest.json").is_file()


def test_non_execute_diagnostic_rejects_a_submission_claim(tmp_path) -> None:
    from scripts.run_libero_p532_diagnostics import DiagnosticRequest, run_diagnostic

    request = DiagnosticRequest(
        config_path="configs/object6.yaml",
        candidate_artifact="candidates/object6.json",
        seed=54,
        object_name="butter",
        target_name="basket",
    )

    with pytest.raises(ValueError, match="exceeded physical execution limit"):
        run_diagnostic(
            request,
            mode="depth",
            output_root=tmp_path,
            mode_runner=lambda _request, _run_dir: {"physical_execution_count": 1},
        )


def test_execute_diagnostic_allows_one_submission_and_remains_ineligible(tmp_path) -> None:
    from scripts.run_libero_p532_diagnostics import DiagnosticRequest, run_diagnostic

    request = DiagnosticRequest(
        config_path="configs/object6.yaml",
        candidate_artifact="candidates/object6.json",
        seed=53,
        object_name="butter",
        target_name="basket",
    )

    outcome = run_diagnostic(
        request,
        mode="execute",
        output_root=tmp_path,
        mode_runner=lambda _request, _run_dir: {
            "physical_execution_count": 1,
            "physical_result": {"completed": False},
        },
    )

    config = json.loads((outcome.run_dir.path / "run_config.json").read_text())
    assert config["physical_execution_limit"] == 1
    assert config["eligible_for_evaluation"] is False
    assert outcome.physical_execution_count == 1
