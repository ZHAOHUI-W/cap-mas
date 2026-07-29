from __future__ import annotations

import hashlib
import json

from capmas.evaluation.phase5_artifacts import Phase5RunDirectory


def test_phase5_run_directory_is_unique_and_creates_expected_subdirectories(tmp_path) -> None:
    first = Phase5RunDirectory.create(tmp_path, "P5.2_geometry_evidence", "same-run")
    second = Phase5RunDirectory.create(tmp_path, "P5.2_geometry_evidence", "same-run")

    assert first.path != second.path
    for name in ("logs", "results", "traces", "evidence", "artifacts"):
        assert (first.path / name).is_dir()


def test_phase5_artifacts_are_atomic_and_manifest_has_sha256(tmp_path) -> None:
    run = Phase5RunDirectory.create(tmp_path, "experiment", "run-1")
    output = run.write_json(
        "run_config.json",
        {
            "model": "gpt-5.5",
            "api_key": "secret-key",
            "authorization": "Bearer secret-token",
            "provider_headers": {"x-provider-secret": "secret-header"},
        },
    )
    log_path = run.log_path()
    log_path.write_text("runner output", encoding="utf-8")

    manifest_path = run.finalize_manifest()
    payload = json.loads(output.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = {item["path"]: item for item in manifest["files"]}
    digest = hashlib.sha256(output.read_bytes()).hexdigest()

    assert "secret-key" not in output.read_text(encoding="utf-8")
    assert "secret-token" not in output.read_text(encoding="utf-8")
    assert "secret-header" not in output.read_text(encoding="utf-8")
    assert payload["api_key"] == "[REDACTED]"
    assert entries["run_config.json"]["sha256"] == digest
    assert entries["logs/runner.log"]["size"] == len("runner output")
