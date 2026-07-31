from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from capmas.evaluation.phase5_artifacts import Phase5RunDirectory
from capmas.contracts.candidates import ArbitrationResult
from capmas.evaluation.online_rehearsal import RehearsalArbitrationReport


@dataclass(frozen=True)
class _DataclassSummary:
    completed: bool
    traces: tuple[str, ...]


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


def test_phase5_json_writer_preserves_structured_dataclass_payloads(tmp_path) -> None:
    run = Phase5RunDirectory.create(tmp_path, "experiment", "run-structured")

    output = run.write_json(
        "summary.json",
        {"result": _DataclassSummary(True, ("trace-1",))},
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["result"] == {"completed": True, "traces": ["trace-1"]}


def test_phase5_writer_serializes_online_rehearsal_report(tmp_path) -> None:
    run = Phase5RunDirectory.create(tmp_path, "experiment", "run-rehearsal")
    report = RehearsalArbitrationReport(
        mode="online_bounded",
        baseline=ArbitrationResult(None),
        evidence_aware=ArbitrationResult(None),
        live=ArbitrationResult(None),
        attached_candidate_ids=("candidate-a",),
        evidence_rejections=("candidate-b: stale",),
        provider_latency_ms=12.5,
        fallback_reason="provider_timeout",
    )

    output = run.write_json(
        "summary.json",
        {
            "run_config": {"rehearsal_mode": "online_bounded"},
            "scheduler_metrics": {"rehearsal_reports": {"first": report}},
        },
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    serialized = payload["scheduler_metrics"]["rehearsal_reports"]["first"]
    assert payload["run_config"]["rehearsal_mode"] == "online_bounded"
    assert serialized["mode"] == "online_bounded"
    assert serialized["attached_candidate_ids"] == ["candidate-a"]
    assert serialized["provider_latency_ms"] == 12.5
