from __future__ import annotations

import json
import os
from dataclasses import replace
from types import SimpleNamespace

import pytest


def _assets(tmp_path):
    from scripts.create_p532_object6_manifest import Object6Assets, asset_sha256

    config = tmp_path / "object6.yaml"
    config.write_text("task: object-6\n", encoding="utf-8")
    candidates = tmp_path / "object6_candidates.json"
    candidates.write_text('{"candidates": []}\n', encoding="utf-8")
    return Object6Assets(
        task_id="libero_object_6",
        config_path=str(config),
        config_sha256=asset_sha256(config),
        candidate_artifact=str(candidates),
        candidate_artifact_sha256=asset_sha256(candidates),
        object_name="butter",
        target_name="basket",
    )


def test_manifest_is_deterministic_and_rejects_duplicate_seed_or_digest_mismatch(tmp_path) -> None:
    from scripts.create_p532_object6_manifest import (
        build_object6_manifest,
        load_and_preflight,
        validate_manifest,
        write_manifest,
    )

    manifest = build_object6_manifest(range(52, 62), _assets(tmp_path))
    path = write_manifest(tmp_path / "manifest.json", manifest)

    assert load_and_preflight(path).manifest_sha256 == manifest.manifest_sha256
    assert write_manifest(path, manifest).read_bytes() == path.read_bytes()
    with pytest.raises(ValueError, match="duplicate collection task/seed"):
        validate_manifest(replace(manifest, cases=(manifest.cases[0], manifest.cases[0])))
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_manifest(replace(manifest, manifest_sha256="0" * 64))


def test_cpu_molmo_profile_changes_the_manifest_identity(tmp_path) -> None:
    from scripts.create_p532_object6_manifest import build_object6_manifest

    cuda_manifest = build_object6_manifest(range(52, 62), _assets(tmp_path), molmo_device="cuda")
    cpu_manifest = build_object6_manifest(range(52, 62), _assets(tmp_path), molmo_device="cpu")

    assert cuda_manifest.schema_version == "p532.collection.v2"
    assert cpu_manifest.manifest_sha256 != cuda_manifest.manifest_sha256
    assert {case.molmo_device for case in cpu_manifest.cases} == {"cpu"}


def test_cpu_molmo_profile_survives_manifest_round_trip_and_preflight(tmp_path) -> None:
    import scripts.run_libero_p532_object6 as module
    from scripts.create_p532_object6_manifest import (
        build_object6_manifest,
        load_and_preflight,
        write_manifest,
    )

    manifest = build_object6_manifest(range(52, 62), _assets(tmp_path), molmo_device="cpu")
    path = write_manifest(tmp_path / "cpu-manifest.json", manifest)
    loaded = load_and_preflight(path)
    result = module.run_capability(path, output_root=tmp_path / "outputs", dry_run=True)
    preflight = json.loads((result.run_dir.path / "results" / "preflight.json").read_text())

    assert loaded.manifest_sha256 == manifest.manifest_sha256
    assert {case.molmo_device for case in loaded.cases} == {"cpu"}
    assert preflight["molmo_device"] == "cpu"


def test_dry_run_constructs_no_live_session(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.run_libero_p532_object6 as module
    from scripts.create_p532_object6_manifest import (
        build_object6_manifest,
        write_manifest,
    )

    path = write_manifest(
        tmp_path / "manifest.json",
        build_object6_manifest(range(52, 62), _assets(tmp_path)),
    )
    monkeypatch.setattr(module, "_live_session_factory", pytest.fail)

    result = module.run_capability(path, output_root=tmp_path / "outputs", dry_run=True)

    assert result.live_session_count == 0
    assert result.case_count == 10
    assert result.run_dir.path.exists()
    assert (result.run_dir.path / "results" / "preflight.json").exists()


def test_dry_run_does_not_configure_molmo_or_import_capx(tmp_path, monkeypatch) -> None:
    import scripts.run_libero_p532_object6 as module
    from scripts.create_p532_object6_manifest import build_object6_manifest, write_manifest

    path = write_manifest(
        tmp_path / "manifest.json",
        build_object6_manifest(range(52, 62), _assets(tmp_path), molmo_device="cpu"),
    )
    monkeypatch.setenv("MOLMO_DEVICE", "unchanged")
    monkeypatch.setattr(module, "_setup_capx_paths", pytest.fail)
    monkeypatch.setattr(module, "_live_session_factory", pytest.fail)

    module.run_capability(path, output_root=tmp_path / "outputs", dry_run=True)

    assert os.environ["MOLMO_DEVICE"] == "unchanged"


def test_selected_fingerprint_is_matched_by_candidate_id(tmp_path) -> None:
    import scripts.run_libero_p532_object6 as module
    from capmas.evaluation.phase5_artifacts import Phase5RunDirectory

    run_dir = Phase5RunDirectory.create(tmp_path, "online", "fingerprint")
    run_dir.write_json(
        "results/selection.json",
        {
            "physical_candidate_id": "candidate-b",
            "selected_program_fingerprint": "program-b",
            "program_fingerprints": {"candidate-a": "program-a", "candidate-b": "program-b"},
        },
    )
    counts = {"fingerprint_mismatch_count": 0}
    outcome = SimpleNamespace(physical_candidate_id="candidate-b", run_dir=run_dir)

    module._check_selected_fingerprint(counts, outcome)

    assert counts["fingerprint_mismatch_count"] == 0


def test_live_runner_configures_capx_paths_before_starting_api_servers(tmp_path, monkeypatch) -> None:
    """The CAP-X package must be importable before the server helper is imported."""

    import scripts.run_libero_p532_object6 as module
    from capmas.evaluation.phase5_artifacts import Phase5RunDirectory
    from scripts.create_p532_object6_manifest import build_object6_manifest

    events: list[str] = []
    manifest = build_object6_manifest(range(52, 62), _assets(tmp_path), molmo_device="cpu")
    run_dir = Phase5RunDirectory.create(tmp_path / "outputs", "p532", "path-order")
    monkeypatch.delenv("MOLMO_DEVICE", raising=False)
    monkeypatch.setattr(
        module,
        "_setup_capx_paths",
        lambda: events.append(f"paths:{os.environ.get('MOLMO_DEVICE')}")
    )
    monkeypatch.setattr(
        module,
        "_start_capx_api_servers",
        lambda _config_path: events.append("servers") or [],
    )
    monkeypatch.setattr(
        module,
        "_run_case",
        lambda _case, _case_dir: events.append("case")
        or SimpleNamespace(
            report=SimpleNamespace(
                live=SimpleNamespace(selection_basis="evidence_score", selected=None)
            ),
            physical_execution_started_at_ns=None,
            physical_result=None,
            physical_candidate_id=None,
            run_dir=run_dir,
        ),
    )

    module._run_live(manifest, run_dir)

    assert events[0] == "paths:cpu"
    assert events[1:] == ["servers", "case"] * 10


def test_live_capability_run_records_a_completed_terminal_status(tmp_path, monkeypatch) -> None:
    """The top-level artifact must not remain running after every case terminates."""

    import scripts.run_libero_p532_object6 as module
    from capmas.evaluation.phase5_artifacts import Phase5RunDirectory
    from scripts.create_p532_object6_manifest import build_object6_manifest, write_manifest

    manifest_path = write_manifest(
        tmp_path / "manifest.json",
        build_object6_manifest(range(52, 62), _assets(tmp_path), molmo_device="cpu"),
    )
    online_run_dir = Phase5RunDirectory.create(tmp_path / "online", "online", "terminal")
    monkeypatch.setattr(module, "_setup_capx_paths", lambda: None)
    monkeypatch.setattr(module, "_start_capx_api_servers", lambda _config_path: [])
    monkeypatch.setattr(
        module,
        "_run_case",
        lambda _case, _case_dir: SimpleNamespace(
            report=SimpleNamespace(
                live=SimpleNamespace(selection_basis="none", selected=None)
            ),
            physical_execution_started_at_ns=None,
            physical_result=None,
            physical_candidate_id=None,
            run_dir=online_run_dir,
            decision_completed_at_ns=1,
        ),
    )

    result = module.run_capability(manifest_path, output_root=tmp_path / "outputs", dry_run=False)

    run_config = json.loads((result.run_dir.path / "run_config.json").read_text())
    assert run_config["status"] == "completed"
