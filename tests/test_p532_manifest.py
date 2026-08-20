from __future__ import annotations

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
