from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from capmas.evaluation.capability import (
    CapabilityCase,
    diagnose_family_capability,
    load_p55_capability_cases,
    run_capability_diagnosis,
)


def _capability_case(
    seed: int,
    *,
    reached: bool,
    success: bool | None,
    family_id: str = "object-6",
    failure: str | None = None,
    infrastructure_unknown: bool = False,
) -> CapabilityCase:
    return CapabilityCase(
        case_id=f"id-{family_id}-seed{seed}",
        family_id=family_id,
        seed=seed,
        split="id",
        reached_physical_execution=reached,
        evaluator_success=success,
        infrastructure_unknown=infrastructure_unknown,
        failure_class=failure,
        evidence_refs=(f"cases/id-{family_id}-seed{seed}/evidence/ood_replay.json",),
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_p55_suite_fixture(suite: Path) -> Path:
    for seed in range(1, 11):
        case_dir = suite / "cases" / f"20260807_0000{seed:02d}_id-spatial-0-seed{seed}"
        _write_json(
            case_dir / "case.json",
            {
                "case_id": f"id-spatial-0-seed{seed}",
                "pair_id": f"spatial-0-seed{seed}",
                "seed": seed,
                "split": "id",
                "task_family": "spatial-0",
            },
        )
        _write_json(
            case_dir / "summary.json",
            {
                "case_id": f"id-spatial-0-seed{seed}",
                "status": "completed",
                "primary_winner": "sg_pick:policy-0:0",
                "evaluator_success": False,
            },
        )
        _write_json(
            case_dir / "evidence" / "ood_replay.json",
            {
                "case_id": f"id-spatial-0-seed{seed}",
                "split": "id",
                "candidate_id": "sg_pick:policy-0:0",
                "evaluator_success": False,
                "failure_class": "POSTCONDITION_FAILED",
            },
        )
    _write_json(suite / "suite_manifest.json", {"manifest_sha256": "c" * 64})
    return suite


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _latest_run_dir(output_root: Path) -> Path:
    runs_root = output_root / "P5.6.0_capability_diagnosis"
    return max((path for path in runs_root.iterdir() if path.is_dir()), key=lambda path: path.name)


def test_capability_gate_requires_execution_reach_and_one_success() -> None:
    cases = tuple(
        _capability_case(
            seed,
            reached=seed <= 8,
            success=seed == 1,
            failure=None if seed == 1 else "POSTCONDITION_FAILED",
        )
        for seed in range(1, 11)
    )

    report, handoff = diagnose_family_capability(
        cases,
        family_id="object-6",
        source_manifest_sha256="a" * 64,
    )

    assert report.eligible is True
    assert report.execution_reach_rate == 0.8
    assert handoff is None


def test_capability_gate_rejects_mixed_success_with_untyped_failure() -> None:
    cases = tuple(
        _capability_case(
            seed,
            reached=True,
            success=seed == 1,
            failure=None if seed in {1, 2} else "POSTCONDITION_FAILED",
        )
        for seed in range(1, 11)
    )

    report, handoff = diagnose_family_capability(
        cases,
        family_id="object-6",
        source_manifest_sha256="a" * 64,
    )

    assert report.eligible is False
    assert report.gate_failures == ("UNTYPED_FAILURE",)
    assert handoff is not None


def test_zero_success_family_emits_typed_p53_2_handoff() -> None:
    cases = tuple(
        _capability_case(
            seed,
            reached=True,
            success=False,
            family_id="spatial-0",
            failure="POSTCONDITION_FAILED",
        )
        for seed in range(1, 11)
    )

    report, handoff = diagnose_family_capability(
        cases,
        family_id="spatial-0",
        source_manifest_sha256="b" * 64,
    )

    assert report.eligible is False
    assert "NO_EVALUATOR_SUCCESS" in report.gate_failures
    assert handoff is not None
    assert handoff.package == "P5.3.2 Task-Family Capability Repair"
    assert handoff.suspected_owner == "verification_or_robot_skill"
    assert handoff.acceptance_test == (
        "rerun the same frozen ten-seed capability manifest with zero infrastructure unknowns, "
        "at least 80% physical execution reach, typed provenance for every failure, and at "
        "least one evaluator success"
    )


def test_capability_cli_does_not_modify_source_suite(tmp_path: Path) -> None:
    suite = _write_p55_suite_fixture(tmp_path / "source")
    before = _tree_digest(suite)

    result = run_capability_diagnosis(
        suite_dir=suite,
        families=("spatial-0",),
        output_root=tmp_path / "outputs",
    )

    assert _tree_digest(suite) == before
    assert (result.run_dir / "artifacts" / "p53_2_spatial-0.json").exists()


def test_loader_resolves_exact_case_summary_not_nested_online_summary(tmp_path: Path) -> None:
    suite = _write_p55_suite_fixture(tmp_path / "source")
    nested_summary = (
        suite
        / "cases"
        / "20260807_000001_id-spatial-0-seed1"
        / "online"
        / "summary.json"
    )
    _write_json(
        nested_summary,
        {
            "case_id": "id-spatial-0-seed1",
            "status": "completed",
            "primary_winner": None,
            "evaluator_success": True,
        },
    )

    cases = load_p55_capability_cases(suite, "spatial-0")

    seed_one = next(case for case in cases if case.seed == 1)
    assert seed_one.reached_physical_execution is True
    assert seed_one.evaluator_success is False


def test_loader_fails_closed_for_incomplete_source(tmp_path: Path) -> None:
    suite = _write_p55_suite_fixture(tmp_path / "source")
    (suite / "cases" / "20260807_000001_id-spatial-0-seed1" / "evidence" / "ood_replay.json").unlink()

    with pytest.raises(ValueError, match="ood_replay.json"):
        load_p55_capability_cases(suite, "spatial-0")


def test_loader_fails_closed_when_summary_winner_mismatches_evidence_candidate(
    tmp_path: Path,
) -> None:
    suite = _write_p55_suite_fixture(tmp_path / "source")
    _write_json(
        suite
        / "cases"
        / "20260807_000001_id-spatial-0-seed1"
        / "evidence"
        / "ood_replay.json",
        {
            "case_id": "id-spatial-0-seed1",
            "split": "id",
            "candidate_id": "different-policy:1",
            "evaluator_success": False,
            "failure_class": "POSTCONDITION_FAILED",
        },
    )

    with pytest.raises(ValueError, match="primary_winner.*candidate_id"):
        load_p55_capability_cases(suite, "spatial-0")


def test_loader_treats_coherent_no_selection_as_no_physical_execution(
    tmp_path: Path,
) -> None:
    suite = _write_p55_suite_fixture(tmp_path / "source")
    case_dir = suite / "cases" / "20260807_000001_id-spatial-0-seed1"
    _write_json(
        case_dir / "summary.json",
        {
            "case_id": "id-spatial-0-seed1",
            "status": "completed",
            "primary_winner": None,
            "evaluator_success": False,
        },
    )
    _write_json(
        case_dir / "evidence" / "ood_replay.json",
        {
            "case_id": "id-spatial-0-seed1",
            "split": "id",
            "candidate_id": "unselected",
            "evaluator_success": False,
            "failure_class": "POSTCONDITION_FAILED",
        },
    )

    cases = load_p55_capability_cases(suite, "spatial-0")

    seed_one = next(case for case in cases if case.seed == 1)
    assert seed_one.reached_physical_execution is False


def test_run_rejects_output_root_inside_suite_before_creating_output(
    tmp_path: Path,
) -> None:
    suite = _write_p55_suite_fixture(tmp_path / "source")
    suite_link = tmp_path / "source-link"
    suite_link.symlink_to(suite, target_is_directory=True)
    before = _tree_digest(suite)

    with pytest.raises(ValueError, match="output_root.*suite_dir"):
        run_capability_diagnosis(
            suite_dir=suite,
            families=("spatial-0",),
            output_root=suite_link / "outputs",
        )

    assert _tree_digest(suite) == before
    assert not (suite / "outputs").exists()


def test_run_finalizes_failed_artifacts_after_malformed_case_json(
    tmp_path: Path,
) -> None:
    suite = _write_p55_suite_fixture(tmp_path / "source")
    output_root = tmp_path / "outputs"
    (
        suite / "cases" / "20260807_000001_id-spatial-0-seed1" / "evidence" / "ood_replay.json"
    ).write_text("{malformed json", encoding="utf-8")

    with pytest.raises(ValueError, match="malformed JSON"):
        run_capability_diagnosis(
            suite_dir=suite,
            families=("spatial-0",),
            output_root=output_root,
        )

    run_dir = _latest_run_dir(output_root)
    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    runner_log = (run_dir / "logs" / "runner.log").read_text(encoding="utf-8")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest_paths = {entry["path"] for entry in manifest["files"]}

    assert run_config["status"] == "failed"
    assert run_config["error_type"] == "ValueError"
    assert "malformed JSON" in run_config["error"]
    assert "api_key" not in run_config["error"]
    assert "status=failed" in runner_log
    assert {"run_config.json", "logs/runner.log"}.issubset(manifest_paths)


def test_run_rejects_duplicate_families_before_creating_run_dir(tmp_path: Path) -> None:
    suite = _write_p55_suite_fixture(tmp_path / "source")
    output_root = tmp_path / "outputs"

    with pytest.raises(ValueError, match="duplicate family"):
        run_capability_diagnosis(
            suite_dir=suite,
            families=("spatial-0", "spatial-0"),
            output_root=output_root,
        )

    assert not output_root.exists()
