from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from capmas.evaluation.ood import OODCase, OODSplitManifest
from scripts.run_libero_p55_ood import OODRunConfig, _start_capx_api_servers, run_ood_suite


ROOT = Path(__file__).parents[1]
CANDIDATE_ARTIFACT = (
    ROOT
    / "outputs"
    / "phase5"
    / "P5.3_process_rehearsal_input_20260730"
    / "matched_candidates.json"
)


def test_p55_rejects_multiworker_same_gpu_rehearsal() -> None:
    with pytest.raises(ValueError, match="max_workers=1"):
        OODRunConfig(max_workers=2, gpu="5")


def test_p55_api_server_helper_uses_manifest_config_and_returns_processes() -> None:
    config_path = "libero_spatial_0.yaml"
    calls: list[object] = []
    processes = [object(), object()]

    result = _start_capx_api_servers(
        config_path,
        loader=lambda path: {"loaded_from": path},
        starter=lambda api_servers: calls.append(api_servers) or processes,
    )

    assert result == processes
    assert calls == [None]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case(
    *,
    case_id: str,
    split: str,
    layout_family: str,
    seed: int,
    parent_case_id: str | None,
    pair_id: str = "pair-1",
    layout_variant: dict[str, object] | None = None,
) -> OODCase:
    return OODCase(
        case_id=case_id,
        split=split,
        ood_type="none" if split == "id" else "layout",
        task_id="libero_spatial_0",
        task_goal="place bowl on plate",
        task_family="spatial-0",
        layout_family=layout_family,
        object_name="akita black bowl",
        target_name="plate",
        seed=seed,
        pair_id=pair_id,
        config_path="libero_spatial_0.yaml",
        candidate_artifact=str(CANDIDATE_ARTIFACT),
        candidate_artifact_sha256=_sha256(CANDIDATE_ARTIFACT),
        environment_version="capx-test",
        generator_version="fixture-v1",
        parent_case_id=parent_case_id,
        layout_variant=layout_variant or {},
    )


def _manifest_with_id_and_layout_ood_pair(*, shared_layout: bool = False) -> OODSplitManifest:
    id_case = _case(
        case_id="id-seed1",
        split="id",
        layout_family="layout-a",
        seed=1,
        parent_case_id=None,
    )
    ood_case = _case(
        case_id="ood-layout-seed2",
        split="ood",
        layout_family="layout-a" if shared_layout else "layout-b",
        seed=2,
        parent_case_id=id_case.case_id,
    )
    return OODSplitManifest(
        suite_id="p55-test-suite",
        manifest_version="1",
        cases=(id_case, ood_case),
        id_task_families=("spatial-0",),
        ood_task_families=("spatial-0",),
        id_layout_families=("layout-a",),
        ood_layout_families=(ood_case.layout_family,),
        memory_snapshot_version="memory-v1",
        robot_skill_snapshot_version="robot-v1",
        prompt_version="prompt-v1",
        code_revision="revision-v1",
        created_at_utc="2026-08-03T00:00:00Z",
    )


def _fake_online_outcome(
    candidates: tuple[object, ...],
    *,
    success: bool | None = True,
    physical_candidate_id: str | None = None,
) -> dict[str, object]:
    winner = physical_candidate_id or (
        candidates[0].candidate_id if success is not None else None
    )
    physical = None
    if winner is not None:
        physical = {
            "completed": success is True,
            "evaluator_success": success,
            "success": success,
        }
    return {
        "mode": "online_bounded",
        "physical_candidate_id": winner,
        "physical_result": physical,
        "provider_call_count": 1,
        "selection_latency_ms": 10.0,
        "live_selection_basis": "evidence_tie_break",
        "cache_stats": {"hits": 0, "misses": 1},
    }


def test_ood_suite_audits_before_running_any_case(tmp_path) -> None:
    calls: list[object] = []

    def fake_runner(**kwargs):
        calls.append(kwargs)
        return _fake_online_outcome(kwargs["candidates"])

    with pytest.raises(ValueError, match="layout family"):
        run_ood_suite(
            _manifest_with_id_and_layout_ood_pair(shared_layout=True),
            output_root=tmp_path,
            run_config=OODRunConfig(),
            online_runner=fake_runner,
        )
    assert calls == []


def test_ood_suite_runs_case_scoped_capmas_replay_and_writes_shadow_evidence(tmp_path) -> None:
    calls: list[dict[str, object]] = []

    def fake_runner(**kwargs):
        calls.append(kwargs)
        return _fake_online_outcome(
            kwargs["candidates"], success=None, physical_candidate_id=None
        )

    report = run_ood_suite(
        _manifest_with_id_and_layout_ood_pair(),
        output_root=tmp_path,
        run_config=OODRunConfig(),
        online_runner=fake_runner,
        executor_factory=lambda **_: None,
    )

    assert len(calls) == 2
    assert {call["seed"] for call in calls} == {1, 2}
    assert all(call["cache_mode"] == "disabled" for call in calls)
    assert report.case_count == 2
    assert report.failed_case_count == 0
    assert (report.suite_dir / "results" / "aggregate.json").exists()
    evidence = list(report.suite_dir.rglob("ood_replay.json"))
    assert evidence
    assert json.loads(evidence[0].read_text())["shadow_only"] is True


def test_ood_suite_does_not_count_reset_failure_as_task_failure(tmp_path) -> None:
    def fake_runner(**kwargs):
        winner = kwargs["candidates"][0].candidate_id
        return {
            "mode": "online_bounded",
            "physical_candidate_id": winner,
            "physical_result": {
                "completed": False,
                "evaluator_success": False,
                "success": False,
                "execution_valid": False,
                "failure_class": "reset_failure",
                "failure_reason": "LIBERO depth initialization failed",
            },
            "rehearsal_results": [
                {
                    "candidate_id": winner,
                    "seed": kwargs["seed"],
                    "failure_class": "reset_failure",
                    "failure_reason": "LIBERO depth initialization failed",
                    "failure_step": 0,
                    "latency_ms": 1.0,
                }
            ],
            "provider_call_count": 1,
            "selection_latency_ms": 10.0,
            "live_selection_basis": "evidence_tie_break",
        }

    report = run_ood_suite(
        _manifest_with_id_and_layout_ood_pair(),
        output_root=tmp_path,
        run_config=OODRunConfig(),
        online_runner=fake_runner,
        executor_factory=lambda **_: None,
    )

    assert report.aggregate.infrastructure_unknown_count == 2
    assert report.aggregate.failure_classes == {"reset_failure": 2}
    assert all(
        evidence.evaluator_success is None
        for result in report.case_results
        for evidence in result.evidence
    )
    summaries = list(report.suite_dir.rglob("rehearsal_failure_summary.json"))
    assert len(summaries) == 2
    summary = json.loads(summaries[0].read_text())
    assert summary["failure_classes"] == {"reset_failure": 1}
    assert summary["failed_count"] == 1


def test_ood_suite_passes_layout_variant_to_online_runner(tmp_path) -> None:
    layout_variant = {
        "variant_id": "translated-v1",
        "layout_family": "layout-b",
        "transforms": [],
    }
    id_case = _case(
        case_id="id-seed1",
        split="id",
        layout_family="layout-a",
        seed=1,
        parent_case_id=None,
        layout_variant={
            "variant_id": "native-v1",
            "layout_family": "layout-a",
            "transforms": [],
        },
    )
    ood_case = _case(
        case_id="ood-layout-seed2",
        split="ood",
        layout_family="layout-b",
        seed=2,
        parent_case_id=id_case.case_id,
        layout_variant=layout_variant,
    )
    manifest = OODSplitManifest(
        suite_id="p55-layout-pass-through",
        manifest_version="1",
        cases=(id_case, ood_case),
        id_task_families=("spatial-0",),
        ood_task_families=("spatial-0",),
        id_layout_families=("layout-a",),
        ood_layout_families=("layout-b",),
        memory_snapshot_version="memory-v1",
        robot_skill_snapshot_version="robot-v1",
        prompt_version="prompt-v1",
        code_revision="revision-v1",
        created_at_utc="2026-08-03T00:00:00Z",
    )
    calls: list[dict[str, object]] = []

    def fake_runner(**kwargs):
        calls.append(kwargs)
        return _fake_online_outcome(kwargs["candidates"])

    run_ood_suite(
        manifest,
        output_root=tmp_path,
        run_config=OODRunConfig(),
        online_runner=fake_runner,
        executor_factory=lambda **_: None,
    )

    assert calls[0]["layout_variant"] == id_case.layout_variant
    assert calls[1]["layout_variant"] == ood_case.layout_variant


def test_ood_suite_retains_case_failure_and_continues(tmp_path) -> None:
    def fake_runner(**kwargs):
        if kwargs["seed"] == 2:
            raise RuntimeError("case failed")
        return _fake_online_outcome(kwargs["candidates"])

    report = run_ood_suite(
        _manifest_with_id_and_layout_ood_pair(),
        output_root=tmp_path,
        run_config=OODRunConfig(),
        online_runner=fake_runner,
        executor_factory=lambda **_: None,
    )

    assert report.failed_case_count == 1
    failed_case = next(
        result.case_dir for result in report.case_results if result.status == "failed"
    )
    assert "seed2" in str(failed_case)
    assert (failed_case / "failure.json").exists()
    assert (failed_case / "case.json").exists()
    assert (failed_case / "run_config.json").exists()
    assert (failed_case / "manifest.json").exists()
