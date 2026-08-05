from __future__ import annotations

import json
from pathlib import Path

from capmas.evaluation.ood import OODCase, OODReplayEvidence, OODSplitManifest
from capmas.evaluation.ood_statistics import aggregate_ood_pairs
from scripts.run_libero_p55_ood import normalize_capx_case


def _case(
    *,
    case_id: str,
    split: str,
    pair_id: str,
    seed: int,
    layout_family: str,
) -> OODCase:
    return OODCase(
        case_id=case_id,
        split=split,
        ood_type="none" if split == "id" else "layout",
        task_id="libero_spatial_0",
        task_goal="place bowl on plate",
        task_family="spatial-0",
        layout_family=layout_family,
        object_name="bowl",
        target_name="plate",
        seed=seed,
        pair_id=pair_id,
        config_path="config.yaml",
        candidate_artifact="candidates.json",
        candidate_artifact_sha256="a" * 64,
        environment_version="capx-test",
        generator_version="fixture-v1",
        parent_case_id="id-1" if split == "ood" else None,
    )


def _manifest() -> OODSplitManifest:
    cases = (
        _case(
            case_id="id-case-one",
            split="id",
            pair_id="pair-one",
            seed=1,
            layout_family="layout-a",
        ),
        _case(
            case_id="ood-case-one",
            split="ood",
            pair_id="pair-one",
            seed=1,
            layout_family="layout-b",
        ),
        _case(
            case_id="id-case-two",
            split="id",
            pair_id="pair-two",
            seed=2,
            layout_family="layout-a",
        ),
        _case(
            case_id="ood-case-two",
            split="ood",
            pair_id="pair-two",
            seed=2,
            layout_family="layout-c",
        ),
    )
    return OODSplitManifest(
        suite_id="p55-aggregation",
        manifest_version="1",
        cases=cases,
        id_task_families=("spatial-0",),
        ood_task_families=("spatial-0",),
        id_layout_families=("layout-a",),
        ood_layout_families=("layout-b", "layout-c"),
        memory_snapshot_version="memory-v1",
        robot_skill_snapshot_version="robot-v1",
        prompt_version="prompt-v1",
        code_revision="revision-v1",
        created_at_utc="2026-08-03T00:00:00Z",
    )


def _evidence(case: OODCase, success: bool | None) -> OODReplayEvidence:
    return OODReplayEvidence(
        case_id=case.case_id,
        pair_id=case.pair_id,
        condition="capmas",
        candidate_id=f"candidate-{case.case_id}",
        split=case.split,
        ood_type=case.ood_type,
        source_scene_version=1,
        candidate_fingerprint="fingerprint",
        evaluator_success=success,
        verifier_success=None,
        graph_completed=success is True,
        failure_class=None if success else "timeout" if success is None else "task_failure",
        recovery_count=0,
        human_intervention_count=0,
        latency_ms=10.0,
        provider_call_count=1,
        cache_hit_count=0,
        selection_basis="evidence_tie_break",
    )


def test_aggregate_reports_id_ood_gap_and_unknowns() -> None:
    manifest = _manifest()
    evidence = (
        _evidence(manifest.cases[0], True),
        _evidence(manifest.cases[1], False),
        _evidence(manifest.cases[2], None),
        _evidence(manifest.cases[3], False),
    )

    report = aggregate_ood_pairs(evidence, manifest=manifest)

    assert report.id_success_count == 1
    assert report.ood_success_count == 0
    assert report.infrastructure_unknown_count == 1
    assert report.ood_gap.estimate == 1.0
    assert report.selection_bases["evidence_tie_break"] == 4


def test_capx_normalization_marks_condition_and_unknown_verifier(tmp_path) -> None:
    trial = tmp_path / "trial_01_sandboxrc_0_reward_1.000_taskcompleted_1"
    trial.mkdir()
    (trial / "summary.txt").write_text("Task Completed: True\nReward: 1.0\n")

    evidence = normalize_capx_case(_manifest().cases[1], trial)

    assert evidence.condition == "capx"
    assert evidence.verifier_success is None
    assert evidence.evaluator_success is True
    assert evidence.shadow_only is True
