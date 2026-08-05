from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest

from capmas.evaluation.ood import (
    OODCase,
    OODReplayEvidence,
    OODSplitManifest,
    dump_ood_manifest,
    load_ood_manifest,
    manifest_sha256,
    validate_ood_manifest,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _case(
    *,
    case_id: str = "id-1",
    split: str = "id",
    ood_type: str = "none",
    pair_id: str = "pair-1",
    parent_case_id: str | None = None,
    task_goal: str = "place bowl on plate",
    task_family: str = "same-id-family",
    layout_family: str = "layout-a",
    seed: int = 1,
) -> OODCase:
    if split == "ood" and ood_type == "none":
        ood_type = "layout"
    return OODCase(
        case_id=case_id,
        split=split,
        ood_type=ood_type,
        task_id="libero_spatial_0",
        task_goal=task_goal,
        task_family=task_family,
        layout_family=layout_family,
        object_name="akita black bowl",
        target_name="plate",
        seed=seed,
        pair_id=pair_id,
        config_path="config.yaml",
        candidate_artifact="candidates.json",
        candidate_artifact_sha256=_digest("candidates"),
        environment_version="capx-test",
        generator_version="fixture-v1",
        parent_case_id=parent_case_id,
    )


def _manifest(cases: tuple[OODCase, ...], **overrides: object) -> OODSplitManifest:
    payload = dict(
        suite_id="p55-test",
        manifest_version="1",
        cases=cases,
        id_task_families=tuple(case.task_family for case in cases if case.split == "id"),
        ood_task_families=tuple(case.task_family for case in cases if case.split == "ood"),
        id_layout_families=tuple(case.layout_family for case in cases if case.split == "id"),
        ood_layout_families=tuple(case.layout_family for case in cases if case.split == "ood"),
        memory_snapshot_version="memory-v1",
        robot_skill_snapshot_version="robot-v1",
        prompt_version="prompt-v1",
        code_revision="revision-v1",
        created_at_utc="2026-08-03T00:00:00Z",
        manifest_sha256="",
    )
    payload.update(overrides)
    return OODSplitManifest(**payload)


def test_layout_pair_requires_same_goal_and_different_layout_family() -> None:
    id_case = _case(
        case_id="id-1",
        split="id",
        ood_type="none",
        pair_id="pair-1",
        task_goal="place bowl on plate",
        task_family="spatial-0",
        layout_family="layout-a",
    )
    ood_case = _case(
        case_id="ood-1",
        split="ood",
        ood_type="layout",
        pair_id="pair-1",
        parent_case_id="id-1",
        task_goal="place bowl on plate",
        task_family="spatial-0",
        layout_family="layout-b",
    )
    manifest = _manifest(
        (id_case, ood_case),
        id_task_families=("spatial-0",),
        ood_task_families=("spatial-0",),
        id_layout_families=("layout-a",),
        ood_layout_families=("layout-b",),
    )
    validate_ood_manifest(manifest)


def test_manifest_digest_excludes_self_digest() -> None:
    manifest = _manifest((_case(),))
    digest = manifest_sha256(manifest)
    assert digest == manifest_sha256(replace(manifest, manifest_sha256=digest))


def test_manifest_rejects_task_family_overlap_for_task_object_ood() -> None:
    id_case = _case(case_id="id-1", task_family="same-id-family")
    ood_case = _case(
        case_id="ood-1",
        split="ood",
        ood_type="task_object",
        pair_id="pair-1",
        parent_case_id="id-1",
        task_family="same-id-family",
        layout_family="layout-b",
    )
    manifest = _manifest(
        (id_case, ood_case),
        id_task_families=("same-id-family",),
        ood_task_families=("same-id-family",),
        id_layout_families=("layout-a",),
        ood_layout_families=("layout-b",),
    )
    with pytest.raises(ValueError, match="task family"):
        validate_ood_manifest(manifest)


def test_ood_replay_evidence_preserves_unknown_evaluator_and_is_shadow_only() -> None:
    evidence = OODReplayEvidence(
        case_id="ood-1",
        pair_id="pair-1",
        condition="capmas",
        candidate_id="candidate-a",
        split="ood",
        ood_type="layout",
        source_scene_version=1,
        candidate_fingerprint="fp",
        evaluator_success=None,
        verifier_success=None,
        graph_completed=False,
        failure_class="timeout",
        recovery_count=0,
        human_intervention_count=0,
        latency_ms=10.0,
        provider_call_count=1,
        cache_hit_count=0,
    )
    assert evidence.evaluator_success is None
    assert evidence.shadow_only is True


def test_manifest_round_trip_recomputes_and_persists_digest(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    manifest = _manifest(
        (
            _case(case_id="id-1"),
            _case(
                case_id="ood-1",
                split="ood",
                ood_type="layout",
                pair_id="pair-1",
                parent_case_id="id-1",
                layout_family="layout-b",
            ),
        ),
        id_task_families=("same-id-family",),
        ood_task_families=("same-id-family",),
        id_layout_families=("layout-a",),
        ood_layout_families=("layout-b",),
    )
    dump_ood_manifest(path, manifest)
    loaded = load_ood_manifest(path)
    assert loaded.manifest_sha256 == manifest_sha256(loaded)
    assert loaded.cases[1].parent_case_id == "id-1"
    assert json.loads(path.read_text())['manifest_sha256'] == loaded.manifest_sha256
