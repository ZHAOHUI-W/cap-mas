from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

import scripts.run_libero_p56_collect as collection_module
from capmas.contracts.calibration import (
    FEATURE_SCHEMA_VERSION,
    CalibrationCollectionCase,
    CalibrationCollectionManifest,
    CandidateFeatureSnapshot,
    HorizonLabel,
    collection_manifest_sha256,
)
from capmas.evaluation.feature_snapshots import FEATURE_GROUPS_V1
from scripts.create_p56_object6_manifests import create_object6_manifests
from scripts.run_libero_p56_collect import (
    CollectionRunConfig,
    run_collection,
    summarize_collection,
)

ROOT = Path(__file__).parents[1]
CONFIG_PATH = ROOT / "configs" / "phase5" / "capx_libero_object_6_nonprivileged.yaml"
CANDIDATE_ARTIFACT = (
    ROOT
    / "outputs"
    / "phase5"
    / "P5.5_real_layout_assets_20260803"
    / "candidates"
    / "object_6.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _collection_manifest(seeds: tuple[int, ...] = (11,)) -> CalibrationCollectionManifest:
    cases = tuple(
        CalibrationCollectionCase(
            case_id=f"object-6-id-seed{seed}",
            lineage_group_id=f"object-6-id-seed{seed}",
            family_id="object-6",
            task_id="libero_object_6",
            seed=seed,
            split_identity="id",
            config_path=str(CONFIG_PATH.relative_to(ROOT)),
            config_sha256=_sha256(CONFIG_PATH),
            candidate_artifact=str(CANDIDATE_ARTIFACT.relative_to(ROOT)),
            candidate_artifact_sha256=_sha256(CANDIDATE_ARTIFACT),
            object_name="butter",
            target_name="basket",
            layout_family="native-object-6",
            layout_variant={
                "variant_id": "native-object-6-zero-delta",
                "layout_family": "native-object-6",
                "generator_version": "capmas-layout-v1",
                "transforms": [
                    {"body_name": "butter_1_main", "translation_delta_xyz": [0.0, 0.0, 0.0]},
                    {"body_name": "basket_1_main", "translation_delta_xyz": [0.0, 0.0, 0.0]},
                ],
            },
        )
        for seed in seeds
    )
    manifest = CalibrationCollectionManifest(
        manifest_id="",
        schema_version="p56.collection.v1",
        cases=cases,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        memory_skill_version="p56-memory-object6-frozen-v1",
        robot_skill_version="p56-robot-object6-frozen-v1",
        prompt_version="p56-object6-prompt-v1",
        environment_version="capx-libero-object6-nonprivileged-v1",
        code_revision="test-revision",
    )
    digest = collection_manifest_sha256(manifest)
    return replace(manifest, manifest_id=f"sha256:{digest}", manifest_sha256=digest)


def _snapshot(
    candidate_id: str,
    *,
    episode_id: str = "object-6-id-seed11",
    fingerprint: str | None = None,
    captured_at_ns: int = 100,
) -> CandidateFeatureSnapshot:
    digest = fingerprint or hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()
    return CandidateFeatureSnapshot(
        episode_id=episode_id,
        episode_epoch=1,
        family_id="object-6",
        candidate_id=candidate_id,
        candidate_fingerprint=digest,
        scene_version=1,
        map_version=1,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        captured_at_ns=captured_at_ns,
        collection_lane="physical",
        features={name: None for name in FEATURE_GROUPS_V1},
        feature_status={name: "unknown" for name in FEATURE_GROUPS_V1},
        correlation_groups=FEATURE_GROUPS_V1,
        memory_skill_version="p56-memory-object6-frozen-v1",
        robot_skill_version="p56-robot-object6-frozen-v1",
        selection_probability=None,
        evidence_refs=("artifact://decision-time/snapshot",),
        evidence_providers={"decision": "fake-online-runner"},
        rewrite_metadata={"candidate_id": candidate_id},
    )


def _horizon() -> HorizonLabel:
    return HorizonLabel(
        planned_critical_path_actions=2,
        planned_critical_path_subgoals=2,
        planned_checkpoint_subgraphs=1,
        attempted_actions=2,
        completed_actions=2,
        attempted_subgoals=2,
        completed_subgoals=2,
        attempted_checkpoints=1,
        completed_checkpoints=1,
        planned_source="mission_graph",
        realized_source="execution_trace",
        planned_valid=True,
        realized_valid=True,
    )


def _online_outcome_with_two_snapshots(
    *,
    selected: str,
    success: bool | None,
    episode_id: str = "object-6-id-seed11",
) -> dict[str, object]:
    snapshots = (
        _snapshot("candidate-a", episode_id=episode_id),
        _snapshot("candidate-b", episode_id=episode_id),
    )
    physical_result = None
    if selected:
        physical_result = {
            "completed": success,
            "graph_completed": success,
            "verifier_success": success,
            "evaluator_success": success,
            "success": success,
            "failure_class": None if success else "task_failure",
            "horizon": _horizon().to_dict(),
            "graph_events": [
                {
                    "sequence": 0,
                    "kind": "subgraph_started",
                    "subgraph_id": "sg_pick",
                    "node_id": None,
                    "node_type": None,
                    "attempt": 1,
                    "outcome": None,
                    "occurred_at_ns": 101,
                }
            ],
        }
    return {
        "mode": "online_bounded",
        "physical_candidate_id": selected,
        "physical_result": physical_result,
        "provider_call_count": 1,
        "selection_latency_ms": 10.0,
        "feature_snapshots": snapshots,
        "decision_completed_at_ns": 100,
        "physical_execution_started_at_ns": 110 if selected else None,
    }


def _online_outcome(
    *,
    success: bool | None,
    episode_id: str = "object-6-id-seed11",
) -> dict[str, object]:
    return _online_outcome_with_two_snapshots(
        selected="candidate-a",
        success=success,
        episode_id=episode_id,
    )


def test_object6_manifests_are_disjoint_complete_fixed_blocks() -> None:
    first, second = create_object6_manifests(ROOT)

    assert tuple(case.seed for case in first.cases) == tuple(range(11, 21))
    assert tuple(case.seed for case in second.cases) == tuple(range(21, 31))
    assert {case.seed for case in first.cases}.isdisjoint(case.seed for case in second.cases)
    assert all(case.split_identity == "id" for case in (*first.cases, *second.cases))
    assert all(case.family_id == "object-6" for case in (*first.cases, *second.cases))
    assert all(case.lineage_group_id == case.case_id for case in (*first.cases, *second.cases))


def test_collection_manifest_round_trip_preserves_digest() -> None:
    first, _ = create_object6_manifests(ROOT)
    restored = CalibrationCollectionManifest.from_dict(first.to_dict())

    assert restored == first
    assert collection_manifest_sha256(restored) == first.manifest_sha256
    assert restored.manifest_id == f"sha256:{first.manifest_sha256}"
    assert list(restored.to_dict()) == sorted(restored.to_dict())


def test_generator_is_byte_identical_and_check_validates_committed_manifests(tmp_path) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/create_p56_object6_manifests.py",
            "--project-root",
            str(ROOT),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=True,
    )
    first_bytes = (tmp_path / "p56_object6_id_seeds_11_20.json").read_bytes()
    subprocess.run(
        [
            sys.executable,
            "scripts/create_p56_object6_manifests.py",
            "--project-root",
            str(ROOT),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=True,
    )
    assert (tmp_path / "p56_object6_id_seeds_11_20.json").read_bytes() == first_bytes
    subprocess.run(
        [
            sys.executable,
            "scripts/create_p56_object6_manifests.py",
            "--project-root",
            str(ROOT),
            "--output-dir",
            str(tmp_path),
            "--check",
        ],
        cwd=ROOT,
        check=True,
    )


def test_collection_captures_all_candidates_but_labels_only_selected(tmp_path) -> None:
    manifest = _collection_manifest(seeds=(11,))

    def fake_runner(**kwargs: object) -> dict[str, object]:
        assert kwargs["calibration_context"].episode_id == "object-6-id-seed11"
        assert kwargs["mode"] == "online_bounded"
        assert kwargs["cache_mode"] == "disabled"
        assert kwargs["selection_repeats"] == 1
        assert kwargs["pool_config"].max_workers == 1
        assert kwargs["gpu"] == "5"
        return _online_outcome_with_two_snapshots(selected="candidate-a", success=True)

    report = run_collection(
        manifest,
        output_root=tmp_path,
        run_config=CollectionRunConfig(max_workers=1, gpu="5"),
        online_runner=fake_runner,
        executor_factory=lambda **kwargs: object(),
    )

    assert report.completed_cases == 1
    rows = json.loads((report.suite_dir / "results" / "outcomes.json").read_text())
    assert [(row["candidate_id"], row["tier"], row["task_success"]) for row in rows] == [
        ("candidate-a", "A", True),
        ("candidate-b", "C", None),
    ]
    case_dirs = [path for path in (report.suite_dir / "cases").iterdir() if path.is_dir()]
    assert len(case_dirs) == 1
    assert (case_dirs[0] / "evidence" / "decision_snapshots.json").exists()
    assert (case_dirs[0] / "evidence" / "physical_payload.json").exists()
    assert (case_dirs[0] / "evidence" / "horizon.json").exists()


@pytest.mark.parametrize("evaluator_success", [None, "true"])
def test_collection_never_uses_generic_success_as_evaluator_label(
    tmp_path, evaluator_success: object
) -> None:
    def fake_runner(**kwargs: object) -> dict[str, object]:
        outcome = _online_outcome(
            success=True,
            episode_id=kwargs["calibration_context"].episode_id,
        )
        physical_result = outcome["physical_result"]
        assert isinstance(physical_result, dict)
        if evaluator_success is None:
            physical_result.pop("evaluator_success")
        else:
            physical_result["evaluator_success"] = evaluator_success
        physical_result["success"] = True
        return outcome

    report = run_collection(
        _collection_manifest(),
        output_root=tmp_path,
        run_config=CollectionRunConfig(),
        online_runner=fake_runner,
        executor_factory=lambda **kwargs: object(),
    )

    selected = next(
        row
        for row in json.loads((report.suite_dir / "results" / "outcomes.json").read_text())
        if row["candidate_id"] == "candidate-a"
    )
    assert selected["execution_status"] == "selected_executed"
    assert selected["tier"] == "C"
    assert selected["task_success"] is None


def test_post_runner_failure_preserves_all_raw_evidence(tmp_path) -> None:
    def fake_runner(**kwargs: object) -> dict[str, object]:
        outcome = _online_outcome(
            success=True,
            episode_id=kwargs["calibration_context"].episode_id,
        )
        physical_result = outcome["physical_result"]
        assert isinstance(physical_result, dict)
        physical_result["horizon"] = {"raw_horizon": "not-a-horizon-contract"}
        return outcome

    report = run_collection(
        _collection_manifest(),
        output_root=tmp_path,
        run_config=CollectionRunConfig(),
        online_runner=fake_runner,
        executor_factory=lambda **kwargs: object(),
    )

    assert report.failed_cases == 1
    case_dir = report.cases[0].case_dir
    online = json.loads((case_dir / "results" / "online.json").read_text())
    snapshots = json.loads((case_dir / "evidence" / "decision_snapshots.json").read_text())
    physical = json.loads((case_dir / "evidence" / "physical_payload.json").read_text())
    horizon = json.loads((case_dir / "evidence" / "horizon.json").read_text())
    graph_events = json.loads((case_dir / "evidence" / "graph_events.json").read_text())
    failure = json.loads((case_dir / "failure.json").read_text())

    assert online["physical_candidate_id"] == "candidate-a"
    assert [snapshot["candidate_id"] for snapshot in snapshots] == [
        "candidate-a",
        "candidate-b",
    ]
    assert physical["success"] is True
    assert horizon == {"raw_horizon": "not-a-horizon-contract"}
    assert graph_events == physical["graph_events"]
    assert failure["stage"] == "normalization"
    assert failure["error_type"] == "ValueError"


def test_raw_artifact_persistence_retry_keeps_original_failure_typed(
    tmp_path, monkeypatch
) -> None:
    persistence_errors = iter(
        [OSError("initial raw artifact write failed"), RuntimeError("retry obscured error")]
    )

    def failing_raw_writer(*args: object, **kwargs: object) -> None:
        raise next(persistence_errors)

    monkeypatch.setattr(collection_module, "_write_raw_artifacts", failing_raw_writer)

    report = run_collection(
        _collection_manifest(),
        output_root=tmp_path,
        run_config=CollectionRunConfig(),
        online_runner=lambda **kwargs: _online_outcome(
            success=True,
            episode_id=kwargs["calibration_context"].episode_id,
        ),
        executor_factory=lambda **kwargs: object(),
    )

    assert report.failed_cases == 1
    failure = json.loads((report.cases[0].case_dir / "failure.json").read_text())
    assert failure["stage"] == "persistence"
    assert failure["error_type"] == "OSError"
    assert failure["error"] == "initial raw artifact write failed"
    assert failure["raw_artifact_persistence_error_type"] == "RuntimeError"
    assert failure["raw_artifact_persistence_error"] == "retry obscured error"


def test_case_finalization_failure_is_typed_and_fail_fast_continues(
    tmp_path, monkeypatch
) -> None:
    original_finalize = collection_module.Phase5RunDirectory.finalize_manifest

    def failing_case_finalize(run_dir: object) -> object:
        path = run_dir.path
        if path.parent.name == "cases" and path.name.endswith("seed12"):
            raise OSError("case manifest persistence failed")
        return original_finalize(run_dir)

    monkeypatch.setattr(
        collection_module.Phase5RunDirectory,
        "finalize_manifest",
        failing_case_finalize,
    )

    report = run_collection(
        _collection_manifest(seeds=(11, 12, 13)),
        output_root=tmp_path,
        run_config=CollectionRunConfig(fail_fast=True),
        online_runner=lambda **kwargs: _online_outcome(
            success=kwargs["seed"] == 11,
            episode_id=kwargs["calibration_context"].episode_id,
        ),
        executor_factory=lambda **kwargs: object(),
    )

    assert [case.seed for case in report.cases] == [11, 12, 13]
    assert report.completed_cases == 2
    assert report.failed_cases == 1
    assert report.positive_count == 1
    assert report.negative_count == 1
    failure = json.loads((report.cases[1].case_dir / "failure.json").read_text())
    assert failure["stage"] == "persistence"
    assert failure["error_type"] == "OSError"


@pytest.mark.parametrize("preparation_failure", ["context", "pool"])
def test_fail_fast_continues_after_online_preparation_failure(
    tmp_path, monkeypatch, preparation_failure: str
) -> None:
    calls = []
    if preparation_failure == "context":
        original_context = collection_module.CalibrationCollectionContext

        def failing_context(**kwargs: object) -> object:
            if kwargs["episode_id"] == "object-6-id-seed12":
                raise ValueError("context construction failed")
            return original_context(**kwargs)

        monkeypatch.setattr(collection_module, "CalibrationCollectionContext", failing_context)
    else:
        original_pool_config = collection_module.RehearsalPoolConfig
        pool_calls = 0

        def failing_pool_config(**kwargs: object) -> object:
            nonlocal pool_calls
            pool_calls += 1
            if pool_calls == 2:
                raise ValueError("online call argument preparation failed")
            return original_pool_config(**kwargs)

        monkeypatch.setattr(collection_module, "RehearsalPoolConfig", failing_pool_config)

    def fake_runner(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs["seed"])
        return _online_outcome(
            success=False,
            episode_id=kwargs["calibration_context"].episode_id,
        )

    report = run_collection(
        _collection_manifest(seeds=(11, 12, 13)),
        output_root=tmp_path,
        run_config=CollectionRunConfig(fail_fast=True),
        online_runner=fake_runner,
        executor_factory=lambda **kwargs: object(),
    )

    assert calls == [11, 13]
    assert [case.seed for case in report.cases] == [11, 12, 13]
    assert report.completed_cases == 2
    assert report.failed_cases == 1
    failure = json.loads((report.cases[1].case_dir / "failure.json").read_text())
    assert failure["stage"] == "validation"


def test_fail_fast_continues_after_normalization_failure(tmp_path) -> None:
    calls = []

    def fake_runner(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs["seed"])
        outcome = _online_outcome(
            success=False,
            episode_id=kwargs["calibration_context"].episode_id,
        )
        if kwargs["seed"] == 12:
            physical_result = outcome["physical_result"]
            assert isinstance(physical_result, dict)
            physical_result["horizon"] = {"invalid": True}
        return outcome

    report = run_collection(
        _collection_manifest(seeds=(11, 12, 13)),
        output_root=tmp_path,
        run_config=CollectionRunConfig(fail_fast=True),
        online_runner=fake_runner,
        executor_factory=lambda **kwargs: object(),
    )

    assert calls == [11, 12, 13]
    assert report.completed_cases == 2
    assert report.failed_cases == 1
    failure = json.loads((report.cases[1].case_dir / "failure.json").read_text())
    assert failure["stage"] == "normalization"


def test_collection_runs_whole_block_without_adaptive_stopping(tmp_path) -> None:
    calls = []

    def fake_runner(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs["seed"])
        return _online_outcome(
            success=kwargs["seed"] == 11,
            episode_id=kwargs["calibration_context"].episode_id,
        )

    report = run_collection(
        _collection_manifest(seeds=(11, 12, 13)),
        output_root=tmp_path,
        run_config=CollectionRunConfig(),
        online_runner=fake_runner,
        executor_factory=lambda **kwargs: object(),
    )

    assert calls == [11, 12, 13]
    assert report.completed_cases == 3
    assert report.positive_count == 1
    assert report.negative_count == 2


def test_collection_retains_typed_failure_and_continues_unless_fail_fast(tmp_path) -> None:
    calls = []

    def fake_runner(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs["seed"])
        if kwargs["seed"] == 12:
            raise RuntimeError("executor lost transport")
        return _online_outcome(
            success=True,
            episode_id=kwargs["calibration_context"].episode_id,
        )

    report = run_collection(
        _collection_manifest(seeds=(11, 12, 13)),
        output_root=tmp_path,
        run_config=CollectionRunConfig(fail_fast=False),
        online_runner=fake_runner,
        executor_factory=lambda **kwargs: object(),
    )

    assert calls == [11, 12, 13]
    assert report.completed_cases == 2
    assert report.failed_cases == 1
    failures = list(report.suite_dir.rglob("failure.json"))
    assert len(failures) == 1
    failure = json.loads(failures[0].read_text())
    assert failure["error_type"] == "RuntimeError"
    assert failure["stage"] == "online_runner"


def test_fail_fast_continues_after_non_infrastructure_failure(tmp_path) -> None:
    calls = []

    def fake_runner(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs["seed"])
        episode_id = kwargs["calibration_context"].episode_id
        if kwargs["seed"] == 12:
            episode_id = "wrong-episode"
        return _online_outcome(success=False, episode_id=episode_id)

    report = run_collection(
        _collection_manifest(seeds=(11, 12, 13)),
        output_root=tmp_path,
        run_config=CollectionRunConfig(fail_fast=True),
        online_runner=fake_runner,
        executor_factory=lambda **kwargs: object(),
    )

    assert calls == [11, 12, 13]
    assert report.completed_cases == 2
    assert report.failed_cases == 1
    assert report.negative_count == 2
    failure = json.loads((report.cases[1].case_dir / "failure.json").read_text())
    assert failure["stage"] == "validation"


@pytest.mark.parametrize("failure_source", ["executor", "runner"])
def test_fail_fast_stops_after_infrastructure_exception(tmp_path, failure_source: str) -> None:
    constructed = []
    calls = []

    def executor_factory(**kwargs: object) -> object:
        constructed.append(kwargs["seed"])
        if failure_source == "executor" and kwargs["seed"] == 12:
            raise RuntimeError("executor construction failed")
        return object()

    def fake_runner(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs["seed"])
        if failure_source == "runner" and kwargs["seed"] == 12:
            raise RuntimeError("online runner failed")
        return _online_outcome(
            success=False,
            episode_id=kwargs["calibration_context"].episode_id,
        )

    with pytest.raises(RuntimeError, match="P5.6 collection case failed"):
        run_collection(
            _collection_manifest(seeds=(11, 12, 13)),
            output_root=tmp_path,
            run_config=CollectionRunConfig(fail_fast=True),
            online_runner=fake_runner,
            executor_factory=executor_factory,
        )

    assert constructed == [11, 12]
    assert calls == ([11] if failure_source == "executor" else [11, 12])
    failure_path = next(tmp_path.rglob("failure.json"))
    failure = json.loads(failure_path.read_text())
    expected_stage = "executor_construction" if failure_source == "executor" else "online_runner"
    assert failure["stage"] == expected_stage


def test_collection_preflight_rejects_candidate_digest_before_runtime(tmp_path) -> None:
    calls = []
    manifest = _collection_manifest(seeds=(11,))
    bad_case = replace(manifest.cases[0], candidate_artifact_sha256="0" * 64)
    bad_manifest = replace(manifest, cases=(bad_case,), manifest_sha256="", manifest_id="")

    with pytest.raises(ValueError, match="candidate artifact digest"):
        run_collection(
            bad_manifest,
            output_root=tmp_path,
            run_config=CollectionRunConfig(),
            online_runner=(
                lambda **kwargs: calls.append(kwargs)
                or _online_outcome(
                    success=True,
                    episode_id=kwargs["calibration_context"].episode_id,
                )
            ),
            executor_factory=lambda **kwargs: object(),
        )

    assert calls == []


def test_summary_combines_sources_and_rejects_duplicates(tmp_path) -> None:
    def fake_runner(**kwargs: object) -> dict[str, object]:
        return _online_outcome(
            success=kwargs["seed"] % 2 == 0,
            episode_id=kwargs["calibration_context"].episode_id,
        )

    first = run_collection(
        _collection_manifest(seeds=(11, 12)),
        output_root=tmp_path,
        run_config=CollectionRunConfig(),
        online_runner=fake_runner,
        executor_factory=lambda **kwargs: object(),
    )
    second = run_collection(
        _collection_manifest(seeds=(13, 14)),
        output_root=tmp_path,
        run_config=CollectionRunConfig(),
        online_runner=fake_runner,
        executor_factory=lambda **kwargs: object(),
    )

    summary = summarize_collection((first.suite_dir, second.suite_dir))

    assert summary.admissible_tier_a_count == 4
    assert summary.positive_count == 2
    assert summary.negative_count == 2
    assert summary.eligible_20_5_5 is False
    with pytest.raises(ValueError, match="duplicate case"):
        summarize_collection((first.suite_dir, first.suite_dir))


def test_summary_rejects_duplicate_identity_for_failed_cases_without_lineages(tmp_path) -> None:
    def failing_runner(**kwargs: object) -> dict[str, object]:
        raise RuntimeError(f"runner unavailable for seed {kwargs['seed']}")

    first = run_collection(
        _collection_manifest(seeds=(11,)),
        output_root=tmp_path,
        run_config=CollectionRunConfig(),
        online_runner=failing_runner,
        executor_factory=lambda **kwargs: object(),
    )
    duplicate_case = run_collection(
        _collection_manifest(seeds=(11,)),
        output_root=tmp_path,
        run_config=CollectionRunConfig(),
        online_runner=failing_runner,
        executor_factory=lambda **kwargs: object(),
    )

    with pytest.raises(ValueError, match="duplicate case"):
        summarize_collection((first.suite_dir, duplicate_case.suite_dir))


def test_summary_rejects_malformed_distinct_case_lineage_identity(tmp_path) -> None:
    def failing_runner(**kwargs: object) -> dict[str, object]:
        raise RuntimeError(f"runner unavailable for seed {kwargs['seed']}")

    first = run_collection(
        _collection_manifest(),
        output_root=tmp_path,
        run_config=CollectionRunConfig(),
        online_runner=failing_runner,
        executor_factory=lambda **kwargs: object(),
    )
    second = run_collection(
        _collection_manifest(seeds=(12,)),
        output_root=tmp_path,
        run_config=CollectionRunConfig(),
        online_runner=failing_runner,
        executor_factory=lambda **kwargs: object(),
    )

    assert first.cases[0].status == "failed"
    assert second.cases[0].status == "failed"

    for report, case_id in (
        (first, "object-6-id-seed11-first"),
        (second, "object-6-id-seed12-second"),
    ):
        case_path = report.cases[0].case_dir / "case.json"
        case_payload = json.loads(case_path.read_text())
        case_payload["case_id"] = case_id
        case_payload["lineage_group_id"] = "shared-failed-lineage"
        case_path.write_text(json.dumps(case_payload, indent=2, sort_keys=True) + "\n")
        cases_path = report.suite_dir / "results" / "cases.json"
        cases_payload = json.loads(cases_path.read_text())
        cases_payload[0]["case_id"] = case_id
        cases_path.write_text(json.dumps(cases_payload, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="lineage_group_id must equal case_id"):
        summarize_collection((first.suite_dir, second.suite_dir))
