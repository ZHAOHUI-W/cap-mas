from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.run_libero_p53_online as online_module
from capmas.contracts.action import SkillCall
from capmas.contracts.calibration import CalibrationCollectionContext
from capmas.contracts.core import SkillRef
from capmas.contracts.graph import CheckpointSpec, MissionGraph, SubgraphNodeSpec, SubgraphSpec
from capmas.contracts.scene import ObjectTrack, SceneSnapshot
from capmas.evaluation.rehearsal import RehearsalResult
from capmas.evaluation.rehearsal_evidence import RehearsalPoolConfig
from capmas.graph.serialization import mission_graph_to_dict
from scripts.run_libero_p53_online import (
    _physical_result_payload,
    _scene_debug_payload,
    _setup_capx_paths,
    load_online_candidates,
    run_online_experiment,
)


def _graph_payload(description: str) -> dict[str, object]:
    node = SubgraphNodeSpec(
        node_id="act",
        description=description,
        skill_calls=(SkillCall(SkillRef("noop", "1.0.0"), {}),),
        postconditions=("done",),
        proposed_by=description,
    )
    subgraph = SubgraphSpec(
        subgraph_id="sg_pick",
        subgoal_id="pick",
        description=description,
        nodes=(node,),
        edges=(),
        entry_node="act",
        success_nodes=("act",),
        failure_nodes=("act",),
        checkpoints=(CheckpointSpec("check", ("done",)),),
    )
    return mission_graph_to_dict(
        MissionGraph(
            mission_id=f"mission-{description}",
            task="pick",
            subgraphs=(subgraph,),
            edges=(),
            bindings=(),
            entry_subgraph="sg_pick",
            success_subgraphs=("sg_pick",),
            failure_subgraphs=("sg_pick",),
            parent_scene_version=4,
        )
    )


def _artifact(tmp_path):
    from capmas.evaluation.candidate_identity import raw_graph_fingerprint

    first = _graph_payload("first")
    second = _graph_payload("second")
    path = tmp_path / "candidates.json"
    path.write_text(
        json.dumps(
            {
                "task_id": "task",
                "scene_version": 4,
                "candidates": [
                    {
                        "candidate_id": "candidate-a",
                        "candidate_fingerprint": raw_graph_fingerprint(first),
                        "fingerprint_scope": "graph",
                        "arbiter_subgraph_id": "sg_pick",
                        "graph": first,
                    },
                    {
                        "candidate_id": "candidate-b",
                        "candidate_fingerprint": raw_graph_fingerprint(second),
                        "fingerprint_scope": "graph",
                        "arbiter_subgraph_id": "sg_pick",
                        "graph": second,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _calibration_context() -> CalibrationCollectionContext:
    return CalibrationCollectionContext(
        episode_id="online-episode",
        episode_epoch=1,
        family_id="online-family",
        feature_schema_version="p56.feature.v1",
        memory_skill_version="memory-v1",
        robot_skill_version="robot-v1",
    )


def test_online_driver_writes_decision_snapshots_before_physical_execution(tmp_path) -> None:
    candidates = load_online_candidates(_artifact(tmp_path))
    observed_artifacts = []

    def fake_run(jobs, worker_factory, pool_config):
        del worker_factory, pool_config
        return tuple(
            RehearsalResult(
                candidate_id=job.candidate_id,
                seed=job.seed,
                success=True,
                latency_ms=1.0,
                scene_version=job.scene_version,
                candidate_fingerprint=job.candidate_fingerprint,
                fingerprint_scope=job.fingerprint_scope,
                arbiter_subgraph_id=job.arbiter_subgraph_id,
                arbiter_fingerprint=job.arbiter_fingerprint,
            )
            for job in jobs
        )

    def physical(_candidate, _graph):
        artifacts = list(tmp_path.rglob("evidence/calibration_feature_snapshots.json"))
        observed_artifacts.append(artifacts)
        return {"completed": True}

    outcome = run_online_experiment(
        config_path="libero.yaml",
        candidates=candidates,
        seed=1,
        scene_version=4,
        mode="online_bounded",
        output_root=tmp_path / "runs",
        pool_config=RehearsalPoolConfig(max_workers=1, timeout_s=1.0),
        run_fn=fake_run,
        calibration_context=_calibration_context(),
        physical_executor=physical,
    )

    assert observed_artifacts and observed_artifacts[0]
    snapshots = json.loads(observed_artifacts[0][0].read_text(encoding="utf-8"))
    assert len(snapshots) == len(candidates)
    assert outcome.feature_snapshots
    assert outcome.decision_completed_at_ns is not None
    assert outcome.physical_execution_started_at_ns is not None
    assert outcome.decision_completed_at_ns <= outcome.physical_execution_started_at_ns
    run_config = json.loads((outcome.run_dir.path / "run_config.json").read_text())
    summary = json.loads((outcome.run_dir.path / "summary.json").read_text())
    assert run_config["feature_snapshot_count"] == len(candidates)
    assert run_config["feature_schema_version"] == "p56.feature.v1"
    assert summary["decision_completed_at_ns"] == outcome.decision_completed_at_ns
    assert "physical_execution_started_at_ns=" in (
        outcome.run_dir.path / "logs" / "runner.log"
    ).read_text()


def test_online_driver_sets_the_decision_boundary_after_snapshot_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = load_online_candidates(_artifact(tmp_path))
    original_capture = online_module.capture_feature_snapshot
    timestamps = iter((100, 200, 300, 400))

    def capture_at_controlled_time(candidate, context):
        return original_capture(candidate, context, clock=online_module.time.time_ns)

    monkeypatch.setattr(online_module, "capture_feature_snapshot", capture_at_controlled_time)
    monkeypatch.setattr(online_module.time, "time_ns", lambda: next(timestamps))

    outcome = run_online_experiment(
        config_path="libero.yaml",
        candidates=candidates,
        seed=1,
        scene_version=4,
        mode="disabled",
        output_root=tmp_path / "runs",
        pool_config=RehearsalPoolConfig(max_workers=1, timeout_s=1.0),
        calibration_context=_calibration_context(),
        physical_executor=lambda _candidate, _graph: {"completed": True},
    )

    assert outcome.decision_completed_at_ns is not None
    assert all(
        snapshot.captured_at_ns <= outcome.decision_completed_at_ns
        for snapshot in outcome.feature_snapshots
    )


def test_online_driver_without_calibration_context_keeps_empty_snapshot_provenance(tmp_path) -> None:
    candidates = load_online_candidates(_artifact(tmp_path))

    outcome = run_online_experiment(
        config_path="libero.yaml",
        candidates=candidates,
        seed=1,
        scene_version=4,
        mode="disabled",
        output_root=tmp_path / "runs",
        pool_config=RehearsalPoolConfig(max_workers=1, timeout_s=1.0),
        physical_executor=lambda _candidate, _graph: {"completed": True},
    )

    assert outcome.feature_snapshots == ()
    assert outcome.decision_completed_at_ns is not None
    assert outcome.physical_execution_started_at_ns is not None
    assert not (outcome.run_dir.path / "evidence/calibration_feature_snapshots.json").exists()


def test_online_driver_rehearses_candidates_then_executes_one_winner(tmp_path) -> None:
    candidates = load_online_candidates(_artifact(tmp_path))
    jobs_seen = []
    physical_calls = []

    def fake_run(jobs, worker_factory, pool_config):
        del worker_factory, pool_config
        jobs_seen.extend(jobs)
        return tuple(
            RehearsalResult(
                candidate_id=job.candidate_id,
                seed=job.seed,
                success=job.candidate_id == "candidate-b",
                latency_ms=3.0,
                scene_version=job.scene_version,
                candidate_fingerprint=job.candidate_fingerprint,
                fingerprint_scope=job.fingerprint_scope,
                arbiter_subgraph_id=job.arbiter_subgraph_id,
                arbiter_fingerprint=job.arbiter_fingerprint,
            )
            for job in jobs
        )

    def physical(candidate, graph):
        physical_calls.append((candidate.candidate_id, graph.mission_id))
        return {"completed": True}

    outcome = run_online_experiment(
        config_path="libero.yaml",
        candidates=candidates,
        seed=1,
        scene_version=4,
        mode="online_bounded",
        output_root=tmp_path / "runs",
        pool_config=RehearsalPoolConfig(max_workers=1, timeout_s=1.0),
        run_fn=fake_run,
        physical_executor=physical,
    )

    assert {job.candidate_id for job in jobs_seen} == {"candidate-a", "candidate-b"}
    assert outcome.report.attached_candidate_ids == ("candidate-a", "candidate-b")
    assert outcome.report.live.selected is not None
    assert outcome.report.live.selected.candidate_id == "candidate-b"
    assert physical_calls == [("candidate-b", "mission-second")]
    assert (outcome.run_dir.path / "results" / "rehearsal.json").exists()
    assert (outcome.run_dir.path / "results" / "selection.json").exists()
    assert (outcome.run_dir.path / "summary.json").exists()
    assert (outcome.run_dir.path / "logs" / "runner.log").exists()
    assert (outcome.run_dir.path / "manifest.json").exists()


def test_physical_payload_preserves_graph_failure_metadata() -> None:
    from types import SimpleNamespace

    from capmas.contracts.failures import FailureArtifact

    failure = FailureArtifact(
        failure_id="failure-1",
        failure_class="EXECUTION_ERROR",
        message="node returned EXECUTION_ERROR",
        scene_version=3,
        node_id="pick-action",
        subgraph_id="pick",
    )
    result = SimpleNamespace(
        completed=False,
        traces=(SimpleNamespace(trace_id="trace-1"),),
        failure=failure,
        terminal_subgraph="pick",
        next_subgraph=None,
    )

    payload = _physical_result_payload(result, evaluator_success=False)

    assert payload["execution_valid"] is True
    assert payload["failure_class"] == "EXECUTION_ERROR"
    assert payload["failure_reason"] == "node returned EXECUTION_ERROR"
    assert payload["failure"]["node_id"] == "pick-action"
    assert payload["failure"]["subgraph_id"] == "pick"
    assert payload["trace_count"] == 1
    assert payload["horizon"]["planned_valid"] is False


def test_physical_payload_persists_graph_events_and_horizon() -> None:
    from types import SimpleNamespace

    from capmas.contracts.trace import GraphExecutionEvent
    from capmas.graph.serialization import mission_graph_from_dict

    graph = mission_graph_from_dict(_graph_payload("telemetry"))
    event = GraphExecutionEvent(
        sequence=0,
        kind="subgraph_started",
        subgraph_id="sg_pick",
        node_id=None,
        node_type=None,
        attempt=1,
        outcome=None,
        occurred_at_ns=1,
    )
    result = SimpleNamespace(
        completed=True,
        traces=(),
        failure=None,
        terminal_subgraph="sg_pick",
        next_subgraph=None,
        events=(event,),
    )

    payload = _physical_result_payload(
        result,
        evaluator_success=True,
        graph=graph,
    )

    assert payload["graph_events"] == [
        {
            "sequence": 0,
            "kind": "subgraph_started",
            "subgraph_id": "sg_pick",
            "node_id": None,
            "node_type": None,
            "attempt": 1,
            "outcome": None,
            "occurred_at_ns": 1,
        }
    ]
    assert payload["horizon"]["planned_valid"] is True


def test_scene_debug_payload_includes_placement_pose() -> None:
    scene = SceneSnapshot(
        episode_id="episode",
        episode_epoch=0,
        scene_version=2,
        sensor_timestamp_ns=10,
        publish_timestamp_ns=11,
        robot={},
        objects=(
            ObjectTrack(
                track_id="basket",
                label="basket",
                pose_wxyz_xyz=(1.0, 0.0, 0.0, 0.0, 0.6, 0.25, 0.0),
                confidence=1.0,
                last_seen_ns=10,
                placement_pose_wxyz_xyz=(
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.6,
                    0.25,
                    0.04,
                ),
                placement_pose_source="geometry_pointcloud",
                placement_pose_reason=None,
            ),
        ),
    )

    payload = _scene_debug_payload(scene, object_ids=("basket",))

    assert payload["objects"][0]["placement_pose_wxyz_xyz"] == (
        1.0,
        0.0,
        0.0,
        0.0,
        0.6,
        0.25,
        0.04,
    )
    assert payload["objects"][0]["placement_pose_source"] == "geometry_pointcloud"
    assert payload["objects"][0]["placement_pose_reason"] is None


def test_physical_payload_preserves_skill_failure_details() -> None:
    from types import SimpleNamespace

    trace = SimpleNamespace(
        trace_id="trace-1",
        status="failed",
        failure_class="EXECUTION_ERROR",
        skill_traces=(
            SimpleNamespace(
                invocation_id="invoke-1",
                skill_id="goto_pose",
                skill_version="capx-compat-1",
                args={"z_approach": 0.08},
                status="failed",
                error_type="ConnectionError",
                error_message="grasp service unavailable",
                output={},
            ),
        ),
    )
    result = SimpleNamespace(
        completed=False,
        traces=(trace,),
        failure=None,
        terminal_subgraph="pick",
        next_subgraph=None,
    )

    payload = _physical_result_payload(result, evaluator_success=False)

    assert payload["traces"][0]["skill_traces"][0]["skill_id"] == "goto_pose"
    assert payload["traces"][0]["skill_traces"][0]["error_type"] == "ConnectionError"
    assert payload["traces"][0]["skill_traces"][0]["error_message"] == "grasp service unavailable"


def test_online_driver_publishes_enabled_cache_artifacts(tmp_path) -> None:
    candidates = load_online_candidates(_artifact(tmp_path))
    provider_calls = 0
    physical_calls = []

    def fake_run(jobs, worker_factory, pool_config):
        nonlocal provider_calls
        provider_calls += 1
        del worker_factory, pool_config
        return tuple(
            RehearsalResult(
                candidate_id=job.candidate_id,
                seed=job.seed,
                success=True,
                latency_ms=1.0,
                scene_version=job.scene_version,
                candidate_fingerprint=job.candidate_fingerprint,
                fingerprint_scope=job.fingerprint_scope,
                arbiter_subgraph_id=job.arbiter_subgraph_id,
                arbiter_fingerprint=job.arbiter_fingerprint,
            )
            for job in jobs
        )

    outcome = run_online_experiment(
        config_path="libero.yaml",
        candidates=candidates,
        seed=1,
        scene_version=4,
        mode="online_bounded",
        cache_mode="enabled",
        selection_repeats=2,
        output_root=tmp_path / "runs",
        pool_config=RehearsalPoolConfig(max_workers=1, timeout_s=1.0),
        run_fn=fake_run,
        physical_executor=lambda candidate, _graph: physical_calls.append(candidate.candidate_id),
    )

    assert outcome.report.cache_stats is not None
    assert outcome.report.cache_stats.stores == 2
    assert outcome.report.cache_stats.hits == 2
    assert provider_calls == 1
    assert outcome.provider_call_count == 1
    assert outcome.selection_latency_ms >= 0.0
    assert len(physical_calls) == 1
    selection = json.loads(
        (outcome.run_dir.path / "results" / "selection.json").read_text()
    )
    assert selection["cache_enabled"] is True
    assert selection["cache_stats"]["stores"] == 2
    assert selection["provider_call_count"] == 1
    assert selection["selection_latency_ms"] >= 0.0
    assert len(selection["selection_history"]) == 2
    assert selection["selection_history"][0]["provider_call_count"] == 1
    assert selection["selection_history"][1]["provider_call_count"] == 1
    run_config = json.loads(
        (outcome.run_dir.path / "run_config.json").read_text()
    )
    assert run_config["provider_call_count"] == 1
    summary = json.loads((outcome.run_dir.path / "summary.json").read_text())
    assert summary["provider_call_count"] == 1
    assert summary["selection_latency_ms"] >= 0.0
    assert "provider_call_count=1" in (
        outcome.run_dir.path / "logs" / "runner.log"
    ).read_text()
    assert selection["selection_history"][1]["cache_stats"]["hits"] == 2
    assert (outcome.run_dir.path / "results" / "cache_events.json").exists()


def test_online_loader_recovers_legacy_graph_scoped_fingerprint(tmp_path) -> None:
    from capmas.evaluation.candidate_identity import raw_graph_fingerprint

    graph = _graph_payload("legacy")
    path = tmp_path / "legacy-candidates.json"
    path.write_text(
        json.dumps(
            {
                "task_id": "task",
                "scene_version": 4,
                "candidates": [
                    {
                        "candidate_id": "sg_pick:policy-0:0",
                        "candidate_fingerprint": raw_graph_fingerprint(graph),
                        "graph": graph,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    candidates = load_online_candidates(path)

    assert candidates[0].fingerprint_scope == "graph"
    assert candidates[0].identity is not None
    assert candidates[0].identity.subgraph_id == "sg_pick"


def test_shadow_driver_keeps_baseline_winner_for_physical_execution(tmp_path) -> None:
    candidates = load_online_candidates(_artifact(tmp_path))
    physical_calls = []

    def fake_run(jobs, worker_factory, pool_config):
        del worker_factory, pool_config
        return tuple(
            RehearsalResult(
                candidate_id=job.candidate_id,
                seed=job.seed,
                success=job.candidate_id == "candidate-a",
                latency_ms=1.0,
                scene_version=job.scene_version,
                candidate_fingerprint=job.candidate_fingerprint,
                fingerprint_scope=job.fingerprint_scope,
                arbiter_subgraph_id=job.arbiter_subgraph_id,
                arbiter_fingerprint=job.arbiter_fingerprint,
            )
            for job in jobs
        )

    def physical(candidate, _graph):
        physical_calls.append(candidate.candidate_id)
        return {"completed": True}

    outcome = run_online_experiment(
        config_path="libero.yaml",
        candidates=candidates,
        seed=1,
        scene_version=4,
        mode="shadow",
        output_root=tmp_path / "runs",
        pool_config=RehearsalPoolConfig(max_workers=1, timeout_s=1.0),
        run_fn=fake_run,
        physical_executor=physical,
    )

    assert outcome.report.evidence_aware is not None
    assert outcome.report.live == outcome.report.baseline
    assert physical_calls == [outcome.report.baseline.selected.candidate_id]


def test_disabled_driver_skips_rehearsal_provider_and_executes_baseline(tmp_path) -> None:
    candidates = load_online_candidates(_artifact(tmp_path))
    run_calls = 0

    def fake_run(*_args, **_kwargs):
        nonlocal run_calls
        run_calls += 1
        return ()

    outcome = run_online_experiment(
        config_path="libero.yaml",
        candidates=candidates,
        seed=1,
        scene_version=4,
        mode="disabled",
        output_root=tmp_path / "runs",
        pool_config=RehearsalPoolConfig(max_workers=1, timeout_s=1.0),
        run_fn=fake_run,
        physical_executor=lambda candidate, _graph: {"candidate_id": candidate.candidate_id},
    )

    assert run_calls == 0
    assert outcome.report.evidence_aware is None


def test_online_driver_persists_failure_artifacts_before_reraising(tmp_path) -> None:
    candidates = load_online_candidates(_artifact(tmp_path))

    def successful_run(jobs, worker_factory, pool_config):
        del worker_factory, pool_config
        return tuple(
            RehearsalResult(
                candidate_id=job.candidate_id,
                seed=job.seed,
                success=True,
                latency_ms=1.0,
                scene_version=job.scene_version,
                candidate_fingerprint=job.candidate_fingerprint,
                fingerprint_scope=job.fingerprint_scope,
                arbiter_subgraph_id=job.arbiter_subgraph_id,
                arbiter_fingerprint=job.arbiter_fingerprint,
            )
            for job in jobs
        )

    def failing_physical_executor(_candidate, _graph):
        raise RuntimeError("physical executor unavailable")

    with pytest.raises(RuntimeError, match="physical executor unavailable"):
        run_online_experiment(
            config_path="libero.yaml",
            candidates=candidates,
            seed=1,
            scene_version=4,
            mode="online_bounded",
            output_root=tmp_path / "runs",
            pool_config=RehearsalPoolConfig(max_workers=1, timeout_s=1.0),
            run_fn=successful_run,
            calibration_context=_calibration_context(),
            physical_executor=failing_physical_executor,
        )

    run_dirs = list((tmp_path / "runs").glob("P5.3.1_online_rehearsal_arbiter/*"))
    assert len(run_dirs) == 1
    run_dir = Path(run_dirs[0])
    failure = json.loads((run_dir / "failure.json").read_text(encoding="utf-8"))
    assert failure["status"] == "failed"
    assert failure["error_type"] == "RuntimeError"
    assert failure["error"] == "physical executor unavailable"
    assert (run_dir / "logs" / "runner.log").exists()
    assert (run_dir / "manifest.json").exists()
    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    log = (run_dir / "logs" / "runner.log").read_text(encoding="utf-8")
    for artifact in (run_config, summary):
        assert artifact["feature_schema_version"] == "p56.feature.v1"
        assert artifact["feature_snapshot_count"] == len(candidates)
        assert artifact["decision_completed_at_ns"] is not None
        assert artifact["physical_execution_started_at_ns"] is not None
    assert "feature_schema_version=p56.feature.v1" in log
    assert "feature_snapshot_count=2" in log
    assert "decision_completed_at_ns=" in log
    assert "physical_execution_started_at_ns=" in log


def test_online_capx_paths_prefer_libero_robosuite(monkeypatch) -> None:
    import sys

    capx_root = Path(__file__).resolve().parents[1].parent / "cap-x"
    libero_robosuite = capx_root / "capx" / "third_party" / "libero_dependencies" / "robosuite"
    generic_robosuite = capx_root / "capx" / "third_party" / "robosuite"
    libero_package_root = capx_root / "capx" / "third_party" / "LIBERO-PRO" / "libero"

    monkeypatch.setattr(sys, "path", [])
    _setup_capx_paths()

    assert str(libero_robosuite) in sys.path
    assert str(libero_package_root) in sys.path
    assert str(generic_robosuite) not in sys.path
