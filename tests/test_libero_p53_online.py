from __future__ import annotations

import json
from pathlib import Path

import pytest

from capmas.contracts.action import SkillCall
from capmas.contracts.core import SkillRef
from capmas.contracts.graph import CheckpointSpec, MissionGraph, SubgraphNodeSpec, SubgraphSpec
from capmas.evaluation.rehearsal import RehearsalResult
from capmas.evaluation.rehearsal_evidence import RehearsalPoolConfig
from scripts.run_libero_p53_online import _physical_result_payload
from capmas.evaluation.evidence_cache import VersionedEvidenceCache
from capmas.graph.serialization import mission_graph_to_dict
from scripts.run_libero_p53_online import (
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
