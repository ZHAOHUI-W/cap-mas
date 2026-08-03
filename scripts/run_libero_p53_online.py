"""Run one bounded online rehearsal-Arbiter decision for CAP-X/LIBERO."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import os
from pathlib import Path
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Literal
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from capmas.agents.arbiter import CandidateArbiter
from capmas.contracts.candidates import GraphCandidate, subgraph_fingerprint
from capmas.contracts.graph import MissionGraph
from capmas.contracts.scene import SceneSnapshot
from capmas.evaluation.candidate_identity import (
    candidate_identity_from_raw_graph,
    raw_graph_fingerprint,
)
from capmas.evaluation.evidence_cache import VersionedEvidenceCache
from capmas.evaluation.evidence_contracts import EvidenceRequestContext
from capmas.evaluation.libero_rehearsal import LiberoRehearsalConfig, LiberoRehearsalWorker
from capmas.evaluation.online_rehearsal import (
    RehearsalArbitrationReport,
    RehearsalMode,
    RehearsalEvidenceProvider,
    select_with_rehearsal,
)
from capmas.evaluation.phase5_artifacts import Phase5RunDirectory
from capmas.evaluation.rehearsal import RehearsalResult
from capmas.evaluation.rehearsal_evidence import (
    RehearsalPoolConfig,
    rehearsal_result_to_evidence,
    run_with_respawn,
)
from capmas.graph.serialization import mission_graph_from_dict
from scripts.run_libero_p53_rehearsal import (
    CandidateSpec,
    build_rehearsal_jobs,
    parse_candidate_mapping,
)


PhysicalExecutor = Callable[[GraphCandidate, MissionGraph], object]
RehearsalRunFn = Callable[..., tuple[RehearsalResult, ...]]


@dataclass(frozen=True)
class OnlineSelectionOutcome:
    run_dir: Phase5RunDirectory
    report: RehearsalArbitrationReport
    rehearsal_results: tuple[RehearsalResult, ...]
    physical_candidate_id: str | None
    physical_result: object | None
    provider_call_count: int
    selection_latency_ms: float


@dataclass(frozen=True)
class _TypedCandidateSet:
    candidates: tuple[GraphCandidate, ...]
    graphs: Mapping[str, MissionGraph]
    specs: Mapping[str, CandidateSpec]


def load_online_candidates(path: str | Path) -> tuple[CandidateSpec, ...]:
    """Load the same graph-scoped candidate artifact used by P5.3 rehearsal."""

    import json

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    candidates = parse_candidate_mapping(raw)
    normalized: list[CandidateSpec] = []
    for candidate in candidates:
        # P5.3 artifacts emitted before the explicit identity closure omitted
        # fingerprint_scope but stored the full graph hash. Recognize only
        # that exact legacy shape, then derive the same local identity gate
        # used by new graph-scoped artifacts.
        if (
            candidate.fingerprint_scope == "subgraph"
            and candidate.candidate_fingerprint == raw_graph_fingerprint(candidate.graph)
        ):
            graph = mission_graph_from_dict(candidate.graph)
            identity = candidate_identity_from_raw_graph(
                candidate.graph,
                graph.entry_subgraph,
                candidate.scene_version,
            )
            normalized.append(
                replace(
                    candidate,
                    fingerprint_scope="graph",
                    identity=identity,
                )
            )
        else:
            normalized.append(candidate)
    return tuple(normalized)


def run_online_experiment(
    *,
    config_path: str,
    candidates: Sequence[CandidateSpec],
    seed: int,
    scene_version: int,
    mode: RehearsalMode,
    cache_mode: Literal["disabled", "enabled"] = "disabled",
    selection_repeats: int = 1,
    output_root: str | Path,
    pool_config: RehearsalPoolConfig,
    max_steps: int = 32,
    object_name: str = "akita black bowl",
    target_name: str = "plate",
    gpu: str = "5",
    run_fn: RehearsalRunFn = run_with_respawn,
    physical_executor: PhysicalExecutor | None = None,
) -> OnlineSelectionOutcome:
    """Run rehearsal, select one live candidate, and execute it at most once."""

    if not candidates:
        raise ValueError("online candidate set must not be empty")
    if scene_version < 0:
        raise ValueError("online scene version must not be negative")
    if seed < 0:
        raise ValueError("online seed must not be negative")
    if max_steps <= 0:
        raise ValueError("online max steps must be positive")
    if cache_mode not in {"disabled", "enabled"}:
        raise ValueError("cache mode must be disabled or enabled")
    if selection_repeats <= 0:
        raise ValueError("selection repeats must be positive")

    run_dir = Phase5RunDirectory.create(
        output_root,
        "P5.3.1_online_rehearsal_arbiter",
        f"seed{seed}_{uuid4().hex[:8]}",
    )
    run_config = {
        "experiment": "P5.3.1_online_rehearsal_arbiter",
        "config_path": str(Path(config_path).resolve()),
        "candidate_ids": [candidate.candidate_id for candidate in candidates],
        "scene_version": scene_version,
        "seed": seed,
        "mode": mode,
        "cache_mode": cache_mode,
        "selection_repeats": selection_repeats,
        "max_workers": pool_config.max_workers,
        "timeout_s": pool_config.timeout_s,
        "max_restarts": pool_config.max_restarts,
        "max_steps": max_steps,
        "gpu": gpu,
        "artifact_dir": str(run_dir.path),
        "provider_call_count": 0,
    }
    run_dir.write_json("run_config.json", {**run_config, "status": "running"})
    rehearsal_results: list[RehearsalResult] = []
    provider_call_count = 0
    selection_latency_ms = 0.0
    evidence_cache = (
        VersionedEvidenceCache()
        if cache_mode == "enabled"
        else None
    )
    stage = "candidate_validation"

    typed = _typed_candidates(candidates, scene_version)
    scene = SceneSnapshot(
        "p53-online",
        1,
        scene_version,
        1,
        1,
        {},
    )
    rehearsal_config = LiberoRehearsalConfig(
        config_path=config_path,
        object_name=object_name,
        target_name=target_name,
        max_steps=max_steps,
    )

    def provider(
        live_candidates: Sequence[GraphCandidate],
        current_scene: SceneSnapshot,
    ) -> Mapping[str, object]:
        nonlocal provider_call_count
        provider_call_count += 1
        jobs = build_rehearsal_jobs(
            tuple(typed.specs[candidate.candidate_id] for candidate in live_candidates),
            seed=seed,
        )
        results = run_fn(
            jobs,
            worker_factory=lambda: LiberoRehearsalWorker(rehearsal_config),
            pool_config=pool_config,
        )
        rehearsal_results.extend(results)
        evidence = {}
        for result in results:
            spec = typed.specs.get(result.candidate_id)
            if spec is None:
                raise ValueError(
                    f"rehearsal result has unknown candidate {result.candidate_id!r}"
                )
            evidence[result.candidate_id] = rehearsal_result_to_evidence(
                result,
                EvidenceRequestContext(
                    candidate_fingerprint=spec.candidate_fingerprint,
                    scene_version=current_scene.scene_version,
                ),
            )
        return evidence

    provider_fn: RehearsalEvidenceProvider | None = provider if mode != "disabled" else None
    report: RehearsalArbitrationReport | None = None
    selection_history: list[dict[str, object]] = []
    for request_index in range(selection_repeats):
        selection_started = time.perf_counter()
        report = select_with_rehearsal(
            typed.candidates,
            scene,
            CandidateArbiter(),
            mode=mode,
            provider=provider_fn,
            evidence_cache=evidence_cache,
        )
        selection_latency_ms += (time.perf_counter() - selection_started) * 1000.0
        selection_history.append(
            {
                "request_index": request_index,
                "provider_call_count": provider_call_count,
                **_selection_payload(report, None),
            }
        )
    assert report is not None
    run_config["provider_call_count"] = provider_call_count

    physical_candidate_id = (
        report.live.selected.candidate_id if report.live.selected is not None else None
    )
    physical_result = None
    stage = "physical_execution"
    try:
        if physical_candidate_id is not None and physical_executor is not None:
            selected = report.live.selected
            assert selected is not None
            physical_result = physical_executor(selected, typed.graphs[physical_candidate_id])
    except Exception as exc:
        _write_failure_artifacts(
            run_dir,
            run_config=run_config,
            stage=stage,
            error=exc,
            rehearsal_results=rehearsal_results,
            evidence_cache=evidence_cache,
        )
        raise

    run_dir.write_json(
        "run_config.json",
        {
            "experiment": "P5.3.1_online_rehearsal_arbiter",
            "config_path": str(Path(config_path).resolve()),
            "candidate_ids": [candidate.candidate_id for candidate in candidates],
            "scene_version": scene_version,
            "seed": seed,
            "mode": mode,
            "cache_mode": cache_mode,
            "selection_repeats": selection_repeats,
            "max_workers": pool_config.max_workers,
            "timeout_s": pool_config.timeout_s,
            "max_restarts": pool_config.max_restarts,
            "max_steps": max_steps,
            "gpu": gpu,
            "physical_execution_count": int(physical_candidate_id is not None),
            "artifact_dir": str(run_dir.path),
            "provider_call_count": provider_call_count,
            "selection_latency_ms": selection_latency_ms,
        },
    )
    run_dir.write_json("results/rehearsal.json", [asdict(item) for item in rehearsal_results])
    if evidence_cache is not None:
        run_dir.write_json(
            "results/cache_events.json",
            [asdict(event) for event in evidence_cache.events()],
        )
    selection_payload = _selection_payload(report, physical_candidate_id)
    selection_payload["provider_call_count"] = provider_call_count
    selection_payload["selection_latency_ms"] = selection_latency_ms
    selection_payload["selection_history"] = selection_history
    run_dir.write_json(
        "results/selection.json",
        selection_payload,
    )
    run_dir.write_text(
        "logs/runner.log",
        "\n".join(
            [
                f"experiment=P5.3.1_online_rehearsal_arbiter seed={seed} mode={mode}",
                f"cache_mode={cache_mode} selection_repeats={selection_repeats}",
                f"rehearsal_results={len(rehearsal_results)}",
                f"baseline_winner={_winner_id(report.baseline)}",
                f"evidence_aware_winner={_winner_id(report.evidence_aware)}",
                f"live_winner={physical_candidate_id}",
                f"physical_execution_count={int(physical_candidate_id is not None)}",
                f"provider_call_count={provider_call_count}",
                f"selection_latency_ms={selection_latency_ms:.3f}",
                f"provider_latency_ms={report.provider_latency_ms:.3f}",
                f"cache_hits={report.cache_stats.hits if report.cache_stats else 0}",
                f"cache_stores={report.cache_stats.stores if report.cache_stats else 0}",
                "",
            ]
        ),
    )
    run_dir.write_json(
        "summary.json",
        {
            "mode": mode,
            "cache_mode": cache_mode,
            "seed": seed,
            "baseline_winner": _winner_id(report.baseline),
            "live_winner": physical_candidate_id,
            "would_change_selection": report.would_change_selection,
            "physical_execution_count": int(physical_candidate_id is not None),
            "provider_call_count": provider_call_count,
            "selection_latency_ms": selection_latency_ms,
            "physical_result": physical_result,
            "selection": selection_payload,
        },
    )
    run_dir.write_text(
        "summary.md",
        "# CAP-MAS P5.3.1 online rehearsal Arbiter\n\n"
        f"- mode: {mode}\n"
        f"- cache_mode: {cache_mode}\n"
        f"- selection_repeats: {selection_repeats}\n"
        f"- seed: {seed}\n"
        f"- baseline_winner: {_winner_id(report.baseline)}\n"
        f"- live_winner: {physical_candidate_id}\n"
        f"- provider_call_count: {provider_call_count}\n"
        f"- selection_latency_ms: {selection_latency_ms:.3f}\n"
        f"- cache_hits: {report.cache_stats.hits if report.cache_stats else 0}\n"
        f"- cache_stores: {report.cache_stats.stores if report.cache_stats else 0}\n"
        f"- physical_execution_count: {int(physical_candidate_id is not None)}\n",
    )
    run_dir.finalize_manifest()
    return OnlineSelectionOutcome(
        run_dir=run_dir,
        report=report,
        rehearsal_results=tuple(rehearsal_results),
        physical_candidate_id=physical_candidate_id,
        physical_result=physical_result,
        provider_call_count=provider_call_count,
        selection_latency_ms=selection_latency_ms,
    )


def _write_failure_artifacts(
    run_dir: Phase5RunDirectory,
    *,
    run_config: Mapping[str, object],
    stage: str,
    error: Exception,
    rehearsal_results: Sequence[RehearsalResult],
    evidence_cache: VersionedEvidenceCache | None = None,
) -> None:
    """Persist failure context without masking the original exception."""

    failure = {
        "status": "failed",
        "stage": stage,
        "error_type": type(error).__name__,
        "error": str(error),
        "rehearsal_result_count": len(rehearsal_results),
    }
    try:
        run_dir.write_json(
            "run_config.json",
            {**run_config, "status": "failed", "stage": stage},
        )
        run_dir.write_json("failure.json", failure)
        run_dir.write_json(
            "results/rehearsal.json", [asdict(item) for item in rehearsal_results]
        )
        if evidence_cache is not None:
            run_dir.write_json(
                "results/cache_events.json",
                [asdict(event) for event in evidence_cache.events()],
            )
        run_dir.write_text(
            "logs/runner.log",
            "\n".join(
                [
                    f"experiment={run_config['experiment']} seed={run_config['seed']} mode={run_config['mode']}",
                    f"status=failed stage={stage}",
                    f"error_type={type(error).__name__}",
                    f"error={error}",
                    f"rehearsal_results={len(rehearsal_results)}",
                    "",
                ]
            ),
        )
        run_dir.finalize_manifest()
    except Exception:
        # Failure reporting must not replace the exception that caused the run
        # to fail. The run directory itself remains available for inspection.
        pass


def _typed_candidates(
    specs: Sequence[CandidateSpec],
    scene_version: int,
) -> _TypedCandidateSet:
    typed: list[GraphCandidate] = []
    graphs: dict[str, MissionGraph] = {}
    by_id: dict[str, CandidateSpec] = {}
    for spec in specs:
        if spec.scene_version != scene_version:
            raise ValueError(
                f"candidate {spec.candidate_id!r} targets scene {spec.scene_version}, "
                f"expected {scene_version}"
            )
        graph = mission_graph_from_dict(spec.graph)
        if spec.identity is not None:
            subgraph_id = spec.identity.subgraph_id
            local_fingerprint = spec.identity.subgraph_fingerprint
        else:
            subgraph_id = graph.entry_subgraph
            local_fingerprint = subgraph_fingerprint(graph.subgraph(subgraph_id))
            if spec.candidate_fingerprint != local_fingerprint:
                raise ValueError(
                    f"subgraph candidate {spec.candidate_id!r} fingerprint mismatch"
                )
        local = graph.subgraph(subgraph_id)
        if subgraph_fingerprint(local) != local_fingerprint:
            raise ValueError(f"candidate {spec.candidate_id!r} local identity mismatch")
        if spec.candidate_id in by_id:
            raise ValueError(f"duplicate candidate id {spec.candidate_id!r}")
        typed.append(
            GraphCandidate(
                candidate_id=spec.candidate_id,
                subgraph=local,
                parent_scene_version=scene_version,
                producer_agent="rehearsal-artifact",
                raw_subgraph=local,
            )
        )
        graphs[spec.candidate_id] = graph
        by_id[spec.candidate_id] = spec
    return _TypedCandidateSet(tuple(typed), graphs, by_id)


def _selection_payload(
    report: RehearsalArbitrationReport,
    physical_candidate_id: str | None,
) -> dict[str, object]:
    return {
        "mode": report.mode,
        "baseline_winner": _winner_id(report.baseline),
        "baseline_selection_basis": report.baseline.selection_basis,
        "evidence_aware_winner": _winner_id(report.evidence_aware),
        "evidence_aware_selection_basis": (
            report.evidence_aware.selection_basis
            if report.evidence_aware is not None
            else None
        ),
        "live_winner": _winner_id(report.live),
        "live_selection_basis": report.live.selection_basis,
        "physical_candidate_id": physical_candidate_id,
        "would_change_selection": report.would_change_selection,
        "attached_candidate_ids": report.attached_candidate_ids,
        "evidence_rejections": report.evidence_rejections,
        "provider_latency_ms": report.provider_latency_ms,
        "fallback_reason": report.fallback_reason,
        "cache_enabled": report.cache_stats is not None,
        "cache_stats": (
            asdict(report.cache_stats) if report.cache_stats is not None else None
        ),
    }


def _winner_id(result: object | None) -> str | None:
    if result is None:
        return None
    selected = getattr(result, "selected", None)
    return getattr(selected, "candidate_id", None)


def _setup_capx_paths() -> None:
    capx_root = PROJECT_ROOT.parent / "cap-x"
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/capmas-mpl")
    for path in (
        capx_root,
        capx_root / "capx" / "third_party" / "contact_graspnet_pytorch",
        capx_root / "capx" / "third_party" / "sam3",
        capx_root / "capx" / "third_party" / "LIBERO-PRO",
        # LIBERO is coupled to cap-x's robosuite 1.4 fork. Do not add the
        # generic robosuite checkout here, because it shadows this path.
        capx_root / "capx" / "third_party" / "libero_dependencies" / "robosuite",
    ):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _build_live_executor(
    *,
    config_path: str,
    object_name: str,
    target_name: str,
    max_steps: int,
    seed: int,
) -> PhysicalExecutor:
    from capmas.backends.capx_libero_factory import build_capx_runtime_from_yaml
    from capmas.runtime.action_lease import ActionLeaseManager
    from capmas.runtime.artifact_bus import ArtifactStore
    from capmas.runtime.graph_interpreter import FixedGraphInterpreter
    from capmas.runtime.orchestrator import RuntimeOrchestrator
    from capmas.runtime.scheduler import FixedGraphScheduler
    from capmas.runtime.state_store import InMemoryStateStore
    from capmas.verification.libero import LiberoObservableVerifier

    def execute(_candidate: GraphCandidate, graph: MissionGraph) -> object:
        bundle = build_capx_runtime_from_yaml(
            config_path,
            object_names=(object_name, target_name),
        )
        try:
            runtime = RuntimeOrchestrator(
                backend=bundle.backend,
                state_store=InMemoryStateStore(),
                skill_registry=bundle.skill_registry,
                lease_manager=ActionLeaseManager(),
                verifier=LiberoObservableVerifier(),
            )
            episode = runtime.backend.reset(seed=seed)
            runtime.start_episode(episode)
            scene = runtime.state_store.latest()
            result = FixedGraphInterpreter(
                FixedGraphScheduler(runtime),
                artifact_store=ArtifactStore(),
                max_steps=max_steps,
            ).run(
                graph,
                scene,
                episode_id=episode.handle.episode_id,
                episode_epoch=episode.handle.episode_epoch,
                task_id=bundle.task_id,
            )
            evaluator_success = bool(bundle.backend.evaluator_success())
            return {
                "completed": bool(result.completed),
                "evaluator_success": evaluator_success,
                "success": bool(result.completed and evaluator_success),
                "trace_count": len(result.traces),
            }
        finally:
            bundle.backend.stop(None)

    return execute


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run P5.3.1 online rehearsal-Arbiter CAP-X/LIBERO smoke"
    )
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--candidate-artifact", required=True)
    parser.add_argument(
        "--mode",
        choices=("disabled", "shadow", "online_bounded"),
        default="online_bounded",
    )
    parser.add_argument(
        "--cache-mode",
        choices=("disabled", "enabled"),
        default="disabled",
    )
    parser.add_argument("--selection-repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--max-restarts", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--gpu", default="5")
    parser.add_argument("--output-root", default="outputs/phase5")
    parser.add_argument("--object-name", default="akita black bowl")
    parser.add_argument("--target-name", default="plate")
    parser.add_argument("--skip-api-servers", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    _setup_capx_paths()
    candidates = load_online_candidates(args.candidate_artifact)
    servers = []
    try:
        if not args.skip_api_servers:
            from capx.envs.configs.loader import DictLoader
            from capx.envs.runner import _start_api_servers

            config = DictLoader.load(args.config_path)
            servers = _start_api_servers(config.get("api_servers"))
        outcome = run_online_experiment(
            config_path=args.config_path,
            candidates=candidates,
            seed=args.seed,
            scene_version=candidates[0].scene_version,
            mode=args.mode,
            cache_mode=args.cache_mode,
            selection_repeats=args.selection_repeats,
            output_root=args.output_root,
            pool_config=RehearsalPoolConfig(
                max_workers=args.max_workers,
                timeout_s=args.timeout_s,
                max_restarts=args.max_restarts,
            ),
            max_steps=args.max_steps,
            object_name=args.object_name,
            target_name=args.target_name,
            gpu=args.gpu,
            physical_executor=_build_live_executor(
                config_path=args.config_path,
                object_name=args.object_name,
                target_name=args.target_name,
                max_steps=args.max_steps,
                seed=args.seed,
            ),
        )
    finally:
        for process in servers:
            process.terminate()
            process.join(timeout=5)
    print(f"CAP-MAS P5.3.1 run: {outcome.run_dir.path}")
    print(f"CAP-MAS P5.3.1 live winner: {outcome.physical_candidate_id}")


if __name__ == "__main__":
    main()
