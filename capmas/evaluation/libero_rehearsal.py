"""CAP-X/LIBERO execution boundary for isolated candidate rehearsal."""

from __future__ import annotations

from dataclasses import dataclass
import os
import time
from collections.abc import Mapping

from capmas.evaluation.rehearsal import (
    RehearsalFailureClass,
    RehearsalJob,
    RehearsalResult,
)
from capmas.graph.serialization import GraphSchemaError, mission_graph_from_dict


@dataclass(frozen=True)
class LiberoRehearsalConfig:
    config_path: str
    object_name: str = "akita black bowl"
    target_name: str = "plate"
    max_steps: int = 32

    def __post_init__(self) -> None:
        if not self.config_path:
            raise ValueError("LIBERO rehearsal config path must not be empty")
        if self.max_steps <= 0:
            raise ValueError("LIBERO rehearsal max_steps must be positive")


def _default_build_runtime(config: LiberoRehearsalConfig):
    from capmas.backends.capx_libero_factory import build_capx_runtime_from_yaml

    return build_capx_runtime_from_yaml(
        config.config_path,
        object_names=(config.object_name, config.target_name),
    )


def run_libero_rehearsal_job(
    job: RehearsalJob,
    config: LiberoRehearsalConfig,
) -> RehearsalResult:
    """Reset CAP-X, execute one serialized graph, and return audit metadata."""

    started_ns = time.monotonic_ns()
    worker_pid = os.getpid()
    bundle = None
    try:
        raw_graph = job.payload.get("graph")
        if not isinstance(raw_graph, Mapping):
            raise GraphSchemaError("rehearsal payload.graph must be an object")
        graph = mission_graph_from_dict(raw_graph)
        bundle = _default_build_runtime(config)

        from capmas.runtime.action_lease import ActionLeaseManager
        from capmas.runtime.artifact_bus import ArtifactStore
        from capmas.runtime.graph_interpreter import FixedGraphInterpreter
        from capmas.runtime.orchestrator import RuntimeOrchestrator
        from capmas.runtime.scheduler import FixedGraphScheduler
        from capmas.runtime.state_store import InMemoryStateStore
        from capmas.verification.libero import LiberoObservableVerifier

        runtime = RuntimeOrchestrator(
            backend=bundle.backend,
            state_store=InMemoryStateStore(),
            skill_registry=bundle.skill_registry,
            lease_manager=ActionLeaseManager(),
            verifier=LiberoObservableVerifier(),
        )
        episode = runtime.backend.reset(seed=job.seed)
        runtime.start_episode(episode)
        scene = runtime.state_store.latest()
        result = FixedGraphInterpreter(
            FixedGraphScheduler(runtime),
            artifact_store=ArtifactStore(),
            max_steps=config.max_steps,
        ).run(
            graph,
            scene,
            episode_id=episode.handle.episode_id,
            episode_epoch=episode.handle.episode_epoch,
            task_id=bundle.task_id,
        )

        checkpoint_results = tuple(
            {
                "trace_id": trace.trace_id,
                "contract_id": trace.contract_id,
                "status": trace.status,
                "start_scene_version": trace.start_scene_version,
                "end_scene_version": trace.end_scene_version,
            }
            for trace in result.traces
        )
        evaluator_success = False
        try:
            evaluator_success = bool(bundle.backend.evaluator_success())
        except Exception:
            evaluator_success = False
        success = bool(result.completed and evaluator_success)
        failure_class: str | None = None
        failure_reason: str | None = None
        failure_step: int | None = None
        if not result.completed:
            failure_class = _failure_class(result.failure.failure_class if result.failure else None)
            failure_reason = result.failure.message if result.failure else "graph execution failed"
            failure_step = len(result.traces)
        elif not evaluator_success:
            failure_class = RehearsalFailureClass.POSTCONDITION_FAILURE
            failure_reason = "CAP-X evaluator reports task incomplete"
            failure_step = len(result.traces)
        checkpoint_results += (
            {
                "terminal_subgraph": result.terminal_subgraph,
                "mission_completed": result.completed,
                "evaluator_success": evaluator_success,
            },
        )
        return RehearsalResult(
            candidate_id=job.candidate_id,
            seed=job.seed,
            success=success,
            latency_ms=(time.monotonic_ns() - started_ns) / 1_000_000,
            failure_class=failure_class,
            worker_pid=worker_pid,
            checkpoint_results=checkpoint_results,
            failure_step=failure_step,
            failure_reason=failure_reason,
            scene_version=job.scene_version,
            candidate_fingerprint=job.candidate_fingerprint,
            fingerprint_scope=job.fingerprint_scope,
            arbiter_subgraph_id=job.arbiter_subgraph_id,
            arbiter_fingerprint=job.arbiter_fingerprint,
        )
    except GraphSchemaError as exc:
        return _failure_result(job, started_ns, worker_pid, RehearsalFailureClass.INVALID_GRAPH, str(exc))
    except Exception as exc:
        return _failure_result(job, started_ns, worker_pid, RehearsalFailureClass.WORKER_CRASH, str(exc))
    finally:
        if bundle is not None:
            try:
                bundle.backend.stop(None)
            except Exception:
                pass


class LiberoRehearsalWorker:
    """Pickle-safe callable wrapper for ``ProcessRehearsalPool``."""

    def __init__(self, config: LiberoRehearsalConfig) -> None:
        self.config = config

    def __call__(self, job: RehearsalJob) -> RehearsalResult:
        return run_libero_rehearsal_job(job, self.config)


def _failure_result(
    job: RehearsalJob,
    started_ns: int,
    worker_pid: int,
    failure_class: RehearsalFailureClass,
    reason: str,
) -> RehearsalResult:
    return RehearsalResult(
        candidate_id=job.candidate_id,
        seed=job.seed,
        success=False,
        latency_ms=(time.monotonic_ns() - started_ns) / 1_000_000,
        failure_class=failure_class,
        worker_pid=worker_pid,
        failure_step=0,
        failure_reason=reason,
        scene_version=job.scene_version,
        candidate_fingerprint=job.candidate_fingerprint,
        fingerprint_scope=job.fingerprint_scope,
        arbiter_subgraph_id=job.arbiter_subgraph_id,
        arbiter_fingerprint=job.arbiter_fingerprint,
    )


def _failure_class(value: str | None) -> RehearsalFailureClass:
    if value == "POSTCONDITION_FAILED":
        return RehearsalFailureClass.POSTCONDITION_FAILURE
    if value in {"PRECONDITION_FAILED", "STALE_STATE"}:
        return RehearsalFailureClass.INVALID_GRAPH
    if value in {"MOTION_TIMEOUT"}:
        return RehearsalFailureClass.TIMEOUT
    if value:
        return RehearsalFailureClass.SKILL_FAILURE
    return RehearsalFailureClass.WORKER_CRASH
