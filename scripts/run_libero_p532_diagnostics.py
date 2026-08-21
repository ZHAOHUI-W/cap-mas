"""Run bounded, non-evaluative diagnostics for frozen P5.3.2 failures."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Literal, cast
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from capmas.evaluation.phase5_artifacts import Phase5RunDirectory

DiagnosticMode = Literal["execute", "depth", "preview"]
ModeRunner = Callable[["DiagnosticRequest", Phase5RunDirectory], Mapping[str, object]]


@dataclass(frozen=True)
class DiagnosticRequest:
    """Immutable inputs for one P5.3.2.1 diagnostic lane."""

    config_path: str
    candidate_artifact: str
    seed: int
    object_name: str
    target_name: str
    max_steps: int = 32
    gpu: str = "5"

    def __post_init__(self) -> None:
        if not self.config_path:
            raise ValueError("diagnostic config path must not be empty")
        if not self.candidate_artifact:
            raise ValueError("diagnostic candidate artifact must not be empty")
        if self.seed < 0:
            raise ValueError("diagnostic seed must not be negative")
        if not self.object_name or not self.target_name:
            raise ValueError("diagnostic object and target names must not be empty")
        if self.max_steps <= 0:
            raise ValueError("diagnostic max steps must be positive")
        if not self.gpu:
            raise ValueError("diagnostic GPU must not be empty")


@dataclass(frozen=True)
class DiagnosticOutcome:
    """Published result of one bounded diagnostic run."""

    run_dir: Phase5RunDirectory
    mode: DiagnosticMode
    physical_execution_count: int
    diagnostic: Mapping[str, object]


def run_diagnostic(
    request: DiagnosticRequest,
    *,
    mode: DiagnosticMode,
    output_root: str | Path,
    mode_runner: ModeRunner | None = None,
) -> DiagnosticOutcome:
    """Run one diagnostic lane and enforce its physical submission budget.

    This is deliberately separate from the P5.3.2 capability runner.  Its
    artifacts cannot be consumed as evaluation or calibration evidence.
    """

    physical_execution_limit = _execution_limit(mode)
    run_dir = Phase5RunDirectory.create(
        output_root,
        "P5.3.2.1_diagnostics",
        f"{mode}_seed{request.seed}_{uuid4().hex[:8]}",
    )
    base_config = _run_config(request, mode, run_dir, physical_execution_limit)
    run_dir.write_json("run_config.json", {**base_config, "status": "running"})

    runner = mode_runner or _default_mode_runner(mode)
    try:
        with _capture_mode_logs(run_dir):
            mode_payload = runner(request, run_dir)
        if not isinstance(mode_payload, Mapping):
            raise TypeError("diagnostic mode runner must return a mapping")
        physical_execution_count = _execution_count(mode_payload)
        if physical_execution_count > physical_execution_limit:
            raise ValueError(
                "diagnostic mode exceeded physical execution limit: "
                f"mode={mode} count={physical_execution_count} "
                f"limit={physical_execution_limit}"
            )
    except Exception as exc:
        _write_failure(run_dir, base_config, mode, exc)
        raise

    diagnostic = {
        "mode": mode,
        "diagnostic_only": True,
        "eligible_for_evaluation": False,
        "physical_execution_limit": physical_execution_limit,
        **dict(mode_payload),
        "physical_execution_count": physical_execution_count,
    }
    run_dir.write_json("results/diagnostic.json", diagnostic)
    if "physical_result" in mode_payload:
        run_dir.write_json("results/physical_result.json", mode_payload["physical_result"])
    run_dir.write_json(
        "run_config.json",
        {
            **base_config,
            "physical_execution_count": physical_execution_count,
            "status": "completed",
        },
    )
    run_dir.write_text(
        "logs/runner.log",
        "\n".join(
            (
                "experiment=P5.3.2.1_diagnostics",
                f"mode={mode}",
                f"seed={request.seed}",
                f"physical_execution_limit={physical_execution_limit}",
                f"physical_execution_count={physical_execution_count}",
                "diagnostic_only=true",
                "eligible_for_evaluation=false",
                "status=completed",
                "",
            )
        ),
    )
    run_dir.finalize_manifest()
    return DiagnosticOutcome(run_dir, mode, physical_execution_count, diagnostic)


def _execution_limit(mode: DiagnosticMode) -> int:
    if mode in {"depth", "preview"}:
        return 0
    if mode == "execute":
        return 1
    raise ValueError(f"unsupported P5.3.2.1 diagnostic mode: {mode!r}")


def _execution_count(payload: Mapping[str, object]) -> int:
    value = payload.get("physical_execution_count", 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("diagnostic physical_execution_count must be a non-negative integer")
    return value


def _run_config(
    request: DiagnosticRequest,
    mode: DiagnosticMode,
    run_dir: Phase5RunDirectory,
    physical_execution_limit: int,
) -> dict[str, object]:
    return {
        "experiment": "P5.3.2.1_diagnostics",
        "mode": mode,
        "config_path": str(Path(request.config_path).resolve()),
        "candidate_artifact": str(Path(request.candidate_artifact).resolve()),
        "seed": request.seed,
        "object_name": request.object_name,
        "target_name": request.target_name,
        "max_steps": request.max_steps,
        "gpu": request.gpu,
        "molmo_device": "cpu",
        "artifact_dir": str(run_dir.path),
        "diagnostic_only": True,
        "eligible_for_evaluation": False,
        "physical_execution_limit": physical_execution_limit,
        "physical_execution_count": 0,
    }


def _write_failure(
    run_dir: Phase5RunDirectory,
    base_config: Mapping[str, object],
    mode: DiagnosticMode,
    error: Exception,
) -> None:
    """Persist a bounded-lane failure before exposing it to the caller."""

    run_dir.write_json(
        "run_config.json",
        {**base_config, "status": "failed", "stage": f"{mode}_diagnostic"},
    )
    run_dir.write_json(
        "failure.json",
        {
            "status": "failed",
            "mode": mode,
            "error_type": type(error).__name__,
            "error": str(error),
            "diagnostic_only": True,
            "eligible_for_evaluation": False,
        },
    )
    run_dir.write_text(
        "logs/runner.log",
        "\n".join(
            (
                "experiment=P5.3.2.1_diagnostics",
                f"mode={mode}",
                "diagnostic_only=true",
                "eligible_for_evaluation=false",
                f"status=failed error_type={type(error).__name__}",
                f"error={error}",
                "",
            )
        ),
    )
    run_dir.finalize_manifest()


@contextmanager
def _capture_mode_logs(run_dir: Phase5RunDirectory) -> Iterator[None]:
    """Keep CAP-X and server output inside the diagnostic run directory."""

    with run_dir.log_path("mode.stdout.log").open("w", encoding="utf-8") as stdout, run_dir.log_path(
        "mode.stderr.log"
    ).open("w", encoding="utf-8") as stderr, redirect_stdout(stdout), redirect_stderr(stderr):
        yield


def _default_mode_runner(mode: DiagnosticMode) -> ModeRunner:
    if mode == "depth":
        return _run_depth_diagnostic
    if mode == "preview":
        return _run_preview_diagnostic
    if mode == "execute":
        return _run_execute_diagnostic
    raise ValueError(f"unsupported P5.3.2.1 diagnostic mode: {mode!r}")


def _configure_live_environment(request: DiagnosticRequest) -> None:
    """Set the frozen diagnostic runtime profile before importing CAP-X."""

    os.environ["CUDA_VISIBLE_DEVICES"] = request.gpu
    os.environ["MOLMO_DEVICE"] = "cpu"
    from scripts.run_libero_p53_online import _setup_capx_paths

    _setup_capx_paths()


def _run_depth_diagnostic(
    request: DiagnosticRequest,
    _run_dir: Phase5RunDirectory,
) -> Mapping[str, object]:
    """Reset a bare CAP-X low-level LIBERO environment under the depth probe."""

    _configure_live_environment(request)
    from capmas.evaluation.libero_depth_probe import installed_depth_probe

    environment = _build_low_level_environment(request.config_path)
    try:
        with installed_depth_probe() as records:
            reset = getattr(environment, "reset", None)
            if not callable(reset):
                raise TypeError("CAP-X low-level environment does not expose reset()")
            reset(seed=request.seed, options={})
        return {
            "physical_execution_count": 0,
            "depth_records": records,
            "depth_record_count": len(records),
        }
    finally:
        _stop_environment(environment)


def _build_low_level_environment(config_path: str) -> object:
    """Instantiate the YAML's low-level CAP-X environment without any API."""

    from capx.envs.configs.instantiate import instantiate
    from capx.envs.configs.loader import DictLoader

    config = DictLoader.load(config_path)
    env_config = config.get("env") if isinstance(config, Mapping) else None
    cfg = env_config.get("cfg") if isinstance(env_config, Mapping) else None
    low_level = cfg.get("low_level") if isinstance(cfg, Mapping) else None
    if not isinstance(low_level, Mapping):
        raise TypeError("CAP-X YAML env.cfg.low_level must be a mapping")
    low_level_config = dict(low_level)
    for key in ("privileged", "enable_render", "viser_debug"):
        if isinstance(cfg, Mapping) and key in cfg:
            low_level_config[key] = cfg[key]

    original_argv = sys.argv[:]
    try:
        # Simulator setup may inspect argv; do not leak diagnostic CLI flags.
        sys.argv = sys.argv[:1]
        return instantiate(low_level_config)
    finally:
        sys.argv = original_argv


def _run_preview_diagnostic(
    request: DiagnosticRequest,
    run_dir: Phase5RunDirectory,
) -> Mapping[str, object]:
    """Prepare and preview every candidate in one retained scene, without acting."""

    _configure_live_environment(request)
    from capmas.evaluation.libero_evidence_session import (
        LiveLiberoEvidenceSession,
        LiveLiberoEvidenceSessionConfig,
    )
    from capmas.evaluation.physical_payload import scene_snapshot_payload
    from scripts.run_libero_p53_online import _typed_candidates, load_online_candidates

    servers = _start_capx_api_servers(request.config_path)
    session = LiveLiberoEvidenceSession(
        LiveLiberoEvidenceSessionConfig(
            config_path=request.config_path,
            object_name=request.object_name,
            target_name=request.target_name,
            seed=request.seed,
            max_steps=request.max_steps,
        )
    )
    try:
        scene = session.start()
        specs = load_online_candidates(request.candidate_artifact)
        typed = _typed_candidates(specs, scene.scene_version)
        candidate_previews: list[dict[str, object]] = []
        for candidate in typed.candidates:
            prepared = session.prepare_candidate(candidate, typed.graphs[candidate.candidate_id])
            preview = session.preview_prepared(prepared)
            candidate_previews.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "program": prepared.program.to_dict(),
                    "geometry_evidence": _jsonable(prepared.evidence.geometry),
                    "preview": _jsonable(preview),
                }
            )
        payload = {
            "physical_execution_count": 0,
            "decision_scene": scene_snapshot_payload(
                scene,
                object_ids=(request.object_name, request.target_name),
            ),
            "candidate_count": len(candidate_previews),
            "candidate_previews": candidate_previews,
        }
        run_dir.write_json("results/preview.json", payload)
        return payload
    finally:
        session.close()
        _terminate_servers(servers)


def _run_execute_diagnostic(
    request: DiagnosticRequest,
    run_dir: Phase5RunDirectory,
) -> Mapping[str, object]:
    """Reproduce one selected graph using the normal same-runtime execution path."""

    _configure_live_environment(request)
    from capmas.evaluation.libero_evidence_session import (
        LiveLiberoEvidenceSession,
        LiveLiberoEvidenceSessionConfig,
    )
    from capmas.evaluation.rehearsal_evidence import RehearsalPoolConfig
    from scripts.run_libero_p53_online import load_online_candidates, run_online_experiment

    servers = _start_capx_api_servers(request.config_path)
    try:
        candidates = load_online_candidates(request.candidate_artifact)
        if not candidates:
            raise ValueError("diagnostic candidate artifact must not be empty")
        scene_versions = {candidate.scene_version for candidate in candidates}
        if len(scene_versions) != 1:
            raise ValueError("diagnostic candidate artifact must use one scene version")
        outcome = run_online_experiment(
            config_path=request.config_path,
            candidates=candidates,
            seed=request.seed,
            scene_version=next(iter(scene_versions)),
            mode="disabled",
            output_root=run_dir.path / "artifacts",
            pool_config=RehearsalPoolConfig(max_workers=1, timeout_s=360.0, max_restarts=0),
            max_steps=request.max_steps,
            object_name=request.object_name,
            target_name=request.target_name,
            gpu=request.gpu,
            evidence_session=LiveLiberoEvidenceSession(
                LiveLiberoEvidenceSessionConfig(
                    config_path=request.config_path,
                    object_name=request.object_name,
                    target_name=request.target_name,
                    seed=request.seed,
                    max_steps=request.max_steps,
                )
            ),
            effective_motion_scope="mission_suffix",
        )
        payload = {
            "physical_execution_count": int(
                outcome.physical_execution_started_at_ns is not None
            ),
            "online_run_dir": str(outcome.run_dir.path),
            "physical_candidate_id": outcome.physical_candidate_id,
            "physical_result": outcome.physical_result,
            "selection_basis": outcome.report.live.selection_basis,
            "decision_completed_at_ns": outcome.decision_completed_at_ns,
            "physical_execution_started_at_ns": outcome.physical_execution_started_at_ns,
        }
        run_dir.write_json("results/execute.json", payload)
        return payload
    finally:
        _terminate_servers(servers)


def _start_capx_api_servers(config_path: str) -> list[object]:
    """Start only missing CAP-X API services and retain only owned processes."""

    from capx.envs.configs.loader import DictLoader
    from capx.envs.runner import _start_api_servers

    config = DictLoader.load(config_path)
    return list(_start_api_servers(config.get("api_servers")))


def _terminate_servers(servers: list[object]) -> None:
    """Terminate only processes returned by this runner's launch call."""

    for process in servers:
        terminate = getattr(process, "terminate", None)
        if callable(terminate):
            terminate()
        join = getattr(process, "join", None)
        if callable(join):
            join(timeout=5.0)


def _stop_environment(environment: object) -> None:
    stop = getattr(environment, "stop", None)
    if callable(stop):
        stop()


def _jsonable(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("execute", "depth", "preview"))
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--candidate-artifact", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--object-name", default="butter")
    parser.add_argument("--target-name", default="basket")
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--gpu", default="5")
    parser.add_argument("--output-root", default="outputs/phase5")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    request = DiagnosticRequest(
        config_path=args.config_path,
        candidate_artifact=args.candidate_artifact,
        seed=args.seed,
        object_name=args.object_name,
        target_name=args.target_name,
        max_steps=args.max_steps,
        gpu=args.gpu,
    )
    outcome = run_diagnostic(request, mode=cast(DiagnosticMode, args.mode), output_root=args.output_root)
    print(
        json.dumps(
            {
                "run_dir": str(outcome.run_dir.path),
                "mode": outcome.mode,
                "physical_execution_count": outcome.physical_execution_count,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
