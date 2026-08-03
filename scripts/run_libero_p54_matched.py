"""Run matched multi-seed online rehearsal with the P5.4 cache disabled/enabled."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Literal
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from capmas.evaluation.phase5_artifacts import Phase5RunDirectory
from capmas.evaluation.rehearsal_evidence import RehearsalPoolConfig, run_with_respawn
from scripts.run_libero_p53_online import (
    OnlineSelectionOutcome,
    _build_live_executor,
    _setup_capx_paths,
    load_online_candidates,
    run_online_experiment,
)


CacheMode = Literal["disabled", "enabled"]
OnlineRunner = Callable[..., object]
ExecutorFactory = Callable[..., object]


@dataclass(frozen=True)
class MatchedCacheTaskSpec:
    task_id: str
    config_path: str | Path
    candidate_artifact: str | Path
    object_name: str
    target_name: str
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("matched cache task id must not be empty")
        if not self.object_name or not self.target_name:
            raise ValueError("matched cache object and target names must not be empty")
        if not self.seeds or any(seed < 0 for seed in self.seeds):
            raise ValueError("matched cache seeds must be non-negative and non-empty")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("matched cache seeds must be unique")


@dataclass(frozen=True)
class MatchedCachePairResult:
    task_id: str
    seed: int
    status: str
    pair_dir: Path
    disabled: Mapping[str, object]
    enabled: Mapping[str, object]
    same_candidate_artifact: bool


@dataclass(frozen=True)
class MatchedCacheEvaluationReport:
    suite_dir: Path
    pairs: tuple[MatchedCachePairResult, ...]
    completed_pairs: int
    failed_pairs: int
    disabled_provider_calls: int
    enabled_provider_calls: int
    enabled_cache_hits: int
    provider_call_reduction: float
    same_candidate_artifact_count: int


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_fingerprints(candidates: Sequence[object]) -> tuple[str, ...]:
    fingerprints = []
    for candidate in candidates:
        identity = getattr(candidate, "identity", None)
        if identity is not None:
            fingerprints.append(identity.subgraph_fingerprint)
        else:
            fingerprints.append(str(getattr(candidate, "candidate_fingerprint", "")))
    return tuple(fingerprints)


def _winner_id(result: object | None) -> str | None:
    if result is None:
        return None
    selected = getattr(result, "selected", None)
    return getattr(selected, "candidate_id", None)


def _physical_success(payload: Mapping[str, object]) -> bool | None:
    physical = payload.get("physical_result")
    if not isinstance(physical, Mapping):
        return None
    value = physical.get("success")
    return value if isinstance(value, bool) else None


def _lane_payload(outcome: object, cache_mode: CacheMode) -> dict[str, object]:
    if isinstance(outcome, Mapping):
        payload = dict(outcome)
        payload.setdefault("cache_mode", cache_mode)
        return payload
    if not isinstance(outcome, OnlineSelectionOutcome):
        raise TypeError("matched cache runner returned an unsupported outcome")
    report = outcome.report
    cache_stats = (
        asdict(report.cache_stats) if report.cache_stats is not None else None
    )
    return {
        "mode": report.mode,
        "cache_mode": cache_mode,
        "run_dir": str(outcome.run_dir.path),
        "physical_candidate_id": outcome.physical_candidate_id,
        "physical_execution_count": int(outcome.physical_candidate_id is not None),
        "physical_result": outcome.physical_result,
        "success": _physical_success({"physical_result": outcome.physical_result}),
        "provider_call_count": outcome.provider_call_count,
        "selection_latency_ms": outcome.selection_latency_ms,
        "cache_stats": cache_stats,
        "baseline_winner": _winner_id(report.baseline),
        "live_winner": _winner_id(report.live),
        "live_selection_basis": report.live.selection_basis,
        "would_change_selection": report.would_change_selection,
        "attached_candidate_ids": report.attached_candidate_ids,
        "evidence_rejections": report.evidence_rejections,
        "rehearsal_result_count": len(outcome.rehearsal_results),
    }


def _error_payload(error: BaseException, cache_mode: CacheMode) -> dict[str, object]:
    return {
        "status": "failed",
        "cache_mode": cache_mode,
        "error_type": type(error).__name__,
        "error": str(error),
    }


def _int_metric(payload: Mapping[str, object], name: str) -> int:
    value = payload.get(name)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _float_metric(payload: Mapping[str, object], name: str) -> float:
    value = payload.get(name)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _cache_stat(payload: Mapping[str, object], name: str) -> int:
    stats = payload.get("cache_stats")
    if not isinstance(stats, Mapping):
        return 0
    value = stats.get(name)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _write_pair_artifacts(
    pair_dir: Phase5RunDirectory,
    *,
    task: MatchedCacheTaskSpec,
    seed: int,
    candidate_artifact_sha256: str,
    config_sha256: str | None,
    candidate_ids: Sequence[str],
    candidate_fingerprints: Sequence[str],
    disabled: Mapping[str, object],
    enabled: Mapping[str, object],
    status: str,
    errors: Sequence[BaseException],
) -> MatchedCachePairResult:
    result = MatchedCachePairResult(
        task_id=task.task_id,
        seed=seed,
        status=status,
        pair_dir=pair_dir.path,
        disabled=dict(disabled),
        enabled=dict(enabled),
        same_candidate_artifact=True,
    )
    pair_dir.write_json(
        "run_config.json",
        {
            "experiment": "P5.4_matched_online_cache_pair",
            "task_id": task.task_id,
            "seed": seed,
            "config_path": str(Path(task.config_path).resolve()),
            "candidate_artifact": str(Path(task.candidate_artifact).resolve()),
            "candidate_artifact_sha256": candidate_artifact_sha256,
            "config_sha256": config_sha256,
            "candidate_ids": list(candidate_ids),
            "candidate_fingerprints": list(candidate_fingerprints),
            "object_name": task.object_name,
            "target_name": task.target_name,
            "modes": ["online_bounded"],
            "cache_modes": ["disabled", "enabled"],
            "status": status,
        },
    )
    pair_dir.write_json(
        "results/pair.json",
        {
            "task_id": task.task_id,
            "seed": seed,
            "status": status,
            "disabled": dict(disabled),
            "enabled": dict(enabled),
            "same_candidate_artifact": True,
            "errors": [
                {"type": type(error).__name__, "message": str(error)}
                for error in errors
            ],
        },
    )
    if errors:
        pair_dir.write_json(
            "failure.json",
            {
                "status": "failed",
                "errors": [
                    {"type": type(error).__name__, "message": str(error)}
                    for error in errors
                ],
            },
        )
    pair_dir.write_text(
        "summary.md",
        "# CAP-MAS P5.4 matched online cache pair\n\n"
        f"- task_id: {task.task_id}\n"
        f"- seed: {seed}\n"
        f"- status: {status}\n"
        f"- disabled_provider_calls: {_int_metric(disabled, 'provider_call_count')}\n"
        f"- enabled_provider_calls: {_int_metric(enabled, 'provider_call_count')}\n"
        f"- enabled_cache_hits: {_cache_stat(enabled, 'hits')}\n"
        f"- disabled_selection_latency_ms: {_float_metric(disabled, 'selection_latency_ms'):.3f}\n"
        f"- enabled_selection_latency_ms: {_float_metric(enabled, 'selection_latency_ms'):.3f}\n",
    )
    pair_dir.write_text(
        "logs/runner.log",
        "\n".join(
            [
                "experiment=P5.4_matched_online_cache_pair",
                f"task_id={task.task_id}",
                f"seed={seed}",
                f"status={status}",
                f"disabled_provider_calls={_int_metric(disabled, 'provider_call_count')}",
                f"enabled_provider_calls={_int_metric(enabled, 'provider_call_count')}",
                f"enabled_cache_hits={_cache_stat(enabled, 'hits')}",
                f"disabled_selection_latency_ms={_float_metric(disabled, 'selection_latency_ms'):.3f}",
                f"enabled_selection_latency_ms={_float_metric(enabled, 'selection_latency_ms'):.3f}",
                "",
            ]
        ),
    )
    pair_dir.finalize_manifest()
    return result


def run_matched_cache_evaluation(
    *,
    tasks: Sequence[MatchedCacheTaskSpec],
    output_root: str | Path,
    selection_repeats: int = 2,
    max_workers: int = 1,
    timeout_s: float = 120.0,
    max_restarts: int = 1,
    max_steps: int = 32,
    gpu: str = "5",
    online_runner: OnlineRunner = run_online_experiment,
    executor_factory: ExecutorFactory = _build_live_executor,
    run_fn: Callable = run_with_respawn,
    fail_fast: bool = False,
) -> MatchedCacheEvaluationReport:
    if not tasks:
        raise ValueError("matched cache evaluation requires at least one task")
    if selection_repeats < 2:
        raise ValueError("matched cache evaluation requires at least two selections")
    pool_config = RehearsalPoolConfig(
        max_workers=max_workers,
        timeout_s=timeout_s,
        max_restarts=max_restarts,
    )
    suite_dir = Phase5RunDirectory.create(
        output_root,
        "P5.4_matched_online_cache",
        f"suite_{uuid4().hex[:8]}",
    )
    suite_config = {
        "experiment": "P5.4_matched_online_cache",
        "tasks": [
            {
                "task_id": task.task_id,
                "config_path": str(Path(task.config_path).resolve()),
                "candidate_artifact": str(Path(task.candidate_artifact).resolve()),
                "seeds": list(task.seeds),
            }
            for task in tasks
        ],
        "mode": "online_bounded",
        "cache_modes": ["disabled", "enabled"],
        "selection_repeats": selection_repeats,
        "max_workers": max_workers,
        "timeout_s": timeout_s,
        "max_restarts": max_restarts,
        "max_steps": max_steps,
        "gpu": gpu,
        "fail_fast": fail_fast,
        "status": "running",
    }
    suite_dir.write_json("run_config.json", suite_config)

    pairs: list[MatchedCachePairResult] = []
    try:
        for task in tasks:
            raw_payload = json.loads(Path(task.candidate_artifact).read_text(encoding="utf-8"))
            artifact_task_id = raw_payload.get("task_id")
            if artifact_task_id is not None and artifact_task_id != task.task_id:
                raise ValueError(
                    f"candidate artifact task {artifact_task_id!r} does not match {task.task_id!r}"
                )
            candidates = load_online_candidates(task.candidate_artifact)
            if not candidates:
                raise ValueError(f"candidate artifact has no candidates: {task.candidate_artifact}")
            candidate_ids = tuple(candidate.candidate_id for candidate in candidates)
            candidate_fingerprints = _candidate_fingerprints(candidates)
            candidate_artifact_sha256 = _sha256(task.candidate_artifact)
            config_path = Path(task.config_path)
            config_sha256 = _sha256(config_path) if config_path.exists() else None
            scene_version = candidates[0].scene_version

            for seed in task.seeds:
                pair_dir = Phase5RunDirectory.create(
                    suite_dir.path,
                    "pairs",
                    f"{task.task_id}_seed{seed}",
                )
                mode_root = pair_dir.path / "mode_runs"
                lanes: dict[CacheMode, dict[str, object]] = {}
                errors: list[BaseException] = []
                for cache_mode in ("disabled", "enabled"):
                    try:
                        physical_executor = executor_factory(
                            config_path=str(task.config_path),
                            object_name=task.object_name,
                            target_name=task.target_name,
                            max_steps=max_steps,
                            seed=seed,
                        )
                        outcome = online_runner(
                            config_path=str(task.config_path),
                            candidates=candidates,
                            seed=seed,
                            scene_version=scene_version,
                            mode="online_bounded",
                            cache_mode=cache_mode,
                            selection_repeats=selection_repeats,
                            output_root=mode_root / f"cache_{cache_mode}",
                            pool_config=pool_config,
                            max_steps=max_steps,
                            object_name=task.object_name,
                            target_name=task.target_name,
                            gpu=gpu,
                            run_fn=run_fn,
                            physical_executor=physical_executor,
                        )
                        lanes[cache_mode] = _lane_payload(outcome, cache_mode)
                    except BaseException as error:
                        errors.append(error)
                        lanes[cache_mode] = _error_payload(error, cache_mode)
                pairs.append(
                    _write_pair_artifacts(
                        pair_dir,
                        task=task,
                        seed=seed,
                        candidate_artifact_sha256=candidate_artifact_sha256,
                        config_sha256=config_sha256,
                        candidate_ids=candidate_ids,
                        candidate_fingerprints=candidate_fingerprints,
                        disabled=lanes.get("disabled", {"cache_mode": "disabled"}),
                        enabled=lanes.get("enabled", {"cache_mode": "enabled"}),
                        status="complete" if not errors else "failed",
                        errors=errors,
                    )
                )
                if errors and fail_fast:
                    raise errors[0]
    except BaseException as error:
        suite_dir.write_json(
            "failure.json",
            {"status": "failed", "error_type": type(error).__name__, "error": str(error)},
        )
        suite_dir.write_text("logs/runner.log", f"status=failed\nerror={error}\n")
        suite_dir.finalize_manifest()
        raise

    complete = [pair for pair in pairs if pair.status == "complete"]
    disabled_provider_calls = sum(
        _int_metric(pair.disabled, "provider_call_count") for pair in complete
    )
    enabled_provider_calls = sum(
        _int_metric(pair.enabled, "provider_call_count") for pair in complete
    )
    enabled_cache_hits = sum(
        _cache_stat(pair.enabled, "hits") for pair in complete
    )
    provider_call_reduction = (
        (disabled_provider_calls - enabled_provider_calls) / disabled_provider_calls
        if disabled_provider_calls
        else 0.0
    )
    aggregate = {
        "pair_count": len(pairs),
        "complete_pair_count": len(complete),
        "failed_pair_count": len(pairs) - len(complete),
        "disabled_provider_calls": disabled_provider_calls,
        "enabled_provider_calls": enabled_provider_calls,
        "enabled_cache_hits": enabled_cache_hits,
        "provider_call_reduction": provider_call_reduction,
        "disabled_selection_latency_ms": sum(
            _float_metric(pair.disabled, "selection_latency_ms") for pair in complete
        ),
        "enabled_selection_latency_ms": sum(
            _float_metric(pair.enabled, "selection_latency_ms") for pair in complete
        ),
        "disabled_physical_executions": sum(
            _int_metric(pair.disabled, "physical_execution_count")
            or int(pair.disabled.get("physical_candidate_id") is not None)
            for pair in complete
        ),
        "enabled_physical_executions": sum(
            _int_metric(pair.enabled, "physical_execution_count")
            or int(pair.enabled.get("physical_candidate_id") is not None)
            for pair in complete
        ),
        "disabled_successes": sum(_physical_success(pair.disabled) is True for pair in complete),
        "enabled_successes": sum(_physical_success(pair.enabled) is True for pair in complete),
        "same_candidate_artifact_count": sum(
            pair.same_candidate_artifact for pair in pairs
        ),
        "pairs": [
            {
                "task_id": pair.task_id,
                "seed": pair.seed,
                "status": pair.status,
                "pair_dir": str(pair.pair_dir),
                "disabled_provider_calls": _int_metric(
                    pair.disabled, "provider_call_count"
                ),
                "enabled_provider_calls": _int_metric(
                    pair.enabled, "provider_call_count"
                ),
                "enabled_cache_hits": _cache_stat(pair.enabled, "hits"),
                "disabled_success": _physical_success(pair.disabled),
                "enabled_success": _physical_success(pair.enabled),
            }
            for pair in pairs
        ],
    }
    suite_dir.write_json("results/aggregate.json", aggregate)
    suite_dir.write_text(
        "summary.md",
        "# CAP-MAS P5.4 matched online cache evaluation\n\n"
        f"- complete_pair_count: {len(complete)}\n"
        f"- failed_pair_count: {len(pairs) - len(complete)}\n"
        f"- disabled_provider_calls: {disabled_provider_calls}\n"
        f"- enabled_provider_calls: {enabled_provider_calls}\n"
        f"- enabled_cache_hits: {enabled_cache_hits}\n"
        f"- provider_call_reduction: {provider_call_reduction:.6f}\n"
        f"- disabled_successes: {aggregate['disabled_successes']}\n"
        f"- enabled_successes: {aggregate['enabled_successes']}\n",
    )
    suite_dir.write_text(
        "logs/runner.log",
        "\n".join(
            [
                "experiment=P5.4_matched_online_cache",
                f"complete_pair_count={len(complete)}",
                f"failed_pair_count={len(pairs) - len(complete)}",
                f"disabled_provider_calls={disabled_provider_calls}",
                f"enabled_provider_calls={enabled_provider_calls}",
                f"enabled_cache_hits={enabled_cache_hits}",
                f"provider_call_reduction={provider_call_reduction:.6f}",
                "",
            ]
        ),
    )
    suite_dir.write_json(
        "run_config.json",
        {
            **suite_config,
            "status": "complete" if not (len(pairs) - len(complete)) else "partial",
            "pair_count": len(pairs),
        },
    )
    suite_dir.finalize_manifest()
    return MatchedCacheEvaluationReport(
        suite_dir=suite_dir.path,
        pairs=tuple(pairs),
        completed_pairs=len(complete),
        failed_pairs=len(pairs) - len(complete),
        disabled_provider_calls=disabled_provider_calls,
        enabled_provider_calls=enabled_provider_calls,
        enabled_cache_hits=enabled_cache_hits,
        provider_call_reduction=provider_call_reduction,
        same_candidate_artifact_count=sum(pair.same_candidate_artifact for pair in pairs),
    )


def load_task_manifest(path: str | Path) -> tuple[MatchedCacheTaskSpec, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list):
        raise ValueError("matched cache task manifest must contain a tasks list")
    return tuple(
        MatchedCacheTaskSpec(
            task_id=str(item["task_id"]),
            config_path=str(item["config_path"]),
            candidate_artifact=str(item["candidate_artifact"]),
            object_name=str(item["object_name"]),
            target_name=str(item["target_name"]),
            seeds=tuple(int(seed) for seed in item["seeds"]),
        )
        for item in raw_tasks
    )


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    return seeds


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-manifest")
    parser.add_argument("--config-path")
    parser.add_argument("--candidate-artifact")
    parser.add_argument("--task-id")
    parser.add_argument("--object-name", default="akita black bowl")
    parser.add_argument("--target-name", default="plate")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--selection-repeats", type=int, default=2)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--max-restarts", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--gpu", default="5")
    parser.add_argument("--output-root", default="outputs/phase5")
    parser.add_argument("--skip-api-servers", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.task_manifest:
        tasks = load_task_manifest(args.task_manifest)
    else:
        required = {
            "--config-path": args.config_path,
            "--candidate-artifact": args.candidate_artifact,
            "--task-id": args.task_id,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise SystemExit(
                "provide --task-manifest or all of " + ", ".join(missing)
            )
        tasks = (
            MatchedCacheTaskSpec(
                task_id=args.task_id,
                config_path=args.config_path,
                candidate_artifact=args.candidate_artifact,
                object_name=args.object_name,
                target_name=args.target_name,
                seeds=_parse_seeds(args.seeds),
            ),
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    _setup_capx_paths()
    servers = []
    try:
        if not args.skip_api_servers:
            from capx.envs.configs.loader import DictLoader
            from capx.envs.runner import _start_api_servers

            config = DictLoader.load(str(tasks[0].config_path))
            servers = _start_api_servers(config.get("api_servers"))
        report = run_matched_cache_evaluation(
            tasks=tasks,
            output_root=args.output_root,
            selection_repeats=args.selection_repeats,
            max_workers=args.max_workers,
            timeout_s=args.timeout_s,
            max_restarts=args.max_restarts,
            max_steps=args.max_steps,
            gpu=args.gpu,
            fail_fast=args.fail_fast,
        )
    finally:
        for process in servers:
            process.terminate()
            process.join(timeout=5)
    print(f"P5.4 matched cache suite: {report.suite_dir}")
    print(f"complete_pairs={report.completed_pairs}")
    print(f"failed_pairs={report.failed_pairs}")
    print(f"disabled_provider_calls={report.disabled_provider_calls}")
    print(f"enabled_provider_calls={report.enabled_provider_calls}")
    print(f"enabled_cache_hits={report.enabled_cache_hits}")
    print(f"provider_call_reduction={report.provider_call_reduction:.6f}")
    return 0 if report.failed_pairs == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
