"""Run matched baseline versus online P5.3.1 CAP-X/LIBERO episodes."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
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


OnlineRunner = Callable[..., object]
ExecutorFactory = Callable[..., object]


@dataclass(frozen=True)
class MatchedTaskSpec:
    task_id: str
    config_path: str | Path
    candidate_artifact: str | Path
    object_name: str
    target_name: str
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("matched task id must not be empty")
        if not self.object_name or not self.target_name:
            raise ValueError("matched object and target names must not be empty")
        if not self.seeds or any(seed < 0 for seed in self.seeds):
            raise ValueError("matched task seeds must be non-negative and non-empty")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("matched task seeds must be unique")


@dataclass(frozen=True)
class MatchedPairResult:
    task_id: str
    seed: int
    status: str
    pair_dir: Path
    baseline: Mapping[str, object]
    online: Mapping[str, object]
    baseline_success: bool | None
    online_success: bool | None
    same_candidate_artifact: bool


@dataclass(frozen=True)
class MatchedEvaluationReport:
    suite_dir: Path
    pairs: tuple[MatchedPairResult, ...]
    completed_pairs: int
    failed_pairs: int
    baseline_successes: int
    online_successes: int
    online_minus_baseline: int
    same_candidate_artifact_count: int


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_fingerprints(candidates: Sequence[object]) -> tuple[str, ...]:
    values = []
    for candidate in candidates:
        identity = getattr(candidate, "identity", None)
        if identity is not None:
            values.append(identity.subgraph_fingerprint)
        else:
            values.append(str(getattr(candidate, "candidate_fingerprint", "")))
    return tuple(values)


def _winner_id(result: object | None) -> str | None:
    if result is None:
        return None
    selected = getattr(result, "selected", None)
    return getattr(selected, "candidate_id", None)


def _outcome_payload(outcome: object) -> dict[str, object]:
    if isinstance(outcome, Mapping):
        return dict(outcome)
    if not isinstance(outcome, OnlineSelectionOutcome):
        raise TypeError("matched online runner returned an unsupported outcome")
    report = outcome.report
    return {
        "mode": report.mode,
        "run_dir": str(outcome.run_dir.path),
        "physical_candidate_id": outcome.physical_candidate_id,
        "physical_result": outcome.physical_result,
        "baseline_winner": _winner_id(report.baseline),
        "evidence_aware_winner": _winner_id(report.evidence_aware),
        "live_winner": _winner_id(report.live),
        "would_change_selection": report.would_change_selection,
        "baseline_selection_basis": report.baseline.selection_basis,
        "live_selection_basis": report.live.selection_basis,
        "attached_candidate_ids": report.attached_candidate_ids,
        "evidence_rejections": report.evidence_rejections,
        "provider_latency_ms": report.provider_latency_ms,
        "fallback_reason": report.fallback_reason,
        "rehearsal_result_count": len(outcome.rehearsal_results),
    }


def _error_payload(error: BaseException) -> dict[str, object]:
    return {
        "status": "failed",
        "error_type": type(error).__name__,
        "error": str(error),
    }


def _physical_success(payload: Mapping[str, object]) -> bool | None:
    physical = payload.get("physical_result")
    if not isinstance(physical, Mapping):
        return None
    value = physical.get("success")
    return bool(value) if isinstance(value, bool) else None


def _selection_field(payload: Mapping[str, object], name: str) -> object | None:
    """Read a selection field from either live or legacy nested payloads."""

    if name in payload:
        return payload[name]
    report = payload.get("report")
    if isinstance(report, Mapping):
        return report.get(name)
    return None


def _winner_field(payload: Mapping[str, object], *, baseline: bool) -> str | None:
    name = "baseline_winner" if baseline else "live_winner"
    value = _selection_field(payload, name)
    if isinstance(value, str) and value:
        return value
    physical = payload.get("physical_candidate_id")
    return physical if isinstance(physical, str) and physical else None


def _count_values(pairs: Sequence["MatchedPairResult"], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    online_fields = {"live_winner", "online_winner", "live_selection_basis", "online_selection_basis"}
    for pair in pairs:
        payload = pair.online if field in online_fields else pair.baseline
        value = _selection_field(payload, field)
        key = value if isinstance(value, str) and value else "unknown"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _write_pair_artifacts(
    pair_dir: Phase5RunDirectory,
    *,
    task: MatchedTaskSpec,
    seed: int,
    candidate_artifact_sha256: str,
    config_sha256: str | None,
    candidate_ids: Sequence[str],
    candidate_fingerprints: Sequence[str],
    baseline: Mapping[str, object],
    online: Mapping[str, object],
    status: str,
) -> MatchedPairResult:
    baseline_success = _physical_success(baseline)
    online_success = _physical_success(online)
    baseline_winner = _winner_field(baseline, baseline=True)
    online_winner = _winner_field(online, baseline=False)
    baseline_selection_basis = _selection_field(baseline, "baseline_selection_basis")
    online_selection_basis = _selection_field(online, "live_selection_basis")
    winner_changed = (
        baseline_winner is not None
        and online_winner is not None
        and baseline_winner != online_winner
    )
    result = MatchedPairResult(
        task_id=task.task_id,
        seed=seed,
        status=status,
        pair_dir=pair_dir.path,
        baseline=dict(baseline),
        online=dict(online),
        baseline_success=baseline_success,
        online_success=online_success,
        same_candidate_artifact=True,
    )
    pair_dir.write_json(
        "run_config.json",
        {
            "experiment": "P5.3.1_matched_evaluation_pair",
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
            "modes": ["disabled", "online_bounded"],
            "status": status,
        },
    )
    pair_dir.write_json(
        "results/pair.json",
        {
            "task_id": task.task_id,
            "seed": seed,
            "status": status,
            "baseline": dict(baseline),
            "online": dict(online),
            "baseline_success": baseline_success,
            "online_success": online_success,
            "online_minus_baseline": (
                int(online_success) - int(baseline_success)
                if baseline_success is not None and online_success is not None
                else None
            ),
            "same_candidate_artifact": True,
            "baseline_winner": baseline_winner,
            "online_winner": online_winner,
            "baseline_selection_basis": baseline_selection_basis,
            "online_selection_basis": online_selection_basis,
            "winner_changed": winner_changed,
        },
    )
    pair_dir.write_text(
        "summary.md",
        "# CAP-MAS P5.3.1 matched pair\n\n"
        f"- task_id: {task.task_id}\n"
        f"- seed: {seed}\n"
        f"- status: {status}\n"
        f"- baseline_success: {baseline_success}\n"
        f"- online_success: {online_success}\n"
        f"- baseline_winner: {baseline_winner}\n"
        f"- online_winner: {online_winner}\n"
        f"- baseline_selection_basis: {baseline_selection_basis}\n"
        f"- online_selection_basis: {online_selection_basis}\n"
        f"- winner_changed: {winner_changed}\n",
    )
    pair_dir.write_text(
        "logs/runner.log",
        "\n".join(
            [
                "experiment=P5.3.1_matched_evaluation_pair",
                f"task_id={task.task_id}",
                f"seed={seed}",
                f"status={status}",
                f"baseline_success={baseline_success}",
                f"online_success={online_success}",
                f"baseline_winner={baseline_winner}",
                f"online_winner={online_winner}",
                f"online_selection_basis={online_selection_basis}",
                "",
            ]
        ),
    )
    pair_dir.finalize_manifest()
    return result


def _write_pair_failure(
    pair_dir: Phase5RunDirectory,
    *,
    task: MatchedTaskSpec,
    seed: int,
    error: BaseException,
) -> None:
    pair_dir.write_json("failure.json", _error_payload(error))
    pair_dir.write_text(
        "logs/runner.log",
        "\n".join(
            [
                "experiment=P5.3.1_matched_evaluation_pair",
                f"task_id={task.task_id}",
                f"seed={seed}",
                "status=failed",
                f"error_type={type(error).__name__}",
                f"error={error}",
                "",
            ]
        ),
    )
    pair_dir.finalize_manifest()


def _load_task_candidates(task: MatchedTaskSpec):
    payload = json.loads(Path(task.candidate_artifact).read_text(encoding="utf-8"))
    artifact_task_id = payload.get("task_id")
    if artifact_task_id is not None and artifact_task_id != task.task_id:
        raise ValueError(
            f"candidate artifact task {artifact_task_id!r} does not match {task.task_id!r}"
        )
    candidates = load_online_candidates(task.candidate_artifact)
    if not candidates:
        raise ValueError(f"candidate artifact has no candidates: {task.candidate_artifact}")
    return candidates


def run_matched_evaluation(
    *,
    tasks: Sequence[MatchedTaskSpec],
    output_root: str | Path,
    max_workers: int = 1,
    timeout_s: float = 120.0,
    max_restarts: int = 1,
    max_steps: int = 32,
    gpu: str = "5",
    online_runner: OnlineRunner = run_online_experiment,
    executor_factory: ExecutorFactory = _build_live_executor,
    run_fn: Callable = run_with_respawn,
    fail_fast: bool = False,
) -> MatchedEvaluationReport:
    if not tasks:
        raise ValueError("matched evaluation requires at least one task")
    pool_config = RehearsalPoolConfig(
        max_workers=max_workers,
        timeout_s=timeout_s,
        max_restarts=max_restarts,
    )
    suite_dir = Phase5RunDirectory.create(
        output_root,
        "P5.3.1_matched_evaluation",
        f"suite_{uuid4().hex[:8]}",
    )
    suite_dir.write_json(
        "run_config.json",
        {
            "experiment": "P5.3.1_matched_evaluation",
            "tasks": [
                {
                    "task_id": task.task_id,
                    "config_path": str(Path(task.config_path).resolve()),
                    "candidate_artifact": str(Path(task.candidate_artifact).resolve()),
                    "seeds": list(task.seeds),
                }
                for task in tasks
            ],
            "modes": ["disabled", "online_bounded"],
            "max_workers": max_workers,
            "timeout_s": timeout_s,
            "max_restarts": max_restarts,
            "max_steps": max_steps,
            "gpu": gpu,
            "fail_fast": fail_fast,
            "status": "running",
        },
    )
    pairs: list[MatchedPairResult] = []
    try:
        for task in tasks:
            candidates = _load_task_candidates(task)
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
                baseline: dict[str, object] = {}
                online: dict[str, object] = {}
                errors: list[BaseException] = []
                mode_root = pair_dir.path / "mode_runs"
                try:
                    baseline_outcome = online_runner(
                        config_path=str(task.config_path),
                        candidates=candidates,
                        seed=seed,
                        scene_version=scene_version,
                        mode="disabled",
                        output_root=mode_root / "baseline",
                        pool_config=pool_config,
                        max_steps=max_steps,
                        object_name=task.object_name,
                        target_name=task.target_name,
                        gpu=gpu,
                        run_fn=run_fn,
                        physical_executor=executor_factory(
                            config_path=str(task.config_path),
                            object_name=task.object_name,
                            target_name=task.target_name,
                            max_steps=max_steps,
                            seed=seed,
                        ),
                    )
                    baseline = _outcome_payload(baseline_outcome)
                except BaseException as error:
                    errors.append(error)
                    baseline = _error_payload(error)
                try:
                    online_outcome = online_runner(
                        config_path=str(task.config_path),
                        candidates=candidates,
                        seed=seed,
                        scene_version=scene_version,
                        mode="online_bounded",
                        output_root=mode_root / "online",
                        pool_config=pool_config,
                        max_steps=max_steps,
                        object_name=task.object_name,
                        target_name=task.target_name,
                        gpu=gpu,
                        run_fn=run_fn,
                        physical_executor=executor_factory(
                            config_path=str(task.config_path),
                            object_name=task.object_name,
                            target_name=task.target_name,
                            max_steps=max_steps,
                            seed=seed,
                        ),
                    )
                    online = _outcome_payload(online_outcome)
                except BaseException as error:
                    errors.append(error)
                    online = _error_payload(error)

                status = "complete" if not errors else "failed"
                result = _write_pair_artifacts(
                    pair_dir,
                    task=task,
                    seed=seed,
                    candidate_artifact_sha256=candidate_artifact_sha256,
                    config_sha256=config_sha256,
                    candidate_ids=candidate_ids,
                    candidate_fingerprints=candidate_fingerprints,
                    baseline=baseline,
                    online=online,
                    status=status,
                )
                pairs.append(result)
                if errors and fail_fast:
                    raise errors[0]

    except BaseException as error:
        suite_dir.write_json("failure.json", _error_payload(error))
        suite_dir.write_text("logs/runner.log", f"status=failed\nerror={error}\n")
        suite_dir.finalize_manifest()
        raise

    complete = [pair for pair in pairs if pair.status == "complete"]
    baseline_successes = sum(pair.baseline_success is True for pair in complete)
    online_successes = sum(pair.online_success is True for pair in complete)
    aggregate = {
        "pair_count": len(pairs),
        "complete_pair_count": len(complete),
        "failed_pair_count": len(pairs) - len(complete),
        "baseline_successes": baseline_successes,
        "online_successes": online_successes,
        "online_minus_baseline": online_successes - baseline_successes,
        "same_candidate_artifact_count": sum(
            pair.same_candidate_artifact for pair in pairs
        ),
        "winner_change_count": sum(
            _winner_field(pair.baseline, baseline=True) is not None
            and _winner_field(pair.online, baseline=False) is not None
            and _winner_field(pair.baseline, baseline=True)
            != _winner_field(pair.online, baseline=False)
            for pair in pairs
        ),
        "baseline_winner_counts": _count_values(pairs, "baseline_winner"),
        "online_winner_counts": _count_values(pairs, "live_winner"),
        "baseline_selection_basis_counts": _count_values(
            pairs, "baseline_selection_basis"
        ),
        "online_selection_basis_counts": _count_values(
            pairs, "live_selection_basis"
        ),
        "pairs": [
            {
                "task_id": pair.task_id,
                "seed": pair.seed,
                "status": pair.status,
                "baseline_success": pair.baseline_success,
                "online_success": pair.online_success,
                "baseline_winner": _winner_field(pair.baseline, baseline=True),
                "online_winner": _winner_field(pair.online, baseline=False),
                "online_selection_basis": _selection_field(
                    pair.online, "live_selection_basis"
                ),
                "pair_dir": str(pair.pair_dir),
            }
            for pair in pairs
        ],
    }
    suite_dir.write_json("results/aggregate.json", aggregate)
    suite_dir.write_text(
        "summary.md",
        "# CAP-MAS P5.3.1 matched evaluation\n\n"
        f"- pair_count: {len(pairs)}\n"
        f"- complete_pair_count: {len(complete)}\n"
        f"- failed_pair_count: {len(pairs) - len(complete)}\n"
        f"- baseline_successes: {baseline_successes}\n"
        f"- online_successes: {online_successes}\n"
        f"- online_minus_baseline: {online_successes - baseline_successes}\n",
    )
    suite_dir.write_text(
        "logs/runner.log",
        "\n".join(
            [
                "experiment=P5.3.1_matched_evaluation",
                f"pair_count={len(pairs)}",
                f"complete_pair_count={len(complete)}",
                f"failed_pair_count={len(pairs) - len(complete)}",
                f"baseline_successes={baseline_successes}",
                f"online_successes={online_successes}",
                "",
            ]
        ),
    )
    suite_dir.write_json(
        "run_config.json",
        {
            "experiment": "P5.3.1_matched_evaluation",
            "tasks": [
                {
                    "task_id": task.task_id,
                    "config_path": str(Path(task.config_path).resolve()),
                    "candidate_artifact": str(Path(task.candidate_artifact).resolve()),
                    "seeds": list(task.seeds),
                }
                for task in tasks
            ],
            "modes": ["disabled", "online_bounded"],
            "max_workers": max_workers,
            "timeout_s": timeout_s,
            "max_restarts": max_restarts,
            "max_steps": max_steps,
            "gpu": gpu,
            "fail_fast": fail_fast,
            "status": "complete",
            "pair_count": len(pairs),
        },
    )
    suite_dir.finalize_manifest()
    return MatchedEvaluationReport(
        suite_dir=suite_dir.path,
        pairs=tuple(pairs),
        completed_pairs=len(complete),
        failed_pairs=len(pairs) - len(complete),
        baseline_successes=baseline_successes,
        online_successes=online_successes,
        online_minus_baseline=online_successes - baseline_successes,
        same_candidate_artifact_count=sum(pair.same_candidate_artifact for pair in pairs),
    )


def load_task_manifest(path: str | Path) -> tuple[MatchedTaskSpec, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list):
        raise ValueError("matched task manifest must contain a tasks list")
    return tuple(
        MatchedTaskSpec(
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
    parser.add_argument("--seeds", default="1,2,3,4,5,6,7,8,9,10")
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
            MatchedTaskSpec(
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
        report = run_matched_evaluation(
            tasks=tasks,
            output_root=args.output_root,
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
    print(f"P5.3.1 matched suite: {report.suite_dir}")
    print(f"complete_pairs={report.completed_pairs}")
    print(f"failed_pairs={report.failed_pairs}")
    print(f"baseline_successes={report.baseline_successes}")
    print(f"online_successes={report.online_successes}")
    print(f"online_minus_baseline={report.online_minus_baseline}")
    return 0 if report.failed_pairs == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
