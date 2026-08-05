"""Run frozen, leakage-audited ID/OOD CAP-MAS replay cases."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
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

from capmas.evaluation.ood import (
    OODCase,
    OODReplayEvidence,
    OODSplitManifest,
    assert_leakage_free,
    audit_leakage,
    load_ood_manifest,
    manifest_sha256,
    validate_ood_manifest,
)
from capmas.evaluation.ood_statistics import OODAggregateReport, aggregate_ood_pairs
from capmas.evaluation.parity import load_capx_trial
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
RunFn = Callable[..., object]


@dataclass(frozen=True)
class OODRunConfig:
    """Execution policy for one frozen OOD suite."""

    selection_repeats: int = 1
    max_workers: int = 1
    timeout_s: float = 360.0
    max_restarts: int = 0
    max_steps: int = 32
    gpu: str = "5"
    cache_mode: CacheMode = "disabled"

    def __post_init__(self) -> None:
        if self.selection_repeats <= 0:
            raise ValueError("OOD selection repeats must be positive")
        if self.max_workers <= 0:
            raise ValueError("OOD max_workers must be positive")
        if self.max_workers != 1:
            raise ValueError(
                "P5.5 CAP-X rehearsal requires max_workers=1 on the configured GPU"
            )
        if self.timeout_s <= 0:
            raise ValueError("OOD timeout_s must be positive")
        if self.max_restarts < 0:
            raise ValueError("OOD max_restarts must be non-negative")
        if self.max_steps <= 0:
            raise ValueError("OOD max_steps must be positive")
        if not self.gpu.strip():
            raise ValueError("OOD gpu must not be empty")
        if self.cache_mode not in {"disabled", "enabled"}:
            raise ValueError("OOD cache_mode must be disabled or enabled")


def _start_capx_api_servers(
    config_path: str | Path,
    *,
    loader: Callable[[str | Path], Mapping[str, object]] | None = None,
    starter: Callable[[object], Sequence[object]] | None = None,
) -> list[object]:
    """Start the CAP-X services required by every isolated replay worker."""
    if loader is None:
        from capx.envs.configs.loader import DictLoader

        loader = DictLoader.load
    if starter is None:
        from capx.envs.runner import _start_api_servers

        starter = _start_api_servers
    config = loader(config_path)
    return list(starter(config.get("api_servers")))


@dataclass(frozen=True)
class OODCaseResult:
    case_id: str
    pair_id: str
    status: str
    case_dir: Path
    evidence: tuple[OODReplayEvidence, ...]
    primary_winner: str | None
    selection_basis: str | None
    physical_execution_count: int
    error: str | None = None


@dataclass(frozen=True)
class OODSuiteReport:
    suite_dir: Path
    case_results: tuple[OODCaseResult, ...]
    aggregate: OODAggregateReport
    case_count: int
    failed_case_count: int

    @property
    def case_dirs(self) -> tuple[Path, ...]:
        return tuple(result.case_dir for result in self.case_results)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_candidate_digest(case: OODCase) -> str:
    path = Path(case.candidate_artifact).expanduser()
    if not path.exists():
        raise ValueError(f"candidate artifact does not exist for case {case.case_id}: {path}")
    return _sha256(path)


def _as_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _outcome_payload(outcome: object) -> dict[str, object]:
    if isinstance(outcome, Mapping):
        return dict(outcome)
    if not isinstance(outcome, OnlineSelectionOutcome):
        raise TypeError("OOD online runner returned an unsupported outcome")
    report = outcome.report
    live = report.live
    selected = getattr(live, "selected", None)
    return {
        "mode": report.mode,
        "physical_candidate_id": outcome.physical_candidate_id,
        "physical_result": outcome.physical_result,
        "rehearsal_results": [asdict(item) for item in outcome.rehearsal_results],
        "provider_call_count": outcome.provider_call_count,
        "selection_latency_ms": outcome.selection_latency_ms,
        "live_winner": getattr(selected, "candidate_id", None),
        "live_selection_basis": live.selection_basis,
        "cache_stats": asdict(report.cache_stats) if report.cache_stats else None,
    }


def _bool_value(payload: Mapping[str, object], name: str) -> bool | None:
    value = payload.get(name)
    return value if isinstance(value, bool) else None


def _int_value(payload: Mapping[str, object], name: str) -> int:
    value = payload.get(name)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _float_value(payload: Mapping[str, object], name: str) -> float:
    value = payload.get(name)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _candidate_fingerprint(candidate: object) -> str:
    identity = getattr(candidate, "identity", None)
    if identity is not None:
        return str(identity.subgraph_fingerprint)
    return str(getattr(candidate, "candidate_fingerprint", ""))


_INFRASTRUCTURE_FAILURE_CLASSES = frozenset(
    {
        "reset_failure",
        "worker_crash",
        "timeout",
        "infrastructure_unknown",
    }
)


def _failure_class_value(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _rehearsal_failure_summary(payload: Mapping[str, object]) -> dict[str, object]:
    """Return candidate-level rehearsal failures for the case audit trail."""

    raw_results = payload.get("rehearsal_results")
    if not isinstance(raw_results, Sequence) or isinstance(raw_results, (str, bytes)):
        raw_results = ()
    failures: list[dict[str, object]] = []
    failure_classes: Counter[str] = Counter()
    for raw_result in raw_results:
        if not isinstance(raw_result, Mapping):
            continue
        failure_class = _failure_class_value(raw_result.get("failure_class"))
        if failure_class is None:
            continue
        failure_classes[failure_class] += 1
        failures.append(
            {
                "candidate_id": raw_result.get("candidate_id"),
                "seed": raw_result.get("seed"),
                "failure_class": failure_class,
                "failure_reason": raw_result.get("failure_reason"),
                "failure_step": raw_result.get("failure_step"),
                "latency_ms": raw_result.get("latency_ms"),
            }
        )
    return {
        "total_count": len(raw_results),
        "failed_count": len(failures),
        "failure_classes": dict(failure_classes),
        "failures": failures,
    }


def _evidence_from_outcome(
    case: OODCase,
    candidates: Sequence[object],
    scene_version: int,
    outcome: object,
) -> tuple[OODReplayEvidence, str | None, str | None, int]:
    payload = _outcome_payload(outcome)
    physical = _as_mapping(payload.get("physical_result"))
    winner_value = payload.get("physical_candidate_id")
    if not isinstance(winner_value, str) or not winner_value:
        winner_value = payload.get("live_winner")
    winner = winner_value if isinstance(winner_value, str) and winner_value else None
    candidates_by_id = {
        str(getattr(candidate, "candidate_id")): candidate for candidate in candidates
    }
    selected = candidates_by_id.get(winner) if winner is not None else None
    if selected is None:
        candidate_id = "unselected"
        candidate_fingerprint = "unselected"
    else:
        candidate_id = str(getattr(selected, "candidate_id"))
        candidate_fingerprint = _candidate_fingerprint(selected)

    evaluator_success = _bool_value(physical, "evaluator_success")
    if evaluator_success is None:
        evaluator_success = _bool_value(physical, "success")
    verifier_success = _bool_value(physical, "verifier_success")
    graph_completed = _bool_value(physical, "completed")
    if graph_completed is None:
        graph_completed = _bool_value(physical, "graph_completed")
    failure_class = _failure_class_value(physical.get("failure_class"))
    if failure_class is None:
        failure_class = _failure_class_value(payload.get("failure_class"))
    if failure_class is None:
        nested_failure = _as_mapping(physical.get("failure"))
        failure_class = _failure_class_value(nested_failure.get("failure_class"))

    if failure_class in _INFRASTRUCTURE_FAILURE_CLASSES:
        # Renderer/reset/worker failures do not tell us whether the task would
        # have succeeded. Keep them out of the success-rate denominator.
        evaluator_success = None
    elif failure_class is None and graph_completed is False:
        # A valid task failure must have an explicit graph failure or a
        # completed graph followed by a failed evaluator. Otherwise provenance
        # is incomplete and the result is infrastructure-unknown.
        failure_class = "infrastructure_unknown"
        evaluator_success = None
    elif failure_class is None and evaluator_success is False:
        failure_class = "task_failure"
    elif failure_class is None and evaluator_success is None:
        failure_class = "infrastructure_unknown"
    cache_stats = _as_mapping(payload.get("cache_stats"))
    layout_report = _as_mapping(physical.get("layout_application"))
    cache_hit_count = _int_value(payload, "cache_hit_count")
    if cache_hit_count == 0:
        cache_hit_count = _int_value(cache_stats, "hits")
    evidence = OODReplayEvidence(
        case_id=case.case_id,
        pair_id=case.pair_id,
        condition="capmas",
        candidate_id=candidate_id,
        split=case.split,
        ood_type=case.ood_type,
        source_scene_version=scene_version,
        candidate_fingerprint=candidate_fingerprint,
        evaluator_success=evaluator_success,
        verifier_success=verifier_success,
        graph_completed=bool(graph_completed),
        failure_class=failure_class,
        recovery_count=_int_value(physical, "recovery_count") or _int_value(payload, "recovery_count"),
        human_intervention_count=(
            _int_value(physical, "human_intervention_count")
            or _int_value(payload, "human_intervention_count")
        ),
        latency_ms=max(
            0.0,
            _float_value(payload, "selection_latency_ms")
            + _float_value(physical, "latency_ms"),
        ),
        provider_call_count=_int_value(payload, "provider_call_count"),
        cache_hit_count=cache_hit_count,
        selection_basis=(
            payload.get("live_selection_basis")
            if isinstance(payload.get("live_selection_basis"), str)
            else payload.get("selection_basis")
            if isinstance(payload.get("selection_basis"), str)
            else None
        ),
        shadow_only=True,
        layout_state_fingerprint=(
            layout_report.get("state_fingerprint")
            if isinstance(layout_report.get("state_fingerprint"), str)
            else None
        ),
    )
    physical_execution_count = int(winner is not None)
    return evidence, winner, evidence.selection_basis, physical_execution_count


def _write_case_failure(
    case_dir: Phase5RunDirectory,
    *,
    case: OODCase,
    error: Exception,
    stage: str,
) -> None:
    case_dir.write_json(
        "failure.json",
        {
            "status": "failed",
            "case_id": case.case_id,
            "stage": stage,
            "error_type": type(error).__name__,
            "error": str(error),
        },
    )
    case_dir.write_text(
        "logs/runner.log",
        "\n".join(
            [
                "experiment=P5.5_frozen_ood_replay",
                f"case_id={case.case_id}",
                "status=failed",
                f"stage={stage}",
                f"error_type={type(error).__name__}",
                f"error={error}",
                "",
            ]
        ),
    )


def _write_case_success(
    case_dir: Phase5RunDirectory,
    *,
    case: OODCase,
    outcome: object,
    evidence: OODReplayEvidence,
) -> None:
    payload = _outcome_payload(outcome)
    rehearsal_summary = _rehearsal_failure_summary(payload)
    case_dir.write_json("results/online.json", payload)
    case_dir.write_json("evidence/ood_replay.json", asdict(evidence))
    case_dir.write_json(
        "evidence/rehearsal_failure_summary.json",
        rehearsal_summary,
    )
    layout_report = _as_mapping(_as_mapping(payload.get("physical_result")).get("layout_application"))
    if layout_report:
        case_dir.write_json("evidence/layout_application.json", dict(layout_report))
    case_dir.write_json(
        "summary.json",
        {
            "case_id": case.case_id,
            "pair_id": case.pair_id,
            "status": "completed",
            "primary_winner": payload.get("physical_candidate_id"),
            "evaluator_success": evidence.evaluator_success,
            "verifier_success": evidence.verifier_success,
            "shadow_only": evidence.shadow_only,
            "provider_call_count": evidence.provider_call_count,
            "cache_hit_count": evidence.cache_hit_count,
            "rehearsal_failure_classes": rehearsal_summary["failure_classes"],
            "rehearsal_failed_count": rehearsal_summary["failed_count"],
        },
    )
    case_dir.write_text(
        "summary.md",
        "# CAP-MAS P5.5 frozen OOD replay case\n\n"
        f"- case_id: {case.case_id}\n"
        f"- pair_id: {case.pair_id}\n"
        "- status: completed\n"
        f"- evaluator_success: {evidence.evaluator_success}\n"
        f"- verifier_success: {evidence.verifier_success}\n"
        f"- shadow_only: {evidence.shadow_only}\n"
        f"- provider_call_count: {evidence.provider_call_count}\n"
        f"- rehearsal_failed_count: {rehearsal_summary['failed_count']}\n"
        f"- rehearsal_failure_classes: {rehearsal_summary['failure_classes']}\n",
    )
    case_dir.write_text(
        "logs/runner.log",
        "\n".join(
            [
                "experiment=P5.5_frozen_ood_replay",
                f"case_id={case.case_id}",
                "status=completed",
                f"primary_winner={payload.get('physical_candidate_id')}",
                f"evaluator_success={evidence.evaluator_success}",
                f"provider_call_count={evidence.provider_call_count}",
                f"cache_hit_count={evidence.cache_hit_count}",
                f"rehearsal_failed_count={rehearsal_summary['failed_count']}",
                f"rehearsal_failure_classes={rehearsal_summary['failure_classes']}",
                "",
            ]
        ),
    )


def normalize_capx_case(case: OODCase, capx_trial_dir: str | Path) -> OODReplayEvidence:
    """Normalize a read-only CAP-X trial into the shared shadow schema."""

    episode = load_capx_trial(
        capx_trial_dir,
        task_id=case.task_id,
        seed=case.seed,
    )
    return OODReplayEvidence(
        case_id=case.case_id,
        pair_id=case.pair_id,
        condition="capx",
        candidate_id=Path(capx_trial_dir).name,
        split=case.split,
        ood_type=case.ood_type,
        source_scene_version=0,
        candidate_fingerprint=case.candidate_artifact_sha256,
        evaluator_success=episode.success,
        verifier_success=None,
        graph_completed=episode.success,
        failure_class=None if episode.success else (episode.failure_reason or "task_failure"),
        recovery_count=0,
        human_intervention_count=0,
        latency_ms=0.0,
        provider_call_count=0,
        cache_hit_count=0,
        selection_basis=None,
        shadow_only=True,
    )


def write_ood_report(
    suite_dir: str | Path | Phase5RunDirectory,
    report: OODSuiteReport,
) -> None:
    """Persist the aggregate view without changing any live execution state."""

    run_dir = suite_dir if isinstance(suite_dir, Phase5RunDirectory) else Phase5RunDirectory(Path(suite_dir))
    run_dir.write_json("results/aggregate.json", asdict(report.aggregate))
    run_dir.write_json(
        "results/cases.json",
        [
            {
                "case_id": result.case_id,
                "pair_id": result.pair_id,
                "status": result.status,
                "case_dir": str(result.case_dir),
                "primary_winner": result.primary_winner,
                "selection_basis": result.selection_basis,
                "physical_execution_count": result.physical_execution_count,
                "error": result.error,
            }
            for result in report.case_results
        ],
    )
    run_dir.write_json(
        "summary.json",
        {
            "experiment": "P5.5_frozen_ood_replay",
            "case_count": report.case_count,
            "failed_case_count": report.failed_case_count,
            "shadow_only": True,
        },
    )


def run_ood_case(
    case: OODCase,
    *,
    suite_dir: str | Path | Phase5RunDirectory,
    run_config: OODRunConfig,
    online_runner: OnlineRunner = run_online_experiment,
    executor_factory: ExecutorFactory = _build_live_executor,
    run_fn: RunFn = run_with_respawn,
) -> OODCaseResult:
    """Replay one manifest case and retain all case-scoped artifacts."""

    parent = suite_dir.path if isinstance(suite_dir, Phase5RunDirectory) else Path(suite_dir)
    case_dir = Phase5RunDirectory.create(parent, "cases", case.case_id)
    case_dir.write_json("case.json", asdict(case))
    case_dir.write_json(
        "run_config.json",
        {
            "experiment": "P5.5_frozen_ood_replay",
            "case": asdict(case),
            "run_config": asdict(run_config),
            "status": "running",
        },
    )
    stage = "candidate_validation"
    try:
        candidate_path = Path(case.candidate_artifact).expanduser()
        actual_digest = _sha256(candidate_path)
        if actual_digest != case.candidate_artifact_sha256:
            raise ValueError(
                f"candidate artifact digest mismatch for case {case.case_id}"
            )
        candidates = load_online_candidates(candidate_path)
        if not candidates:
            raise ValueError(f"candidate artifact has no candidates: {candidate_path}")
        if any(candidate.task_id != case.task_id for candidate in candidates):
            raise ValueError(f"candidate task id mismatch for case {case.case_id}")
        scene_versions = {candidate.scene_version for candidate in candidates}
        if len(scene_versions) != 1:
            raise ValueError(f"candidate artifact has mixed scene versions for case {case.case_id}")
        scene_version = next(iter(scene_versions))
        stage = "executor_factory"
        physical_executor = executor_factory(
            config_path=case.config_path,
            object_name=case.object_name,
            target_name=case.target_name,
            max_steps=run_config.max_steps,
            seed=case.seed,
            layout_variant=case.layout_variant,
        )
        stage = "online_replay"
        pool_config = RehearsalPoolConfig(
            max_workers=run_config.max_workers,
            timeout_s=run_config.timeout_s,
            max_restarts=run_config.max_restarts,
        )
        outcome = online_runner(
            config_path=case.config_path,
            candidates=candidates,
            seed=case.seed,
            scene_version=scene_version,
            mode="online_bounded",
            cache_mode=run_config.cache_mode,
            selection_repeats=run_config.selection_repeats,
            output_root=case_dir.path / "online",
            pool_config=pool_config,
            max_steps=run_config.max_steps,
            object_name=case.object_name,
            target_name=case.target_name,
            layout_variant=case.layout_variant,
            gpu=run_config.gpu,
            run_fn=run_fn,
            physical_executor=physical_executor,
        )
        evidence, winner, selection_basis, execution_count = _evidence_from_outcome(
            case, candidates, scene_version, outcome
        )
        _write_case_success(
            case_dir,
            case=case,
            outcome=outcome,
            evidence=evidence,
        )
        result = OODCaseResult(
            case_id=case.case_id,
            pair_id=case.pair_id,
            status="completed",
            case_dir=case_dir.path,
            evidence=(evidence,),
            primary_winner=winner,
            selection_basis=selection_basis,
            physical_execution_count=execution_count,
        )
    except Exception as error:
        _write_case_failure(case_dir, case=case, error=error, stage=stage)
        result = OODCaseResult(
            case_id=case.case_id,
            pair_id=case.pair_id,
            status="failed",
            case_dir=case_dir.path,
            evidence=(),
            primary_winner=None,
            selection_basis=None,
            physical_execution_count=0,
            error=str(error),
        )
    finally:
        case_dir.finalize_manifest()
    return result


def run_ood_suite(
    manifest: OODSplitManifest,
    *,
    output_root: str | Path,
    run_config: OODRunConfig,
    online_runner: OnlineRunner = run_online_experiment,
    executor_factory: ExecutorFactory = _build_live_executor,
    run_fn: RunFn = run_with_respawn,
    fail_fast: bool = False,
) -> OODSuiteReport:
    """Run every frozen case sequentially after fail-closed preflight."""

    os.environ["CUDA_VISIBLE_DEVICES"] = run_config.gpu
    validate_ood_manifest(manifest, candidate_digest_resolver=_resolve_candidate_digest)
    assert_leakage_free(audit_leakage(manifest))
    suite_dir = Phase5RunDirectory.create(
        output_root,
        "P5.5_frozen_ood_replay",
        f"suite_{uuid4().hex[:8]}",
    )
    finalized_manifest = replace(manifest, manifest_sha256=manifest_sha256(manifest))
    suite_dir.write_json("suite_manifest.json", asdict(finalized_manifest))
    suite_dir.write_json(
        "run_config.json",
        {
            "experiment": "P5.5_frozen_ood_replay",
            "manifest_sha256": finalized_manifest.manifest_sha256,
            **asdict(run_config),
            "status": "running",
        },
    )

    results: list[OODCaseResult] = []
    stop_error: RuntimeError | None = None
    for case in manifest.cases:
        result = run_ood_case(
            case,
            suite_dir=suite_dir,
            run_config=run_config,
            online_runner=online_runner,
            executor_factory=executor_factory,
            run_fn=run_fn,
        )
        results.append(result)
        if result.status == "failed" and fail_fast:
            stop_error = RuntimeError(f"OOD case failed: {case.case_id}: {result.error}")
            break

    evidence = tuple(item for result in results for item in result.evidence)
    aggregate = aggregate_ood_pairs(evidence, manifest=manifest)
    failed_count = sum(result.status == "failed" for result in results)
    report = OODSuiteReport(
        suite_dir=suite_dir.path,
        case_results=tuple(results),
        aggregate=aggregate,
        case_count=len(results),
        failed_case_count=failed_count,
    )
    write_ood_report(suite_dir, report)
    suite_dir.write_json(
        "summary.json",
        {
            "experiment": "P5.5_frozen_ood_replay",
            "manifest_sha256": finalized_manifest.manifest_sha256,
            "case_count": report.case_count,
            "failed_case_count": report.failed_case_count,
            "shadow_only": True,
        },
    )
    suite_dir.write_text(
        "summary.md",
        "# CAP-MAS P5.5 frozen OOD replay\n\n"
        f"- case_count: {report.case_count}\n"
        f"- failed_case_count: {report.failed_case_count}\n"
        f"- id_success_rate: {aggregate.id_rate.estimate:.6f}\n"
        f"- ood_success_rate: {aggregate.ood_rate.estimate:.6f}\n"
        f"- shadow_only: True\n",
    )
    suite_dir.write_text(
        "logs/runner.log",
        "\n".join(
            [
                "experiment=P5.5_frozen_ood_replay",
                f"case_count={report.case_count}",
                f"failed_case_count={report.failed_case_count}",
                f"infrastructure_unknown_count={aggregate.infrastructure_unknown_count}",
                "shadow_only=True",
                "",
            ]
        ),
    )
    suite_dir.finalize_manifest()
    if stop_error is not None:
        raise stop_error
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CAP-MAS P5.5 frozen OOD replay")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", default="outputs/phase5")
    parser.add_argument("--selection-repeats", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=360.0)
    parser.add_argument("--max-restarts", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--gpu", default="5")
    parser.add_argument("--cache-mode", choices=("disabled", "enabled"), default="disabled")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    _setup_capx_paths()
    manifest = load_ood_manifest(args.manifest)
    servers = _start_capx_api_servers(manifest.cases[0].config_path)
    try:
        report = run_ood_suite(
            manifest,
            output_root=args.output_root,
            run_config=OODRunConfig(
                selection_repeats=args.selection_repeats,
                max_workers=args.max_workers,
                timeout_s=args.timeout_s,
                max_restarts=args.max_restarts,
                max_steps=args.max_steps,
                gpu=args.gpu,
                cache_mode=args.cache_mode,
            ),
            fail_fast=args.fail_fast,
        )
    finally:
        for process in servers:
            terminate = getattr(process, "terminate", None)
            if callable(terminate):
                terminate()
            join = getattr(process, "join", None)
            if callable(join):
                join(timeout=5)
    print(f"CAP-MAS P5.5 suite: {report.suite_dir}")
    print(f"CAP-MAS P5.5 cases: {report.case_count}")
    print(f"CAP-MAS P5.5 failed cases: {report.failed_case_count}")


if __name__ == "__main__":
    main()
