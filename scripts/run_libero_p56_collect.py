"""Run or summarize CAP-MAS P5.6 object-6 physical collection suites."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from capmas.contracts.calibration import (
    CalibrationCollectionCase,
    CalibrationCollectionContext,
    CalibrationCollectionManifest,
    CalibrationLineage,
    CalibrationOutcome,
    CandidateFeatureSnapshot,
    HorizonLabel,
    collection_manifest_sha256,
)
from capmas.evaluation.dataset import normalize_physical_outcomes
from capmas.evaluation.phase5_artifacts import Phase5RunDirectory
from capmas.evaluation.rehearsal_evidence import RehearsalPoolConfig

EXPERIMENT_NAME = "P5.6.2a_object6_collection"
_INFRASTRUCTURE_STAGES = frozenset({"executor_construction", "online_runner"})
OnlineRunner = Callable[..., object]
ExecutorFactory = Callable[..., object]


@dataclass(frozen=True)
class CollectionRunConfig:
    max_workers: int = 1
    timeout_s: float = 360.0
    max_restarts: int = 0
    max_steps: int = 32
    gpu: str = "5"
    fail_fast: bool = False

    def __post_init__(self) -> None:
        if self.max_workers <= 0:
            raise ValueError("collection max_workers must be positive")
        if self.max_workers != 1:
            raise ValueError("P5.6 object-6 collection requires max_workers=1")
        if self.timeout_s <= 0:
            raise ValueError("collection timeout_s must be positive")
        if self.max_restarts < 0:
            raise ValueError("collection max_restarts must be non-negative")
        if self.max_steps <= 0:
            raise ValueError("collection max_steps must be positive")
        if not self.gpu.strip():
            raise ValueError("collection gpu must not be empty")
        if not isinstance(self.fail_fast, bool):
            raise TypeError("collection fail_fast must be a boolean")


@dataclass(frozen=True)
class CollectionCaseResult:
    case_id: str
    seed: int
    status: Literal["completed", "failed"]
    case_dir: Path
    outcomes: tuple[CalibrationOutcome, ...]
    error: str | None = None


@dataclass(frozen=True)
class CollectionSuiteReport:
    suite_dir: Path
    cases: tuple[CollectionCaseResult, ...]
    completed_cases: int
    failed_cases: int
    tier_a_count: int
    positive_count: int
    negative_count: int
    eligible_20_5_5: bool


@dataclass(frozen=True)
class CollectionEligibilityReport:
    source_suites: tuple[str, ...]
    history_audit: str | None
    admissible_tier_a_count: int
    positive_count: int
    negative_count: int
    eligible_20_5_5: bool


def run_online_experiment(**kwargs: object) -> object:
    """Lazy wrapper for P5.3 online runner to keep imports simulator-free."""

    from scripts.run_libero_p53_online import run_online_experiment as live_runner

    return live_runner(**kwargs)


def _build_live_executor(**kwargs: object) -> object:
    """Lazy wrapper for the P5.3 physical executor factory."""

    from scripts.run_libero_p53_online import _build_live_executor as live_factory

    return live_factory(**kwargs)


def _setup_capx_paths() -> None:
    from scripts.run_libero_p53_online import _setup_capx_paths as setup

    setup()


def _start_capx_api_servers(config_path: str | Path) -> list[object]:
    from capx.envs.configs.loader import DictLoader
    from capx.envs.runner import _start_api_servers

    config = DictLoader.load(config_path)
    return list(_start_api_servers(config.get("api_servers")))


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve(strict=True)


def load_collection_manifest(path: str | Path) -> CalibrationCollectionManifest:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TypeError("collection manifest must contain a JSON object")
    manifest = CalibrationCollectionManifest.from_dict(raw)
    digest = collection_manifest_sha256(manifest)
    return replace(manifest, manifest_id=f"sha256:{digest}", manifest_sha256=digest)


def _finalize_manifest(manifest: CalibrationCollectionManifest) -> CalibrationCollectionManifest:
    digest = collection_manifest_sha256(manifest)
    return replace(manifest, manifest_id=f"sha256:{digest}", manifest_sha256=digest)


def _preflight_manifest(manifest: CalibrationCollectionManifest) -> CalibrationCollectionManifest:
    finalized = _finalize_manifest(manifest)
    case_keys: set[tuple[str, int]] = set()
    for case in finalized.cases:
        key = (case.task_id, case.seed)
        if key in case_keys:
            raise ValueError(f"duplicate collection task/seed entry: {key}")
        case_keys.add(key)
        config_path = _resolve_project_path(case.config_path)
        candidate_path = _resolve_project_path(case.candidate_artifact)
        if _sha256(config_path) != case.config_sha256:
            raise ValueError(f"config digest mismatch for case {case.case_id}")
        if _sha256(candidate_path) != case.candidate_artifact_sha256:
            raise ValueError(f"candidate artifact digest mismatch for case {case.case_id}")
    return finalized


def _load_candidates(path: str | Path) -> tuple[object, ...]:
    from scripts.run_libero_p53_online import load_online_candidates

    return tuple(load_online_candidates(path))


def _candidate_scene_version(candidates: Sequence[object], *, case_id: str) -> int:
    versions = {
        getattr(candidate, "scene_version", None)
        for candidate in candidates
        if isinstance(getattr(candidate, "scene_version", None), int)
    }
    if len(versions) != 1:
        raise ValueError(f"candidate artifact has mixed or missing scene versions for {case_id}")
    scene_version = next(iter(versions))
    if scene_version < 0:
        raise ValueError(f"candidate artifact has invalid scene version for {case_id}")
    return scene_version


def _candidate_task_ids(candidates: Sequence[object]) -> set[str]:
    return {
        str(candidate.task_id)
        for candidate in candidates
        if isinstance(getattr(candidate, "task_id", None), str)
    }


def _write_case_failure(
    case_dir: Phase5RunDirectory,
    *,
    case: CalibrationCollectionCase,
    stage: str,
    error: Exception,
    payload: Mapping[str, object] | None = None,
) -> None:
    if payload is None:
        case_dir.write_json("results/outcomes.json", [])
        case_dir.write_json("evidence/decision_snapshots.json", [])
        case_dir.write_json("evidence/physical_payload.json", {})
        case_dir.write_json("evidence/horizon.json", _unknown_horizon().to_dict())
        case_dir.write_json("evidence/graph_events.json", [])
    else:
        _write_raw_artifacts(case_dir, payload=payload)
    case_dir.write_json(
        "failure.json",
        {
            "status": "failed",
            "case_id": case.case_id,
            "seed": case.seed,
            "stage": stage,
            "error_type": type(error).__name__,
            "error": str(error),
        },
    )
    case_dir.write_json("results/outcomes.json", [])
    case_dir.write_json(
        "summary.json",
        {
            "case_id": case.case_id,
            "seed": case.seed,
            "status": "failed",
            "stage": stage,
            "error_type": type(error).__name__,
            "error": str(error),
        },
    )
    case_dir.write_text(
        "logs/runner.log",
        "\n".join(
            [
                f"experiment={EXPERIMENT_NAME}",
                f"case_id={case.case_id}",
                f"seed={case.seed}",
                "status=failed",
                f"stage={stage}",
                f"error_type={type(error).__name__}",
                f"error={error}",
                "",
            ]
        ),
    )


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _json_plain(value: object) -> object:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _json_plain(to_dict())
        except Exception:  # noqa: BLE001
            return repr(value)
    if isinstance(value, Mapping):
        return {str(key): _json_plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_plain(item) for item in value]
    return value


def _write_raw_artifacts(
    case_dir: Phase5RunDirectory,
    *,
    payload: Mapping[str, object],
) -> None:
    physical_payload = payload.get("physical_result")
    physical_mapping = physical_payload if isinstance(physical_payload, Mapping) else {}
    raw_horizon = physical_mapping.get("horizon", _unknown_horizon())
    raw_graph_events = physical_mapping.get("graph_events", [])
    case_dir.write_json("results/online.json", _json_plain(payload))
    case_dir.write_json(
        "evidence/decision_snapshots.json",
        _json_plain(payload.get("feature_snapshots", [])),
    )
    case_dir.write_json("evidence/physical_payload.json", _json_plain(physical_payload or {}))
    case_dir.write_json("evidence/horizon.json", _json_plain(raw_horizon))
    case_dir.write_json("evidence/graph_events.json", _json_plain(raw_graph_events))


def _outcome_payload(outcome: object) -> dict[str, object]:
    if isinstance(outcome, Mapping):
        return dict(outcome)
    payload = {
        "mode": getattr(getattr(outcome, "report", None), "mode", None),
        "physical_candidate_id": getattr(outcome, "physical_candidate_id", None),
        "physical_result": getattr(outcome, "physical_result", None),
        "provider_call_count": getattr(outcome, "provider_call_count", None),
        "selection_latency_ms": getattr(outcome, "selection_latency_ms", None),
        "feature_snapshots": getattr(outcome, "feature_snapshots", ()),
        "decision_completed_at_ns": getattr(outcome, "decision_completed_at_ns", None),
        "physical_execution_started_at_ns": getattr(
            outcome, "physical_execution_started_at_ns", None
        ),
    }
    run_dir = getattr(outcome, "run_dir", None)
    path = getattr(run_dir, "path", None)
    if path is not None:
        payload["online_run_dir"] = str(path)
    return payload


def _snapshot_tuple(raw_snapshots: object) -> tuple[CandidateFeatureSnapshot, ...]:
    if not isinstance(raw_snapshots, (tuple, list)):
        return ()
    snapshots: list[CandidateFeatureSnapshot] = []
    for raw in raw_snapshots:
        if isinstance(raw, CandidateFeatureSnapshot):
            snapshots.append(raw)
        elif isinstance(raw, Mapping):
            snapshots.append(CandidateFeatureSnapshot.from_dict(raw))
        else:
            raise TypeError("feature snapshots must use P5.6 snapshot contracts")
    return tuple(snapshots)


def _validate_snapshots(
    snapshots: Sequence[CandidateFeatureSnapshot],
    *,
    case: CalibrationCollectionCase,
    manifest: CalibrationCollectionManifest,
) -> None:
    for snapshot in snapshots:
        if (
            snapshot.episode_id != case.case_id
            or snapshot.family_id != case.family_id
            or snapshot.feature_schema_version != manifest.feature_schema_version
            or snapshot.memory_skill_version != manifest.memory_skill_version
            or snapshot.robot_skill_version != manifest.robot_skill_version
            or snapshot.collection_lane != "physical"
        ):
            raise ValueError(f"decision snapshot identity/version mismatch for {case.case_id}")


def _bool_value(payload: Mapping[str, object], *names: str) -> bool | None:
    for name in names:
        value = payload.get(name)
        if isinstance(value, bool):
            return value
    return None


def _string_or_none(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _horizon_from_payload(physical_payload: Mapping[str, object]) -> HorizonLabel:
    raw_horizon = physical_payload.get("horizon")
    if isinstance(raw_horizon, HorizonLabel):
        return raw_horizon
    if isinstance(raw_horizon, Mapping):
        return HorizonLabel.from_dict(raw_horizon)
    return _unknown_horizon()


def _unknown_horizon() -> HorizonLabel:
    return HorizonLabel(
        planned_critical_path_actions=None,
        planned_critical_path_subgoals=None,
        planned_checkpoint_subgraphs=None,
        attempted_actions=None,
        completed_actions=None,
        attempted_subgoals=None,
        completed_subgoals=None,
        attempted_checkpoints=None,
        completed_checkpoints=None,
        planned_source="unknown",
        realized_source="unknown",
        planned_valid=False,
        realized_valid=False,
    )


def _normalize_outcomes(
    *,
    snapshots: tuple[CandidateFeatureSnapshot, ...],
    payload: Mapping[str, object],
) -> tuple[CalibrationOutcome, ...]:
    physical_payload = _mapping(payload.get("physical_result"))
    selected = _string_or_none(payload.get("physical_candidate_id"))
    physical_started_at_ns = payload.get("physical_execution_started_at_ns")
    execution_started = (
        selected is not None
        and isinstance(physical_started_at_ns, int)
        and not isinstance(physical_started_at_ns, bool)
    )
    task_success = _bool_value(physical_payload, "evaluator_success")
    graph_completed = _bool_value(physical_payload, "graph_completed", "completed")
    verifier_success = _bool_value(physical_payload, "verifier_success")
    failure_class = _string_or_none(physical_payload.get("failure_class"))
    return normalize_physical_outcomes(
        snapshots=snapshots,
        selected_candidate_id=selected,
        execution_started=execution_started,
        task_success=task_success,
        graph_completed=graph_completed,
        verifier_success=verifier_success,
        failure_class=failure_class,
        horizon=_horizon_from_payload(physical_payload),
    )


def _lineage_from_case(
    case: CalibrationCollectionCase,
    *,
    payload: Mapping[str, object],
) -> CalibrationLineage:
    decision_boundary_ns = payload.get("decision_completed_at_ns")
    evaluator_observed_at_ns = payload.get("physical_execution_started_at_ns")
    if not isinstance(decision_boundary_ns, int) or isinstance(decision_boundary_ns, bool):
        decision_boundary_ns = 0
    if not isinstance(evaluator_observed_at_ns, int) or isinstance(evaluator_observed_at_ns, bool):
        evaluator_observed_at_ns = None
    return CalibrationLineage(
        episode_id=case.case_id,
        lineage_group_id=case.lineage_group_id,
        seed=case.seed,
        split_identity="id",
        layout_pair_id=None,
        retry_of_episode_id=None,
        candidate_artifact_sha256=case.candidate_artifact_sha256,
        decision_boundary_ns=decision_boundary_ns,
        evaluator_observed_at_ns=evaluator_observed_at_ns,
    )


def _write_case_success(
    case_dir: Phase5RunDirectory,
    *,
    case: CalibrationCollectionCase,
    payload: Mapping[str, object],
    snapshots: tuple[CandidateFeatureSnapshot, ...],
    outcomes: tuple[CalibrationOutcome, ...],
    lineage: CalibrationLineage,
) -> None:
    physical_payload = _mapping(payload.get("physical_result"))
    case_dir.write_json("results/outcomes.json", [outcome.to_dict() for outcome in outcomes])
    case_dir.write_json("evidence/lineage.json", lineage.to_dict())
    tier_a = tuple(outcome for outcome in outcomes if outcome.tier == "A")
    task_success = tier_a[0].task_success if tier_a else None
    case_dir.write_json(
        "summary.json",
        {
            "case_id": case.case_id,
            "seed": case.seed,
            "status": "completed",
            "tier_a_count": len(tier_a),
            "task_success": task_success,
            "selected_candidate_id": payload.get("physical_candidate_id"),
            "decision_completed_at_ns": payload.get("decision_completed_at_ns"),
            "physical_execution_started_at_ns": payload.get("physical_execution_started_at_ns"),
            "feature_snapshot_count": len(snapshots),
        },
    )
    case_dir.write_text(
        "summary.md",
        "# CAP-MAS P5.6 object-6 collection case\n\n"
        f"- case_id: {case.case_id}\n"
        f"- seed: {case.seed}\n"
        "- status: completed\n"
        f"- tier_a_count: {len(tier_a)}\n"
        f"- task_success: {task_success}\n",
    )
    case_dir.write_text(
        "logs/runner.log",
        "\n".join(
            [
                f"experiment={EXPERIMENT_NAME}",
                f"case_id={case.case_id}",
                f"seed={case.seed}",
                "status=completed",
                f"selected_candidate_id={payload.get('physical_candidate_id')}",
                f"tier_a_count={len(tier_a)}",
                f"task_success={task_success}",
                f"decision_completed_at_ns={payload.get('decision_completed_at_ns')}",
                (
                    "physical_execution_started_at_ns="
                    f"{payload.get('physical_execution_started_at_ns')}"
                ),
                "",
            ]
        ),
    )


def run_collection_case(
    case: CalibrationCollectionCase,
    *,
    suite_dir: str | Path | Phase5RunDirectory,
    manifest: CalibrationCollectionManifest,
    run_config: CollectionRunConfig,
    online_runner: OnlineRunner,
    executor_factory: ExecutorFactory,
) -> tuple[CollectionCaseResult, CalibrationLineage | None]:
    parent = suite_dir.path if isinstance(suite_dir, Phase5RunDirectory) else Path(suite_dir)
    case_dir = Phase5RunDirectory.create(parent, "cases", case.case_id)
    case_dir.write_json("case.json", case.to_dict())
    case_dir.write_json(
        "run_config.json",
        {
            "experiment": EXPERIMENT_NAME,
            "case": case.to_dict(),
            "run_config": asdict(run_config),
            "status": "running",
        },
    )
    stage = "candidate_validation"
    payload: Mapping[str, object] | None = None
    try:
        candidate_path = _resolve_project_path(case.candidate_artifact)
        config_path = _resolve_project_path(case.config_path)
        candidates = _load_candidates(candidate_path)
        if not candidates:
            raise ValueError(f"candidate artifact has no candidates: {candidate_path}")
        if _candidate_task_ids(candidates) != {case.task_id}:
            raise ValueError(f"candidate task id mismatch for case {case.case_id}")
        scene_version = _candidate_scene_version(candidates, case_id=case.case_id)
        layout_variant = _json_plain(case.layout_variant)
        if not isinstance(layout_variant, Mapping):
            raise TypeError("collection layout_variant must be a mapping")
        stage = "executor_construction"
        physical_executor = executor_factory(
            config_path=str(config_path),
            object_name=case.object_name,
            target_name=case.target_name,
            max_steps=run_config.max_steps,
            seed=case.seed,
            layout_variant=layout_variant,
        )
        stage = "online_runner"
        context = CalibrationCollectionContext(
            episode_id=case.case_id,
            episode_epoch=1,
            family_id=case.family_id,
            feature_schema_version=manifest.feature_schema_version,
            memory_skill_version=manifest.memory_skill_version,
            robot_skill_version=manifest.robot_skill_version,
            collection_lane="physical",
        )
        outcome = online_runner(
            config_path=str(config_path),
            candidates=candidates,
            seed=case.seed,
            scene_version=scene_version,
            mode="online_bounded",
            cache_mode="disabled",
            selection_repeats=1,
            output_root=case_dir.path / "online",
            pool_config=RehearsalPoolConfig(
                max_workers=1,
                timeout_s=run_config.timeout_s,
                max_restarts=run_config.max_restarts,
            ),
            max_steps=run_config.max_steps,
            object_name=case.object_name,
            target_name=case.target_name,
            layout_variant=layout_variant,
            gpu=run_config.gpu,
            physical_executor=physical_executor,
            calibration_context=context,
        )
        stage = "validation"
        payload = _outcome_payload(outcome)
        stage = "persistence"
        _write_raw_artifacts(case_dir, payload=payload)
        stage = "validation"
        snapshots = _snapshot_tuple(payload.get("feature_snapshots"))
        if not snapshots:
            raise ValueError(
                f"online runner returned no decision-time snapshots for {case.case_id}"
            )
        _validate_snapshots(snapshots, case=case, manifest=manifest)
        stage = "normalization"
        outcomes = _normalize_outcomes(snapshots=snapshots, payload=payload)
        stage = "validation"
        lineage = _lineage_from_case(case, payload=payload)
        stage = "persistence"
        _write_case_success(
            case_dir,
            case=case,
            payload=payload,
            snapshots=snapshots,
            outcomes=outcomes,
            lineage=lineage,
        )
        case_dir.write_json(
            "run_config.json",
            {
                "experiment": EXPERIMENT_NAME,
                "case": case.to_dict(),
                "run_config": asdict(run_config),
                "status": "completed",
            },
        )
        result = CollectionCaseResult(
            case_id=case.case_id,
            seed=case.seed,
            status="completed",
            case_dir=case_dir.path,
            outcomes=outcomes,
        )
    except Exception as error:  # noqa: BLE001
        _write_case_failure(
            case_dir,
            case=case,
            stage=stage,
            error=error,
            payload=payload,
        )
        case_dir.write_json(
            "run_config.json",
            {
                "experiment": EXPERIMENT_NAME,
                "case": case.to_dict(),
                "run_config": asdict(run_config),
                "status": "failed",
                "stage": stage,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        result = CollectionCaseResult(
            case_id=case.case_id,
            seed=case.seed,
            status="failed",
            case_dir=case_dir.path,
            outcomes=(),
            error=str(error),
        )
        lineage = None
    finally:
        case_dir.finalize_manifest()
    return result, lineage


def _is_infrastructure_failure(result: CollectionCaseResult) -> bool:
    if result.status != "failed":
        return False
    try:
        failure = json.loads((result.case_dir / "failure.json").read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return False
    return isinstance(failure, Mapping) and failure.get("stage") in _INFRASTRUCTURE_STAGES


def _suite_summary_payload(
    *,
    manifest: CalibrationCollectionManifest,
    results: Sequence[CollectionCaseResult],
    lineages: Sequence[CalibrationLineage],
) -> dict[str, object]:
    outcomes = tuple(outcome for result in results for outcome in result.outcomes)
    tier_a = tuple(outcome for outcome in outcomes if outcome.tier == "A")
    positives = sum(outcome.task_success is True for outcome in tier_a)
    negatives = sum(outcome.task_success is False for outcome in tier_a)
    return {
        "experiment": EXPERIMENT_NAME,
        "manifest_sha256": manifest.manifest_sha256,
        "case_count": len(results),
        "completed_cases": sum(result.status == "completed" for result in results),
        "failed_cases": sum(result.status == "failed" for result in results),
        "tier_a_count": len(tier_a),
        "positive_count": positives,
        "negative_count": negatives,
        "eligible_20_5_5": len(tier_a) >= 20 and positives >= 5 and negatives >= 5,
        "lineage_count": len(lineages),
    }


def run_collection(
    manifest: CalibrationCollectionManifest,
    *,
    output_root: str | Path,
    run_config: CollectionRunConfig,
    online_runner: OnlineRunner = run_online_experiment,
    executor_factory: ExecutorFactory = _build_live_executor,
) -> CollectionSuiteReport:
    """Run every pre-registered case once without outcome-adaptive stopping."""

    os.environ["CUDA_VISIBLE_DEVICES"] = run_config.gpu
    finalized = _preflight_manifest(manifest)
    suite_dir = Phase5RunDirectory.create(
        output_root,
        EXPERIMENT_NAME,
        f"suite_{uuid4().hex[:8]}",
    )
    suite_dir.write_json("suite_manifest.json", finalized.to_dict())
    suite_dir.write_json(
        "run_config.json",
        {
            "experiment": EXPERIMENT_NAME,
            "manifest_sha256": finalized.manifest_sha256,
            **asdict(run_config),
            "status": "running",
        },
    )

    results: list[CollectionCaseResult] = []
    lineages: list[CalibrationLineage] = []
    stop_error: RuntimeError | None = None
    for case in finalized.cases:
        result, lineage = run_collection_case(
            case,
            suite_dir=suite_dir,
            manifest=finalized,
            run_config=run_config,
            online_runner=online_runner,
            executor_factory=executor_factory,
        )
        results.append(result)
        if lineage is not None:
            lineages.append(lineage)
        if run_config.fail_fast and _is_infrastructure_failure(result):
            stop_error = RuntimeError(
                f"P5.6 collection case failed: {case.case_id}: {result.error}"
            )
            break

    outcomes = tuple(outcome for result in results for outcome in result.outcomes)
    tier_a = tuple(outcome for outcome in outcomes if outcome.tier == "A")
    positives = sum(outcome.task_success is True for outcome in tier_a)
    negatives = sum(outcome.task_success is False for outcome in tier_a)
    report = CollectionSuiteReport(
        suite_dir=suite_dir.path,
        cases=tuple(results),
        completed_cases=sum(result.status == "completed" for result in results),
        failed_cases=sum(result.status == "failed" for result in results),
        tier_a_count=len(tier_a),
        positive_count=positives,
        negative_count=negatives,
        eligible_20_5_5=len(tier_a) >= 20 and positives >= 5 and negatives >= 5,
    )
    summary_payload = _suite_summary_payload(
        manifest=finalized,
        results=results,
        lineages=lineages,
    )
    suite_dir.write_json("results/outcomes.json", [outcome.to_dict() for outcome in outcomes])
    suite_dir.write_json("results/lineages.json", [lineage.to_dict() for lineage in lineages])
    suite_dir.write_json(
        "results/cases.json",
        [
            {
                "case_id": result.case_id,
                "seed": result.seed,
                "status": result.status,
                "case_dir": str(result.case_dir),
                "outcome_count": len(result.outcomes),
                "error": result.error,
            }
            for result in results
        ],
    )
    suite_dir.write_json("summary.json", summary_payload)
    suite_dir.write_json(
        "run_config.json",
        {**summary_payload, **asdict(run_config), "status": "completed"},
    )
    suite_dir.write_text(
        "summary.md",
        "# CAP-MAS P5.6 object-6 collection suite\n\n"
        f"- case_count: {len(results)}\n"
        f"- completed_cases: {report.completed_cases}\n"
        f"- failed_cases: {report.failed_cases}\n"
        f"- tier_a_count: {report.tier_a_count}\n"
        f"- positive_count: {report.positive_count}\n"
        f"- negative_count: {report.negative_count}\n"
        f"- eligible_20_5_5: {report.eligible_20_5_5}\n",
    )
    suite_dir.write_text(
        "logs/runner.log",
        "\n".join(
            [
                f"experiment={EXPERIMENT_NAME}",
                f"manifest_sha256={finalized.manifest_sha256}",
                f"case_count={len(results)}",
                f"completed_cases={report.completed_cases}",
                f"failed_cases={report.failed_cases}",
                f"tier_a_count={report.tier_a_count}",
                f"positive_count={report.positive_count}",
                f"negative_count={report.negative_count}",
                f"eligible_20_5_5={report.eligible_20_5_5}",
                "",
            ]
        ),
    )
    suite_dir.finalize_manifest()
    if stop_error is not None:
        raise stop_error
    return report


def _read_mapping_json(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _read_sequence_json(path: Path) -> Sequence[object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"expected JSON list: {path}")
    return payload


def _register_case(
    *,
    case_id: str,
    lineage_group_id: str,
    seen_cases: set[str],
    seen_lineages: set[str],
) -> None:
    if case_id in seen_cases:
        raise ValueError(f"duplicate case id across collection sources: {case_id}")
    if lineage_group_id in seen_lineages:
        raise ValueError(f"duplicate lineage group across collection sources: {lineage_group_id}")
    seen_cases.add(case_id)
    seen_lineages.add(lineage_group_id)


def _suite_case_identities(
    suite_path: Path,
) -> tuple[CalibrationCollectionCase, ...]:
    cases: list[CalibrationCollectionCase] = []
    for row in _read_sequence_json(suite_path / "results" / "cases.json"):
        if not isinstance(row, Mapping):
            raise TypeError(f"collection case result must be a mapping: {suite_path}")
        case_id = row.get("case_id")
        case_dir_value = row.get("case_dir")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"collection case result lacks case_id: {suite_path}")
        if not isinstance(case_dir_value, str) or not case_dir_value:
            raise ValueError(f"collection case result lacks case_dir: {suite_path}")
        case_dir = Path(case_dir_value)
        if not case_dir.is_absolute():
            case_dir = suite_path / case_dir
        case = CalibrationCollectionCase.from_dict(_read_mapping_json(case_dir / "case.json"))
        if case.case_id != case_id:
            raise ValueError(f"persisted collection case identity mismatch: {case_id}")
        cases.append(case)
    return tuple(cases)


def _history_audit_path(path: str | Path) -> Path:
    raw = Path(path)
    if raw.is_dir():
        return raw / "results" / "history_audit.json"
    return raw


def _history_label(audit_dir: Path, row: Mapping[str, object]) -> bool:
    source_suite = row.get("source_suite")
    source_refs = row.get("source_refs")
    if not isinstance(source_suite, str) or not isinstance(source_refs, list):
        raise TypeError("admissible history rows require source_suite and source_refs")
    for ref in source_refs:
        if not isinstance(ref, str) or not ref.endswith(("summary.json", "ood_replay.json")):
            continue
        payload = _read_mapping_json(Path(source_suite) / ref)
        value = payload.get("evaluator_success")
        if isinstance(value, bool):
            return value
    raise ValueError(f"admissible history row lacks a conclusive evaluator label in {audit_dir}")


def _history_rows(path: str | Path) -> tuple[Mapping[str, object], ...]:
    audit_path = _history_audit_path(path)
    payload = _read_mapping_json(audit_path)
    source_suite = payload.get("source_suite")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise TypeError(f"history audit rows must be a list: {audit_path}")
    normalized: list[Mapping[str, object]] = []
    for row in rows:
        if not isinstance(row, Mapping) or row.get("admissible") is not True:
            continue
        normalized.append({**row, "source_suite": source_suite})
    return tuple(normalized)


def summarize_collection(
    suite_dirs: Sequence[str | Path],
    *,
    history_audit: str | Path | None = None,
) -> CollectionEligibilityReport:
    """Summarize fixed suites and optional audit rows without choosing new seeds."""

    if not suite_dirs:
        raise ValueError("at least one collection suite is required for summary")
    seen_cases: set[str] = set()
    seen_lineages: set[str] = set()
    positive_count = 0
    negative_count = 0
    for suite_dir in suite_dirs:
        suite_path = Path(suite_dir)
        suite_cases = _suite_case_identities(suite_path)
        suite_identities: dict[str, str] = {}
        for case in suite_cases:
            _register_case(
                case_id=case.case_id,
                lineage_group_id=case.lineage_group_id,
                seen_cases=seen_cases,
                seen_lineages=seen_lineages,
            )
            suite_identities[case.case_id] = case.lineage_group_id
        lineages = tuple(
            CalibrationLineage.from_dict(lineage)
            for lineage in _read_sequence_json(suite_path / "results" / "lineages.json")
            if isinstance(lineage, Mapping)
        )
        for lineage in lineages:
            expected_group = suite_identities.get(lineage.episode_id)
            if expected_group is None:
                raise ValueError(f"lineage lacks persisted case: {lineage.episode_id}")
            if lineage.lineage_group_id != expected_group:
                raise ValueError(f"persisted lineage identity mismatch: {lineage.episode_id}")
        outcomes = tuple(
            CalibrationOutcome.from_dict(outcome)
            for outcome in _read_sequence_json(suite_path / "results" / "outcomes.json")
            if isinstance(outcome, Mapping)
        )
        for outcome in outcomes:
            if outcome.tier != "A" or outcome.task_success is None:
                continue
            if outcome.episode_id not in suite_identities:
                raise ValueError(f"Tier A outcome lacks lineage: {outcome.episode_id}")
            if outcome.task_success:
                positive_count += 1
            else:
                negative_count += 1

    audit_path: Path | None = None
    if history_audit is not None:
        audit_path = _history_audit_path(history_audit)
        for row in _history_rows(audit_path):
            case_id = row.get("case_id")
            if not isinstance(case_id, str) or not case_id:
                raise ValueError("admissible history row lacks case_id")
            _register_case(
                case_id=case_id,
                lineage_group_id=case_id,
                seen_cases=seen_cases,
                seen_lineages=seen_lineages,
            )
            if _history_label(audit_path, row):
                positive_count += 1
            else:
                negative_count += 1

    admissible = positive_count + negative_count
    report = CollectionEligibilityReport(
        source_suites=tuple(str(Path(path)) for path in suite_dirs),
        history_audit=str(audit_path) if audit_path is not None else None,
        admissible_tier_a_count=admissible,
        positive_count=positive_count,
        negative_count=negative_count,
        eligible_20_5_5=admissible >= 20 and positive_count >= 5 and negative_count >= 5,
    )
    output_dir = Path(suite_dirs[0]) / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "eligibility.json").write_text(
        json.dumps(asdict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest")
    parser.add_argument("--output-root", default="outputs/phase5")
    parser.add_argument("--gpu", default="5")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=360.0)
    parser.add_argument("--max-restarts", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--summarize-suite", action="append", default=[])
    parser.add_argument("--history-audit", default=None)
    return parser.parse_args()


def _terminate_servers(servers: Sequence[object]) -> None:
    for process in servers:
        terminate = getattr(process, "terminate", None)
        if callable(terminate):
            terminate()
        join = getattr(process, "join", None)
        if callable(join):
            join(timeout=5)


def main() -> None:
    args = parse_args()
    if args.summarize_suite:
        report = summarize_collection(args.summarize_suite, history_audit=args.history_audit)
        print(json.dumps(asdict(report), indent=2, sort_keys=True))
        return
    if not args.manifest:
        raise SystemExit("--manifest is required unless --summarize-suite is provided")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    _setup_capx_paths()
    manifest = load_collection_manifest(args.manifest)
    finalized = _preflight_manifest(manifest)
    servers: list[object] = []
    try:
        servers = _start_capx_api_servers(_resolve_project_path(finalized.cases[0].config_path))
        report = run_collection(
            finalized,
            output_root=args.output_root,
            run_config=CollectionRunConfig(
                max_workers=args.max_workers,
                timeout_s=args.timeout_s,
                max_restarts=args.max_restarts,
                max_steps=args.max_steps,
                gpu=args.gpu,
                fail_fast=args.fail_fast,
            ),
        )
    finally:
        _terminate_servers(servers)
    print(f"CAP-MAS P5.6 suite: {report.suite_dir}")
    print(f"CAP-MAS P5.6 completed cases: {report.completed_cases}")
    print(f"CAP-MAS P5.6 failed cases: {report.failed_cases}")
    print(f"CAP-MAS P5.6 eligible_20_5_5: {report.eligible_20_5_5}")


if __name__ == "__main__":
    main()
