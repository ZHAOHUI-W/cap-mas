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
SessionFactory = Callable[..., object]


class RawArtifactPersistenceError(RuntimeError):
    """Carries all raw evidence writes that failed without aborting later writes."""

    def __init__(self, errors: Sequence[Mapping[str, str]]) -> None:
        self.errors = tuple(dict(error) for error in errors)
        super().__init__(f"raw artifact persistence failed for {len(self.errors)} artifact(s)")


class ExternalProcessInterruptionError(RuntimeError):
    """Records a case whose collection process ended outside Python handling."""


@dataclass(frozen=True)
class CollectionRunConfig:
    max_workers: int = 1
    timeout_s: float = 360.0
    max_restarts: int = 0
    max_steps: int = 32
    gpu: str = "5"
    fail_fast: bool = False
    evidence_mode: Literal["same_runtime", "rehearsal_only"] = "same_runtime"

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
        if self.evidence_mode not in {"same_runtime", "rehearsal_only"}:
            raise ValueError("collection evidence_mode must be same_runtime or rehearsal_only")


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
    excluded_transport_smoke_suites: tuple[str, ...] = ()


def run_online_experiment(**kwargs: object) -> object:
    """Lazy wrapper for P5.3 online runner to keep imports simulator-free."""

    from scripts.run_libero_p53_online import run_online_experiment as live_runner

    return live_runner(**kwargs)


def _build_live_executor(**kwargs: object) -> object:
    """Lazy wrapper for the P5.3 physical executor factory."""

    from scripts.run_libero_p53_online import _build_live_executor as live_factory

    return live_factory(**kwargs)


def _build_live_evidence_session(**kwargs: object) -> object:
    """Build a lazy same-runtime session without importing LIBERO at startup."""

    from capmas.evaluation.libero_evidence_session import (
        LiveLiberoEvidenceSession,
        LiveLiberoEvidenceSessionConfig,
    )

    return LiveLiberoEvidenceSession(LiveLiberoEvidenceSessionConfig(**kwargs))


def _effective_evidence_mode(
    run_config: CollectionRunConfig,
    session_factory: SessionFactory | None,
) -> Literal["same_runtime", "rehearsal_only"]:
    """Keep pre-P5.6D injected runners on the legacy seam in unit tests."""

    if run_config.evidence_mode == "same_runtime" and session_factory is not None:
        return "same_runtime"
    return "rehearsal_only"


def _setup_capx_paths() -> None:
    from scripts.run_libero_p53_online import _setup_capx_paths as setup

    setup()


def _start_capx_api_servers(config_path: str | Path) -> list[object]:
    from capx.envs.configs.loader import DictLoader
    from capx.envs.runner import _start_api_servers

    config = DictLoader.load(str(config_path))
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
    raw_errors: dict[str, list[dict[str, str]]] = {}
    if isinstance(error, RawArtifactPersistenceError):
        raw_errors["initial"] = [dict(item) for item in error.errors]
    raw_artifact_error: Exception | None = None
    if payload is None:
        placeholder_errors = _write_evidence_artifacts(
            case_dir,
            {
                "evidence/decision_snapshots.json": [],
                "evidence/physical_payload.json": {},
                "evidence/horizon.json": _unknown_horizon().to_dict(),
                "evidence/graph_events.json": [],
            },
        )
        if placeholder_errors:
            raw_errors["placeholder"] = placeholder_errors
    else:
        try:
            retry_errors = _write_raw_artifacts(case_dir, payload=payload)
        except Exception as persistence_error:  # noqa: BLE001
            raw_artifact_error = persistence_error
        else:
            if retry_errors:
                raw_errors["retry"] = retry_errors

    ancillary_errors: list[dict[str, str]] = []
    _write_json_best_effort(case_dir, "results/outcomes.json", [], ancillary_errors)
    _write_json_best_effort(
        case_dir,
        "summary.json",
        {
            "case_id": case.case_id,
            "seed": case.seed,
            "status": "failed",
            "stage": stage,
            "error_type": type(error).__name__,
            "error": str(error),
        },
        ancillary_errors,
    )
    _write_text_best_effort(
        case_dir,
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
        ancillary_errors,
    )
    failure_payload: dict[str, object] = {
        "status": "failed",
        "case_id": case.case_id,
        "seed": case.seed,
        "stage": stage,
        "error_type": type(error).__name__,
        "error": str(error),
    }
    if raw_artifact_error is not None:
        failure_payload.update(
            {
                "raw_artifact_persistence_error_type": type(raw_artifact_error).__name__,
                "raw_artifact_persistence_error": str(raw_artifact_error),
            }
        )
    if raw_errors:
        failure_payload["raw_artifact_persistence_errors"] = raw_errors
    if ancillary_errors:
        failure_payload["failure_artifact_persistence_errors"] = ancillary_errors
    _write_json_best_effort(case_dir, "failure.json", failure_payload, [])


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


def _persistence_error(name: str, error: Exception) -> dict[str, str]:
    return {"artifact": name, "error_type": type(error).__name__, "error": str(error)}


def _write_json_best_effort(
    case_dir: Phase5RunDirectory,
    name: str,
    payload: object,
    errors: list[dict[str, str]],
) -> None:
    try:
        case_dir.write_json(name, payload)
    except Exception as error:  # noqa: BLE001
        errors.append(_persistence_error(name, error))


def _write_text_best_effort(
    case_dir: Phase5RunDirectory,
    name: str,
    content: str,
    errors: list[dict[str, str]],
) -> None:
    try:
        case_dir.write_text(name, content)
    except Exception as error:  # noqa: BLE001
        errors.append(_persistence_error(name, error))


def _write_evidence_artifacts(
    case_dir: Phase5RunDirectory,
    artifacts: Mapping[str, object],
) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    for name, artifact in artifacts.items():
        _write_json_best_effort(case_dir, name, artifact, errors)
    return errors


def _write_raw_artifacts(
    case_dir: Phase5RunDirectory,
    *,
    payload: Mapping[str, object],
) -> list[dict[str, str]]:
    physical_payload = payload.get("physical_result")
    physical_mapping = physical_payload if isinstance(physical_payload, Mapping) else {}
    raw_horizon = physical_mapping.get("horizon", _unknown_horizon())
    raw_graph_events = physical_mapping.get("graph_events", [])
    return _write_evidence_artifacts(
        case_dir,
        {
            "results/online.json": _json_plain(payload),
            "evidence/decision_snapshots.json": _json_plain(payload.get("feature_snapshots", [])),
            "evidence/physical_payload.json": _json_plain(physical_payload or {}),
            "evidence/horizon.json": _json_plain(raw_horizon),
            "evidence/graph_events.json": _json_plain(raw_graph_events),
        },
    )


def _annotate_finalization_failure(
    case_dir: Phase5RunDirectory,
    *,
    case: CalibrationCollectionCase,
    error: Exception,
) -> None:
    try:
        failure_payload = dict(_read_mapping_json(case_dir.path / "failure.json"))
    except Exception:  # noqa: BLE001
        failure_payload = {
            "status": "failed",
            "case_id": case.case_id,
            "seed": case.seed,
            "stage": "persistence",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    failure_payload["manifest_finalization_error_type"] = type(error).__name__
    failure_payload["manifest_finalization_error"] = str(error)
    _write_json_best_effort(case_dir, "failure.json", failure_payload, [])


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


def _read_optional_mapping(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _physical_execution_started_at_ns(case_dir: Path) -> int | None:
    paths = [case_dir / "results" / "online.json"]
    online_dir = case_dir / "online"
    if online_dir.is_dir():
        paths.extend(sorted(online_dir.rglob("run_config.json")))
    started_at = []
    for path in paths:
        value = _read_optional_mapping(path).get("physical_execution_started_at_ns")
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            started_at.append(value)
    return min(started_at) if started_at else None


def _case_status(case_dir: Path) -> str:
    summary_status = _read_optional_mapping(case_dir / "summary.json").get("status")
    if isinstance(summary_status, str):
        return summary_status
    run_status = _read_optional_mapping(case_dir / "run_config.json").get("status")
    return run_status if isinstance(run_status, str) else "unknown"


def _restore_completed_case(
    case_dir: Path,
    *,
    case: CalibrationCollectionCase,
) -> tuple[CollectionCaseResult, CalibrationLineage]:
    outcomes_payload = json.loads((case_dir / "results" / "outcomes.json").read_text(encoding="utf-8"))
    if not isinstance(outcomes_payload, list):
        raise TypeError(f"completed case outcomes must be a list: {case_dir}")
    outcomes = tuple(
        CalibrationOutcome.from_dict(item)
        for item in outcomes_payload
        if isinstance(item, Mapping)
    )
    lineage = CalibrationLineage.from_dict(
        _read_mapping_json(case_dir / "evidence" / "lineage.json")
    )
    if lineage.episode_id != case.case_id or lineage.lineage_group_id != case.lineage_group_id:
        raise ValueError(f"completed case lineage mismatch: {case.case_id}")
    return (
        CollectionCaseResult(
            case_id=case.case_id,
            seed=case.seed,
            status="completed",
            case_dir=case_dir,
            outcomes=outcomes,
        ),
        lineage,
    )


def _restore_failed_case(
    case_dir: Path,
    *,
    case: CalibrationCollectionCase,
) -> CollectionCaseResult:
    failure = _read_optional_mapping(case_dir / "failure.json")
    error = failure.get("error")
    return CollectionCaseResult(
        case_id=case.case_id,
        seed=case.seed,
        status="failed",
        case_dir=case_dir,
        outcomes=(),
        error=error if isinstance(error, str) else "collection case failed",
    )


def _mark_interrupted_case(
    case_dir: Phase5RunDirectory,
    *,
    case: CalibrationCollectionCase,
    physical_execution_started_at_ns: int | None,
) -> None:
    error = ExternalProcessInterruptionError(
        "collection process ended before the online runner completed"
    )
    _write_case_failure(
        case_dir,
        case=case,
        stage="online_runner",
        error=error,
    )
    prior_config = _read_optional_mapping(case_dir.path / "run_config.json")
    case_dir.write_json(
        "run_config.json",
        {
            **prior_config,
            "status": "failed",
            "stage": "online_runner",
            "error_type": type(error).__name__,
            "error": str(error),
        },
    )
    failure = dict(_read_optional_mapping(case_dir.path / "failure.json"))
    failure["physical_execution_started_at_ns"] = physical_execution_started_at_ns
    case_dir.write_json("failure.json", failure)
    case_dir.finalize_manifest()


def _suite_report(
    suite_dir: Phase5RunDirectory,
    *,
    manifest: CalibrationCollectionManifest,
    run_config: CollectionRunConfig,
    results: Sequence[CollectionCaseResult],
    lineages: Sequence[CalibrationLineage],
) -> CollectionSuiteReport:
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
        manifest=manifest,
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
                "case_dir": str(result.case_dir.relative_to(suite_dir.path)),
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
                f"manifest_sha256={manifest.manifest_sha256}",
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
    return report


def run_collection_case(
    case: CalibrationCollectionCase,
    *,
    suite_dir: str | Path | Phase5RunDirectory,
    manifest: CalibrationCollectionManifest,
    run_config: CollectionRunConfig,
    online_runner: OnlineRunner,
    executor_factory: ExecutorFactory,
    session_factory: SessionFactory | None = None,
) -> tuple[CollectionCaseResult, CalibrationLineage | None]:
    parent = suite_dir.path if isinstance(suite_dir, Phase5RunDirectory) else Path(suite_dir)
    case_dir = Phase5RunDirectory.create(parent, "cases", case.case_id)
    evidence_mode = _effective_evidence_mode(run_config, session_factory)
    case_dir.write_json("case.json", case.to_dict())
    case_dir.write_json(
        "run_config.json",
        {
            "experiment": EXPERIMENT_NAME,
            "case": case.to_dict(),
            "run_config": asdict(run_config),
            "evidence_mode": evidence_mode,
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
        physical_executor: object | None = None
        evidence_session: object | None = None
        if evidence_mode == "same_runtime":
            assert session_factory is not None
            stage = "evidence_session_construction"
            evidence_session = session_factory(
                config_path=str(config_path),
                object_name=case.object_name,
                target_name=case.target_name,
                max_steps=run_config.max_steps,
                seed=case.seed,
                layout_variant=layout_variant,
            )
        else:
            stage = "executor_construction"
            physical_executor = executor_factory(
                config_path=str(config_path),
                object_name=case.object_name,
                target_name=case.target_name,
                max_steps=run_config.max_steps,
                seed=case.seed,
                layout_variant=layout_variant,
            )
        stage = "validation"
        context = CalibrationCollectionContext(
            episode_id=case.case_id,
            episode_epoch=1,
            family_id=case.family_id,
            feature_schema_version=manifest.feature_schema_version,
            memory_skill_version=manifest.memory_skill_version,
            robot_skill_version=manifest.robot_skill_version,
            collection_lane="physical",
        )
        pool_config = RehearsalPoolConfig(
            max_workers=1,
            timeout_s=run_config.timeout_s,
            max_restarts=run_config.max_restarts,
        )
        stage = "online_runner"
        outcome = online_runner(
            config_path=str(config_path),
            candidates=candidates,
            seed=case.seed,
            scene_version=scene_version,
            mode="online_bounded",
            cache_mode="disabled",
            selection_repeats=1,
            output_root=case_dir.path / "online",
            pool_config=pool_config,
            max_steps=run_config.max_steps,
            object_name=case.object_name,
            target_name=case.target_name,
            layout_variant=layout_variant,
            gpu=run_config.gpu,
            physical_executor=physical_executor,
            calibration_context=context,
            evidence_session=evidence_session,
        )
        stage = "validation"
        payload = _outcome_payload(outcome)
        stage = "persistence"
        raw_write_errors = _write_raw_artifacts(case_dir, payload=payload)
        if raw_write_errors:
            raise RawArtifactPersistenceError(raw_write_errors)
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
                "evidence_mode": evidence_mode,
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
                "evidence_mode": evidence_mode,
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
    try:
        case_dir.finalize_manifest()
    except Exception as finalization_error:  # noqa: BLE001
        failure_write_error: Exception | None = None
        if result.status == "completed":
            try:
                _write_case_failure(
                    case_dir,
                    case=case,
                    stage="persistence",
                    error=finalization_error,
                    payload=payload,
                )
            except Exception as write_error:  # noqa: BLE001
                failure_write_error = write_error
            result = CollectionCaseResult(
                case_id=case.case_id,
                seed=case.seed,
                status="failed",
                case_dir=case_dir.path,
                outcomes=(),
                error=(
                    str(finalization_error)
                    if failure_write_error is None
                    else f"{finalization_error}; typed failure write also failed: {failure_write_error}"
                ),
            )
            lineage = None
        else:
            _annotate_finalization_failure(
                case_dir,
                case=case,
                error=finalization_error,
            )
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
    session_factory: SessionFactory | None = None,
) -> CollectionSuiteReport:
    """Run every pre-registered case once without outcome-adaptive stopping."""

    os.environ["CUDA_VISIBLE_DEVICES"] = run_config.gpu
    finalized = _preflight_manifest(manifest)
    evidence_mode = _effective_evidence_mode(run_config, session_factory)
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
            "evidence_mode": evidence_mode,
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
            session_factory=session_factory,
        )
        results.append(result)
        if lineage is not None:
            lineages.append(lineage)
        if run_config.fail_fast and _is_infrastructure_failure(result):
            stop_error = RuntimeError(
                f"P5.6 collection case failed: {case.case_id}: {result.error}"
            )
            break

    report = _suite_report(
        suite_dir,
        manifest=finalized,
        run_config=run_config,
        results=results,
        lineages=lineages,
    )
    if stop_error is not None:
        raise stop_error
    return report


def _load_persisted_manifest(path: Path) -> CalibrationCollectionManifest:
    return CalibrationCollectionManifest.from_dict(_read_mapping_json(path / "suite_manifest.json"))


def _existing_case_attempts(
    suite_path: Path,
    *,
    manifest: CalibrationCollectionManifest,
) -> Mapping[str, tuple[Path, ...]]:
    known_cases = {case.case_id: case for case in manifest.cases}
    grouped: dict[str, list[Path]] = {case.case_id: [] for case in manifest.cases}
    cases_dir = suite_path / "cases"
    if not cases_dir.is_dir():
        return {case_id: () for case_id in grouped}
    for case_dir in sorted(path for path in cases_dir.iterdir() if path.is_dir()):
        case_path = case_dir / "case.json"
        if not case_path.is_file():
            continue
        persisted_case = CalibrationCollectionCase.from_dict(_read_mapping_json(case_path))
        expected = known_cases.get(persisted_case.case_id)
        if expected is None:
            raise ValueError(f"interrupted suite contains an unknown case: {persisted_case.case_id}")
        if persisted_case != expected:
            raise ValueError(f"interrupted suite case contract mismatch: {persisted_case.case_id}")
        grouped[persisted_case.case_id].append(case_dir)
    return {case_id: tuple(paths) for case_id, paths in grouped.items()}


def resume_collection(
    manifest: CalibrationCollectionManifest,
    *,
    suite_dir: str | Path,
    run_config: CollectionRunConfig,
    online_runner: OnlineRunner = run_online_experiment,
    executor_factory: ExecutorFactory = _build_live_executor,
    session_factory: SessionFactory | None = None,
) -> CollectionSuiteReport:
    """Finish an interrupted immutable suite without replaying started execution."""

    os.environ["CUDA_VISIBLE_DEVICES"] = run_config.gpu
    finalized = _preflight_manifest(manifest)
    evidence_mode = _effective_evidence_mode(run_config, session_factory)
    existing_path = Path(suite_dir).resolve(strict=True)
    existing_dir = Phase5RunDirectory(existing_path)
    persisted = _load_persisted_manifest(existing_path)
    if collection_manifest_sha256(persisted) != finalized.manifest_sha256:
        raise ValueError("resume manifest digest does not match the interrupted suite")
    if _read_optional_mapping(existing_path / "run_config.json").get("status") == "completed":
        raise ValueError("cannot resume a completed collection suite")

    existing_dir.write_json(
        "run_config.json",
        {
            "experiment": EXPERIMENT_NAME,
            "manifest_sha256": finalized.manifest_sha256,
            **asdict(run_config),
            "evidence_mode": evidence_mode,
            "status": "resuming",
        },
    )
    attempts = _existing_case_attempts(existing_path, manifest=finalized)
    results_by_case: dict[str, CollectionCaseResult] = {}
    lineages_by_case: dict[str, CalibrationLineage] = {}
    pending: list[CalibrationCollectionCase] = []
    interruption_records: list[dict[str, object]] = []

    for case in finalized.cases:
        case_attempts = attempts[case.case_id]
        completed = tuple(path for path in case_attempts if _case_status(path) == "completed")
        if len(completed) > 1:
            raise ValueError(f"interrupted suite has multiple completed attempts: {case.case_id}")
        if completed:
            result, lineage = _restore_completed_case(completed[0], case=case)
            results_by_case[case.case_id] = result
            lineages_by_case[case.case_id] = lineage
            continue

        running = tuple(path for path in case_attempts if _case_status(path) == "running")
        if len(running) > 1:
            raise ValueError(f"interrupted suite has multiple running attempts: {case.case_id}")
        if running:
            interrupted_path = running[0]
            started_at = _physical_execution_started_at_ns(interrupted_path)
            _mark_interrupted_case(
                Phase5RunDirectory(interrupted_path),
                case=case,
                physical_execution_started_at_ns=started_at,
            )
            disposition = (
                "retry_unstarted" if started_at is None else "do_not_retry_physical_started"
            )
            interruption_records.append(
                {
                    "case_id": case.case_id,
                    "disposition": disposition,
                    "physical_execution_started_at_ns": started_at,
                    "prior_case_dir": str(interrupted_path),
                }
            )
            if started_at is None:
                pending.append(case)
            else:
                results_by_case[case.case_id] = _restore_failed_case(
                    interrupted_path,
                    case=case,
                )
            continue

        failed = tuple(path for path in case_attempts if _case_status(path) == "failed")
        if len(failed) > 1:
            raise ValueError(f"interrupted suite has multiple failed attempts: {case.case_id}")
        if failed:
            results_by_case[case.case_id] = _restore_failed_case(failed[0], case=case)
            continue
        pending.append(case)

    existing_dir.write_json("results/interrupted_attempts.json", interruption_records)
    stop_error: RuntimeError | None = None
    for case in pending:
        result, lineage = run_collection_case(
            case,
            suite_dir=existing_dir,
            manifest=finalized,
            run_config=run_config,
            online_runner=online_runner,
            executor_factory=executor_factory,
            session_factory=session_factory,
        )
        results_by_case[case.case_id] = result
        if lineage is not None:
            lineages_by_case[case.case_id] = lineage
        if run_config.fail_fast and _is_infrastructure_failure(result):
            stop_error = RuntimeError(
                f"P5.6 collection case failed: {case.case_id}: {result.error}"
            )
            break

    ordered_results = [results_by_case[case.case_id] for case in finalized.cases if case.case_id in results_by_case]
    ordered_lineages = [
        lineages_by_case[case.case_id]
        for case in finalized.cases
        if case.case_id in lineages_by_case
    ]
    report = _suite_report(
        existing_dir,
        manifest=finalized,
        run_config=run_config,
        results=ordered_results,
        lineages=ordered_lineages,
    )
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
        raw_case_dir = Path(case_dir_value)
        candidates = (
            (raw_case_dir,)
            if raw_case_dir.is_absolute()
            else (suite_path / raw_case_dir, raw_case_dir)
        )
        case_dir = next(
            (candidate for candidate in candidates if (candidate / "case.json").is_file()),
            None,
        )
        if case_dir is None:
            raise FileNotFoundError(
                f"persisted collection case directory cannot be resolved: {case_dir_value}"
            )
        case = CalibrationCollectionCase.from_dict(_read_mapping_json(case_dir / "case.json"))
        if case.case_id != case_id:
            raise ValueError(f"persisted collection case identity mismatch: {case_id}")
        cases.append(case)
    return tuple(cases)


def _suite_collection_purpose(suite_path: Path) -> str:
    manifest = CalibrationCollectionManifest.from_dict(
        _read_mapping_json(suite_path / "suite_manifest.json")
    )
    return manifest.collection_purpose


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
    excluded_transport_smoke_suites: list[str] = []
    for suite_dir in suite_dirs:
        suite_path = Path(suite_dir)
        collection_purpose = _suite_collection_purpose(suite_path)
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
        if collection_purpose == "transport_smoke":
            excluded_transport_smoke_suites.append(str(suite_path))
            continue
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
        excluded_transport_smoke_suites=tuple(excluded_transport_smoke_suites),
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
    parser.add_argument(
        "--evidence-mode",
        choices=("same_runtime", "rehearsal_only"),
        default="same_runtime",
    )
    parser.add_argument("--resume-suite")
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
    if args.resume_suite and args.summarize_suite:
        raise SystemExit("--resume-suite cannot be combined with --summarize-suite")
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
        run_config = CollectionRunConfig(
            max_workers=args.max_workers,
            timeout_s=args.timeout_s,
            max_restarts=args.max_restarts,
            max_steps=args.max_steps,
            gpu=args.gpu,
            fail_fast=args.fail_fast,
            evidence_mode=args.evidence_mode,
        )
        session_factory = (
            _build_live_evidence_session
            if args.evidence_mode == "same_runtime"
            else None
        )
        if args.resume_suite:
            report = resume_collection(
                finalized,
                suite_dir=args.resume_suite,
                run_config=run_config,
                session_factory=session_factory,
            )
        else:
            report = run_collection(
                finalized,
                output_root=args.output_root,
                run_config=run_config,
                session_factory=session_factory,
            )
    finally:
        _terminate_servers(servers)
    print(f"CAP-MAS P5.6 suite: {report.suite_dir}")
    print(f"CAP-MAS P5.6 completed cases: {report.completed_cases}")
    print(f"CAP-MAS P5.6 failed cases: {report.failed_cases}")
    print(f"CAP-MAS P5.6 eligible_20_5_5: {report.eligible_20_5_5}")


if __name__ == "__main__":
    main()
