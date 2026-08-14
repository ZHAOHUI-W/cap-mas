"""Read-only compatibility audit for historical P5.5 rows entering P5.6 datasets."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

from capmas.contracts.calibration import CandidateFeatureSnapshot, HorizonLabel
from capmas.evaluation.phase5_artifacts import Phase5RunDirectory

SCHEMA_VERSION = "p5.6.2a.history_audit.v1"
EXPERIMENT_NAME = "P5.6.2a_object6_history_audit"

MISSING_CANDIDATE_ID = "MISSING_CANDIDATE_ID"
MISSING_CANDIDATE_FINGERPRINT = "MISSING_CANDIDATE_FINGERPRINT"
INVALID_CANDIDATE_FINGERPRINT = "INVALID_CANDIDATE_FINGERPRINT"
MISSING_SELECTION_EVENT = "MISSING_SELECTION_EVENT"
SELECTION_IDENTITY_MISMATCH = "SELECTION_IDENTITY_MISMATCH"
MISSING_PREEXECUTION_FEATURE_SNAPSHOT = "MISSING_PREEXECUTION_FEATURE_SNAPSHOT"
FEATURE_SNAPSHOT_IDENTITY_MISMATCH = "FEATURE_SNAPSHOT_IDENTITY_MISMATCH"
INVALID_FEATURE_SNAPSHOT = "INVALID_FEATURE_SNAPSHOT"
MISSING_EXECUTION_START_TIMESTAMP = "MISSING_EXECUTION_START_TIMESTAMP"
FEATURE_SNAPSHOT_AFTER_EXECUTION_START = "FEATURE_SNAPSHOT_AFTER_EXECUTION_START"
INCONCLUSIVE_EVALUATOR = "INCONCLUSIVE_EVALUATOR"
EVALUATOR_OUTCOME_MISMATCH = "EVALUATOR_OUTCOME_MISMATCH"
MISSING_EVALUATOR_OBSERVATION_TIMESTAMP = "MISSING_EVALUATOR_OBSERVATION_TIMESTAMP"
EVALUATOR_OBSERVED_BEFORE_EXECUTION_START = "EVALUATOR_OBSERVED_BEFORE_EXECUTION_START"
MISSING_GRAPH_EVENTS = "MISSING_GRAPH_EVENTS"
INVALID_GRAPH_EVENTS = "INVALID_GRAPH_EVENTS"
MISSING_HORIZON_LINEAGE = "MISSING_HORIZON_LINEAGE"
INVALID_HORIZON_LINEAGE = "INVALID_HORIZON_LINEAGE"
MISSING_SKILL_VERSION = "MISSING_SKILL_VERSION"
MISSING_CANDIDATE_ARTIFACT_DIGEST = "MISSING_CANDIDATE_ARTIFACT_DIGEST"
INVALID_CANDIDATE_ARTIFACT_DIGEST = "INVALID_CANDIDATE_ARTIFACT_DIGEST"


@dataclass(frozen=True)
class HistoricalRowDecision:
    case_id: str
    family_id: str
    candidate_id: str | None
    candidate_fingerprint: str | None
    admissible: bool
    reasons: tuple[str, ...]
    source_refs: tuple[str, ...]


@dataclass(frozen=True)
class HistoricalCompatibilityAudit:
    family_id: str
    source_suite: str
    source_manifest_sha256: str
    examined_count: int
    admissible_tier_a_count: int
    rejected_count: int
    rejection_counts: Mapping[str, int]
    rows: tuple[HistoricalRowDecision, ...]


@dataclass(frozen=True)
class HistoricalAuditRunResult:
    run_dir: Path
    audit: HistoricalCompatibilityAudit


def audit_p55_history(
    suite_dir: str | Path,
    *,
    family_id: str,
) -> HistoricalCompatibilityAudit:
    """Inspect historical family rows without mutating or backfilling source artifacts."""

    suite_path = Path(suite_dir)
    cases_root = suite_path / "cases"
    if not cases_root.is_dir():
        raise ValueError(f"P5.5 suite cases directory does not exist: {cases_root}")
    source_manifest_sha256 = _load_source_manifest_sha256(suite_path)

    rows: list[HistoricalRowDecision] = []
    for case_dir in sorted(path for path in cases_root.iterdir() if path.is_dir()):
        case_path = case_dir / "case.json"
        if not case_path.is_file():
            continue
        case_payload = _read_mapping_json(case_path)
        if case_payload.get("task_family") != family_id:
            continue
        rows.append(_audit_case(suite_path=suite_path, case_dir=case_dir, case=case_payload))

    rejection_counts = Counter(reason for row in rows for reason in row.reasons)
    admissible_count = sum(row.admissible for row in rows)
    return HistoricalCompatibilityAudit(
        family_id=family_id,
        source_suite=str(suite_path),
        source_manifest_sha256=source_manifest_sha256,
        examined_count=len(rows),
        admissible_tier_a_count=admissible_count,
        rejected_count=len(rows) - admissible_count,
        rejection_counts=dict(sorted(rejection_counts.items())),
        rows=tuple(rows),
    )


def run_history_audit(
    *,
    suite_dir: str | Path,
    family_id: str,
    output_root: str | Path,
) -> HistoricalAuditRunResult:
    """Run the read-only audit and write an isolated Phase 5 artifact directory."""

    suite_path = Path(suite_dir)
    output_path = Path(output_root)
    _reject_output_inside_suite(suite_path, output_path)
    run_dir = Phase5RunDirectory.create(
        output_path,
        EXPERIMENT_NAME,
        f"history_{uuid4().hex[:8]}",
    )
    log_lines = [
        f"experiment={EXPERIMENT_NAME}",
        f"suite_dir={suite_path}",
        f"family_id={family_id}",
        "read_only_source=True",
    ]
    try:
        source_manifest_sha256 = _load_source_manifest_sha256(suite_path)
        run_dir.write_json(
            "run_config.json",
            _run_config_payload(
                suite_dir=suite_path,
                family_id=family_id,
                source_manifest_sha256=source_manifest_sha256,
                status="running",
            ),
        )
        audit = audit_p55_history(suite_path, family_id=family_id)
        run_dir.write_json(
            "results/history_audit.json",
            {"schema_version": SCHEMA_VERSION, **_audit_payload(audit)},
        )
        run_dir.write_json(
            "results/admissible_rows.json",
            [asdict(row) for row in audit.rows if row.admissible],
        )
        run_dir.write_text("summary.md", _summary_markdown(audit))
        run_dir.write_json(
            "run_config.json",
            _run_config_payload(
                suite_dir=suite_path,
                family_id=family_id,
                source_manifest_sha256=audit.source_manifest_sha256,
                status="completed",
            ),
        )
        log_lines.extend(
            [
                f"source_manifest_sha256={audit.source_manifest_sha256}",
                f"examined_count={audit.examined_count}",
                f"admissible_tier_a_count={audit.admissible_tier_a_count}",
                f"rejected_count={audit.rejected_count}",
                f"rejection_counts={dict(audit.rejection_counts)}",
                "status=completed",
                "",
            ]
        )
        run_dir.write_text("logs/runner.log", "\n".join(log_lines))
        run_dir.finalize_manifest()
    except BaseException as error:
        manifest = _safe_source_manifest_sha256(suite_path)
        run_dir.write_json(
            "run_config.json",
            _run_config_payload(
                suite_dir=suite_path,
                family_id=family_id,
                source_manifest_sha256=manifest,
                status="failed",
                error_type=type(error).__name__,
                error=str(error),
            ),
        )
        log_lines.extend(["status=failed", f"error={type(error).__name__}: {error}", ""])
        run_dir.write_text("logs/runner.log", "\n".join(log_lines))
        run_dir.finalize_manifest()
        raise

    return HistoricalAuditRunResult(run_dir=run_dir.path, audit=audit)


def _audit_case(
    *,
    suite_path: Path,
    case_dir: Path,
    case: Mapping[str, object],
) -> HistoricalRowDecision:
    summary_path = case_dir / "summary.json"
    evidence_path = case_dir / "evidence" / "ood_replay.json"
    snapshot_path = case_dir / "evidence" / "calibration_feature_snapshots.json"
    summary = _read_mapping_json(summary_path)
    evidence = _read_mapping_json(evidence_path)
    case_id = _required_str(case, "case_id")
    family_id = _required_str(case, "task_family")
    reasons: list[str] = []
    source_refs = [
        _relative_ref(suite_path, case_dir / "case.json"),
        _relative_ref(suite_path, summary_path),
        _relative_ref(suite_path, evidence_path),
    ]

    _check_matching_case_ids(case_id, summary=summary, evidence=evidence)
    candidate_id = _candidate_id(summary=summary, evidence=evidence)
    candidate_fingerprint = _optional_non_empty_str(evidence.get("candidate_fingerprint"))
    candidate_artifact_digest = _optional_non_empty_str(case.get("candidate_artifact_sha256"))

    if candidate_id is None:
        reasons.append(MISSING_CANDIDATE_ID)
    if candidate_fingerprint is None:
        reasons.append(MISSING_CANDIDATE_FINGERPRINT)
    elif not _is_sha256(candidate_fingerprint):
        reasons.append(INVALID_CANDIDATE_FINGERPRINT)
    if candidate_artifact_digest is None:
        reasons.append(MISSING_CANDIDATE_ARTIFACT_DIGEST)
    elif not _is_sha256(candidate_artifact_digest):
        reasons.append(INVALID_CANDIDATE_ARTIFACT_DIGEST)

    if not _has_selection_event(summary=summary, evidence=evidence):
        reasons.append(MISSING_SELECTION_EVENT)
    elif _selection_mismatch(summary=summary, evidence=evidence, candidate_id=candidate_id):
        reasons.append(SELECTION_IDENTITY_MISMATCH)

    snapshot: CandidateFeatureSnapshot | None = None
    if not snapshot_path.is_file():
        reasons.append(MISSING_PREEXECUTION_FEATURE_SNAPSHOT)
    else:
        source_refs.append(_relative_ref(suite_path, snapshot_path))
        snapshot, snapshot_reasons = _load_matching_snapshot(
            snapshot_path=snapshot_path,
            case_id=case_id,
            family_id=family_id,
            candidate_id=candidate_id,
            candidate_fingerprint=candidate_fingerprint,
            source_scene_version=evidence.get("source_scene_version"),
        )
        reasons.extend(snapshot_reasons)

    execution_started_at_ns = _execution_started_at_ns(evidence)
    if snapshot is not None:
        if execution_started_at_ns is None:
            reasons.append(MISSING_EXECUTION_START_TIMESTAMP)
        elif snapshot.captured_at_ns > execution_started_at_ns:
            reasons.append(FEATURE_SNAPSHOT_AFTER_EXECUTION_START)
        evaluator_observed_at_ns = _evaluator_observed_at_ns(evidence)
        if evaluator_observed_at_ns is None:
            reasons.append(MISSING_EVALUATOR_OBSERVATION_TIMESTAMP)
        elif (
            execution_started_at_ns is not None
            and evaluator_observed_at_ns < execution_started_at_ns
        ):
            reasons.append(EVALUATOR_OBSERVED_BEFORE_EXECUTION_START)

    if snapshot is not None and (
        not snapshot.memory_skill_version or not snapshot.robot_skill_version
    ):
        reasons.append(MISSING_SKILL_VERSION)

    reasons.extend(_evaluator_reasons(summary=summary, evidence=evidence))
    reasons.extend(_graph_event_reasons(evidence))
    horizon_reasons = _horizon_reasons(evidence)
    reasons.extend(horizon_reasons)

    unique_reasons = tuple(dict.fromkeys(reasons))
    return HistoricalRowDecision(
        case_id=case_id,
        family_id=family_id,
        candidate_id=candidate_id,
        candidate_fingerprint=candidate_fingerprint,
        admissible=not unique_reasons,
        reasons=unique_reasons,
        source_refs=tuple(source_refs),
    )


def _load_matching_snapshot(
    *,
    snapshot_path: Path,
    case_id: str,
    family_id: str,
    candidate_id: str | None,
    candidate_fingerprint: str | None,
    source_scene_version: object,
) -> tuple[CandidateFeatureSnapshot | None, tuple[str, ...]]:
    reasons: list[str] = []
    try:
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None, (INVALID_FEATURE_SNAPSHOT,)
    if not isinstance(payload, list):
        return None, (INVALID_FEATURE_SNAPSHOT,)
    matching_snapshot: CandidateFeatureSnapshot | None = None
    for raw_snapshot in payload:
        if not isinstance(raw_snapshot, Mapping):
            reasons.append(INVALID_FEATURE_SNAPSHOT)
            continue
        if candidate_id is not None and raw_snapshot.get("candidate_id") != candidate_id:
            continue
        try:
            snapshot = CandidateFeatureSnapshot.from_dict(raw_snapshot)
        except (TypeError, ValueError):
            reasons.append(INVALID_FEATURE_SNAPSHOT)
            continue
        matching_snapshot = snapshot
        break
    if matching_snapshot is None:
        reasons.append(FEATURE_SNAPSHOT_IDENTITY_MISMATCH)
        return None, tuple(dict.fromkeys(reasons))
    identity_matches = (
        matching_snapshot.episode_id == case_id
        and matching_snapshot.family_id == family_id
        and matching_snapshot.candidate_id == candidate_id
        and matching_snapshot.candidate_fingerprint == candidate_fingerprint
        and _scene_version_matches(matching_snapshot.scene_version, source_scene_version)
        and matching_snapshot.collection_lane == "physical"
    )
    if not identity_matches:
        reasons.append(FEATURE_SNAPSHOT_IDENTITY_MISMATCH)
    return matching_snapshot, tuple(dict.fromkeys(reasons))


def _horizon_reasons(evidence: Mapping[str, object]) -> tuple[str, ...]:
    raw_horizon = evidence.get("horizon")
    if not isinstance(raw_horizon, Mapping):
        return (MISSING_HORIZON_LINEAGE,)
    try:
        horizon = HorizonLabel.from_dict(raw_horizon)
    except (TypeError, ValueError):
        return (INVALID_HORIZON_LINEAGE,)
    if (
        horizon.planned_source != "mission_graph"
        or horizon.realized_source != "execution_trace"
        or not horizon.planned_valid
        or not horizon.realized_valid
        or not _has_complete_horizon_counts(horizon)
    ):
        return (INVALID_HORIZON_LINEAGE,)
    return ()


def _has_complete_horizon_counts(horizon: HorizonLabel) -> bool:
    return all(
        getattr(horizon, field_name) is not None
        for field_name in (
            "planned_critical_path_actions",
            "planned_critical_path_subgoals",
            "planned_checkpoint_subgraphs",
            "attempted_actions",
            "completed_actions",
            "attempted_subgoals",
            "completed_subgoals",
            "attempted_checkpoints",
            "completed_checkpoints",
        )
    )


def _graph_event_reasons(evidence: Mapping[str, object]) -> tuple[str, ...]:
    events = evidence.get("graph_events")
    if not isinstance(events, list) or not events:
        return (MISSING_GRAPH_EVENTS,)
    if not all(_is_graph_event_record(event) for event in events):
        return (INVALID_GRAPH_EVENTS,)
    return ()


def _is_graph_event_record(event: object) -> bool:
    if not isinstance(event, Mapping):
        return False
    kind = event.get("kind")
    node_type = event.get("node_type")
    outcome = event.get("outcome")
    return (
        _nonnegative_int(event.get("sequence"))
        and kind
        in {
            "subgraph_started",
            "subgraph_completed",
            "subgraph_failed",
            "node_started",
            "node_completed",
            "node_failed",
        }
        and _optional_non_empty_str(event.get("subgraph_id")) is not None
        and (
            event.get("node_id") is None
            or _optional_non_empty_str(event.get("node_id")) is not None
        )
        and (node_type is None or node_type in {"action", "checkpoint", "router"})
        and _nonnegative_int(event.get("attempt"))
        and (outcome is None or isinstance(outcome, str))
        and _nonnegative_int(event.get("occurred_at_ns"))
    )


def _evaluator_reasons(
    *,
    summary: Mapping[str, object],
    evidence: Mapping[str, object],
) -> tuple[str, ...]:
    summary_outcome = summary.get("evaluator_success")
    evidence_outcome = evidence.get("evaluator_success")
    if not isinstance(summary_outcome, bool) or not isinstance(evidence_outcome, bool):
        return (INCONCLUSIVE_EVALUATOR,)
    if summary_outcome != evidence_outcome:
        return (EVALUATOR_OUTCOME_MISMATCH,)
    return ()


def _candidate_id(
    *,
    summary: Mapping[str, object],
    evidence: Mapping[str, object],
) -> str | None:
    evidence_candidate = _optional_non_empty_str(evidence.get("candidate_id"))
    if evidence_candidate is not None and evidence_candidate != "unselected":
        return evidence_candidate
    summary_winner = _optional_non_empty_str(summary.get("primary_winner"))
    if summary_winner is not None and summary_winner != "unselected":
        return summary_winner
    return None


def _has_selection_event(
    *,
    summary: Mapping[str, object],
    evidence: Mapping[str, object],
) -> bool:
    return (
        _optional_non_empty_str(summary.get("primary_winner")) not in {None, "unselected"}
        or _optional_non_empty_str(evidence.get("candidate_id")) not in {None, "unselected"}
        or _optional_non_empty_str(evidence.get("physical_candidate_id"))
        not in {None, "unselected"}
    )


def _selection_mismatch(
    *,
    summary: Mapping[str, object],
    evidence: Mapping[str, object],
    candidate_id: str | None,
) -> bool:
    if candidate_id is None:
        return False
    selected_values = [
        _optional_non_empty_str(summary.get("primary_winner")),
        _optional_non_empty_str(evidence.get("candidate_id")),
        _optional_non_empty_str(evidence.get("physical_candidate_id")),
    ]
    return any(value not in {None, "unselected", candidate_id} for value in selected_values)


def _execution_started_at_ns(evidence: Mapping[str, object]) -> int | None:
    for key in (
        "execution_started_at_ns",
        "physical_execution_started_at_ns",
    ):
        value = evidence.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _evaluator_observed_at_ns(evidence: Mapping[str, object]) -> int | None:
    for key in ("evaluator_observed_at_ns", "physical_evaluator_observed_at_ns"):
        value = evidence.get(key)
        if _nonnegative_int(value):
            return value
    return None


def _scene_version_matches(snapshot_scene_version: int, source_scene_version: object) -> bool:
    return (
        isinstance(source_scene_version, int)
        and not isinstance(source_scene_version, bool)
        and source_scene_version >= 0
        and snapshot_scene_version == source_scene_version
    )


def _audit_payload(audit: HistoricalCompatibilityAudit) -> dict[str, object]:
    payload = asdict(audit)
    payload["rejection_counts"] = dict(audit.rejection_counts)
    return payload


def _summary_markdown(audit: HistoricalCompatibilityAudit) -> str:
    lines = [
        "# CAP-MAS P5.6 historical compatibility audit",
        "",
        f"- family_id: {audit.family_id}",
        f"- source_manifest_sha256: {audit.source_manifest_sha256}",
        f"- examined_count: {audit.examined_count}",
        f"- admissible_tier_a_count: {audit.admissible_tier_a_count}",
        f"- rejected_count: {audit.rejected_count}",
        "",
        "| rejection reason | count |",
        "| --- | ---: |",
    ]
    for reason, count in sorted(audit.rejection_counts.items()):
        lines.append(f"| {reason} | {count} |")
    lines.append("")
    return "\n".join(lines)


def _run_config_payload(
    *,
    suite_dir: Path,
    family_id: str,
    source_manifest_sha256: str,
    status: str,
    error_type: str | None = None,
    error: str | None = None,
) -> Mapping[str, object]:
    payload: dict[str, object] = {
        "experiment": EXPERIMENT_NAME,
        "suite_dir": str(suite_dir),
        "family_id": family_id,
        "source_manifest_sha256": source_manifest_sha256,
        "status": status,
        "read_only_source": True,
    }
    if error_type is not None:
        payload["error_type"] = error_type
    if error is not None:
        payload["error"] = error
    return payload


def _reject_output_inside_suite(suite_dir: Path, output_root: Path) -> None:
    suite_resolved = suite_dir.resolve()
    output_resolved = output_root.resolve(strict=False)
    if output_resolved == suite_resolved or suite_resolved in output_resolved.parents:
        raise ValueError("output_root must not be equal to or nested inside suite_dir")


def _load_source_manifest_sha256(suite_dir: Path) -> str:
    payload = _read_mapping_json(suite_dir / "suite_manifest.json")
    digest = payload.get("manifest_sha256")
    if not isinstance(digest, str) or not _is_sha256(digest):
        raise ValueError("suite_manifest.json must contain a valid manifest_sha256")
    return digest


def _safe_source_manifest_sha256(suite_dir: Path) -> str:
    try:
        return _load_source_manifest_sha256(suite_dir)
    except (OSError, TypeError, ValueError):
        return ""


def _read_mapping_json(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        raise ValueError(f"required retained artifact is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"required retained artifact is malformed JSON: {path}") from error
    if not isinstance(payload, Mapping):
        raise TypeError(f"required retained artifact must contain a JSON object: {path}")
    return payload


def _check_matching_case_ids(
    case_id: str,
    *,
    summary: Mapping[str, object],
    evidence: Mapping[str, object],
) -> None:
    _require_equal(case_id, _required_str(summary, "case_id"), "summary case_id")
    _require_equal(case_id, _required_str(evidence, "case_id"), "evidence case_id")


def _required_str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"required string field is missing or invalid: {key}")
    return value


def _optional_non_empty_str(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _require_equal(left: object, right: object, label: str) -> None:
    if left != right:
        raise ValueError(f"case artifact mismatch for {label}: {left!r} != {right!r}")


def _nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _relative_ref(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


__all__ = [
    "EXPERIMENT_NAME",
    "SCHEMA_VERSION",
    "HistoricalAuditRunResult",
    "HistoricalCompatibilityAudit",
    "HistoricalRowDecision",
    "audit_p55_history",
    "run_history_audit",
]
