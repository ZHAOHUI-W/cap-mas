"""Read-only P5.6 task-family capability diagnosis from retained P5.5 artifacts."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

from capmas.evaluation.phase5_artifacts import Phase5RunDirectory

Split = Literal["id", "ood"]

SCHEMA_VERSION = "p5.6.0.capability_diagnosis.v1"
HANDOFF_PACKAGE = "P5.3.2 Task-Family Capability Repair"
HANDOFF_ACCEPTANCE_TEST = (
    "rerun the same frozen ten-seed capability manifest with zero infrastructure unknowns, "
    "at least 80% physical execution reach, typed provenance for every failure, and at least "
    "one evaluator success"
)
REQUIRED_SEED_COUNT = 10
MIN_EXECUTION_REACH_RATE = 0.80
GATE_INFRASTRUCTURE_UNKNOWN = "INFRASTRUCTURE_UNKNOWN"
GATE_UNTYPED_FAILURE = "UNTYPED_FAILURE"
GATE_EXECUTION_REACH_BELOW_0_80 = "EXECUTION_REACH_BELOW_0_80"
GATE_NO_EVALUATOR_SUCCESS = "NO_EVALUATOR_SUCCESS"

_INFRASTRUCTURE_FAILURE_CLASSES = frozenset(
    {
        "infrastructure_unknown",
        "reset_failure",
        "worker_crash",
        "timeout",
    }
)
_PRECONDITION_FAILURE_CLASSES = frozenset({"PRECONDITION_FAILED", "precondition_failure"})
_POSTCONDITION_FAILURE_CLASSES = frozenset({"POSTCONDITION_FAILED", "postcondition_failure"})
_ORDINARY_TASK_FAILURE_CLASSES = frozenset({"task_failure", "graph_failure"})


@dataclass(frozen=True)
class CapabilityCase:
    case_id: str
    family_id: str
    seed: int
    split: Split
    reached_physical_execution: bool
    evaluator_success: bool | None
    infrastructure_unknown: bool
    failure_class: str | None
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class CapabilityDiagnosticReport:
    schema_version: str
    family_id: str
    source_manifest_sha256: str
    case_count: int
    physical_execution_count: int
    execution_reach_rate: float
    evaluator_success_count: int
    infrastructure_unknown_count: int
    typed_failure_count: int
    failure_histogram: Mapping[str, int]
    representative_evidence_refs: tuple[str, ...]
    eligible: bool
    gate_failures: tuple[str, ...]


@dataclass(frozen=True)
class TaskFamilyRepairHandoff:
    package: Literal["P5.3.2 Task-Family Capability Repair"]
    family_id: str
    source_manifest_sha256: str
    suspected_owner: str
    failure_histogram: Mapping[str, int]
    representative_evidence_refs: tuple[str, ...]
    acceptance_test: str


@dataclass(frozen=True)
class CapabilityRunResult:
    run_dir: Path
    reports: tuple[CapabilityDiagnosticReport, ...]
    handoffs: tuple[TaskFamilyRepairHandoff, ...]


def load_p55_capability_cases(
    suite_dir: str | Path,
    family_id: str,
    *,
    split: Split = "id",
) -> tuple[CapabilityCase, ...]:
    """Load one family from retained P5.5 case artifacts without mutating the suite."""

    if split not in {"id", "ood"}:
        raise ValueError("split must be 'id' or 'ood'")
    suite_path = Path(suite_dir)
    cases_root = suite_path / "cases"
    if not cases_root.is_dir():
        raise ValueError(f"P5.5 suite cases directory does not exist: {cases_root}")

    cases: list[CapabilityCase] = []
    for case_dir in sorted(path for path in cases_root.iterdir() if path.is_dir()):
        case_path = case_dir / "case.json"
        if not case_path.exists():
            continue
        case_payload = _read_mapping_json(case_path)
        if case_payload.get("task_family") != family_id or case_payload.get("split") != split:
            continue
        summary_path = case_dir / "summary.json"
        evidence_path = case_dir / "evidence" / "ood_replay.json"
        summary_payload = _read_mapping_json(summary_path)
        evidence_payload = _read_mapping_json(evidence_path)
        cases.append(
            _case_from_payloads(
                suite_path=suite_path,
                case_dir=case_dir,
                case=case_payload,
                summary=summary_payload,
                evidence=evidence_payload,
            )
        )
    return tuple(sorted(cases, key=lambda item: item.seed))


def diagnose_family_capability(
    cases: Sequence[CapabilityCase],
    *,
    family_id: str,
    source_manifest_sha256: str,
) -> tuple[CapabilityDiagnosticReport, TaskFamilyRepairHandoff | None]:
    """Apply the formal ten-seed P5.6 capability gate to one task family."""

    if not _is_sha256(source_manifest_sha256):
        raise ValueError("source_manifest_sha256 must be a 64-character lowercase hex digest")
    if any(case.family_id != family_id for case in cases):
        raise ValueError(f"all capability cases must belong to family {family_id}")
    seeds = [case.seed for case in cases]
    unique_seeds = set(seeds)
    if len(cases) != REQUIRED_SEED_COUNT or len(unique_seeds) != REQUIRED_SEED_COUNT:
        raise ValueError("formal capability diagnosis requires exactly ten unique seeds")

    physical_execution_count = sum(case.reached_physical_execution for case in cases)
    execution_reach_rate = physical_execution_count / REQUIRED_SEED_COUNT
    evaluator_success_count = sum(case.evaluator_success is True for case in cases)
    infrastructure_unknown_count = sum(case.infrastructure_unknown for case in cases)
    typed_failure_count = sum(
        case.evaluator_success is False and case.failure_class is not None for case in cases
    )
    histogram = Counter(
        case.failure_class for case in cases if case.failure_class is not None
    )
    representative_refs = _representative_evidence_refs(cases)

    gate_failures: list[str] = []
    if infrastructure_unknown_count:
        gate_failures.append(GATE_INFRASTRUCTURE_UNKNOWN)
    if any(
        case.evaluator_success is False and case.failure_class is None for case in cases
    ):
        gate_failures.append(GATE_UNTYPED_FAILURE)
    if execution_reach_rate < MIN_EXECUTION_REACH_RATE:
        gate_failures.append(GATE_EXECUTION_REACH_BELOW_0_80)
    if evaluator_success_count == 0:
        gate_failures.append(GATE_NO_EVALUATOR_SUCCESS)

    report = CapabilityDiagnosticReport(
        schema_version=SCHEMA_VERSION,
        family_id=family_id,
        source_manifest_sha256=source_manifest_sha256,
        case_count=len(cases),
        physical_execution_count=physical_execution_count,
        execution_reach_rate=execution_reach_rate,
        evaluator_success_count=evaluator_success_count,
        infrastructure_unknown_count=infrastructure_unknown_count,
        typed_failure_count=typed_failure_count,
        failure_histogram=dict(sorted(histogram.items())),
        representative_evidence_refs=representative_refs,
        eligible=not gate_failures,
        gate_failures=tuple(gate_failures),
    )
    handoff = None
    if not report.eligible:
        handoff = TaskFamilyRepairHandoff(
            package=HANDOFF_PACKAGE,
            family_id=family_id,
            source_manifest_sha256=source_manifest_sha256,
            suspected_owner=_suspected_owner(report),
            failure_histogram=report.failure_histogram,
            representative_evidence_refs=representative_refs,
            acceptance_test=HANDOFF_ACCEPTANCE_TEST,
        )
    return report, handoff


def run_capability_diagnosis(
    *,
    suite_dir: str | Path,
    families: Sequence[str],
    output_root: str | Path,
    split: Split = "id",
) -> CapabilityRunResult:
    """Run read-only family diagnosis and write a fresh Phase 5 artifact directory."""

    if not families:
        raise ValueError("at least one family is required")
    family_tuple = tuple(families)
    duplicate_families = sorted(
        family_id for family_id, count in Counter(family_tuple).items() if count > 1
    )
    if duplicate_families:
        raise ValueError(f"duplicate family arguments are not allowed: {duplicate_families}")
    suite_path = Path(suite_dir)
    output_path = Path(output_root)
    _reject_output_inside_suite(suite_path, output_path)
    source_manifest_sha256 = _load_source_manifest_sha256(suite_path)
    run_dir = Phase5RunDirectory.create(
        output_path,
        "P5.6.0_capability_diagnosis",
        f"capability_{uuid4().hex[:8]}",
    )
    run_dir.write_json(
        "run_config.json",
        _run_config_payload(
            suite_dir=suite_path,
            families=family_tuple,
            split=split,
            source_manifest_sha256=source_manifest_sha256,
            status="running",
        ),
    )

    reports: list[CapabilityDiagnosticReport] = []
    handoffs: list[TaskFamilyRepairHandoff] = []
    log_lines = [
        "experiment=P5.6.0_capability_diagnosis",
        f"suite_dir={Path(suite_dir)}",
        f"split={split}",
        f"source_manifest_sha256={source_manifest_sha256}",
    ]
    try:
        for family_id in family_tuple:
            cases = load_p55_capability_cases(suite_dir, family_id, split=split)
            report, handoff = diagnose_family_capability(
                cases,
                family_id=family_id,
                source_manifest_sha256=source_manifest_sha256,
            )
            reports.append(report)
            log_lines.append(
                "family="
                f"{family_id} case_count={report.case_count} "
                f"physical_execution_count={report.physical_execution_count} "
                f"evaluator_success_count={report.evaluator_success_count} "
                f"eligible={report.eligible} gate_failures={list(report.gate_failures)}"
            )
            if handoff is not None:
                handoffs.append(handoff)
                run_dir.write_json(f"artifacts/p53_2_{family_id}.json", asdict(handoff))

        run_dir.write_json(
            "results/capability.json",
            {
                "schema_version": SCHEMA_VERSION,
                "experiment": "P5.6.0_capability_diagnosis",
                "source_manifest_sha256": source_manifest_sha256,
                "split": split,
                "reports": [asdict(report) for report in reports],
                "handoffs": [asdict(handoff) for handoff in handoffs],
            },
        )
        run_dir.write_text(
            "summary.md",
            _summary_markdown(
                reports=tuple(reports),
                handoffs=tuple(handoffs),
                source_manifest_sha256=source_manifest_sha256,
                split=split,
            ),
        )
        run_dir.write_json(
            "run_config.json",
            _run_config_payload(
                suite_dir=suite_path,
                families=family_tuple,
                split=split,
                source_manifest_sha256=source_manifest_sha256,
                status="completed",
            ),
        )
        log_lines.extend(["status=completed", ""])
        run_dir.write_text("logs/runner.log", "\n".join(log_lines))
        run_dir.finalize_manifest()
    except BaseException as error:
        run_dir.write_json(
            "run_config.json",
            _run_config_payload(
                suite_dir=suite_path,
                families=family_tuple,
                split=split,
                source_manifest_sha256=source_manifest_sha256,
                status="failed",
                error_type=type(error).__name__,
                error=str(error),
            ),
        )
        log_lines.extend(["status=failed", ""])
        run_dir.write_text("logs/runner.log", "\n".join(log_lines))
        run_dir.finalize_manifest()
        raise

    return CapabilityRunResult(
        run_dir=run_dir.path,
        reports=tuple(reports),
        handoffs=tuple(handoffs),
    )


def _case_from_payloads(
    *,
    suite_path: Path,
    case_dir: Path,
    case: Mapping[str, object],
    summary: Mapping[str, object],
    evidence: Mapping[str, object],
) -> CapabilityCase:
    case_id = _required_str(case, "case_id")
    family_id = _required_str(case, "task_family")
    seed = _required_int(case, "seed")
    split = _required_split(case, "split")
    _require_equal(case_id, _required_str(summary, "case_id"), "summary case_id")
    _require_equal(case_id, _required_str(evidence, "case_id"), "evidence case_id")
    _require_equal(split, _required_split(evidence, "split"), "evidence split")
    status = _required_str(summary, "status")
    if status != "completed":
        raise ValueError(f"case {case_id} is not a completed retained case")

    summary_success = _optional_bool(summary, "evaluator_success", allow_missing=False)
    evidence_success = _optional_bool(evidence, "evaluator_success", allow_missing=False)
    if (
        summary_success is not None
        and evidence_success is not None
        and summary_success != evidence_success
    ):
        raise ValueError(f"case {case_id} evaluator_success mismatch between summary and evidence")

    failure_class = _optional_failure_class(evidence.get("failure_class"))
    infrastructure_unknown = (
        evidence_success is None or failure_class in _INFRASTRUCTURE_FAILURE_CLASSES
    )
    if failure_class in _INFRASTRUCTURE_FAILURE_CLASSES:
        evidence_success = None

    primary_winner = _optional_non_empty_str(summary.get("primary_winner"))
    candidate_id = _optional_non_empty_str(evidence.get("candidate_id"))
    reached_physical_execution = _reached_physical_execution(
        case_id=case_id,
        primary_winner=primary_winner,
        candidate_id=candidate_id,
    )

    return CapabilityCase(
        case_id=case_id,
        family_id=family_id,
        seed=seed,
        split=split,
        reached_physical_execution=reached_physical_execution,
        evaluator_success=evidence_success,
        infrastructure_unknown=infrastructure_unknown,
        failure_class=failure_class,
        evidence_refs=(
            (case_dir / "case.json").relative_to(suite_path).as_posix(),
            (case_dir / "summary.json").relative_to(suite_path).as_posix(),
            (case_dir / "evidence" / "ood_replay.json").relative_to(suite_path).as_posix(),
        ),
    )


def _summary_markdown(
    *,
    reports: tuple[CapabilityDiagnosticReport, ...],
    handoffs: tuple[TaskFamilyRepairHandoff, ...],
    source_manifest_sha256: str,
    split: Split,
) -> str:
    lines = [
        "# CAP-MAS P5.6 capability diagnosis",
        "",
        f"- source_manifest_sha256: {source_manifest_sha256}",
        f"- split: {split}",
        f"- family_count: {len(reports)}",
        f"- handoff_count: {len(handoffs)}",
        "",
        "| family | cases | physical execution | evaluator success | eligible | gate failures |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for report in reports:
        failures = ", ".join(report.gate_failures) if report.gate_failures else "-"
        lines.append(
            f"| {report.family_id} | {report.case_count} | "
            f"{report.physical_execution_count}/{REQUIRED_SEED_COUNT} | "
            f"{report.evaluator_success_count}/{REQUIRED_SEED_COUNT} | "
            f"{report.eligible} | {failures} |"
        )
    lines.append("")
    return "\n".join(lines)


def _representative_evidence_refs(cases: Sequence[CapabilityCase]) -> tuple[str, ...]:
    refs: list[str] = []
    seen: set[str] = set()
    prioritized = sorted(
        cases,
        key=lambda case: (
            case.evaluator_success is True,
            not case.infrastructure_unknown,
            case.seed,
        ),
    )
    for case in prioritized:
        for evidence_ref in case.evidence_refs:
            if evidence_ref not in seen:
                refs.append(evidence_ref)
                seen.add(evidence_ref)
            if len(refs) >= REQUIRED_SEED_COUNT:
                return tuple(refs)
    return tuple(refs)


def _suspected_owner(report: CapabilityDiagnosticReport) -> str:
    if report.infrastructure_unknown_count:
        return "runtime_infrastructure"
    for failure_class, _count in sorted(
        report.failure_histogram.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        if failure_class in _INFRASTRUCTURE_FAILURE_CLASSES:
            return "runtime_infrastructure"
        if failure_class in _PRECONDITION_FAILURE_CLASSES:
            return "perception_or_contract"
        if failure_class in _POSTCONDITION_FAILURE_CLASSES:
            return "verification_or_robot_skill"
        if failure_class in _ORDINARY_TASK_FAILURE_CLASSES:
            return "task_mapping_or_motion"
    return "task_mapping_or_motion"


def _run_config_payload(
    *,
    suite_dir: Path,
    families: tuple[str, ...],
    split: Split,
    source_manifest_sha256: str,
    status: str,
    error_type: str | None = None,
    error: str | None = None,
) -> Mapping[str, object]:
    payload: dict[str, object] = {
        "experiment": "P5.6.0_capability_diagnosis",
        "suite_dir": str(suite_dir),
        "families": families,
        "split": split,
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


def _reached_physical_execution(
    *,
    case_id: str,
    primary_winner: str | None,
    candidate_id: str | None,
) -> bool:
    summary_claims_selection = primary_winner is not None
    evidence_claims_selection = candidate_id is not None and candidate_id != "unselected"
    if summary_claims_selection or evidence_claims_selection:
        if primary_winner != candidate_id:
            raise ValueError(
                f"case {case_id} primary_winner must match evidence candidate_id when "
                "physical selection is claimed"
            )
        return True
    return False


def _load_source_manifest_sha256(suite_dir: Path) -> str:
    payload = _read_mapping_json(suite_dir / "suite_manifest.json")
    digest = payload.get("manifest_sha256")
    if not isinstance(digest, str) or not _is_sha256(digest):
        raise ValueError("suite_manifest.json must contain a valid manifest_sha256")
    return digest


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


def _required_str(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"required string field is missing or invalid: {key}")
    return value


def _required_split(payload: Mapping[str, object], key: str) -> Split:
    value = _required_str(payload, key)
    if value not in {"id", "ood"}:
        raise ValueError(f"split must be 'id' or 'ood': {value}")
    return value


def _required_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"required integer field is missing or invalid: {key}")
    return value


def _optional_bool(
    payload: Mapping[str, object],
    key: str,
    *,
    allow_missing: bool,
) -> bool | None:
    if key not in payload:
        if allow_missing:
            return None
        raise ValueError(f"required boolean/null field is missing: {key}")
    value = payload[key]
    if value is None or isinstance(value, bool):
        return value
    raise ValueError(f"field must be boolean or null: {key}")


def _optional_failure_class(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("failure_class must be a string or null")
    stripped = value.strip()
    return stripped or None


def _optional_non_empty_str(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _require_equal(left: object, right: object, label: str) -> None:
    if left != right:
        raise ValueError(f"case artifact mismatch for {label}: {left!r} != {right!r}")


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)
