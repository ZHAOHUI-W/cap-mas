"""Lineage-safe construction and fail-closed auditing for P5.6 datasets."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType

from capmas.contracts.calibration import (
    DATASET_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION,
    CalibrationDatasetManifest,
    CalibrationLineage,
    CalibrationOutcome,
    CandidateFeatureSnapshot,
    DatasetSplit,
    HorizonLabel,
)
from capmas.evaluation.feature_snapshots import FEATURE_GROUPS_V1

_SUPERVISED_SPLITS = frozenset({"train", "calibration", "test"})
_STALE_REJECTION_CODES = frozenset({"STALE_SCENE", "STALE_EVIDENCE"})
_SAFETY_REJECTION_CODES = frozenset(
    {"GEOMETRY_GATE", "PERCEPTION_GATE", "MISSING_EVIDENCE"}
)
_DECODER_SCHEMA_REJECTION_CODES = frozenset(
    {
        "JSON_INVALID",
        "JSON_NOT_OBJECT",
        "STRUCTURED_PAYLOAD_INVALID",
        "REQUEST_ID_MISMATCH",
        "MISSION_ID_MISMATCH",
        "MISSING_PARENT_SCENE",
        "EMPTY_RESPONSE",
        "SUBGRAPH_ID_MISMATCH",
        "SUBGOAL_ID_MISMATCH",
        "GRAPH_SCHEMA_INVALID",
        "TOPOLOGY_SCHEMA_INVALID",
        "SUBGRAPH_SCHEMA_INVALID",
        "SUBGRAPH_CONDITION_ENRICHMENT_FAILED",
    }
)
_GRAPH_VALIDATION_REJECTION_CODES = frozenset(
    {
        "ACTION_WITHOUT_POSTCONDITION",
        "ACTION_WITHOUT_SKILL",
        "AMBIGUOUS_MISSION_TRANSITION",
        "DANGLING_BINDING",
        "DANGLING_EDGE",
        "DANGLING_MISSION_BINDING",
        "DANGLING_MISSION_EDGE",
        "DANGLING_OUTPUT_BINDING",
        "DUPLICATE_LOOP_ENTRY",
        "DUPLICATE_NODE",
        "DUPLICATE_PORT",
        "DUPLICATE_RESOURCE",
        "DUPLICATE_SUBGRAPH",
        "EMPTY_MISSION",
        "EMPTY_SUBGRAPH",
        "INVALID_LOOP_BUDGET",
        "INVALID_MISSION_BINDING_ORDER",
        "MISSING_FAILURE_NODE",
        "MISSING_LOOP_EXIT",
        "MISSING_MISSION_TERMINAL",
        "MISSING_SUCCESS_NODE",
        "MISSING_VALID_CHECKPOINT",
        "MISSION_BINDING_SOURCE_NOT_PREDECESSOR",
        "MULTIPLE_MISSION_INPUT_BINDINGS",
        "PARALLEL_RESOURCE_CONFLICT",
        "PORT_TYPE_MISMATCH",
        "UNBOUNDED_CYCLE",
        "UNBOUND_INPUT",
        "UNBOUND_MISSION_INPUT",
        "UNBOUND_OUTPUT",
        "UNESTABLISHED_PRECONDITION",
        "UNKNOWN_ENTRY",
        "UNKNOWN_ENTRY_NODE",
        "UNKNOWN_LOOP_ENTRY",
        "UNKNOWN_MISSION_PORT",
        "UNKNOWN_MISSION_TERMINAL",
        "UNKNOWN_OUTPUT_PORT",
        "UNKNOWN_PORT",
        "UNKNOWN_TERMINAL_NODE",
        "UNREACHABLE_BINDING_SOURCE",
        "UNREACHABLE_NODE",
        "UNREACHABLE_SUBGRAPH",
    }
)
_EVALUATOR_FEATURE_MARKERS = (
    "evaluator",
    "task_success",
    "graph_completed",
    "verifier_success",
    "failure_class",
    "postcondition",
)
_FORBIDDEN_V1_FEATURE_MARKERS = ("dynamic_verifier", "ood_aggregate", "ood_success")


@dataclass(frozen=True)
class LeakageFinding:
    code: str
    episode_ids: tuple[str, ...]
    detail: str


@dataclass(frozen=True)
class DatasetAudit:
    passed: bool
    findings: tuple[LeakageFinding, ...]
    tier_counts: Mapping[str, int]
    split_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        if self.passed != (not self.findings):
            raise ValueError("passed must equal the absence of findings")
        object.__setattr__(self, "tier_counts", MappingProxyType(dict(self.tier_counts)))
        object.__setattr__(self, "split_counts", MappingProxyType(dict(self.split_counts)))


def normalize_physical_outcomes(
    snapshots: Sequence[CandidateFeatureSnapshot],
    *,
    selected_candidate_id: str | None,
    execution_started: bool,
    task_success: bool | None,
    graph_completed: bool | None,
    verifier_success: bool | None,
    failure_class: str | None,
    horizon: HorizonLabel,
    rehearsal_labels: Mapping[str, bool | None] | None = None,
    rejection_codes: Mapping[str, str] | None = None,
) -> tuple[CalibrationOutcome, ...]:
    """Normalize candidate rows without inferring labels for unexecuted candidates."""

    rehearsal_labels = rehearsal_labels or {}
    rejection_codes = rejection_codes or {}
    outcomes: list[CalibrationOutcome] = []
    for snapshot in snapshots:
        is_selected = snapshot.candidate_id == selected_candidate_id
        if is_selected:
            execution_status = "selected_executed" if execution_started else "selected_not_started"
        else:
            execution_status = _rejection_status(rejection_codes.get(snapshot.candidate_id))

        has_physical_label = (
            is_selected
            and execution_started
            and isinstance(task_success, bool)
            and snapshot.collection_lane == "physical"
        )
        rehearsal_success = rehearsal_labels.get(snapshot.candidate_id)
        has_rehearsal_label = (
            snapshot.collection_lane == "rehearsal" and isinstance(rehearsal_success, bool)
        )
        if has_physical_label:
            tier = "A"
            normalized_task_success = task_success
            normalized_graph_completed = graph_completed
            normalized_verifier_success = verifier_success
            normalized_failure_class = failure_class
            normalized_rehearsal_success = None
        elif has_rehearsal_label:
            tier = "B"
            normalized_task_success = None
            normalized_graph_completed = None
            normalized_verifier_success = None
            normalized_failure_class = None
            normalized_rehearsal_success = rehearsal_success
        else:
            tier = "C"
            normalized_task_success = None
            normalized_rehearsal_success = None
            normalized_graph_completed = None
            normalized_verifier_success = None
            normalized_failure_class = None

        outcomes.append(
            CalibrationOutcome(
                episode_id=snapshot.episode_id,
                family_id=snapshot.family_id,
                candidate_id=snapshot.candidate_id,
                candidate_fingerprint=snapshot.candidate_fingerprint,
                tier=tier,
                execution_status=execution_status,
                task_success=normalized_task_success,
                graph_completed=normalized_graph_completed,
                verifier_success=normalized_verifier_success,
                rehearsal_success=normalized_rehearsal_success,
                failure_class=normalized_failure_class,
                horizon=horizon,
                feature_snapshot=snapshot,
                dataset_split="unassigned",
            )
        )
    return tuple(outcomes)


def _rejection_status(code: str | None) -> str:
    if code is None:
        return "not_selected"
    if code in _STALE_REJECTION_CODES:
        return "stale"
    if code in _SAFETY_REJECTION_CODES:
        return "rejected_safety"
    if code in _DECODER_SCHEMA_REJECTION_CODES or code in _GRAPH_VALIDATION_REJECTION_CODES:
        return "rejected_schema"
    return "not_selected"


def assign_lineage_splits(
    lineages: Sequence[CalibrationLineage],
    *,
    salt: str,
    train_fraction: float = 0.6,
    calibration_fraction: float = 0.2,
) -> dict[str, DatasetSplit]:
    """Assign complete lineage groups with a deterministic salted hash."""

    if not salt:
        raise ValueError("split salt must not be empty")
    if not 0.0 < train_fraction < 1.0 or not 0.0 < calibration_fraction < 1.0:
        raise ValueError("split fractions must be inside (0, 1)")
    if train_fraction + calibration_fraction >= 1.0:
        raise ValueError("split fractions must sum to less than 1.0")

    group_splits: dict[str, DatasetSplit] = {}
    assignments: dict[str, DatasetSplit] = {}
    for lineage in lineages:
        split = group_splits.get(lineage.lineage_group_id)
        if split is None:
            digest = hashlib.sha256(f"{salt}:{lineage.lineage_group_id}".encode()).digest()
            value = int.from_bytes(digest, "big") / (1 << 256)
            if value < train_fraction:
                split = "train"
            elif value < train_fraction + calibration_fraction:
                split = "calibration"
            else:
                split = "test"
            group_splits[lineage.lineage_group_id] = split
        previous = assignments.get(lineage.episode_id)
        if previous is not None and previous != split:
            raise ValueError(f"conflicting split assignment for episode {lineage.episode_id}")
        assignments[lineage.episode_id] = split
    return assignments


def build_calibration_dataset(
    outcomes: Sequence[CalibrationOutcome],
    lineages: Sequence[CalibrationLineage],
    *,
    split_assignments: Mapping[str, DatasetSplit],
    memory_skill_version: str,
    robot_skill_version: str,
    prompt_version: str,
    environment_version: str,
    code_revision: str,
    split_salt: str,
) -> CalibrationDatasetManifest:
    """Build an immutable manifest and bind its ID to the canonical digest."""

    expected_assignments = assign_lineage_splits(lineages, salt=split_salt)
    if dict(split_assignments) != expected_assignments:
        raise ValueError("split assignments must match the default lineage split")

    normalized_outcomes: list[CalibrationOutcome] = []
    for outcome in outcomes:
        if outcome.tier == "A":
            split = split_assignments.get(outcome.episode_id)
            if split not in _SUPERVISED_SPLITS:
                raise ValueError(f"Tier A episode {outcome.episode_id} lacks a supervised split")
        else:
            split = "shadow"
        normalized_outcomes.append(replace(outcome, dataset_split=split))

    manifest = CalibrationDatasetManifest(
        dataset_id="",
        dataset_schema_version=DATASET_SCHEMA_VERSION,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        outcomes=tuple(normalized_outcomes),
        lineages=tuple(lineages),
        memory_skill_version=memory_skill_version,
        robot_skill_version=robot_skill_version,
        prompt_version=prompt_version,
        environment_version=environment_version,
        code_revision=code_revision,
        split_salt=split_salt,
    )
    digest = _manifest_digest(manifest)
    return replace(manifest, dataset_id=f"sha256:{digest}", manifest_sha256=digest)


def _manifest_digest(manifest: CalibrationDatasetManifest) -> str:
    payload = manifest.to_dict()
    payload.pop("manifest_sha256", None)
    # The content address is derived before dataset_id is populated, avoiding
    # a self-referential fixed point while retaining dataset_id in the schema.
    payload["dataset_id"] = ""
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def audit_calibration_dataset(manifest: CalibrationDatasetManifest) -> DatasetAudit:
    """Collect all leakage, provenance, tier, and v1 feature-schema violations."""

    findings: list[LeakageFinding] = []
    lineages_by_episode: dict[str, list[CalibrationLineage]] = defaultdict(list)
    outcomes_by_episode: dict[str, list[CalibrationOutcome]] = defaultdict(list)
    for lineage in manifest.lineages:
        lineages_by_episode[lineage.episode_id].append(lineage)
    for outcome in manifest.outcomes:
        outcomes_by_episode[outcome.episode_id].append(outcome)

    _audit_manifest_identity(manifest, findings)
    _audit_lineages(lineages_by_episode, findings)
    _audit_group_splits(manifest, lineages_by_episode, outcomes_by_episode, findings)
    _audit_default_split_assignments(manifest, lineages_by_episode, findings)
    _audit_supervised_keys(manifest.outcomes, findings)
    _audit_outcomes(manifest, lineages_by_episode, findings)

    tier_counts = Counter(outcome.tier for outcome in manifest.outcomes)
    split_counts = Counter(outcome.dataset_split for outcome in manifest.outcomes)
    return DatasetAudit(
        passed=not findings,
        findings=tuple(findings),
        tier_counts=dict(sorted(tier_counts.items())),
        split_counts=dict(sorted(split_counts.items())),
    )


def _audit_manifest_identity(
    manifest: CalibrationDatasetManifest, findings: list[LeakageFinding]
) -> None:
    missing = [
        name
        for name in (
            "dataset_schema_version",
            "feature_schema_version",
            "memory_skill_version",
            "robot_skill_version",
            "prompt_version",
            "environment_version",
            "code_revision",
            "split_salt",
        )
        if not getattr(manifest, name, "")
    ]
    if missing:
        _add_finding(findings, "MISSING_PROVENANCE", (), f"missing manifest fields: {missing}")
    if manifest.dataset_schema_version != DATASET_SCHEMA_VERSION:
        _add_finding(findings, "DATASET_SCHEMA_MISMATCH", (), "unsupported dataset schema")
    if manifest.feature_schema_version != FEATURE_SCHEMA_VERSION:
        _add_finding(findings, "FEATURE_SCHEMA_MISMATCH", (), "unsupported feature schema")
    expected_digest = _manifest_digest(manifest)
    if (
        manifest.manifest_sha256 != expected_digest
        or manifest.dataset_id != f"sha256:{expected_digest}"
    ):
        _add_finding(findings, "MANIFEST_DIGEST_MISMATCH", (), "dataset ID or digest is invalid")


def _audit_lineages(
    lineages_by_episode: Mapping[str, list[CalibrationLineage]],
    findings: list[LeakageFinding],
) -> None:
    for episode_id, lineages in sorted(lineages_by_episode.items()):
        identities = {tuple(lineage.to_dict().items()) for lineage in lineages}
        if len(identities) > 1:
            _add_finding(
                findings,
                "CONFLICTING_EPISODE_LINEAGE",
                (episode_id,),
                "duplicate episode ID has conflicting lineage",
            )
        for lineage in lineages:
            missing = []
            if not lineage.lineage_group_id:
                missing.append("lineage_group_id")
            if not lineage.candidate_artifact_sha256:
                missing.append("candidate_artifact_sha256")
            if missing:
                _add_finding(
                    findings,
                    "MISSING_PROVENANCE",
                    (episode_id,),
                    f"missing lineage fields: {missing}",
                )


def _audit_group_splits(
    manifest: CalibrationDatasetManifest,
    lineages_by_episode: Mapping[str, list[CalibrationLineage]],
    outcomes_by_episode: Mapping[str, list[CalibrationOutcome]],
    findings: list[LeakageFinding],
) -> None:
    groups: dict[str, set[str]] = defaultdict(set)
    group_episodes: dict[str, set[str]] = defaultdict(set)
    for episode_id, lineages in lineages_by_episode.items():
        supervised_splits = {
            outcome.dataset_split
            for outcome in outcomes_by_episode.get(episode_id, ())
            if outcome.dataset_split in _SUPERVISED_SPLITS
        }
        for lineage in lineages:
            groups[lineage.lineage_group_id].update(supervised_splits)
            group_episodes[lineage.lineage_group_id].add(episode_id)
    for group_id, splits in sorted(groups.items()):
        if len(splits) > 1:
            _add_finding(
                findings,
                "LINEAGE_GROUP_SPLIT",
                tuple(sorted(group_episodes[group_id])),
                f"lineage group {group_id} crosses supervised splits {sorted(splits)}",
            )


def _audit_supervised_keys(
    outcomes: Sequence[CalibrationOutcome], findings: list[LeakageFinding]
) -> None:
    observed: dict[tuple[str, str], set[str]] = defaultdict(set)
    for outcome in outcomes:
        if outcome.dataset_split in _SUPERVISED_SPLITS:
            observed[(outcome.episode_id, outcome.candidate_fingerprint)].add(
                outcome.dataset_split
            )
    for (episode_id, _fingerprint), splits in sorted(observed.items()):
        if len(splits) > 1:
            _add_finding(
                findings,
                "SUPERVISED_SPLIT_LEAKAGE",
                (episode_id,),
                f"episode candidate occurs in supervised splits {sorted(splits)}",
            )


def _audit_default_split_assignments(
    manifest: CalibrationDatasetManifest,
    lineages_by_episode: Mapping[str, list[CalibrationLineage]],
    findings: list[LeakageFinding],
) -> None:
    expected_assignments = assign_lineage_splits(manifest.lineages, salt=manifest.split_salt)
    for outcome in manifest.outcomes:
        if outcome.tier != "A":
            continue
        expected_split = expected_assignments.get(outcome.episode_id)
        if expected_split is None or outcome.dataset_split != expected_split:
            _add_finding(
                findings,
                "LINEAGE_SPLIT_ASSIGNMENT_MISMATCH",
                (outcome.episode_id,),
                "Tier A split does not match the default lineage split",
            )


def _audit_outcomes(
    manifest: CalibrationDatasetManifest,
    lineages_by_episode: Mapping[str, list[CalibrationLineage]],
    findings: list[LeakageFinding],
) -> None:
    for outcome in manifest.outcomes:
        episode_ids = (outcome.episode_id,)
        snapshot = outcome.feature_snapshot
        lineages = lineages_by_episode.get(outcome.episode_id, ())
        if outcome.tier not in {"A", "B", "C"}:
            _add_finding(
                findings,
                "INVALID_TIER",
                episode_ids,
                f"tier must be one of A, B, or C; got {outcome.tier!r}",
            )
        if not lineages:
            _add_finding(findings, "MISSING_PROVENANCE", episode_ids, "missing episode lineage")
        elif any(snapshot.captured_at_ns > lineage.decision_boundary_ns for lineage in lineages):
            _add_finding(
                findings,
                "FUTURE_STATE_FEATURE",
                episode_ids,
                "feature capture occurred after the decision boundary",
            )
        if outcome.tier == "A" and any(
            lineage.evaluator_observed_at_ns is None for lineage in lineages
        ):
            _add_finding(
                findings,
                "MISSING_PROVENANCE",
                episode_ids,
                "Tier A requires evaluator observation provenance",
            )

        if not _outcome_snapshot_identity_matches(outcome, snapshot):
            _add_finding(
                findings,
                "OUTCOME_SNAPSHOT_IDENTITY",
                episode_ids,
                "outcome and feature snapshot identities do not match",
            )
        schema_matches = (
            snapshot.feature_schema_version == manifest.feature_schema_version
            and set(snapshot.features) == set(snapshot.feature_status)
            and set(snapshot.features) == set(snapshot.correlation_groups)
        )
        if manifest.feature_schema_version == FEATURE_SCHEMA_VERSION:
            schema_matches = (
                schema_matches
                and set(snapshot.features) == set(FEATURE_GROUPS_V1)
                and set(snapshot.feature_status) == set(FEATURE_GROUPS_V1)
                and dict(snapshot.correlation_groups) == FEATURE_GROUPS_V1
            )
        if not schema_matches:
            _add_finding(
                findings,
                "FEATURE_SCHEMA_MISMATCH",
                episode_ids,
                "snapshot schema version or feature keys do not match",
            )
        for name, status in snapshot.feature_status.items():
            if status == "unknown" and snapshot.features.get(name) is not None:
                _add_finding(
                    findings,
                    "UNKNOWN_FEATURE_VALUE",
                    episode_ids,
                    f"unknown feature {name} must remain None",
                )
        for name in snapshot.features:
            normalized = name.lower().replace("-", "_")
            if any(marker in normalized for marker in _EVALUATOR_FEATURE_MARKERS):
                _add_finding(
                    findings,
                    "EVALUATOR_DERIVED_FEATURE",
                    episode_ids,
                    f"feature {name} is evaluator-derived",
                )
            if manifest.feature_schema_version == FEATURE_SCHEMA_VERSION and any(
                marker in normalized for marker in _FORBIDDEN_V1_FEATURE_MARKERS
            ):
                _add_finding(
                    findings,
                    "FORBIDDEN_FEATURE_V1",
                    episode_ids,
                    f"feature {name} is forbidden in feature schema v1",
                )

        missing_snapshot_provenance = [
            name
            for name in ("feature_schema_version", "memory_skill_version", "robot_skill_version")
            if not getattr(snapshot, name, "")
        ]
        if not snapshot.evidence_refs or not snapshot.evidence_providers:
            missing_snapshot_provenance.append("evidence provenance")
        if missing_snapshot_provenance:
            _add_finding(
                findings,
                "MISSING_PROVENANCE",
                episode_ids,
                f"missing snapshot fields: {missing_snapshot_provenance}",
            )
        if (
            snapshot.memory_skill_version != manifest.memory_skill_version
            or snapshot.robot_skill_version != manifest.robot_skill_version
        ):
            _add_finding(
                findings,
                "VERSION_PROVENANCE_MISMATCH",
                episode_ids,
                "snapshot skill versions do not match the manifest",
            )

        if outcome.tier == "A" and (
            outcome.execution_status != "selected_executed"
            or outcome.task_success is None
            or snapshot.collection_lane != "physical"
        ):
            _add_finding(
                findings,
                "INVALID_TIER_A",
                episode_ids,
                "Tier A requires selected physical execution and conclusive task_success",
            )
        if outcome.tier in {"B", "C"} and any(
            value is not None
            for value in (
                outcome.task_success,
                outcome.graph_completed,
                outcome.verifier_success,
                outcome.failure_class,
            )
        ):
            _add_finding(
                findings,
                "PHYSICAL_LABEL_ON_UNSUPERVISED_TIER",
                episode_ids,
                f"Tier {outcome.tier} must not contain a physical label",
            )
        if outcome.tier != "A" and outcome.dataset_split in _SUPERVISED_SPLITS:
            _add_finding(
                findings,
                "UNSUPERVISED_TIER_IN_SUPERVISED_SPLIT",
                episode_ids,
                f"Tier {outcome.tier} cannot be used as a supervised label",
            )


def _outcome_snapshot_identity_matches(
    outcome: CalibrationOutcome, snapshot: CandidateFeatureSnapshot
) -> bool:
    return (
        outcome.episode_id == snapshot.episode_id
        and outcome.family_id == snapshot.family_id
        and outcome.candidate_id == snapshot.candidate_id
        and outcome.candidate_fingerprint == snapshot.candidate_fingerprint
    )


def _add_finding(
    findings: list[LeakageFinding], code: str, episode_ids: tuple[str, ...], detail: str
) -> None:
    finding = LeakageFinding(code, episode_ids, detail)
    if finding not in findings:
        findings.append(finding)


def assert_dataset_eligible(audit: DatasetAudit) -> None:
    if audit.passed != (not audit.findings):
        raise ValueError("calibration dataset audit has an inconsistent passed flag")
    if audit.passed:
        return
    details = "; ".join(f"{finding.code}: {finding.detail}" for finding in audit.findings)
    raise ValueError(f"calibration dataset is ineligible: {details}")


__all__ = [
    "DatasetAudit",
    "LeakageFinding",
    "assert_dataset_eligible",
    "assign_lineage_splits",
    "audit_calibration_dataset",
    "build_calibration_dataset",
    "normalize_physical_outcomes",
]
