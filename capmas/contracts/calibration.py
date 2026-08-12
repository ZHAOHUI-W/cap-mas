"""Immutable, JSON-safe contracts for P5.6 calibration data collection."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any, Literal, TypeVar, cast

FEATURE_SCHEMA_VERSION = "p56.feature.v1"
HORIZON_SCHEMA_VERSION = "p56.horizon.v1"
DATASET_SCHEMA_VERSION = "p56.dataset.v1"
CAPABILITY_SCHEMA_VERSION = "p56.capability.v1"
COLLECTION_SCHEMA_VERSION = "p56.collection.v1"

ExecutionStatus = Literal[
    "selected_executed",
    "selected_not_started",
    "not_selected",
    "rejected_safety",
    "rejected_schema",
    "stale",
    "unknown",
]
DatasetSplit = Literal["train", "calibration", "test", "shadow", "unassigned"]
FeatureStatus = Literal["present", "unknown", "invalid"]
CollectionLane = Literal["physical", "rehearsal", "shadow"]

_T = TypeVar("_T")
_HORIZON_FIELDS = {
    "planned_critical_path_actions",
    "planned_critical_path_subgoals",
    "planned_checkpoint_subgraphs",
    "attempted_actions",
    "completed_actions",
    "attempted_subgoals",
    "completed_subgoals",
    "attempted_checkpoints",
    "completed_checkpoints",
    "planned_source",
    "realized_source",
    "planned_valid",
    "realized_valid",
}
_SNAPSHOT_FIELDS = {
    "episode_id",
    "episode_epoch",
    "family_id",
    "candidate_id",
    "candidate_fingerprint",
    "scene_version",
    "map_version",
    "feature_schema_version",
    "captured_at_ns",
    "collection_lane",
    "features",
    "feature_status",
    "correlation_groups",
    "memory_skill_version",
    "robot_skill_version",
    "selection_probability",
    "evidence_refs",
    "evidence_providers",
    "rewrite_metadata",
}
_OUTCOME_FIELDS = {
    "episode_id",
    "family_id",
    "candidate_id",
    "candidate_fingerprint",
    "tier",
    "execution_status",
    "task_success",
    "graph_completed",
    "verifier_success",
    "rehearsal_success",
    "failure_class",
    "horizon",
    "feature_snapshot",
    "dataset_split",
}
_LINEAGE_FIELDS = {
    "episode_id",
    "lineage_group_id",
    "seed",
    "split_identity",
    "layout_pair_id",
    "retry_of_episode_id",
    "candidate_artifact_sha256",
    "decision_boundary_ns",
    "evaluator_observed_at_ns",
}
_DATASET_MANIFEST_FIELDS = {
    "dataset_id",
    "dataset_schema_version",
    "feature_schema_version",
    "outcomes",
    "lineages",
    "memory_skill_version",
    "robot_skill_version",
    "prompt_version",
    "environment_version",
    "code_revision",
    "split_salt",
    "manifest_sha256",
}
_PREDICTION_FIELDS = {
    "candidate_id",
    "rank_score",
    "success_probability",
    "uncertainty",
    "abstained",
    "reason",
    "model_version",
    "feature_schema_version",
    "snapshot_id",
    "eligible_family",
}
_CONTEXT_FIELDS = {
    "episode_id",
    "episode_epoch",
    "family_id",
    "feature_schema_version",
    "memory_skill_version",
    "robot_skill_version",
    "collection_lane",
}
_SECRET_MARKERS = (
    "access_key",
    "api_key",
    "authorization",
    "credential",
    "password",
    "private_key",
    "secret",
    "token",
)


def _require_fields(raw: Mapping[str, object], expected: set[str], name: str) -> None:
    if not isinstance(raw, Mapping):
        raise TypeError(f"{name} must be a mapping")
    actual = set(raw)
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={sorted(missing)}")
        if extra:
            details.append(f"unexpected={sorted(extra)}")
        raise ValueError(f"{name} fields invalid: {', '.join(details)}")


def _as_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} keys must be strings")
    return {key: item for key, item in value.items()}


def _mapping_proxy(value: Mapping[str, _T], name: str) -> Mapping[str, _T]:
    return MappingProxyType(dict(_as_mapping(value, name)))


def _string_mapping_proxy(value: Mapping[str, str], name: str) -> Mapping[str, str]:
    raw = _as_mapping(value, name)
    for key, item in raw.items():
        normalized_key = key.lower().replace("-", "_")
        if any(marker in normalized_key for marker in _SECRET_MARKERS):
            raise ValueError(f"{name}.{key} must not contain secret material")
        if not isinstance(item, str):
            raise TypeError(f"{name} values must be strings")
    return MappingProxyType(raw)


def _plain(value: object) -> object:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    if isinstance(value, Mapping):
        return {key: _plain(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _sorted_dict(instance: object) -> dict[str, object]:
    return {
        key: _plain(getattr(instance, key))
        for key in sorted(field.name for field in fields(instance))
    }


def _require_nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must not be empty")


def _require_nonnegative(value: int | None, name: str) -> None:
    if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
        raise ValueError(f"{name} must not be negative")


def _require_finite(value: float | None, name: str) -> None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite when available")


def _require_sha256(value: str, name: str) -> None:
    _require_nonempty(value, name)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 fingerprint")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a lowercase SHA-256 fingerprint") from error


def _require_bool_or_none(value: object, name: str) -> None:
    if value is not None and not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean or None")


def _require_string_or_none(value: object, name: str) -> None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{name} must be a string or None")


def _require_probability(value: float | None, name: str) -> None:
    _require_finite(value, name)
    if value is not None and not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1] when available")


@dataclass(frozen=True)
class HorizonLabel:
    planned_critical_path_actions: int | None
    planned_critical_path_subgoals: int | None
    planned_checkpoint_subgraphs: int | None
    attempted_actions: int | None
    completed_actions: int | None
    attempted_subgoals: int | None
    completed_subgoals: int | None
    attempted_checkpoints: int | None
    completed_checkpoints: int | None
    planned_source: Literal["mission_graph", "unknown"]
    realized_source: Literal["execution_trace", "unknown"]
    planned_valid: bool
    realized_valid: bool

    def __post_init__(self) -> None:
        for name in (
            "planned_critical_path_actions",
            "planned_critical_path_subgoals",
            "planned_checkpoint_subgraphs",
            "attempted_actions",
            "completed_actions",
            "attempted_subgoals",
            "completed_subgoals",
            "attempted_checkpoints",
            "completed_checkpoints",
        ):
            _require_nonnegative(getattr(self, name), name)
        for completed, attempted in (
            ("completed_actions", "attempted_actions"),
            ("completed_subgoals", "attempted_subgoals"),
            ("completed_checkpoints", "attempted_checkpoints"),
        ):
            completed_value = getattr(self, completed)
            attempted_value = getattr(self, attempted)
            if (
                completed_value is not None
                and attempted_value is not None
                and completed_value > attempted_value
            ):
                raise ValueError(f"{completed} must not exceed {attempted}")
        if self.planned_source not in {"mission_graph", "unknown"}:
            raise ValueError("planned_source is invalid")
        if self.realized_source not in {"execution_trace", "unknown"}:
            raise ValueError("realized_source is invalid")
        if not isinstance(self.planned_valid, bool) or not isinstance(self.realized_valid, bool):
            raise TypeError("horizon validity flags must be booleans")

    def to_dict(self) -> dict[str, object]:
        return _sorted_dict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> HorizonLabel:
        _require_fields(raw, _HORIZON_FIELDS, "horizon")
        return cls(**cast(dict[str, Any], dict(raw)))


def horizon_bucket(label: HorizonLabel) -> Literal["H1", "H2-3", "H4-6", "H7+", "N/A"]:
    if not label.planned_valid or label.planned_critical_path_subgoals is None:
        return "N/A"
    count = label.planned_critical_path_subgoals
    if count == 0:
        return "N/A"
    if count == 1:
        return "H1"
    if count <= 3:
        return "H2-3"
    if count <= 6:
        return "H4-6"
    return "H7+"


@dataclass(frozen=True)
class CandidateFeatureSnapshot:
    episode_id: str
    episode_epoch: int
    family_id: str
    candidate_id: str
    candidate_fingerprint: str
    scene_version: int
    map_version: int | None
    feature_schema_version: str
    captured_at_ns: int
    collection_lane: CollectionLane
    features: Mapping[str, float | None]
    feature_status: Mapping[str, FeatureStatus]
    correlation_groups: Mapping[str, str]
    memory_skill_version: str
    robot_skill_version: str
    selection_probability: float | None = None
    evidence_refs: tuple[str, ...] = ()
    evidence_providers: Mapping[str, str] = field(default_factory=dict)
    rewrite_metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "episode_id",
            "family_id",
            "candidate_id",
            "feature_schema_version",
            "memory_skill_version",
            "robot_skill_version",
        ):
            _require_nonempty(getattr(self, name), name)
        _require_sha256(self.candidate_fingerprint, "candidate_fingerprint")
        for name in ("episode_epoch", "scene_version", "captured_at_ns", "map_version"):
            _require_nonnegative(getattr(self, name), name)
        if self.collection_lane not in {"physical", "rehearsal", "shadow"}:
            raise ValueError("collection_lane is invalid")
        _require_probability(self.selection_probability, "selection_probability")
        features = _mapping_proxy(self.features, "features")
        statuses = _mapping_proxy(self.feature_status, "feature_status")
        groups = _mapping_proxy(self.correlation_groups, "correlation_groups")
        if set(features) != set(statuses) or set(features) != set(groups):
            raise ValueError("features, feature_status, and correlation_groups keys must match")
        for key, value in features.items():
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
            ):
                raise ValueError(f"feature {key} must be a float or None")
            _require_finite(value, f"feature {key}")
        for key, status in statuses.items():
            if not isinstance(status, str) or status not in {"present", "unknown", "invalid"}:
                raise ValueError(f"feature status {key} is invalid")
            if status == "unknown" and features[key] is not None:
                raise ValueError(f"unknown feature {key} must remain None")
            if status == "present" and features[key] is None:
                raise ValueError(f"present feature {key} must have a finite value")
        if any(not isinstance(group, str) or not group for group in groups.values()):
            raise ValueError("correlation_groups values must be non-empty strings")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "feature_status", statuses)
        object.__setattr__(self, "correlation_groups", groups)
        object.__setattr__(
            self,
            "evidence_providers",
            _string_mapping_proxy(self.evidence_providers, "evidence_providers"),
        )
        object.__setattr__(
            self, "rewrite_metadata", _deep_freeze(self.rewrite_metadata, "rewrite_metadata")
        )
        object.__setattr__(
            self, "evidence_refs", _as_string_sequence(self.evidence_refs, "evidence_refs")
        )

    def to_dict(self) -> dict[str, object]:
        return _sorted_dict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> CandidateFeatureSnapshot:
        _require_fields(raw, _SNAPSHOT_FIELDS, "feature snapshot")
        data = dict(raw)
        data["evidence_refs"] = tuple(_as_string_sequence(data["evidence_refs"], "evidence_refs"))
        data["features"] = _as_mapping(data["features"], "features")
        data["feature_status"] = _as_mapping(data["feature_status"], "feature_status")
        data["correlation_groups"] = _as_mapping(data["correlation_groups"], "correlation_groups")
        data["evidence_providers"] = _as_mapping(data["evidence_providers"], "evidence_providers")
        data["rewrite_metadata"] = _as_mapping(data["rewrite_metadata"], "rewrite_metadata")
        return cls(**cast(dict[str, Any], data))


def _as_string_sequence(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a sequence of strings")
    return tuple(value)


def _deep_freeze(value: Mapping[str, object], name: str) -> Mapping[str, object]:
    raw = _as_mapping(value, name)
    frozen: dict[str, object] = {}
    for key, item in raw.items():
        normalized_key = key.lower().replace("-", "_")
        if any(marker in normalized_key for marker in _SECRET_MARKERS):
            raise ValueError(f"{name}.{key} must not contain secret material")
        if isinstance(item, Mapping):
            frozen[key] = _deep_freeze(item, f"{name}.{key}")
        elif isinstance(item, (list, tuple)):
            frozen[key] = tuple(
                _deep_freeze(entry, f"{name}.{key}[]")
                if isinstance(entry, Mapping)
                else _freeze_json_value(entry, f"{name}.{key}[]")
                for entry in item
            )
        else:
            frozen[key] = _freeze_json_value(item, f"{name}.{key}")
    return MappingProxyType(frozen)


def _freeze_json_value(value: object, name: str) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must contain only finite JSON values")
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_value(item, f"{name}[]") for item in value)
    raise ValueError(f"{name} must contain only JSON-safe values")


@dataclass(frozen=True)
class CalibrationOutcome:
    episode_id: str
    family_id: str
    candidate_id: str
    candidate_fingerprint: str
    tier: Literal["A", "B", "C"]
    execution_status: ExecutionStatus
    task_success: bool | None
    graph_completed: bool | None
    verifier_success: bool | None
    rehearsal_success: bool | None
    failure_class: str | None
    horizon: HorizonLabel
    feature_snapshot: CandidateFeatureSnapshot
    dataset_split: DatasetSplit

    def __post_init__(self) -> None:
        for name in ("episode_id", "family_id", "candidate_id"):
            _require_nonempty(getattr(self, name), name)
        _require_sha256(self.candidate_fingerprint, "candidate_fingerprint")
        if self.tier not in {"A", "B", "C"}:
            raise ValueError("tier is invalid")
        if self.execution_status not in {
            "selected_executed",
            "selected_not_started",
            "not_selected",
            "rejected_safety",
            "rejected_schema",
            "stale",
            "unknown",
        }:
            raise ValueError("execution_status is invalid")
        if self.dataset_split not in {"train", "calibration", "test", "shadow", "unassigned"}:
            raise ValueError("dataset_split is invalid")
        for name in ("task_success", "graph_completed", "verifier_success", "rehearsal_success"):
            _require_bool_or_none(getattr(self, name), name)
        _require_string_or_none(self.failure_class, "failure_class")
        if not isinstance(self.horizon, HorizonLabel) or not isinstance(
            self.feature_snapshot, CandidateFeatureSnapshot
        ):
            raise TypeError("outcome horizon and feature_snapshot must use contract types")
        if self.tier == "A" and (
            self.execution_status != "selected_executed"
            or self.task_success is None
            or self.feature_snapshot.collection_lane != "physical"
        ):
            raise ValueError(
                "Tier A requires selected_executed, physical collection, and a conclusive task_success"
            )
        if self.tier == "B" and (
            self.task_success is not None
            or self.rehearsal_success is None
            or self.feature_snapshot.collection_lane != "rehearsal"
        ):
            raise ValueError(
                "Tier B requires unknown task_success, rehearsal collection, and a conclusive rehearsal_success"
            )
        if self.tier == "C" and (
            self.task_success is not None or self.rehearsal_success is not None
        ):
            raise ValueError("Tier C requires unknown task_success and rehearsal_success")
        if (
            self.feature_snapshot.episode_id != self.episode_id
            or self.feature_snapshot.family_id != self.family_id
            or self.feature_snapshot.candidate_id != self.candidate_id
            or self.feature_snapshot.candidate_fingerprint != self.candidate_fingerprint
        ):
            raise ValueError("outcome and feature_snapshot identities must match")

    def to_dict(self) -> dict[str, object]:
        return _sorted_dict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> CalibrationOutcome:
        _require_fields(raw, _OUTCOME_FIELDS, "calibration outcome")
        data = dict(raw)
        data["horizon"] = HorizonLabel.from_dict(cast(Mapping[str, object], data["horizon"]))
        data["feature_snapshot"] = CandidateFeatureSnapshot.from_dict(
            cast(Mapping[str, object], data["feature_snapshot"])
        )
        return cls(**cast(dict[str, Any], data))


@dataclass(frozen=True)
class CalibrationLineage:
    episode_id: str
    lineage_group_id: str
    seed: int
    split_identity: Literal["id", "ood", "native"]
    layout_pair_id: str | None
    retry_of_episode_id: str | None
    candidate_artifact_sha256: str
    decision_boundary_ns: int
    evaluator_observed_at_ns: int | None

    def __post_init__(self) -> None:
        for name in ("episode_id", "lineage_group_id"):
            _require_nonempty(getattr(self, name), name)
        _require_nonnegative(self.seed, "seed")
        _require_nonnegative(self.decision_boundary_ns, "decision_boundary_ns")
        _require_nonnegative(self.evaluator_observed_at_ns, "evaluator_observed_at_ns")
        _require_string_or_none(self.layout_pair_id, "layout_pair_id")
        _require_string_or_none(self.retry_of_episode_id, "retry_of_episode_id")
        _require_sha256(self.candidate_artifact_sha256, "candidate_artifact_sha256")
        if self.split_identity not in {"id", "ood", "native"}:
            raise ValueError("split_identity is invalid")

    def to_dict(self) -> dict[str, object]:
        return _sorted_dict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> CalibrationLineage:
        _require_fields(raw, _LINEAGE_FIELDS, "calibration lineage")
        return cls(**cast(dict[str, Any], dict(raw)))


@dataclass(frozen=True)
class CalibrationDatasetManifest:
    dataset_id: str
    dataset_schema_version: str
    feature_schema_version: str
    outcomes: tuple[CalibrationOutcome, ...]
    lineages: tuple[CalibrationLineage, ...]
    memory_skill_version: str
    robot_skill_version: str
    prompt_version: str
    environment_version: str
    code_revision: str
    split_salt: str
    manifest_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.outcomes, tuple) or any(
            not isinstance(outcome, CalibrationOutcome) for outcome in self.outcomes
        ):
            raise TypeError("outcomes must be a tuple of CalibrationOutcome values")
        if not isinstance(self.lineages, tuple) or any(
            not isinstance(lineage, CalibrationLineage) for lineage in self.lineages
        ):
            raise TypeError("lineages must be a tuple of CalibrationLineage values")
        for name in (
            "dataset_schema_version",
            "feature_schema_version",
            "memory_skill_version",
            "robot_skill_version",
            "prompt_version",
            "environment_version",
            "code_revision",
            "split_salt",
        ):
            _require_nonempty(getattr(self, name), name)
        if self.dataset_id:
            if not self.dataset_id.startswith("sha256:"):
                raise ValueError("dataset_id must use the sha256: prefix")
            _require_sha256(self.dataset_id.removeprefix("sha256:"), "dataset_id digest")
        if self.manifest_sha256:
            _require_sha256(self.manifest_sha256, "manifest_sha256")

    def to_dict(self) -> dict[str, object]:
        return _sorted_dict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> CalibrationDatasetManifest:
        _require_fields(raw, _DATASET_MANIFEST_FIELDS, "calibration dataset manifest")
        data = dict(raw)
        outcomes = data["outcomes"]
        lineages = data["lineages"]
        if not isinstance(outcomes, (tuple, list)) or not isinstance(lineages, (tuple, list)):
            raise TypeError("manifest outcomes and lineages must be sequences")
        data["outcomes"] = tuple(
            CalibrationOutcome.from_dict(cast(Mapping[str, object], outcome))
            for outcome in outcomes
        )
        data["lineages"] = tuple(
            CalibrationLineage.from_dict(cast(Mapping[str, object], lineage))
            for lineage in lineages
        )
        return cls(**cast(dict[str, Any], data))


@dataclass(frozen=True)
class CalibrationPrediction:
    candidate_id: str
    rank_score: float | None
    success_probability: float | None
    uncertainty: float
    abstained: bool
    reason: str
    model_version: str
    feature_schema_version: str
    snapshot_id: str | None
    eligible_family: bool

    def __post_init__(self) -> None:
        _require_nonempty(self.candidate_id, "candidate_id")
        _require_nonempty(self.reason, "reason")
        _require_nonempty(self.model_version, "model_version")
        _require_nonempty(self.feature_schema_version, "feature_schema_version")
        _require_string_or_none(self.snapshot_id, "snapshot_id")
        _require_finite(self.rank_score, "rank_score")
        _require_probability(self.success_probability, "success_probability")
        _require_probability(self.uncertainty, "uncertainty")
        if not isinstance(self.abstained, bool) or not isinstance(self.eligible_family, bool):
            raise TypeError("prediction flags must be booleans")
        if self.abstained and (self.rank_score is not None or self.success_probability is not None):
            raise ValueError("abstained prediction cannot publish score or probability")

    def to_dict(self) -> dict[str, object]:
        return _sorted_dict(self)


@dataclass(frozen=True)
class CalibrationCollectionContext:
    episode_id: str
    episode_epoch: int
    family_id: str
    feature_schema_version: str
    memory_skill_version: str
    robot_skill_version: str
    collection_lane: CollectionLane = "physical"

    def __post_init__(self) -> None:
        for name in (
            "episode_id",
            "family_id",
            "feature_schema_version",
            "memory_skill_version",
            "robot_skill_version",
        ):
            _require_nonempty(getattr(self, name), name)
        _require_nonnegative(self.episode_epoch, "episode_epoch")
        if self.collection_lane not in {"physical", "rehearsal", "shadow"}:
            raise ValueError("collection_lane is invalid")

    def to_dict(self) -> dict[str, object]:
        return _sorted_dict(self)


__all__ = [
    "CAPABILITY_SCHEMA_VERSION",
    "COLLECTION_SCHEMA_VERSION",
    "DATASET_SCHEMA_VERSION",
    "FEATURE_SCHEMA_VERSION",
    "HORIZON_SCHEMA_VERSION",
    "CalibrationCollectionContext",
    "CalibrationDatasetManifest",
    "CalibrationLineage",
    "CalibrationOutcome",
    "CalibrationPrediction",
    "CandidateFeatureSnapshot",
    "CollectionLane",
    "DatasetSplit",
    "ExecutionStatus",
    "FeatureStatus",
    "HorizonLabel",
    "horizon_bucket",
]
