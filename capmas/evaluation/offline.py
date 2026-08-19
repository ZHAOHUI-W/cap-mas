"""Offline-only P5.6B partitioning and calibration orchestration."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Literal

from capmas.contracts.calibration import (
    CalibrationDatasetManifest,
    CalibrationOutcome,
    CalibrationPrediction,
)
from capmas.evaluation.calibration import (
    ConstrainedLogisticModel,
    IsotonicCalibration,
    brier_score,
    expected_calibration_error,
    fit_constrained_logistic,
    fit_isotonic,
    predict_offline,
)
from capmas.evaluation.correlation import ReducedFeatureVector, reduce_feature_snapshot
from capmas.evaluation.dataset import (
    assert_dataset_eligible,
    audit_calibration_dataset,
)

OFFLINE_SPLITTER_VERSION = "p56b.exact_quota_sha256.v1"
_OFFLINE_SPLITS = ("train", "calibration", "test")
OFFLINE_REPORT_VERSION = "p56b.offline_calibration_report.v1"


@dataclass(frozen=True)
class ExactQuotaSplitConfig:
    """Locked family-scoped P5.6B lineage split configuration."""

    family_id: str
    split_salt: str
    train_count: int
    calibration_count: int
    test_count: int
    splitter_version: str = OFFLINE_SPLITTER_VERSION

    def __post_init__(self) -> None:
        if not self.family_id:
            raise ValueError("family_id must not be empty")
        if not self.split_salt:
            raise ValueError("split_salt must not be empty")
        if self.splitter_version != OFFLINE_SPLITTER_VERSION:
            raise ValueError("splitter_version is not supported")
        for name in ("train_count", "calibration_count", "test_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    @classmethod
    def object6_v1(cls) -> ExactQuotaSplitConfig:
        return cls(
            family_id="object-6",
            split_salt="p56b-object6-split-v1",
            train_count=12,
            calibration_count=4,
            test_count=4,
        )

    @property
    def total_count(self) -> int:
        return self.train_count + self.calibration_count + self.test_count

    def to_dict(self) -> dict[str, object]:
        return {
            "calibration_count": self.calibration_count,
            "family_id": self.family_id,
            "split_salt": self.split_salt,
            "splitter_version": self.splitter_version,
            "test_count": self.test_count,
            "train_count": self.train_count,
        }


@dataclass(frozen=True)
class OfflineExample:
    """One immutable physical outcome assigned to a P5.6B offline split."""

    outcome: CalibrationOutcome
    dataset_split: Literal["train", "calibration", "test"]
    lineage_group_id: str
    reduced: ReducedFeatureVector | None = None

    def __post_init__(self) -> None:
        if self.dataset_split not in _OFFLINE_SPLITS:
            raise ValueError("offline dataset_split is invalid")
        if not self.lineage_group_id:
            raise ValueError("lineage_group_id must not be empty")
        if self.outcome.tier != "A" or self.outcome.task_success is None:
            raise ValueError("offline example requires a conclusive Tier A outcome")
        if self.reduced is not None and (
            self.reduced.episode_id != self.outcome.episode_id
            or self.reduced.candidate_id != self.outcome.candidate_id
            or self.reduced.candidate_fingerprint != self.outcome.candidate_fingerprint
        ):
            raise ValueError("reduced feature vector identity must match the outcome")

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_split": self.dataset_split,
            "lineage_group_id": self.lineage_group_id,
            "outcome": self.outcome.to_dict(),
            "reduced": None if self.reduced is None else self.reduced.to_dict(),
        }


@dataclass(frozen=True)
class OfflineCalibrationReport:
    """Content-addressed result of a calibration run with no runtime effect."""

    report_version: str
    report_sha256: str
    source_dataset_id: str
    source_manifest_sha256: str
    split_config: ExactQuotaSplitConfig
    split_counts: Mapping[str, int]
    label_counts: Mapping[str, Mapping[str, int]]
    reduced_rows: tuple[OfflineExample, ...]
    model: ConstrainedLogisticModel | None
    isotonic: IsotonicCalibration | None
    predictions: Mapping[str, tuple[CalibrationPrediction, ...]]
    metrics: Mapping[str, float | None]
    abstention_counts: Mapping[str, int]
    fit_reason: str | None
    online_effect: bool = False

    def __post_init__(self) -> None:
        if self.report_version != OFFLINE_REPORT_VERSION:
            raise ValueError("offline report version is not supported")
        if len(self.report_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.report_sha256
        ):
            raise ValueError("offline report digest must be a lowercase SHA-256 value")
        if not self.source_dataset_id or not self.source_manifest_sha256:
            raise ValueError("offline report source identity must not be empty")
        if not isinstance(self.split_config, ExactQuotaSplitConfig):
            raise TypeError("offline report split_config must be ExactQuotaSplitConfig")
        if not isinstance(self.online_effect, bool) or self.online_effect:
            raise ValueError("offline calibration reports must have no online effect")
        if (self.model is None) != (self.isotonic is None):
            raise ValueError("offline model and isotonic artifacts must be both present or absent")
        if self.model is None and any(self.predictions.get(split) for split in _OFFLINE_SPLITS):
            raise ValueError("rejected offline fit must not publish predictions")
        object.__setattr__(self, "split_counts", _freeze_int_mapping(self.split_counts))
        object.__setattr__(self, "label_counts", _freeze_label_counts(self.label_counts))
        object.__setattr__(self, "predictions", _freeze_predictions(self.predictions))
        object.__setattr__(self, "metrics", _freeze_metrics(self.metrics))
        object.__setattr__(self, "abstention_counts", _freeze_int_mapping(self.abstention_counts))

    def to_dict(self) -> dict[str, object]:
        return {
            "abstention_counts": dict(self.abstention_counts),
            "fit_reason": self.fit_reason,
            "isotonic": None if self.isotonic is None else self.isotonic.to_dict(),
            "label_counts": {
                split: dict(counts) for split, counts in self.label_counts.items()
            },
            "metrics": dict(self.metrics),
            "model": None if self.model is None else self.model.to_dict(),
            "online_effect": self.online_effect,
            "predictions": {
                split: [prediction.to_dict() for prediction in predictions]
                for split, predictions in self.predictions.items()
            },
            "reduced_rows": [row.to_dict() for row in self.reduced_rows],
            "report_sha256": self.report_sha256,
            "report_version": self.report_version,
            "source_dataset_id": self.source_dataset_id,
            "source_manifest_sha256": self.source_manifest_sha256,
            "split_config": self.split_config.to_dict(),
            "split_counts": dict(self.split_counts),
        }


def partition_tier_a_outcomes(
    manifest: CalibrationDatasetManifest,
    config: ExactQuotaSplitConfig,
) -> tuple[OfflineExample, ...]:
    """Assign exactly one qualified physical outcome per frozen lineage group."""

    audit = audit_calibration_dataset(manifest)
    assert_dataset_eligible(audit)

    families = {outcome.family_id for outcome in manifest.outcomes}
    if families != {config.family_id}:
        raise ValueError(f"offline partition requires family {config.family_id}")

    lineages_by_episode = _lineages_by_episode(manifest)
    selected_by_group: dict[str, CalibrationOutcome] = {}
    for outcome in manifest.outcomes:
        if outcome.tier != "A":
            continue
        if outcome.execution_status != "selected_executed":
            raise ValueError("Tier A offline outcome must be selected and executed")
        lineages = lineages_by_episode.get(outcome.episode_id)
        if lineages is None or len(lineages) != 1:
            raise ValueError(f"outcome {outcome.episode_id} must have exactly one lineage")
        group_id = lineages[0]
        if group_id in selected_by_group:
            raise ValueError(f"lineage {group_id} has multiple Tier A outcomes")
        selected_by_group[group_id] = outcome

    if len(selected_by_group) != config.total_count:
        raise ValueError(
            f"offline partition requires exactly {config.total_count} Tier A lineage groups"
        )

    group_ids = sorted(selected_by_group, key=lambda group_id: _split_digest(config, group_id))
    assignments = _assign_exact_quotas(group_ids, config)
    examples = tuple(
        OfflineExample(
            outcome=selected_by_group[group_id],
            dataset_split=assignments[group_id],
            lineage_group_id=group_id,
        )
        for group_id in group_ids
    )
    _require_fitting_class_support(examples)
    return examples


def run_offline_calibration(
    manifest: CalibrationDatasetManifest,
    config: ExactQuotaSplitConfig,
) -> OfflineCalibrationReport:
    """Fit a locked offline model or return a report with no published artifacts."""

    partitioned = partition_tier_a_outcomes(manifest, config)
    reduced_rows = tuple(
        replace(example, reduced=reduce_feature_snapshot(example.outcome.feature_snapshot))
        for example in partitioned
    )
    split_counts = _split_counts(reduced_rows)
    label_counts = _label_counts(reduced_rows)
    eligible_by_split = {
        split: tuple(
            example
            for example in reduced_rows
            if example.dataset_split == split
            and example.reduced is not None
            and example.reduced.dimension("action_feasibility").value is not None
        )
        for split in _OFFLINE_SPLITS
    }

    rejection = _fit_gate_reason(eligible_by_split)
    if rejection is not None:
        return _report(
            manifest=manifest,
            config=config,
            split_counts=split_counts,
            label_counts=label_counts,
            reduced_rows=reduced_rows,
            fit_reason=rejection,
        )

    try:
        model = fit_constrained_logistic(eligible_by_split["train"])
    except ValueError as error:
        return _report(
            manifest=manifest,
            config=config,
            split_counts=split_counts,
            label_counts=label_counts,
            reduced_rows=reduced_rows,
            fit_reason=_fit_exception_reason(error),
        )
    if not model.converged:
        return _report(
            manifest=manifest,
            config=config,
            split_counts=split_counts,
            label_counts=label_counts,
            reduced_rows=reduced_rows,
            fit_reason="fit_rejected_nonconverged",
        )

    try:
        isotonic = fit_isotonic(model, eligible_by_split["calibration"])
    except ValueError as error:
        return _report(
            manifest=manifest,
            config=config,
            split_counts=split_counts,
            label_counts=label_counts,
            reduced_rows=reduced_rows,
            fit_reason=_fit_exception_reason(error, stage="isotonic"),
        )

    predictions = {
        split: tuple(predict_offline(model, isotonic, example) for example in rows)
        for split, rows in _rows_by_split(reduced_rows).items()
    }
    return _report(
        manifest=manifest,
        config=config,
        split_counts=split_counts,
        label_counts=label_counts,
        reduced_rows=reduced_rows,
        model=model,
        isotonic=isotonic,
        predictions=predictions,
    )


def _lineages_by_episode(manifest: CalibrationDatasetManifest) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for lineage in manifest.lineages:
        grouped[lineage.episode_id].append(lineage.lineage_group_id)
    return {episode_id: tuple(group_ids) for episode_id, group_ids in grouped.items()}


def _split_digest(config: ExactQuotaSplitConfig, group_id: str) -> str:
    return hashlib.sha256(f"{config.split_salt}:{group_id}".encode()).hexdigest()


def _assign_exact_quotas(
    group_ids: list[str], config: ExactQuotaSplitConfig
) -> Mapping[str, Literal["train", "calibration", "test"]]:
    train_end = config.train_count
    calibration_end = train_end + config.calibration_count
    assignments: dict[str, Literal["train", "calibration", "test"]] = {}
    for index, group_id in enumerate(group_ids):
        if index < train_end:
            assignments[group_id] = "train"
        elif index < calibration_end:
            assignments[group_id] = "calibration"
        else:
            assignments[group_id] = "test"
    return assignments


def _require_fitting_class_support(examples: tuple[OfflineExample, ...]) -> None:
    for split in ("train", "calibration"):
        labels = {
            example.outcome.task_success
            for example in examples
            if example.dataset_split == split
        }
        if labels != {False, True}:
            raise ValueError(f"offline {split} split must contain both task_success classes")


def _rows_by_split(
    rows: tuple[OfflineExample, ...],
) -> dict[str, tuple[OfflineExample, ...]]:
    return {
        split: tuple(row for row in rows if row.dataset_split == split)
        for split in _OFFLINE_SPLITS
    }


def _split_counts(rows: tuple[OfflineExample, ...]) -> dict[str, int]:
    return {split: sum(row.dataset_split == split for row in rows) for split in _OFFLINE_SPLITS}


def _label_counts(rows: tuple[OfflineExample, ...]) -> dict[str, dict[str, int]]:
    return {
        split: {
            "negative": sum(
                row.dataset_split == split and row.outcome.task_success is False for row in rows
            ),
            "positive": sum(
                row.dataset_split == split and row.outcome.task_success is True for row in rows
            ),
        }
        for split in _OFFLINE_SPLITS
    }


def _fit_gate_reason(eligible_by_split: Mapping[str, tuple[OfflineExample, ...]]) -> str | None:
    for split in ("train", "calibration"):
        labels = {example.outcome.task_success for example in eligible_by_split[split]}
        if labels != {False, True}:
            return f"fit_rejected_{split}_class_support"
    return None


def _fit_exception_reason(error: ValueError, *, stage: str = "logistic") -> str:
    message = str(error)
    if message.startswith("fit_rejected_"):
        return message
    if "missing required action_feasibility" in message:
        return "fit_rejected_missing_required_evidence"
    return f"fit_rejected_{stage}_invalid_input"


def _report(
    *,
    manifest: CalibrationDatasetManifest,
    config: ExactQuotaSplitConfig,
    split_counts: Mapping[str, int],
    label_counts: Mapping[str, Mapping[str, int]],
    reduced_rows: tuple[OfflineExample, ...],
    model: ConstrainedLogisticModel | None = None,
    isotonic: IsotonicCalibration | None = None,
    predictions: Mapping[str, tuple[CalibrationPrediction, ...]] | None = None,
    fit_reason: str | None = None,
) -> OfflineCalibrationReport:
    normalized_predictions = predictions or {split: () for split in _OFFLINE_SPLITS}
    report = OfflineCalibrationReport(
        report_version=OFFLINE_REPORT_VERSION,
        report_sha256="0" * 64,
        source_dataset_id=manifest.dataset_id,
        source_manifest_sha256=manifest.manifest_sha256,
        split_config=config,
        split_counts=split_counts,
        label_counts=label_counts,
        reduced_rows=reduced_rows,
        model=model,
        isotonic=isotonic,
        predictions=normalized_predictions,
        metrics=_test_metrics(normalized_predictions, reduced_rows),
        abstention_counts=_abstention_counts(normalized_predictions),
        fit_reason=fit_reason,
    )
    digest = _sha256_payload(report.to_dict())
    return replace(report, report_sha256=digest)


def _test_metrics(
    predictions: Mapping[str, tuple[CalibrationPrediction, ...]],
    rows: tuple[OfflineExample, ...],
) -> dict[str, float | None]:
    labels = {row.outcome.candidate_id: row.outcome.task_success for row in rows}
    pairs = tuple(
        (prediction.success_probability, labels[prediction.candidate_id])
        for prediction in predictions.get("test", ())
        if not prediction.abstained and prediction.success_probability is not None
    )
    return {
        "test_brier_score": brier_score(pairs),
        "test_expected_calibration_error": expected_calibration_error(pairs),
    }


def _abstention_counts(
    predictions: Mapping[str, tuple[CalibrationPrediction, ...]],
) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                prediction.reason
                for split_predictions in predictions.values()
                for prediction in split_predictions
                if prediction.abstained
            ).items()
        )
    )


def _sha256_payload(payload: Mapping[str, object]) -> str:
    canonical = dict(payload)
    canonical["report_sha256"] = ""
    encoded = json.dumps(
        canonical,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _freeze_int_mapping(raw: Mapping[str, int]) -> Mapping[str, int]:
    normalized: dict[str, int] = {}
    for name, value in sorted(raw.items()):
        if not isinstance(name, str) or not name or not isinstance(value, int) or value < 0:
            raise ValueError("offline report integer mappings must be non-negative")
        normalized[name] = value
    return MappingProxyType(normalized)


def _freeze_label_counts(
    raw: Mapping[str, Mapping[str, int]],
) -> Mapping[str, Mapping[str, int]]:
    if set(raw) != set(_OFFLINE_SPLITS):
        raise ValueError("offline report label counts must cover every split")
    normalized: dict[str, Mapping[str, int]] = {}
    for split in _OFFLINE_SPLITS:
        counts = _freeze_int_mapping(raw[split])
        if set(counts) != {"positive", "negative"}:
            raise ValueError("offline report label counts must include positive and negative")
        normalized[split] = counts
    return MappingProxyType(normalized)


def _freeze_predictions(
    raw: Mapping[str, tuple[CalibrationPrediction, ...]],
) -> Mapping[str, tuple[CalibrationPrediction, ...]]:
    if set(raw) != set(_OFFLINE_SPLITS):
        raise ValueError("offline report predictions must cover every split")
    normalized: dict[str, tuple[CalibrationPrediction, ...]] = {}
    for split in _OFFLINE_SPLITS:
        values = tuple(raw[split])
        if any(not isinstance(value, CalibrationPrediction) for value in values):
            raise TypeError("offline report predictions must use CalibrationPrediction")
        normalized[split] = values
    return MappingProxyType(normalized)


def _freeze_metrics(raw: Mapping[str, float | None]) -> Mapping[str, float | None]:
    if set(raw) != {"test_brier_score", "test_expected_calibration_error"}:
        raise ValueError("offline report metrics must use the fixed test-only keys")
    normalized: dict[str, float | None] = {}
    for name, value in sorted(raw.items()):
        if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise ValueError("offline report metrics must be finite or None")
        normalized[name] = None if value is None else float(value)
    return MappingProxyType(normalized)


__all__ = [
    "OFFLINE_REPORT_VERSION",
    "OFFLINE_SPLITTER_VERSION",
    "ExactQuotaSplitConfig",
    "OfflineCalibrationReport",
    "OfflineExample",
    "partition_tier_a_outcomes",
    "run_offline_calibration",
]
