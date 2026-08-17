"""Offline-only P5.6B partitioning and calibration orchestration."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from capmas.contracts.calibration import (
    CalibrationDatasetManifest,
    CalibrationOutcome,
)
from capmas.evaluation.correlation import ReducedFeatureVector
from capmas.evaluation.dataset import (
    assert_dataset_eligible,
    audit_calibration_dataset,
)

OFFLINE_SPLITTER_VERSION = "p56b.exact_quota_sha256.v1"
_OFFLINE_SPLITS = ("train", "calibration", "test")


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


__all__ = [
    "OFFLINE_SPLITTER_VERSION",
    "ExactQuotaSplitConfig",
    "OfflineExample",
    "partition_tier_a_outcomes",
]
