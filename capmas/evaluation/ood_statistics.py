"""Dependency-free statistics for paired frozen OOD replay."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math
import random
from statistics import NormalDist
from typing import TYPE_CHECKING

from capmas.evaluation.ood import OODReplayEvidence

if TYPE_CHECKING:
    from capmas.evaluation.ood import OODSplitManifest


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_level <= 1.0:
            raise ValueError("confidence level must be in [0, 1]")
        if self.lower > self.upper:
            raise ValueError("confidence interval lower bound must not exceed upper bound")


@dataclass(frozen=True)
class OODAggregateReport:
    id_success_count: int
    id_success_total: int
    ood_success_count: int
    ood_success_total: int
    id_rate: ConfidenceInterval
    ood_rate: ConfidenceInterval
    ood_gap: ConfidenceInterval
    paired_id_only: int
    paired_ood_only: int
    paired_ties: int
    paired_mcnemar_pvalue: float
    infrastructure_unknown_count: int
    failure_classes: dict[str, int]
    graph_failure_classes: dict[str, int]
    task_failure_classes: dict[str, int]
    verifier_false_negative_classes: dict[str, int]
    evaluator_graph_disagreement_count: int
    id_graph_completed_count: int
    id_graph_completed_total: int
    ood_graph_completed_count: int
    ood_graph_completed_total: int
    id_verifier_success_count: int
    id_verifier_success_total: int
    ood_verifier_success_count: int
    ood_verifier_success_total: int
    selection_bases: dict[str, int]


def wilson_interval(
    successes: int,
    total: int,
    confidence_level: float = 0.95,
) -> ConfidenceInterval:
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("successes must be between zero and total")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence level must be between zero and one")
    if total == 0:
        return ConfidenceInterval(0.0, 0.0, 0.0, confidence_level)
    estimate = successes / total
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    denominator = 1.0 + (z * z / total)
    center = (estimate + (z * z / (2.0 * total))) / denominator
    margin = (
        z
        * math.sqrt(
            estimate * (1.0 - estimate) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return ConfidenceInterval(estimate, max(0.0, center - margin), min(1.0, center + margin), confidence_level)


def paired_success_delta(
    id_results: Sequence[bool | None],
    ood_results: Sequence[bool | None],
) -> tuple[int, int, int]:
    if len(id_results) != len(ood_results):
        raise ValueError("paired ID/OOD result sequences must have equal length")
    id_only = 0
    ood_only = 0
    ties = 0
    for id_result, ood_result in zip(id_results, ood_results):
        if id_result is None or ood_result is None:
            continue
        if id_result and not ood_result:
            id_only += 1
        elif ood_result and not id_result:
            ood_only += 1
        else:
            ties += 1
    return id_only, ood_only, ties


def exact_mcnemar_pvalue(id_only: int, ood_only: int) -> float:
    if id_only < 0 or ood_only < 0:
        raise ValueError("McNemar discordant counts must be non-negative")
    total = id_only + ood_only
    if total == 0:
        return 1.0
    lower_tail = sum(math.comb(total, index) for index in range(min(id_only, ood_only) + 1)) / (2**total)
    upper_tail = sum(math.comb(total, index) for index in range(max(id_only, ood_only), total + 1)) / (2**total)
    return min(1.0, 2.0 * min(lower_tail, upper_tail))


def _bootstrap_gap(
    pairs: Sequence[tuple[bool, bool]],
    confidence_level: float,
    samples: int = 2000,
) -> ConfidenceInterval:
    if not pairs:
        return ConfidenceInterval(0.0, -1.0, 1.0, confidence_level)
    estimate = sum(int(id_result) - int(ood_result) for id_result, ood_result in pairs) / len(pairs)
    rng = random.Random(0)
    distribution: list[float] = []
    for _ in range(samples):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        distribution.append(
            sum(int(id_result) - int(ood_result) for id_result, ood_result in sample) / len(sample)
        )
    distribution.sort()
    lower_index = int((1.0 - confidence_level) / 2.0 * (samples - 1))
    upper_index = int((1.0 - (1.0 - confidence_level) / 2.0) * (samples - 1))
    return ConfidenceInterval(
        estimate,
        distribution[lower_index],
        distribution[upper_index],
        confidence_level,
    )


def aggregate_ood_pairs(
    evidence: Iterable[OODReplayEvidence],
    *,
    manifest: "OODSplitManifest | None" = None,
    confidence_level: float = 0.95,
) -> OODAggregateReport:
    """Aggregate one selected-candidate evidence record per replay case."""

    records = tuple(evidence)
    if len({record.case_id for record in records}) != len(records):
        raise ValueError("OOD aggregation expects one selected record per case")
    if manifest is not None:
        manifest_cases = {case.case_id: case for case in manifest.cases}
        unknown_case_ids = sorted(
            record.case_id for record in records if record.case_id not in manifest_cases
        )
        if unknown_case_ids:
            raise ValueError(f"OOD evidence contains unknown case ids: {unknown_case_ids}")
        mismatches = [
            record.case_id
            for record in records
            if (
                record.pair_id != manifest_cases[record.case_id].pair_id
                or record.split != manifest_cases[record.case_id].split
                or record.ood_type != manifest_cases[record.case_id].ood_type
            )
        ]
        if mismatches:
            raise ValueError(f"OOD evidence does not match manifest: {sorted(mismatches)}")

    id_records: dict[str, OODReplayEvidence] = {}
    ood_records: dict[str, OODReplayEvidence] = {}
    for record in records:
        target = id_records if record.split == "id" else ood_records
        if record.pair_id in target:
            raise ValueError(
                f"OOD aggregation expects one {record.split} record per pair: {record.pair_id}"
            )
        target[record.pair_id] = record
    id_successes = [record.evaluator_success for record in id_records.values()]
    ood_successes = [record.evaluator_success for record in ood_records.values()]
    id_known = [value for value in id_successes if value is not None]
    ood_known = [value for value in ood_successes if value is not None]
    paired = [
        (id_records[pair_id].evaluator_success, ood_records[pair_id].evaluator_success)
        for pair_id in sorted(set(id_records) & set(ood_records))
        if id_records[pair_id].evaluator_success is not None
        and ood_records[pair_id].evaluator_success is not None
    ]
    id_only, ood_only, ties = paired_success_delta(
        tuple(item[0] for item in paired), tuple(item[1] for item in paired)
    )
    failure_classes = Counter(
        record.failure_class
        for record in records
        if record.failure_class is not None
    )
    graph_failure_classes = Counter(
        record.failure_class
        for record in records
        if (
            not record.graph_completed
            and record.evaluator_success is not None
            and record.failure_class is not None
        )
    )
    task_failure_classes = Counter(
        record.failure_class
        for record in records
        if record.evaluator_success is False and record.failure_class is not None
    )
    verifier_false_negative_classes = Counter(
        record.failure_class
        for record in records
        if (
            record.evaluator_success is True
            and record.verifier_success is False
            and record.failure_class is not None
        )
    )
    id_verifier_known = [
        record.verifier_success
        for record in id_records.values()
        if record.verifier_success is not None
    ]
    ood_verifier_known = [
        record.verifier_success
        for record in ood_records.values()
        if record.verifier_success is not None
    ]
    selection_bases = Counter(
        record.selection_basis
        for record in records
        if record.selection_basis is not None
    )
    return OODAggregateReport(
        id_success_count=sum(id_known),
        id_success_total=len(id_known),
        ood_success_count=sum(ood_known),
        ood_success_total=len(ood_known),
        id_rate=wilson_interval(sum(id_known), len(id_known), confidence_level),
        ood_rate=wilson_interval(sum(ood_known), len(ood_known), confidence_level),
        ood_gap=_bootstrap_gap(paired, confidence_level),
        paired_id_only=id_only,
        paired_ood_only=ood_only,
        paired_ties=ties,
        paired_mcnemar_pvalue=exact_mcnemar_pvalue(id_only, ood_only),
        infrastructure_unknown_count=sum(value is None for value in id_successes + ood_successes),
        failure_classes=dict(failure_classes),
        graph_failure_classes=dict(graph_failure_classes),
        task_failure_classes=dict(task_failure_classes),
        verifier_false_negative_classes=dict(verifier_false_negative_classes),
        evaluator_graph_disagreement_count=sum(
            record.evaluator_success is not None
            and record.evaluator_success != record.graph_completed
            for record in records
        ),
        id_graph_completed_count=sum(record.graph_completed for record in id_records.values()),
        id_graph_completed_total=len(id_records),
        ood_graph_completed_count=sum(record.graph_completed for record in ood_records.values()),
        ood_graph_completed_total=len(ood_records),
        id_verifier_success_count=sum(id_verifier_known),
        id_verifier_success_total=len(id_verifier_known),
        ood_verifier_success_count=sum(ood_verifier_known),
        ood_verifier_success_total=len(ood_verifier_known),
        selection_bases=dict(selection_bases),
    )


__all__ = [
    "ConfidenceInterval",
    "OODAggregateReport",
    "aggregate_ood_pairs",
    "exact_mcnemar_pvalue",
    "paired_success_delta",
    "wilson_interval",
]
