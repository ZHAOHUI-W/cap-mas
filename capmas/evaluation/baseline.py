"""Fixed-weight offline baseline for P5.6 qualification comparisons."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from capmas.evaluation.calibration import IsotonicBlock, IsotonicCalibration, wilson_interval_width
from capmas.evaluation.correlation import ReducedFeatureVector

BASELINE_VERSION = "p56b.fixed_weight_baseline.v1"
BASELINE_PAVA_VERSION = "p56b.fixed_weight_pava.v1"

_SUPPORT_WEIGHTS = {
    "scene_grounding": 0.25,
    "action_feasibility": 0.25,
}
_RISK_WEIGHTS = {
    "collision_risk": 0.40,
    "expected_latency_risk": 0.03,
    "recovery_cost_risk": 0.02,
}


@dataclass(frozen=True)
class FixedWeightBaseline:
    """Frozen balanced-profile projection onto the reduced feature schema."""

    baseline_version: str
    family_id: str
    feature_schema_version: str
    support_weights: Mapping[str, float]
    risk_weights: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.baseline_version != BASELINE_VERSION:
            raise ValueError("unsupported fixed baseline version")
        if not self.family_id or not self.feature_schema_version:
            raise ValueError("baseline identity fields must not be empty")
        object.__setattr__(self, "support_weights", _freeze_weights(self.support_weights, _SUPPORT_WEIGHTS))
        object.__setattr__(self, "risk_weights", _freeze_weights(self.risk_weights, _RISK_WEIGHTS))

    @classmethod
    def object6_v1(cls) -> FixedWeightBaseline:
        return cls(
            baseline_version=BASELINE_VERSION,
            family_id="object-6",
            feature_schema_version="p56.feature.v1",
            support_weights=_SUPPORT_WEIGHTS,
            risk_weights=_RISK_WEIGHTS,
        )

    def score(self, vector: ReducedFeatureVector) -> float:
        """Return a monotonic [0, 1] transform of the fixed evidence score."""

        if vector.family_id != self.family_id:
            raise ValueError("baseline vector family does not match baseline")
        if vector.feature_schema_version != self.feature_schema_version:
            raise ValueError("baseline vector schema does not match baseline")
        action = vector.dimension("action_feasibility")
        if action.value is None:
            raise ValueError("missing required action_feasibility evidence")
        linear_score = 0.0
        for name, weight in self.support_weights.items():
            value = vector.dimension(name).value
            if value is not None:
                linear_score += weight * value
        for name, weight in self.risk_weights.items():
            value = vector.dimension(name).value
            if value is not None:
                linear_score -= weight * value
        return _sigmoid(linear_score)

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline_version": self.baseline_version,
            "family_id": self.family_id,
            "feature_schema_version": self.feature_schema_version,
            "risk_weights": dict(self.risk_weights),
            "support_weights": dict(self.support_weights),
        }


def fit_fixed_weight_mapping(
    baseline: FixedWeightBaseline,
    rows: Sequence[tuple[ReducedFeatureVector, bool]],
) -> IsotonicCalibration:
    """Fit a monotonic PAVA map on the supplied train/calibration rows."""

    if not rows:
        raise ValueError("fixed baseline mapping rows must not be empty")
    labels = {label for _, label in rows}
    if labels != {False, True}:
        raise ValueError("fixed baseline mapping requires both task_success classes")
    family_ids = {vector.family_id for vector, _ in rows}
    schema_versions = {vector.feature_schema_version for vector, _ in rows}
    if family_ids != {baseline.family_id}:
        raise ValueError("fixed baseline mapping rows do not match baseline family")
    if schema_versions != {baseline.feature_schema_version}:
        raise ValueError("fixed baseline mapping rows do not match baseline schema")

    grouped: list[_MutableBlock] = []
    for score, label in sorted(
        ((baseline.score(vector), label) for vector, label in rows),
        key=lambda item: item[0],
    ):
        if grouped and score == grouped[-1].upper:
            grouped[-1].sample_count += 1
            grouped[-1].positive_count += int(label)
        else:
            grouped.append(_MutableBlock(score, score, int(label), 1))
        while len(grouped) >= 2 and grouped[-2].rate > grouped[-1].rate:
            right = grouped.pop()
            left = grouped.pop()
            grouped.append(
                _MutableBlock(
                    lower=left.lower,
                    upper=right.upper,
                    positive_count=left.positive_count + right.positive_count,
                    sample_count=left.sample_count + right.sample_count,
                )
            )

    return IsotonicCalibration(
        calibration_version=BASELINE_PAVA_VERSION,
        feature_schema_version=baseline.feature_schema_version,
        blocks=tuple(
            IsotonicBlock(
                lower=block.lower,
                upper=block.upper,
                probability=block.rate,
                sample_count=block.sample_count,
                positive_count=block.positive_count,
                uncertainty=wilson_interval_width(block.positive_count, block.sample_count),
            )
            for block in grouped
        ),
    )


@dataclass
class _MutableBlock:
    lower: float
    upper: float
    positive_count: int
    sample_count: int

    @property
    def rate(self) -> float:
        return self.positive_count / self.sample_count


def _freeze_weights(raw: Mapping[str, float], expected: Mapping[str, float]) -> Mapping[str, float]:
    if set(raw) != set(expected):
        raise ValueError("baseline weight keys do not match the fixed policy")
    values: dict[str, float] = {}
    for name in expected:
        value = raw[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"baseline weight {name} must be finite")
        if value < 0.0:
            raise ValueError(f"baseline weight {name} must be non-negative")
        values[name] = float(value)
    return MappingProxyType(values)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


__all__ = ["BASELINE_PAVA_VERSION", "BASELINE_VERSION", "FixedWeightBaseline", "fit_fixed_weight_mapping"]
