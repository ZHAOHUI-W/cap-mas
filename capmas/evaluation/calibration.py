"""Pure-Python P5.6B constrained calibration primitives."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from capmas.contracts.calibration import CalibrationPrediction
from capmas.evaluation.calibration_diagnostics import (
    TrainDesignDiagnostics,
    analyze_train_design,
)
from capmas.evaluation.correlation import ReducedFeatureVector

if TYPE_CHECKING:
    from capmas.evaluation.offline import OfflineExample

CALIBRATION_MODEL_VERSION = "p56b.constrained_logistic.v2"
ISOTONIC_CALIBRATION_VERSION = "p56b.pava.v1"
MAX_ITERATIONS = 5_000
INITIAL_LEARNING_RATE = 0.10
LEARNING_RATE_DECAY = 0.995
L2_REGULARIZATION = 0.01
CONVERGENCE_TOLERANCE = 1e-9
PROJECTED_GRADIENT_TOLERANCE = 1e-8
WILSON_Z = 1.959963984540054

_SUPPORT_DIMENSIONS = ("scene_grounding", "action_feasibility")
_RISK_DIMENSIONS = (
    "collision_risk",
    "expected_latency_risk",
    "recovery_cost_risk",
)
_OPTIONAL_DIMENSIONS = ("scene_grounding", *_RISK_DIMENSIONS)
_REQUIRED_DIMENSION = "action_feasibility"


@dataclass(frozen=True)
class CalibrationFitDiagnostics:
    """Train-only geometry and final stationarity information for one fit."""

    train_design: TrainDesignDiagnostics
    final_loss_delta: float | None
    projected_gradient_inf_norm: float
    loss_delta_tolerance: float
    projected_gradient_tolerance: float
    convergence_rule: str = "loss_delta_and_projected_kkt.v1"

    def __post_init__(self) -> None:
        if self.final_loss_delta is not None and (
            not math.isfinite(self.final_loss_delta) or self.final_loss_delta < 0.0
        ):
            raise ValueError("fit final_loss_delta must be finite and non-negative")
        for name in (
            "projected_gradient_inf_norm",
            "loss_delta_tolerance",
            "projected_gradient_tolerance",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"fit {name} must be finite and non-negative")
        if self.convergence_rule != "loss_delta_and_projected_kkt.v1":
            raise ValueError("fit convergence_rule is not supported")

    @property
    def frozen_parameters(self) -> tuple[str, ...]:
        return self.train_design.frozen_parameters

    @property
    def availability_signature(self) -> Mapping[str, str]:
        return self.train_design.availability_signature

    def to_dict(self) -> dict[str, object]:
        return {
            "convergence_rule": self.convergence_rule,
            "final_loss_delta": self.final_loss_delta,
            "loss_delta_tolerance": self.loss_delta_tolerance,
            "projected_gradient_inf_norm": self.projected_gradient_inf_norm,
            "projected_gradient_tolerance": self.projected_gradient_tolerance,
            "train_design": self.train_design.to_dict(),
        }


@dataclass(frozen=True)
class ConstrainedLogisticModel:
    """A family-scoped logistic model with fixed evidence sign constraints."""

    model_version: str
    family_id: str
    feature_schema_version: str
    intercept: float
    support_weights: Mapping[str, float]
    risk_weights: Mapping[str, float]
    missing_penalties: Mapping[str, float]
    iterations: int
    final_loss: float
    converged: bool
    fit_diagnostics: CalibrationFitDiagnostics

    def __post_init__(self) -> None:
        if not self.model_version or not self.family_id or not self.feature_schema_version:
            raise ValueError("model identity fields must not be empty")
        if not math.isfinite(self.intercept) or not math.isfinite(self.final_loss):
            raise ValueError("model intercept and loss must be finite")
        if self.iterations < 1 or self.iterations > MAX_ITERATIONS:
            raise ValueError("model iteration count is invalid")
        if not isinstance(self.converged, bool):
            raise TypeError("model converged flag must be boolean")
        if not isinstance(self.fit_diagnostics, CalibrationFitDiagnostics):
            raise TypeError("model fit_diagnostics must be CalibrationFitDiagnostics")
        object.__setattr__(
            self,
            "support_weights",
            _freeze_weights(self.support_weights, _SUPPORT_DIMENSIONS, "support_weights"),
        )
        object.__setattr__(
            self,
            "risk_weights",
            _freeze_weights(self.risk_weights, _RISK_DIMENSIONS, "risk_weights"),
        )
        object.__setattr__(
            self,
            "missing_penalties",
            _freeze_weights(
                self.missing_penalties,
                _OPTIONAL_DIMENSIONS,
                "missing_penalties",
            ),
        )

    def raw_probability(self, vector: ReducedFeatureVector) -> float:
        """Score a reduced decision-time vector without calibration remapping."""

        _validate_vector_compatibility(vector, self.family_id, self.feature_schema_version)
        score = self.intercept
        for name in _SUPPORT_DIMENSIONS:
            dimension = vector.dimension(name)
            if name == _REQUIRED_DIMENSION and dimension.value is None:
                raise ValueError("missing required action_feasibility evidence")
            if dimension.value is None:
                score -= self.missing_penalties[name]
            else:
                score += self.support_weights[name] * dimension.value
        for name in _RISK_DIMENSIONS:
            dimension = vector.dimension(name)
            if dimension.value is None:
                score -= self.missing_penalties[name]
            else:
                score -= self.risk_weights[name] * dimension.value
        return _sigmoid(score)

    def to_dict(self) -> dict[str, object]:
        return {
            "converged": self.converged,
            "family_id": self.family_id,
            "feature_schema_version": self.feature_schema_version,
            "final_loss": self.final_loss,
            "fit_diagnostics": self.fit_diagnostics.to_dict(),
            "intercept": self.intercept,
            "iterations": self.iterations,
            "missing_penalties": dict(self.missing_penalties),
            "model_version": self.model_version,
            "risk_weights": dict(self.risk_weights),
            "support_weights": dict(self.support_weights),
        }


@dataclass(frozen=True)
class IsotonicBlock:
    """One monotonic PAVA block with empirical Wilson uncertainty."""

    lower: float
    upper: float
    probability: float
    sample_count: int
    positive_count: int
    uncertainty: float

    def __post_init__(self) -> None:
        for name in ("lower", "upper", "probability", "uncertainty"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"isotonic block {name} must be finite")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"isotonic block {name} must be in [0, 1]")
        if self.lower > self.upper:
            raise ValueError("isotonic block lower must not exceed upper")
        if self.sample_count < 1 or self.positive_count < 0 or self.positive_count > self.sample_count:
            raise ValueError("isotonic block counts are invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "lower": self.lower,
            "positive_count": self.positive_count,
            "probability": self.probability,
            "sample_count": self.sample_count,
            "uncertainty": self.uncertainty,
            "upper": self.upper,
        }


@dataclass(frozen=True)
class IsotonicCalibration:
    """Frozen PAVA transform produced only from calibration-split labels."""

    calibration_version: str
    feature_schema_version: str
    blocks: tuple[IsotonicBlock, ...]

    def __post_init__(self) -> None:
        if not self.calibration_version or not self.feature_schema_version or not self.blocks:
            raise ValueError("isotonic calibration fields must not be empty")
        previous_upper = -1.0
        previous_probability = -1.0
        for block in self.blocks:
            if block.lower < previous_upper or block.probability < previous_probability:
                raise ValueError("isotonic blocks must be monotonic")
            previous_upper = block.upper
            previous_probability = block.probability

    def calibrate(self, raw_probability: float) -> tuple[float, float]:
        if not isinstance(raw_probability, (int, float)) or not math.isfinite(raw_probability):
            raise ValueError("raw probability must be finite")
        if not 0.0 <= raw_probability <= 1.0:
            raise ValueError("raw probability must be in [0, 1]")
        for block in self.blocks:
            if raw_probability <= block.upper:
                return block.probability, block.uncertainty
        final = self.blocks[-1]
        return final.probability, final.uncertainty

    def to_dict(self) -> dict[str, object]:
        return {
            "blocks": [block.to_dict() for block in self.blocks],
            "calibration_version": self.calibration_version,
            "feature_schema_version": self.feature_schema_version,
        }


def fit_constrained_logistic(examples: Sequence[OfflineExample]) -> ConstrainedLogisticModel:
    """Fit sign-constrained logistic weights from train-split physical labels only."""

    rows = _validated_rows(examples, expected_split="train")
    labels = {label for _, label in rows}
    if labels != {False, True}:
        raise ValueError("train split must contain both task_success classes")

    family_id = rows[0][0].family_id
    feature_schema_version = rows[0][0].feature_schema_version
    train_design = analyze_train_design(tuple(vector for vector, _ in rows))
    frozen_parameters = set(train_design.frozen_parameters)
    positives = sum(label for _, label in rows)
    intercept = _logit((positives + 0.5) / (len(rows) + 1.0))
    support_weights = {name: 0.0 for name in _SUPPORT_DIMENSIONS}
    risk_weights = {name: 0.0 for name in _RISK_DIMENSIONS}
    missing_penalties = {name: 0.0 for name in _OPTIONAL_DIMENSIONS}
    previous_loss: float | None = None
    loss_delta: float | None = None
    projected_gradient_norm = math.inf
    converged = False

    for iteration in range(1, MAX_ITERATIONS + 1):
        gradients = _objective_gradients(
            rows,
            intercept,
            support_weights,
            risk_weights,
            missing_penalties,
        )
        rate = INITIAL_LEARNING_RATE * LEARNING_RATE_DECAY ** ((iteration - 1) // 100)
        intercept -= rate * gradients["intercept"]
        for name in _SUPPORT_DIMENSIONS:
            if f"support.{name}" not in frozen_parameters:
                support_weights[name] = max(
                    0.0,
                    support_weights[name] - rate * gradients["support_weights"][name],
                )
        for name in _RISK_DIMENSIONS:
            if f"risk.{name}" not in frozen_parameters:
                risk_weights[name] = max(
                    0.0,
                    risk_weights[name] - rate * gradients["risk_weights"][name],
                )
        for name in _OPTIONAL_DIMENSIONS:
            if f"missing.{name}" not in frozen_parameters:
                missing_penalties[name] = max(
                    0.0,
                    missing_penalties[name] - rate * gradients["missing_penalties"][name],
                )

        loss = _loss(rows, intercept, support_weights, risk_weights, missing_penalties)
        if not math.isfinite(loss) or not math.isfinite(intercept):
            raise ValueError("fit_rejected_nonfinite_optimizer_state")
        loss_delta = None if previous_loss is None else abs(previous_loss - loss)
        gradients = _objective_gradients(
            rows,
            intercept,
            support_weights,
            risk_weights,
            missing_penalties,
        )
        projected_gradient_norm = projected_gradient_inf_norm(
            gradients["intercept"],
            _active_constrained_gradients(gradients, frozen_parameters),
            _active_constrained_values(
                support_weights,
                risk_weights,
                missing_penalties,
                frozen_parameters,
            ),
        )
        if _convergence_reached(loss_delta, projected_gradient_norm):
            converged = True
            break
        previous_loss = loss
    else:
        iteration = MAX_ITERATIONS

    final_loss = _loss(rows, intercept, support_weights, risk_weights, missing_penalties)
    if not math.isfinite(final_loss):
        raise ValueError("fit_rejected_nonfinite_loss")
    gradients = _objective_gradients(
        rows,
        intercept,
        support_weights,
        risk_weights,
        missing_penalties,
    )
    projected_gradient_norm = projected_gradient_inf_norm(
        gradients["intercept"],
        _active_constrained_gradients(gradients, frozen_parameters),
        _active_constrained_values(
            support_weights,
            risk_weights,
            missing_penalties,
            frozen_parameters,
        ),
    )
    converged = _convergence_reached(loss_delta, projected_gradient_norm)
    return ConstrainedLogisticModel(
        model_version=CALIBRATION_MODEL_VERSION,
        family_id=family_id,
        feature_schema_version=feature_schema_version,
        intercept=intercept,
        support_weights=support_weights,
        risk_weights=risk_weights,
        missing_penalties=missing_penalties,
        iterations=iteration,
        final_loss=final_loss,
        converged=converged,
        fit_diagnostics=CalibrationFitDiagnostics(
            train_design=train_design,
            final_loss_delta=loss_delta,
            projected_gradient_inf_norm=projected_gradient_norm,
            loss_delta_tolerance=CONVERGENCE_TOLERANCE,
            projected_gradient_tolerance=PROJECTED_GRADIENT_TOLERANCE,
        ),
    )


def fit_isotonic(model: ConstrainedLogisticModel, examples: Sequence[OfflineExample]) -> IsotonicCalibration:
    """Fit a monotonic PAVA map using calibration-split labels only."""

    rows = _validated_rows(examples, expected_split="calibration")
    labels = {label for _, label in rows}
    if labels != {False, True}:
        raise ValueError("calibration split must contain both task_success classes")
    if any(vector.family_id != model.family_id for vector, _ in rows):
        raise ValueError("calibration rows do not match model family")
    if any(vector.feature_schema_version != model.feature_schema_version for vector, _ in rows):
        raise ValueError("calibration rows do not match model feature schema")

    grouped: list[_MutableBlock] = []
    for score, label in sorted(
        ((model.raw_probability(vector), label) for vector, label in rows),
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
        calibration_version=ISOTONIC_CALIBRATION_VERSION,
        feature_schema_version=model.feature_schema_version,
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


def predict_offline(
    model: ConstrainedLogisticModel,
    isotonic: IsotonicCalibration,
    example: OfflineExample,
) -> CalibrationPrediction:
    """Generate an offline-only prediction without runtime qualification."""

    vector = _require_reduced(example)
    if vector.dimension(_REQUIRED_DIMENSION).value is None:
        return _abstained_prediction(model, vector, "prediction_abstained_missing_required_evidence")
    if isotonic.feature_schema_version != model.feature_schema_version:
        return _abstained_prediction(model, vector, "prediction_abstained_schema_mismatch")
    try:
        raw_probability = model.raw_probability(vector)
    except ValueError:
        return _abstained_prediction(model, vector, "prediction_abstained_model_mismatch")
    probability, uncertainty = isotonic.calibrate(raw_probability)
    return CalibrationPrediction(
        candidate_id=vector.candidate_id,
        rank_score=raw_probability,
        success_probability=probability,
        uncertainty=uncertainty,
        abstained=False,
        reason="offline_calibrated",
        model_version=model.model_version,
        feature_schema_version=model.feature_schema_version,
        snapshot_id=None,
        eligible_family=False,
    )


def wilson_interval_width(positive_count: int, sample_count: int) -> float:
    """Return the full two-sided Wilson 95% interval width in [0, 1]."""

    if sample_count < 1 or positive_count < 0 or positive_count > sample_count:
        raise ValueError("Wilson counts are invalid")
    proportion = positive_count / sample_count
    denominator = 1.0 + WILSON_Z**2 / sample_count
    center = (proportion + WILSON_Z**2 / (2.0 * sample_count)) / denominator
    radius = (
        WILSON_Z
        * math.sqrt(
            proportion * (1.0 - proportion) / sample_count
            + WILSON_Z**2 / (4.0 * sample_count**2)
        )
        / denominator
    )
    return min(1.0, max(0.0, (center + radius) - (center - radius)))


def brier_score(predictions: Sequence[tuple[float, bool]]) -> float | None:
    """Compute Brier score without treating an empty split as a zero error."""

    if not predictions:
        return None
    _validate_probability_labels(predictions)
    return sum((probability - float(label)) ** 2 for probability, label in predictions) / len(predictions)


def expected_calibration_error(
    predictions: Sequence[tuple[float, bool]], *, bins: int = 10
) -> float | None:
    """Compute fixed-bin ECE; it is descriptive and never a fitting input."""

    if not predictions:
        return None
    if not isinstance(bins, int) or isinstance(bins, bool) or bins < 1:
        raise ValueError("ECE bins must be a positive integer")
    _validate_probability_labels(predictions)
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for probability, label in predictions:
        buckets[min(int(probability * bins), bins - 1)].append((probability, label))
    return sum(
        len(bucket) / len(predictions)
        * abs(
            sum(probability for probability, _ in bucket) / len(bucket)
            - sum(float(label) for _, label in bucket) / len(bucket)
        )
        for bucket in buckets
        if bucket
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


def _validated_rows(
    examples: Sequence[OfflineExample], *, expected_split: str
) -> tuple[tuple[ReducedFeatureVector, bool], ...]:
    if not examples:
        raise ValueError(f"{expected_split} split must not be empty")
    rows: list[tuple[ReducedFeatureVector, bool]] = []
    for example in examples:
        if example.dataset_split != expected_split:
            raise ValueError(f"expected only {expected_split} examples")
        vector = _require_reduced(example)
        if vector.dimension(_REQUIRED_DIMENSION).value is None:
            raise ValueError("missing required action_feasibility evidence")
        rows.append((vector, example.outcome.task_success))
    return tuple(rows)


def _require_reduced(example: OfflineExample) -> ReducedFeatureVector:
    if example.reduced is None:
        raise ValueError("offline example is missing a reduced feature vector")
    return example.reduced


def _raw_score(
    vector: ReducedFeatureVector,
    intercept: float,
    support_weights: Mapping[str, float],
    risk_weights: Mapping[str, float],
    missing_penalties: Mapping[str, float],
) -> float:
    score = intercept
    for name in _SUPPORT_DIMENSIONS:
        dimension = vector.dimension(name)
        if name == _REQUIRED_DIMENSION and dimension.value is None:
            raise ValueError("missing required action_feasibility evidence")
        if dimension.value is None:
            score -= missing_penalties[name]
        else:
            score += support_weights[name] * dimension.value
    for name in _RISK_DIMENSIONS:
        dimension = vector.dimension(name)
        if dimension.value is None:
            score -= missing_penalties[name]
        else:
            score -= risk_weights[name] * dimension.value
    return score


def _loss(
    rows: Sequence[tuple[ReducedFeatureVector, bool]],
    intercept: float,
    support_weights: Mapping[str, float],
    risk_weights: Mapping[str, float],
    missing_penalties: Mapping[str, float],
) -> float:
    total = 0.0
    for vector, label in rows:
        probability = _sigmoid(_raw_score(vector, intercept, support_weights, risk_weights, missing_penalties))
        probability = min(1.0 - 1e-15, max(1e-15, probability))
        total -= float(label) * math.log(probability) + (1.0 - float(label)) * math.log(1.0 - probability)
    regularization = sum(weight * weight for weight in support_weights.values())
    regularization += sum(weight * weight for weight in risk_weights.values())
    regularization += sum(weight * weight for weight in missing_penalties.values())
    return total / len(rows) + L2_REGULARIZATION * regularization


def _objective_gradients(
    rows: Sequence[tuple[ReducedFeatureVector, bool]],
    intercept: float,
    support_weights: Mapping[str, float],
    risk_weights: Mapping[str, float],
    missing_penalties: Mapping[str, float],
) -> dict[str, object]:
    gradients = _zero_gradients()
    for vector, label in rows:
        score = _raw_score(vector, intercept, support_weights, risk_weights, missing_penalties)
        error = _sigmoid(score) - float(label)
        gradients["intercept"] += error
        for name in _SUPPORT_DIMENSIONS:
            dimension = vector.dimension(name)
            if dimension.value is None:
                gradients["missing_penalties"][name] -= error
            else:
                gradients["support_weights"][name] += error * dimension.value
        for name in _RISK_DIMENSIONS:
            dimension = vector.dimension(name)
            if dimension.value is None:
                gradients["missing_penalties"][name] -= error
            else:
                gradients["risk_weights"][name] -= error * dimension.value

    count = len(rows)
    gradients["intercept"] /= count
    for name in _SUPPORT_DIMENSIONS:
        gradients["support_weights"][name] = (
            gradients["support_weights"][name] / count
            + 2.0 * L2_REGULARIZATION * support_weights[name]
        )
    for name in _RISK_DIMENSIONS:
        gradients["risk_weights"][name] = (
            gradients["risk_weights"][name] / count + 2.0 * L2_REGULARIZATION * risk_weights[name]
        )
    for name in _OPTIONAL_DIMENSIONS:
        gradients["missing_penalties"][name] = (
            gradients["missing_penalties"][name] / count
            + 2.0 * L2_REGULARIZATION * missing_penalties[name]
        )
    return gradients


def projected_gradient_inf_norm(
    intercept_gradient: float,
    constrained_gradients: Mapping[str, float],
    constrained_values: Mapping[str, float],
) -> float:
    """Return the first-order KKT residual for non-negative coefficients."""

    if set(constrained_gradients) != set(constrained_values):
        raise ValueError("KKT gradients and values must use identical keys")
    residuals = [abs(_finite_value(intercept_gradient, "intercept_gradient"))]
    for parameter in sorted(constrained_gradients):
        gradient = _finite_value(constrained_gradients[parameter], parameter)
        value = _finite_value(constrained_values[parameter], parameter)
        if value < 0.0:
            raise ValueError(f"{parameter} must be non-negative")
        residuals.append(abs(gradient) if value > 0.0 else max(0.0, -gradient))
    return max(residuals)


def _convergence_reached(
    loss_delta: float | None,
    projected_gradient_norm: float,
) -> bool:
    return (
        loss_delta is not None
        and loss_delta <= CONVERGENCE_TOLERANCE
        and projected_gradient_norm <= PROJECTED_GRADIENT_TOLERANCE
    )


def _active_constrained_gradients(
    gradients: Mapping[str, object], frozen_parameters: set[str]
) -> dict[str, float]:
    active: dict[str, float] = {}
    for name in _SUPPORT_DIMENSIONS:
        parameter = f"support.{name}"
        if parameter not in frozen_parameters:
            active[parameter] = gradients["support_weights"][name]
    for name in _RISK_DIMENSIONS:
        parameter = f"risk.{name}"
        if parameter not in frozen_parameters:
            active[parameter] = gradients["risk_weights"][name]
    for name in _OPTIONAL_DIMENSIONS:
        parameter = f"missing.{name}"
        if parameter not in frozen_parameters:
            active[parameter] = gradients["missing_penalties"][name]
    return active


def _active_constrained_values(
    support_weights: Mapping[str, float],
    risk_weights: Mapping[str, float],
    missing_penalties: Mapping[str, float],
    frozen_parameters: set[str],
) -> dict[str, float]:
    active: dict[str, float] = {}
    for name in _SUPPORT_DIMENSIONS:
        parameter = f"support.{name}"
        if parameter not in frozen_parameters:
            active[parameter] = support_weights[name]
    for name in _RISK_DIMENSIONS:
        parameter = f"risk.{name}"
        if parameter not in frozen_parameters:
            active[parameter] = risk_weights[name]
    for name in _OPTIONAL_DIMENSIONS:
        parameter = f"missing.{name}"
        if parameter not in frozen_parameters:
            active[parameter] = missing_penalties[name]
    return active


def _finite_value(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _zero_gradients() -> dict[str, object]:
    return {
        "intercept": 0.0,
        "support_weights": {name: 0.0 for name in _SUPPORT_DIMENSIONS},
        "risk_weights": {name: 0.0 for name in _RISK_DIMENSIONS},
        "missing_penalties": {name: 0.0 for name in _OPTIONAL_DIMENSIONS},
    }


def _freeze_weights(
    weights: Mapping[str, float], expected: tuple[str, ...], name: str
) -> Mapping[str, float]:
    if set(weights) != set(expected):
        raise ValueError(f"{name} keys do not match the calibration policy")
    frozen: dict[str, float] = {}
    for key in expected:
        value = weights[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name}.{key} must be a finite non-negative value")
        frozen[key] = float(value)
    return MappingProxyType(frozen)


def _validate_vector_compatibility(
    vector: ReducedFeatureVector, family_id: str, feature_schema_version: str
) -> None:
    if vector.family_id != family_id:
        raise ValueError("reduced vector family does not match model")
    if vector.feature_schema_version != feature_schema_version:
        raise ValueError("reduced vector schema does not match model")


def _abstained_prediction(
    model: ConstrainedLogisticModel,
    vector: ReducedFeatureVector,
    reason: str,
) -> CalibrationPrediction:
    return CalibrationPrediction(
        candidate_id=vector.candidate_id,
        rank_score=None,
        success_probability=None,
        uncertainty=1.0,
        abstained=True,
        reason=reason,
        model_version=model.model_version,
        feature_schema_version=model.feature_schema_version,
        snapshot_id=None,
        eligible_family=False,
    )


def _validate_probability_labels(predictions: Sequence[tuple[float, bool]]) -> None:
    for probability, label in predictions:
        if not isinstance(probability, (int, float)) or not math.isfinite(probability):
            raise ValueError("metric probability must be finite")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("metric probability must be in [0, 1]")
        if not isinstance(label, bool):
            raise TypeError("metric label must be boolean")


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


__all__ = [
    "CALIBRATION_MODEL_VERSION",
    "CONVERGENCE_TOLERANCE",
    "INITIAL_LEARNING_RATE",
    "ISOTONIC_CALIBRATION_VERSION",
    "L2_REGULARIZATION",
    "LEARNING_RATE_DECAY",
    "MAX_ITERATIONS",
    "PROJECTED_GRADIENT_TOLERANCE",
    "WILSON_Z",
    "CalibrationFitDiagnostics",
    "ConstrainedLogisticModel",
    "IsotonicBlock",
    "IsotonicCalibration",
    "brier_score",
    "expected_calibration_error",
    "fit_constrained_logistic",
    "fit_isotonic",
    "predict_offline",
    "projected_gradient_inf_norm",
    "wilson_interval_width",
]
