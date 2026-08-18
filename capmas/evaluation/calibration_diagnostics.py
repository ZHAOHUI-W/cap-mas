"""Train-only identifiability diagnostics for constrained calibration."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from capmas.evaluation.correlation import ReducedFeatureVector

RANK_TOLERANCE = 1e-12

AvailabilityClass = Literal["all_present", "all_unknown", "mixed"]

_DIMENSIONS = (
    "scene_grounding",
    "action_feasibility",
    "collision_risk",
    "expected_latency_risk",
    "recovery_cost_risk",
)
_SUPPORT_DIMENSIONS = ("scene_grounding", "action_feasibility")
_RISK_DIMENSIONS = (
    "collision_risk",
    "expected_latency_risk",
    "recovery_cost_risk",
)
_OPTIONAL_DIMENSIONS = ("scene_grounding", *_RISK_DIMENSIONS)
_PARAMETER_ORDER = (
    "intercept",
    *(f"support.{name}" for name in _SUPPORT_DIMENSIONS),
    *(f"risk.{name}" for name in _RISK_DIMENSIONS),
    *(f"missing.{name}" for name in _OPTIONAL_DIMENSIONS),
)


@dataclass(frozen=True)
class TrainColumnDiagnostic:
    """Identifiability summary for one signed train design column."""

    parameter: str
    nonzero_count: int
    minimum: float
    maximum: float
    frozen: bool

    def __post_init__(self) -> None:
        if self.parameter not in _PARAMETER_ORDER:
            raise ValueError("diagnostic parameter is not supported")
        if self.nonzero_count < 0:
            raise ValueError("diagnostic nonzero_count must not be negative")
        if not math.isfinite(self.minimum) or not math.isfinite(self.maximum):
            raise ValueError("diagnostic bounds must be finite")
        if self.minimum > self.maximum:
            raise ValueError("diagnostic minimum must not exceed maximum")
        if not isinstance(self.frozen, bool):
            raise TypeError("diagnostic frozen flag must be boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "frozen": self.frozen,
            "maximum": self.maximum,
            "minimum": self.minimum,
            "nonzero_count": self.nonzero_count,
            "parameter": self.parameter,
        }


@dataclass(frozen=True)
class TrainDesignDiagnostics:
    """Immutable train-only design geometry and availability information."""

    row_count: int
    column_count: int
    active_parameter_count: int
    matrix_rank: int
    rank_tolerance: float
    columns: tuple[TrainColumnDiagnostic, ...]
    frozen_parameters: tuple[str, ...]
    availability_signature: Mapping[str, AvailabilityClass]

    def __post_init__(self) -> None:
        if self.row_count < 1:
            raise ValueError("diagnostic row_count must be positive")
        if self.column_count != len(_PARAMETER_ORDER):
            raise ValueError("diagnostic column_count is invalid")
        if not 1 <= self.active_parameter_count <= self.column_count:
            raise ValueError("diagnostic active_parameter_count is invalid")
        if not 1 <= self.matrix_rank <= min(self.row_count, self.column_count):
            raise ValueError("diagnostic matrix_rank is invalid")
        if not math.isfinite(self.rank_tolerance) or self.rank_tolerance <= 0.0:
            raise ValueError("diagnostic rank_tolerance must be positive and finite")
        if tuple(column.parameter for column in self.columns) != _PARAMETER_ORDER:
            raise ValueError("diagnostic column order is invalid")
        expected_frozen = tuple(
            column.parameter
            for column in self.columns
            if column.parameter != "intercept" and column.frozen
        )
        if self.frozen_parameters != expected_frozen:
            raise ValueError("diagnostic frozen parameters do not match columns")
        if self.active_parameter_count != self.column_count - len(self.frozen_parameters):
            raise ValueError("diagnostic active parameter count does not match freezing")
        signature = dict(self.availability_signature)
        if tuple(signature) != _DIMENSIONS:
            raise ValueError("diagnostic availability dimensions are invalid")
        if any(value not in {"all_present", "all_unknown", "mixed"} for value in signature.values()):
            raise ValueError("diagnostic availability class is invalid")
        object.__setattr__(self, "availability_signature", MappingProxyType(signature))

    def to_dict(self) -> dict[str, object]:
        return {
            "active_parameter_count": self.active_parameter_count,
            "availability_signature": dict(self.availability_signature),
            "column_count": self.column_count,
            "columns": [column.to_dict() for column in self.columns],
            "frozen_parameters": list(self.frozen_parameters),
            "matrix_rank": self.matrix_rank,
            "rank_tolerance": self.rank_tolerance,
            "row_count": self.row_count,
        }


def analyze_train_design(vectors: Sequence[ReducedFeatureVector]) -> TrainDesignDiagnostics:
    """Describe only the train-vector geometry used by a constrained fit."""

    _validate_vector_cohort(vectors)
    matrix = tuple(_design_row(vector) for vector in vectors)
    columns = tuple(
        _column_diagnostic(parameter, tuple(row[index] for row in matrix))
        for index, parameter in enumerate(_PARAMETER_ORDER)
    )
    frozen_parameters = tuple(
        column.parameter
        for column in columns
        if column.parameter != "intercept" and column.frozen
    )
    return TrainDesignDiagnostics(
        row_count=len(matrix),
        column_count=len(_PARAMETER_ORDER),
        active_parameter_count=len(_PARAMETER_ORDER) - len(frozen_parameters),
        matrix_rank=_matrix_rank(matrix, RANK_TOLERANCE),
        rank_tolerance=RANK_TOLERANCE,
        columns=columns,
        frozen_parameters=frozen_parameters,
        availability_signature=_availability_signature(vectors),
    )


def _validate_vector_cohort(vectors: Sequence[ReducedFeatureVector]) -> None:
    if not vectors:
        raise ValueError("train vector cohort must not be empty")
    first = vectors[0]
    for vector in vectors:
        if vector.family_id != first.family_id:
            raise ValueError("train vector cohort family must match")
        if vector.feature_schema_version != first.feature_schema_version:
            raise ValueError("train vector cohort schema must match")
        if tuple(dimension.name for dimension in vector.dimensions) != _DIMENSIONS:
            raise ValueError("train vector cohort dimensions are invalid")
        for dimension in vector.dimensions:
            if dimension.value is not None and not math.isfinite(dimension.value):
                raise ValueError("train vector cohort values must be finite")


def _design_row(vector: ReducedFeatureVector) -> tuple[float, ...]:
    values = {name: vector.dimension(name).value for name in _DIMENSIONS}
    return (
        1.0,
        *(float(values[name]) if values[name] is not None else 0.0 for name in _SUPPORT_DIMENSIONS),
        *(-float(values[name]) if values[name] is not None else 0.0 for name in _RISK_DIMENSIONS),
        *(-1.0 if values[name] is None else 0.0 for name in _OPTIONAL_DIMENSIONS),
    )


def _column_diagnostic(parameter: str, values: tuple[float, ...]) -> TrainColumnDiagnostic:
    minimum = min(values)
    maximum = max(values)
    return TrainColumnDiagnostic(
        parameter=parameter,
        nonzero_count=sum(abs(value) > RANK_TOLERANCE for value in values),
        minimum=minimum,
        maximum=maximum,
        frozen=abs(maximum - minimum) <= RANK_TOLERANCE,
    )


def _availability_signature(
    vectors: Sequence[ReducedFeatureVector],
) -> Mapping[str, AvailabilityClass]:
    signature: dict[str, AvailabilityClass] = {}
    for name in _DIMENSIONS:
        present_count = sum(vector.dimension(name).value is not None for vector in vectors)
        if present_count == 0:
            signature[name] = "all_unknown"
        elif present_count == len(vectors):
            signature[name] = "all_present"
        else:
            signature[name] = "mixed"
    return signature


def _matrix_rank(matrix: Sequence[Sequence[float]], tolerance: float) -> int:
    work = [list(row) for row in matrix]
    row_count = len(work)
    column_count = len(work[0])
    rank = 0
    for column in range(column_count):
        if rank == row_count:
            break
        pivot_row = max(range(rank, row_count), key=lambda row: abs(work[row][column]))
        pivot = work[pivot_row][column]
        if abs(pivot) <= tolerance:
            continue
        work[rank], work[pivot_row] = work[pivot_row], work[rank]
        for row in range(rank + 1, row_count):
            factor = work[row][column] / pivot
            if abs(factor) <= tolerance:
                continue
            for trailing in range(column, column_count):
                work[row][trailing] -= factor * work[rank][trailing]
        rank += 1
    return rank


__all__ = [
    "RANK_TOLERANCE",
    "AvailabilityClass",
    "TrainColumnDiagnostic",
    "TrainDesignDiagnostics",
    "analyze_train_design",
]
