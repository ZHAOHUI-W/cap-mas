# P5.6C Fit Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add train-only identifiability diagnostics, zero-variance parameter freezing, availability guards, and projected-KKT convergence to a versioned offline calibration model without changing the locked P5.6B dataset or runtime.

**Architecture:** Keep `calibration.py` as the fitting and prediction facade. Put immutable train design diagnostics and pure-Python matrix-rank helpers in a focused `calibration_diagnostics.py` module. The fitter consumes the diagnostics, skips frozen coefficient updates, recomputes a projected-KKT residual after each update, and embeds the final diagnostics in a V2 model. Prediction checks the train availability signature before using the unchanged raw-score formula.

**Tech Stack:** Python 3.10+, standard library only, existing CAP-MAS calibration contracts, pytest, Ruff.

## Global Constraints

- Keep the locked `object-6` `12 train / 4 calibration / 4 test` lineage split and `p56b-object6-split-v1` salt unchanged.
- Preserve `None` and explicit `unknown` status; never impute unknown evidence as `0.0`.
- Keep support, risk, and missingness coefficients non-negative.
- Preserve `max_iterations = 5_000`, `initial_learning_rate = 0.10`, `learning_rate_decay = 0.995`, `l2_regularization = 0.01`, and `loss_delta_tolerance = 1e-9`.
- Add only `projected_gradient_tolerance = 1e-8` and `rank_tolerance = 1e-12` as serialized V2 constants.
- Do not use calibration/test labels, evaluator state, post-action state, OOD features, runtime Arbiter state, CAP-X, LIBERO, LLMs, NumPy, SciPy, or any external optimizer.
- Preserve V1 artifacts; V2 uses model version `p56b.constrained_logistic.v2`.
- Do not publish a non-converged model, PAVA transform, or runtime prediction from the later offline orchestrator.

---

### Task 1: Train Design Diagnostics

**Files:**

- Create: `capmas/evaluation/calibration_diagnostics.py`
- Create: `tests/test_p56_calibration_diagnostics.py`
- Modify: `capmas/evaluation/__init__.py`

**Interfaces:**

- Consume: `Sequence[ReducedFeatureVector]` from the train partition.
- Produce: `TrainColumnDiagnostic`, `TrainDesignDiagnostics`, and
  `analyze_train_design(vectors) -> TrainDesignDiagnostics`.
- Use the fixed parameter order `intercept`, `support.scene_grounding`,
  `support.action_feasibility`, `risk.collision_risk`,
  `risk.expected_latency_risk`, `risk.recovery_cost_risk`,
  `missing.scene_grounding`, `missing.collision_risk`,
  `missing.expected_latency_risk`, `missing.recovery_cost_risk`.

- [ ] **Step 1: Write failing diagnostic tests**

```python
def test_design_diagnostics_freeze_constant_unknown_columns() -> None:
    diagnostics = analyze_train_design(tuple(example.reduced for example in _train_examples()))

    assert diagnostics.row_count == 6
    assert diagnostics.column_count == 10
    assert diagnostics.availability_signature["scene_grounding"] == "all_unknown"
    assert diagnostics.availability_signature["action_feasibility"] == "all_present"
    frozen = set(diagnostics.frozen_parameters)
    assert "support.scene_grounding" in frozen
    assert "missing.scene_grounding" in frozen
    assert "missing.collision_risk" in frozen
    assert diagnostics.matrix_rank < diagnostics.column_count


def test_design_diagnostics_is_deterministic_and_json_safe() -> None:
    vectors = tuple(example.reduced for example in _train_examples())
    first = analyze_train_design(vectors)
    second = analyze_train_design(vectors)

    assert first == second
    assert json.loads(json.dumps(first.to_dict()))["rank_tolerance"] == 1e-12
    assert list(first.to_dict()) == sorted(first.to_dict())
```

Add rejection tests for an empty sequence, mixed family IDs, mixed feature
schema versions, missing dimensions, non-finite values, and a vector with an
unexpected dimension name.

- [ ] **Step 2: Run the diagnostic tests and verify the expected import failure**

Run:

```bash
pytest -q tests/test_p56_calibration_diagnostics.py
```

Expected: collection fails because `capmas.evaluation.calibration_diagnostics`
does not exist.

- [ ] **Step 3: Implement immutable diagnostics and rank calculation**

Implement the following shapes:

```python
AvailabilityClass = Literal["all_present", "all_unknown", "mixed"]

@dataclass(frozen=True)
class TrainColumnDiagnostic:
    parameter: str
    nonzero_count: int
    minimum: float
    maximum: float
    frozen: bool

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
    row_count: int
    column_count: int
    active_parameter_count: int
    matrix_rank: int
    rank_tolerance: float
    columns: tuple[TrainColumnDiagnostic, ...]
    frozen_parameters: tuple[str, ...]
    availability_signature: Mapping[str, AvailabilityClass]

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

def analyze_train_design(
    vectors: Sequence[ReducedFeatureVector],
) -> TrainDesignDiagnostics:
    _validate_vector_cohort(vectors)
    matrix = tuple(_design_row(vector) for vector in vectors)
    columns = tuple(
        _column_diagnostic(parameter, tuple(row[index] for row in matrix))
        for index, parameter in enumerate(_PARAMETER_ORDER)
    )
    frozen = tuple(
        column.parameter
        for column in columns
        if column.parameter != "intercept" and column.frozen
    )
    return TrainDesignDiagnostics(
        row_count=len(matrix),
        column_count=len(_PARAMETER_ORDER),
        active_parameter_count=len(_PARAMETER_ORDER) - len(frozen),
        matrix_rank=_matrix_rank(matrix, RANK_TOLERANCE),
        rank_tolerance=RANK_TOLERANCE,
        columns=columns,
        frozen_parameters=frozen,
        availability_signature=_availability_signature(vectors),
    )
```

Build the signed design matrix exactly as the spec defines it. Use
deterministic partial-pivot Gaussian elimination: for each column, choose the
largest absolute pivot among remaining rows, skip it when it is at most
`1e-12`, eliminate below it, and count accepted pivots. Mark every
non-intercept column with `abs(maximum - minimum) <= rank_tolerance` as frozen.
Compute `nonzero_count` with the same tolerance and return sorted mapping keys
in `to_dict()`.

Implement these private helpers in the same module: `_validate_vector_cohort`
rejects empty, mixed-family, mixed-schema, or malformed cohorts;
`_design_row` emits the ten columns in `_PARAMETER_ORDER`; `_column_diagnostic`
computes minimum, maximum, nonzero count, and the frozen flag from one column;
`_availability_signature` returns `all_present`, `all_unknown`, or `mixed` for
each of the five fixed reduced dimensions; and `_matrix_rank` performs the
partial-pivot elimination above without mutating its input matrix.

- [ ] **Step 4: Run focused tests and Ruff**

Run:

```bash
pytest -q tests/test_p56_calibration_diagnostics.py
ruff check capmas/evaluation/calibration_diagnostics.py tests/test_p56_calibration_diagnostics.py
```

Expected: all diagnostic tests pass and Ruff exits zero.

- [ ] **Step 5: Commit the diagnostic module**

```bash
git add capmas/evaluation/calibration_diagnostics.py capmas/evaluation/__init__.py tests/test_p56_calibration_diagnostics.py
git commit -m "feat: add P5.6C train design diagnostics"
```

### Task 2: V2 Frozen-Parameter Fit and KKT Residual

**Files:**

- Modify: `capmas/evaluation/calibration.py`
- Modify: `tests/test_p56_calibration.py`
- Modify: `capmas/evaluation/__init__.py`

**Interfaces:**

- Consume: `TrainDesignDiagnostics` from Task 1 and eligible train examples.
- Produce: `CalibrationFitDiagnostics`, V2 `ConstrainedLogisticModel`, and
  `projected_gradient_inf_norm` as a deterministic pure-Python helper.

- [ ] **Step 1: Add failing V2 fit tests**

```python
def test_v2_freezes_non_identifiable_parameters_and_records_fit_diagnostics() -> None:
    model = fit_constrained_logistic(_train_examples())

    assert model.model_version == "p56b.constrained_logistic.v2"
    assert model.support_weights["scene_grounding"] == 0.0
    assert model.missing_penalties["scene_grounding"] == 0.0
    assert model.fit_diagnostics.frozen_parameters
    assert model.fit_diagnostics.availability_signature["scene_grounding"] == "all_unknown"
    assert model.fit_diagnostics.projected_gradient_inf_norm is not None
    assert model.to_dict()["fit_diagnostics"] == model.fit_diagnostics.to_dict()


def test_kkt_residual_blocks_loss_only_false_convergence() -> None:
    assert _convergence_reached(1e-12, 1e-2) is False
    assert _convergence_reached(1e-12, 1e-9) is True
```

Keep the existing deterministic and non-negative-weight assertions. Update
the budget fixture to assert V2 reaches `MAX_ITERATIONS` without claiming that
the synthetic separable fixture converges.

- [ ] **Step 2: Run the V2 tests and verify the expected failures**

Run:

```bash
pytest -q tests/test_p56_calibration.py
```

Expected: failures for the missing V2 diagnostics/model version and the
unimplemented KKT convergence helper.

- [ ] **Step 3: Refactor gradient calculation into one normalized helper**

Add `_objective_gradients(rows, intercept, support_weights, risk_weights,
missing_penalties)` returning the intercept gradient and all coefficient
gradients after sample averaging and L2 terms. This helper must be the only
source of gradients for both updates and KKT diagnostics.

Create:

```python
PROJECTED_GRADIENT_TOLERANCE = 1e-8
CALIBRATION_MODEL_VERSION = "p56b.constrained_logistic.v2"

def _finite_value(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)

def projected_gradient_inf_norm(
    intercept_gradient: float,
    constrained_gradients: Mapping[str, float],
    constrained_values: Mapping[str, float],
) -> float:
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
```

For a positive constrained value return `abs(gradient)`; at the zero lower
bound return `max(0.0, -gradient)`. Include the absolute intercept gradient in
the maximum. Validate finite inputs and matching keys.

- [ ] **Step 4: Freeze constant columns and add KKT convergence**

Call `analyze_train_design()` on the validated train vectors before
initializing weights. Initialize every frozen non-intercept weight at zero and
skip its update. After each update, call `_objective_gradients()` again at the
new point. Set:

```python
loss_delta = None if previous_loss is None else abs(previous_loss - loss)
kkt_norm = projected_gradient_inf_norm(
    gradients["intercept"],
    active_constrained_gradients,
    active_constrained_values,
)
converged = (
    loss_delta is not None
    and loss_delta <= CONVERGENCE_TOLERANCE
    and kkt_norm <= PROJECTED_GRADIENT_TOLERANCE
)
```

After the loop, recompute the final loss delta and KKT norm and construct
`CalibrationFitDiagnostics` with the train design report, final metrics,
constants, and convergence rule. Add that immutable field to
`ConstrainedLogisticModel` and include it in `to_dict()`.

- [ ] **Step 5: Run focused tests and Ruff**

Run:

```bash
pytest -q tests/test_p56_calibration.py tests/test_p56_calibration_diagnostics.py
ruff check capmas/evaluation/calibration.py capmas/evaluation/calibration_diagnostics.py tests/test_p56_calibration.py tests/test_p56_calibration_diagnostics.py
```

Expected: all focused tests pass. The six-row fixture may remain
`converged=False`; that is expected and must be visible in its diagnostics.

- [ ] **Step 6: Commit the V2 fitter**

```bash
git add capmas/evaluation/calibration.py capmas/evaluation/calibration_diagnostics.py capmas/evaluation/__init__.py tests/test_p56_calibration.py tests/test_p56_calibration_diagnostics.py
git commit -m "feat: add P5.6C KKT-stable calibration fit"
```

### Task 3: Availability Guard and Offline Abstention

**Files:**

- Modify: `capmas/evaluation/calibration.py`
- Modify: `tests/test_p56_calibration.py`

**Interfaces:**

- Consume: `model.fit_diagnostics.availability_signature` and a reduced vector.
- Produce: stable `prediction_abstained_availability_mismatch` behavior.

- [ ] **Step 1: Write failing availability tests**

```python
def test_prediction_abstains_when_all_unknown_dimension_becomes_present() -> None:
    model = fit_constrained_logistic(_train_examples())
    calibration = fit_isotonic(
        model,
        (_example("calibration", False, 0.2, candidate_id="d"),
         _example("calibration", True, 0.8, candidate_id="e")),
    )

    prediction = predict_offline(
        model, calibration,
        _example("test", True, 0.9, candidate_id="g", scene=0.4),
    )

    assert prediction.abstained is True
    assert prediction.reason == "prediction_abstained_availability_mismatch"


def test_mixed_availability_accepts_present_and_unknown_rows() -> None:
    examples = (
        _example("train", False, 0.1, candidate_id="a"),
        _example("train", False, 0.2, candidate_id="b", scene=0.2),
        _example("train", True, 0.8, candidate_id="c"),
        _example("train", True, 0.9, candidate_id="d", scene=0.8),
    )
    model = fit_constrained_logistic(examples)

    assert model.fit_diagnostics.availability_signature["scene_grounding"] == "mixed"
    assert model.raw_probability(examples[0].reduced) >= 0.0
    assert model.raw_probability(examples[1].reduced) >= 0.0
```

- [ ] **Step 2: Run the tests and verify the expected availability failure**

Run: `pytest -q tests/test_p56_calibration.py`

Expected: the new mismatch test fails because V1 raw scoring has no
availability guard or stable abstention reason.

- [ ] **Step 3: Validate availability before scoring**

Add a private `_validate_availability(vector, signature)` helper called from
`ConstrainedLogisticModel.raw_probability()`. Compare only present/unknown
status with non-`mixed` train classes and raise
`ValueError("reduced vector availability signature does not match model")` on
mismatch.

In `predict_offline()`, catch that exact compatibility error before the generic
model mismatch branch and return an abstained `CalibrationPrediction` with
`rank_score=None`, `success_probability=None`, `uncertainty=1.0`, and the
stable reason `prediction_abstained_availability_mismatch`.

- [ ] **Step 4: Run focused tests and inspect serialization**

Run:

```bash
pytest -q tests/test_p56_calibration.py
python -c "from capmas.evaluation import CALIBRATION_MODEL_VERSION; assert CALIBRATION_MODEL_VERSION.endswith('.v2')"
```

Expected: all calibration tests pass and the public model version is V2.

- [ ] **Step 5: Commit the availability guard**

```bash
git add capmas/evaluation/calibration.py tests/test_p56_calibration.py
git commit -m "feat: guard P5.6C scoring by train availability"
```

### Task 4: Documentation, Offline Gate, and Full Verification

**Files:**

- Modify: `docs/superpowers/specs/2026-08-18-p5-6c-fit-stability-design.md`
- Modify: `docs/implementation-roadmap.md`
- Modify: `docs/phase5-evidence-evolution.md`
- Modify: `docs/experiments.md`
- Create: `tests/test_p56_c_fit_stability_docs.py`

**Interfaces:**

- Consume: V2 model diagnostics and the existing Phase5RunDirectory contract.
- Produce: documented V2 offline-only gate and explicit no-real-data status.

- [ ] **Step 1: Add documentation assertions**

```python
def test_p56c_docs_keep_v1_baseline_and_v2_boundaries() -> None:
    from pathlib import Path

    spec = Path("docs/superpowers/specs/2026-08-18-p5-6c-fit-stability-design.md").read_text()
    roadmap = Path("docs/implementation-roadmap.md").read_text()
    experiments = Path("docs/experiments.md").read_text()
    assert "p56b.constrained_logistic.v1" in spec
    assert "p56b.constrained_logistic.v2" in spec
    assert "P5.6C can validate its diagnostic machinery only against synthetic" in spec
    assert "NumPy" in spec and "SciPy" in spec
    assert "P5.6C" in roadmap
    assert "real 12-row" in experiments
```

- [ ] **Step 2: Run the documentation test and verify the expected stale-status failure**

Run: `pytest -q tests/test_p56_c_fit_stability_docs.py`

Expected: FAIL on the missing `P5.6C` roadmap or `real 12-row` experiment
status until Step 3 updates those documents.

- [ ] **Step 3: Update project status without claiming a real fit**

Record V2 implementation status, the synthetic-fixture verification commands,
the absence of an admissible 20-row P5.6A collection artifact, and the exact
future gate: a real Phase5 run must produce a manifest, diagnostics, model
digest, convergence status, and verifier-checked run directory before any
12-row convergence or calibration result is reported.

- [ ] **Step 4: Run the full verification suite**

Run:

```bash
pytest -q
ruff check capmas/evaluation/calibration.py capmas/evaluation/calibration_diagnostics.py tests/test_p56_calibration.py tests/test_p56_calibration_diagnostics.py
git diff --check
```

Expected: all tests pass, touched-file Ruff passes, and the diff has no
whitespace errors. Do not claim the real object-6 calibration gate is closed.

- [ ] **Step 5: Commit documentation and final status**

```bash
git add docs/implementation-roadmap.md docs/phase5-evidence-evolution.md docs/experiments.md tests/test_p56_c_fit_stability_docs.py
git commit -m "docs: record P5.6C fit stability gate"
```

## Completion Checklist

- [ ] Train diagnostics are deterministic, JSON-safe, and train-only.
- [ ] Constant non-intercept design columns freeze at zero.
- [ ] V2 reports matrix rank, availability signature, final loss delta, and
  projected-KKT norm.
- [ ] Loss delta alone cannot mark a fit converged.
- [ ] Availability mismatch abstains offline and never reaches runtime.
- [ ] V1 baseline remains identifiable by model version.
- [ ] Documentation does not claim a real 12-row result without a verified
  Phase 5 artifact.
