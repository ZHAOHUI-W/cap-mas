# P5.6C Fit Stability Design

## 1. Purpose

P5.6C strengthens the offline-only P5.6B constrained logistic fit without
changing the locked `object-6` `12/4/4` lineage split, evidence reducer,
labels, or runtime behavior. It addresses two separate failure modes:

1. A train split can contain non-identifiable coefficient columns, especially
   optional dimensions that are always unknown. Their missingness indicators
   are constant and can be traded against the unregularized intercept.
2. A small loss change alone can mean that a learning-rate schedule moved
   slowly, not that the projected constrained optimum was reached.

The implementation remains standard-library-only and emits diagnostics for
every eligible train split. It produces no online score, calibration snapshot,
Arbiter change, CAP-X call, LIBERO action, LLM request, OOD feature, or new
physical data.

## 2. Version and Compatibility

The existing `p56b.constrained_logistic.v1` remains the historical baseline.
The stabilized model uses `p56b.constrained_logistic.v2`. P5.6C does not
overwrite V1 artifacts or compare a V2 result with a V1 result without
recording both model versions.

The V2 raw-score formula is unchanged for supported availability patterns:

```text
z = intercept
    + sum(support_weight[d] * value[d] for present support dimensions)
    - sum(risk_weight[d] * value[d] for present risk dimensions)
    - sum(missing_penalty[d] for unknown optional dimensions)
probability_raw = sigmoid(z)
```

All support, risk, and missingness coefficients remain non-negative. Unknown
remains `None` in every reduced vector and is never turned into `0.0`.

## 3. Train-Only Identifiability Diagnostics

`capmas.evaluation.calibration` will expose immutable JSON-safe diagnostics
derived only from the eligible `train` examples passed to fitting.

For each coefficient, the diagnostic records:

- `parameter`: one of `support.<dimension>`, `risk.<dimension>`, or
  `missing.<optional_dimension>`.
- `nonzero_count`, `minimum`, and `maximum` of its signed design column.
- `frozen`: whether the column is constant over all train rows.

The design columns are deterministic:

```text
intercept                    = 1
support.<d>                  = value[d] when present, else 0
risk.<d>                     = -value[d] when present, else 0
missing.<optional d>         = -1 when unknown, else 0
```

The report also records `row_count`, full column count, active coefficient
count after freezing, and a deterministic partial-pivot Gaussian-elimination
matrix rank using `rank_tolerance = 1e-12`. Rank is diagnostic only: P5.6C
does not silently remove arbitrary collinear non-constant columns.

## 4. Frozen Parameters and Availability Signature

Every non-intercept coefficient whose design column is constant in the train
split is frozen to exactly `0.0`. This includes a missingness penalty for an
optional dimension that is unknown in every train row. The intercept is never
frozen.

This rule is valid for the fitted train distribution: a constant feature term
can be absorbed into the unregularized intercept, while L2 selects zero for
its separate coefficient. It prevents an optimizer from drifting along an
intercept/missingness direction that carries no train information.

The model stores one train availability class for every reduced dimension:

- `all_present`: all train rows contain the dimension;
- `all_unknown`: all train rows omit the dimension;
- `mixed`: both conditions occur.

At scoring time, an `all_present` dimension must remain present and an
`all_unknown` dimension must remain unknown. A mismatch raises a stable model
compatibility error; offline prediction converts it to abstention. `mixed`
dimensions accept either state. This is an availability guard, not a
value-equality guard: a present value may vary at scoring time.

## 5. Projected-KKT Convergence

P5.6C preserves V1 batch projected-gradient updates and these constants:

```text
max_iterations = 5_000
initial_learning_rate = 0.10
learning_rate(iteration) = 0.10 * 0.995 ** floor(iteration / 100)
l2_regularization = 0.01
loss_delta_tolerance = 1e-9
```

It adds the serialized constant `projected_gradient_tolerance = 1e-8`.
After each projected update, the fitter recomputes the objective gradient at
the new point. For the unconstrained intercept, the residual is the absolute
gradient. For each non-negative coefficient with gradient `g` and current
value `w`, the residual is:

```text
abs(g)               when w > 0
max(0, -g)           when w == 0
```

The projected-KKT infinity norm is the maximum residual over all active
parameters. A V2 model is converged only when both the final loss change is at
most `1e-9` and that projected-KKT norm is at most `1e-8`. A low loss change
cannot by itself mark a run converged.

The model serializes final loss delta, projected-KKT norm, frozen parameters,
availability signature, rank diagnostics, iteration count, and convergence
status. Non-finite values or an exhausted budget remain fail-closed in the
future offline orchestrator: they yield no published model/PAVA/prediction.

## 6. Interface Boundaries

`fit_constrained_logistic(examples)` remains the fitting entry point and
returns a `ConstrainedLogisticModel`. V2 extends that immutable result with a
`fit_diagnostics` value. `raw_probability(vector)` validates family, feature
schema, and availability signature before computing the existing raw-score
formula. `predict_offline()` translates an availability mismatch to the stable
reason `prediction_abstained_availability_mismatch`.

PAVA receives only a converged logistic model from the later offline
orchestrator. The current primitive unit tests may inspect a non-converged
model to test diagnostics, but no Phase 5 artifact may publish it as a fitted
calibration model.

## 7. Testing and Experiment Gates

Unit tests must prove all of the following:

- All-unknown optional dimensions produce frozen zero support/risk/missing
  coefficients and are listed in diagnostics.
- A present-vs-unknown mismatch for an `all_unknown` or `all_present`
  dimension abstains rather than scoring outside train availability.
- Mixed availability accepts either state.
- The rank report is deterministic and detects the constant missingness
  columns in the regression fixture.
- The projected-KKT residual handles active positive coefficients and lower
  bound coefficients correctly.
- A low loss change with an above-threshold projected-KKT residual is not
  converged.
- Deterministic serialization contains the new V2 fields and no OOD field.

There is currently no admissible, versioned object-6 collection artifact with
the 20 required Tier A rows. Therefore P5.6C can validate its diagnostic
machinery only against synthetic unit fixtures now. A real Phase 5 run is
required before claiming a 12-row rank, convergence iteration, calibration
quality, or downstream success result.

## 8. Non-Goals

- Do not tune the learning-rate schedule, L2 value, loss tolerance, split
  salt, split quotas, or sign constraints from held-out metrics.
- Do not implement Newton, IRLS, L-BFGS, SciPy, NumPy, or any external
  optimizer in P5.6C.
- Do not pool task families, add OOD features, or use evaluator/post-action
  state as an input feature.
- Do not use calibration or test labels to choose frozen parameters,
  convergence thresholds, or model structure.
