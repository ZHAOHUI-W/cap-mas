# P5.6B Offline Calibration Foundation Design

## 1. Status and decision

**Status:** Approved design for implementation planning.

P5.6B implements the offline foundation for P5.6.3 correlation reduction and
P5.6.4 constrained calibration. It consumes the 20 qualified object-6 Tier A
outcomes collected by P5.6A. Its purpose is to verify feature lineage,
deterministic fitting, calibration, reproducibility, and leakage prevention.

This is not an online-ranking experiment. P5.6B must not activate a
calibrated probability, alter Arbiter selection, replace `evidence_tie_break`,
start a shadow run, or execute a robot action. Its metrics are descriptive
only; the 20 outcomes do not support a Brier/ECE improvement or downstream
success-rate claim.

The implementation adds no NumPy, SciPy, pandas, scikit-learn, or other
runtime dependency. All fitting is deterministic pure Python.

## 2. Scope and inputs

### In scope

- Deterministic reduction of correlated decision-time evidence.
- A family-scoped constrained logistic model.
- PAVA isotonic calibration on a separate locked split.
- Wilson interval width per isotonic block.
- Immutable JSON artifacts for reduced rows, the fitted model, predictions,
  metrics, and audit.
- Focused unit and integration tests.

### Out of scope

- Changing `spatial-0` or `goal-1` task capability; those remain P5.3.2 work.
- New data collection, new physical execution, or a capability repair.
- OOD as an active feature; `ood_weight` remains exactly `0.0`.
- Snapshot publication, registry activation, episode pinning, shadow Arbiter,
  canary execution, or changes to the active Arbiter path.
- TSDF, semantic adapters, topology adaptation, RL, or skill evolution.

### Admissible data

The input is a `CalibrationDatasetManifest` built only from `family_id ==
"object-6"` rows that pass the existing `audit_calibration_dataset()` checks.
The supervised set contains only Tier A rows with:

- `execution_status == "selected_executed"`;
- `collection_lane == "physical"`;
- a conclusive boolean `task_success`;
- an immutable decision-time `CandidateFeatureSnapshot`; and
- a distinct valid `CalibrationLineage`.

Tier B rehearsal outcomes remain non-label diagnostics. Tier C candidates,
unselected candidates, evaluator-derived feature fields, dynamic verifier
results, and post-action observations are rejected before reduction.

The P5.6A collection has 20 Tier A rows with ten positive and ten negative
physical labels. The current rows predominantly expose
`rehearsal_success_rate`; perception, geometry, and cost-risk dimensions are
explicitly `unknown`. P5.6B must support this valid partial-evidence case. It
must not manufacture scene, geometry, or risk values in order to fit a model.

## 3. Locked partitions and leakage controls

The splitting unit is `lineage_group_id`, never individual candidate rows.
All retries, layout pairs, and episodes sharing a lineage group must remain in
one partition.

P5.6B introduces an exact-quota lineage splitter for this fixed 20-row run:

```text
12 lineage groups -> train
 4 lineage groups -> calibration
 4 lineage groups -> test
```

It orders unique lineage IDs by `SHA-256("p56b-object6-split-v1:" +
lineage_group_id)` and assigns the first 12, next 4, and final 4 to the three
partitions. The salt, ordered lineage IDs, assignments, source manifest digest,
and split counts are recorded in the output artifact. The splitter is based
only on lineage identity, not on `task_success`, candidate evidence, or any
fitted result.

The fit gate requires both labels in `train` and `calibration`. If the locked
partition does not satisfy that condition, P5.6B emits a fail-closed report
and no model. The test partition may have one observed class, but its metrics
must be labelled descriptive and its class counts must be reported. Changing
the salt after observing metrics is prohibited; a replacement split requires a
new approved design and a new manifest identifier.

The approved salt was read-only verified against the P5.6A source manifests:
`train` contains five positive and seven negative labels, `calibration`
contains two positive and two negative labels, and `test` contains three
positive and one negative label. These counts are provenance checks, not model
quality metrics.

No training function may receive rows outside `train`. No isotonic function
may receive rows outside `calibration`. `test` can only receive the frozen
model and transform for prediction and report generation.

## 4. Correlation reduction

`capmas/evaluation/correlation.py` owns the reduction policy. It accepts only
`CandidateFeatureSnapshot` values with `feature_status` equal to `present` or
`unknown`; `invalid` is a hard error.

The version-one reducer produces these reduced dimensions:

| Reduced dimension | Source policy | Availability |
| --- | --- | --- |
| `scene_grounding` | Equal-weight mean of available scene/geometry support dimensions, capped at 1.0 | Optional |
| `action_feasibility` | Static verifier pass rate when its coverage is present and positive; otherwise rehearsal success rate | Required for P5.6B fit |
| `collision_risk` | Candidate-conditioned geometry risk in `[0, 1]` | Optional |
| `expected_latency_risk` | `1 - exp(-expected_latency_ms / 1000.0)` | Optional |
| `recovery_cost_risk` | `1 - exp(-recovery_cost)` | Optional |

`scene_grounding` includes only features assigned to the existing
`scene_grounding` correlation group. `action_feasibility` uses precedence,
rather than summing static-verifier and rehearsal evidence. This prevents two
measurements of the same predicted failure from being counted twice. The three
cost-risk dimensions remain separate because they have different units and
their sign is safety-negative.

For every reduced dimension, the reducer records source feature names,
available-feature coverage, policy version, and a `present`/`unknown` status.
No all-unknown group becomes `0.0`. A record whose required
`action_feasibility` dimension is unknown is excluded from fitting and reported
as `abstained_missing_required_evidence`; it is never relabelled or silently
imputed.

Optional unknown dimensions are represented as an omitted evidence
contribution plus an explicit missingness indicator. This is not a quality
score of zero: the reduction artifact still contains `value: null` and
`status: "unknown"`, while the model records that the dimension was absent.

## 5. Constrained logistic model

`capmas/evaluation/calibration.py` owns fitting and prediction. The model is
family-scoped and has model type `constrained_logistic_isotonic`.

For a reduced row, the raw score is:

```text
z = intercept
    + sum(support_weight[d] * value[d] for present support dimensions)
    - sum(risk_weight[d] * value[d] for present risk dimensions)
    - sum(missing_penalty[d] for unknown optional dimensions)
probability_raw = sigmoid(z)
```

All support weights, risk weights, and missing penalties are non-negative.
The formula makes support evidence monotonic non-decreasing and risk or
missingness monotonic non-increasing. Disabled OOD has no reduced dimension,
no coefficient, and no missingness indicator.

The implementation uses batch projected-gradient descent over binary
cross-entropy with fixed L2 regularization. Version-one constants are part of
the serialized model:

```text
max_iterations = 5_000
initial_learning_rate = 0.10
learning_rate(iteration) = 0.10 * 0.995 ** floor(iteration / 100)
l2_regularization = 0.01
convergence_tolerance = 1e-9
```

The iteration budget is an upper bound, not a convergence claim. With the
fixed schedule, the versioned six-row regression fixture reaches a final
learning rate of `0.0909156262` and a final single-step loss change of
`9.881077e-7` at iteration 2,000; at iteration 5,000 those values are
`0.0782223675` and `1.219500e-7`. Both loss changes exceed the `1e-9`
tolerance, so that fixture correctly remains non-converged and only verifies
the configured budget. The worktree contains no eligible, versioned 12-row
object-6 training artifact from which to reproduce a separate convergence
claim. A Phase 5 offline run must record that measurement before the
specification states an iteration-to-convergence result for the locked train
split. This revision changes no optimizer, schedule, L2 value, tolerance, sign
constraint, or pure-Python boundary.

The initial intercept is the logit of the smoothed train positive rate,
`(positive_count + 0.5) / (train_count + 1.0)`. All non-intercept parameters
start at zero. After each update, the implementation projects every constrained
parameter to `[0, infinity)`. The run records iteration count, final loss,
convergence status, constants, feature order, and model digest. A non-finite
input, non-finite loss, failed convergence, or missing train class emits no
model and a stable fail-closed reason.

## 6. Isotonic calibration and uncertainty

The logistic model is frozen before it sees the calibration partition. It
scores the four calibration rows, which are then used to fit a deterministic
Pool Adjacent Violators Algorithm (PAVA) transform. The implementation keeps
ascending raw-score blocks; adjacent blocks with decreasing empirical positive
rates are merged. Each final block stores:

- Its lower and upper raw-score boundary.
- Its calibrated empirical rate.
- Its sample count and positive count.
- Its Wilson 95% interval width, using a frozen two-sided normal constant of
  `1.959963984540054`.

A prediction first uses the frozen logistic model and then the frozen PAVA
step function. It returns an offline `CalibrationPrediction` plus a reduction
audit. The prediction's uncertainty is the normalized Wilson interval width of
the selected PAVA block. A row with unavailable required evidence, incompatible
schema/version, or a model/transform integrity mismatch abstains with a stable
reason. Its `success_probability` is `None`, never `0.0`.

P5.6B writes predictions only to offline artifacts. It does not instantiate a
`CalibrationSnapshot`, and it does not publish a probability to the runtime.

## 7. Artifacts and public interfaces

The new modules expose small immutable values rather than a runtime service:

```python
reduce_feature_snapshot(snapshot) -> ReducedFeatureVector
fit_constrained_logistic(rows) -> ConstrainedLogisticModel
fit_isotonic(model, calibration_rows) -> IsotonicCalibration
predict_offline(model, isotonic, row) -> CalibrationPrediction
run_offline_calibration(manifest, split_config) -> OfflineCalibrationReport
```

`OfflineCalibrationReport` contains source and output digests, split audit,
row eligibility counts, reduced-row provenance, model metadata, PAVA blocks,
per-split predictions, Brier/ECE values where defined, coverage, abstention
reasons, and an explicit statement that the report has no online effect.

The report is JSON-safe and content-addressed. It contains no secret material
and does not persist raw RGB-D data. A planned `scripts/run_p56_offline.py`
will create a new Phase 5 run directory and preserve its complete log,
configuration, input manifest references, and report artifacts.

## 8. Failure handling

The following conditions terminate fitting or abstain prediction without a
fallback model being silently substituted:

- Invalid dataset audit, non-object-6 family, non-Tier-A label, or duplicate
  lineage across partitions.
- Source manifest, feature schema, Memory Skill, or Robot Skill mismatch.
- Forbidden future/evaluator/dynamic-verifier/OOD feature.
- Missing required `action_feasibility` evidence.
- Absent positive or negative class in a fitting partition.
- Non-finite value, unstable optimizer, invalid PAVA input, or artifact digest
  mismatch.

The output must distinguish `fit_rejected_*` from `prediction_abstained_*`.
It must never replace unknown evidence with a score, reinterpret an
unexecuted candidate as a failed physical outcome, or affect physical
execution.

## 9. Tests and acceptance

Tests will cover:

- Exact 12/4/4 deterministic lineage assignment and no lineage leakage.
- Reducer precedence, cap, transformations, provenance, and no double count.
- Preservation of all-unknown optional groups and rejection of invalid or
  future-state input.
- Projected logistic signs, determinism, finite-output checks, and fit gate.
- PAVA monotonicity, calibration-split isolation, Wilson-width reproducibility,
  and no test-set fitting.
- End-to-end reconstruction of the object-6 manifest into a report with no
  Arbiter, Executor, or CAP-X environment invocation.

Completion requires focused tests, the full regression suite, `compileall`,
`git diff --check`, and a manifest-integrity verification of every input and
output artifact. The result may say only that the offline calibration pipeline
is functional, deterministic, and leakage-audited. It must not state that
calibration improves probability quality or downstream robot success.

## 10. Handoff

P5.6B unlocks only implementation planning for P5.6.5 snapshot lifecycle,
P5.6.6 offline qualification reporting, and eventually P5.6.7 shadow
integration. Each requires a separate approved design. A model built here is
not eligible for activation merely because it fits or produces a report.
