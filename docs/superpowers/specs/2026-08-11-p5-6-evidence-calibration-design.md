# P5.6 Evidence Calibration, Abstention, and Snapshot Activation Design

## 1. Status and purpose

**Status:** Design approved in discussion; pending repository review.

P5.6 turns candidate-conditioned evidence into a qualified, leakage-safe
ranking signal for the CAP-MAS Arbiter. It does not assume that an evidence
score is a probability, and it does not claim a downstream success-rate gain
from calibration alone.

The primary online output of P5.6 is a candidate `rank_score`. A calibrated
`success_probability` is published only when the candidate family, feature
schema, data coverage, uncertainty, and runtime snapshot all pass the
qualification gates. Otherwise the system abstains from calibrated
probability and follows the existing fixed-weight Arbiter path.

P5.6 also closes the missing measurement contract from P5.5 by recording a
verified planned horizon and realized execution horizon. The current P5.5
`max_steps=32` remains an execution budget and must never be interpreted as a
horizon label.

## 2. Context and current boundary

The corrected P5.5 formal suite completed 60 cases across `spatial-0`,
`goal-1`, and `object-6`. Evaluator success was ID `4/30` and OOD `10/30`.
The difference came entirely from `object-6`; both `spatial-0` and `goal-1`
were `0/10` in both splits. Arbitration used `evidence_tie_break` 57/60
times. These results close P5.5 measurement and provenance gates, but they do
not qualify an online calibrated Arbiter.

P5.6 therefore has two parallel lanes with a hard dependency at promotion:

```text
capability diagnosis for spatial-0/goal-1 ----+
                                               +--> family eligibility
calibration infrastructure and offline data --+
                                                       |
                                                       v
                                         shadow --> bounded canary
```

`object-6` is a pipeline smoke family until it meets the same capability and
data gates as every other family. It is not allowed to define a global model
or compensate for a zero-success family.

## 3. Goals and non-goals

### Goals

1. Define a versioned feature and label contract for candidate ranking.
2. Prevent correlated perception, geometry, verifier, rehearsal, and OOD
   signals from being double-counted.
3. Train a constrained offline calibrator only from qualified physical
   outcomes, with explicit uncertainty and abstention.
4. Preserve the safety hard gates, single-owner physical Executor, ActionLease,
   stale-scene checks, and deterministic tie-break behavior.
5. Add leakage-safe splits, family-level qualification, immutable calibration
   snapshots, atomic activation, episode pinning, and rollback.
6. Measure calibration, latency, coverage, horizon behavior, and downstream
   safety before any online promotion claim.

### Non-goals

- Fixing the physical execution capability of `spatial-0` or `goal-1` inside
  the calibration model. P5.6.0 diagnoses and recollects those failures; it
  does not hide them with a learned score.
- Treating P5.5 OOD success as an online feature. Initial `ood_weight=0`.
- Using unselected candidates as physical failures.
- Treating Tier B rehearsal outcomes as physical success labels.
- Adding TSDF, a semantic model, adaptive topology, Memory Skill evolution,
  Robot Skill evolution, RL, or a second physical Executor.
- Claiming that better Brier/ECE or a changed Arbiter winner is a downstream
  task-success improvement.

## 4. Terminology and label semantics

### 4.1 Task and graph outcomes are separate

Every episode records these fields independently:

- `task_success`: the physical benchmark/evaluator result. It is the primary
  calibration label and is `None` when physical execution did not produce a
  conclusive evaluator result.
- `graph_completed`: whether the typed Mission Graph reached its declared
  terminal state.
- `verifier_success`: whether observable predicates accepted the execution;
  it may be `None` when the verifier did not receive conclusive evidence.
- `failure_class`: typed runtime provenance, not a substitute for any label.

The labels are not combined into one `task_success AND graph_completed AND
verifier_success` target. A graph-level postcondition failure can coexist
with evaluator success and is reported as a verifier false negative when the
evaluator result is conclusive.

### 4.2 Candidate execution status

An outcome row contains an explicit `execution_status`:

```python
ExecutionStatus = Literal[
    "selected_executed",
    "selected_not_started",
    "not_selected",
    "rejected_safety",
    "rejected_schema",
    "stale",
    "unknown",
]
```

For `not_selected`, `rejected_*`, and `stale` candidates,
`task_success=None`. They are not physical failures. If a future bounded
top-2 exploration mode selects a candidate with non-zero probability, the
episode must record that selection probability for propensity correction.

### 4.3 Data tiers

P5.6 uses three explicit data tiers:

| Tier | Contents | Calibration use |
| --- | --- | --- |
| A | Physical selected-candidate outcome, feature snapshot captured before execution, and conclusive evaluator label | Primary supervised calibration data |
| B | Isolated process rehearsal outcome and candidate evidence | Diagnostic/shadow analysis and prior feature validation; never a physical-success label |
| C | Candidate evidence without a conclusive outcome | Coverage, drift, and abstention analysis only |

Tier B records use `rehearsal_success`, not `task_success`. Tier C records
remain unlabeled. A record cannot change tier because a field is missing.

## 5. Contracts

P5.6 contracts live in `capmas/contracts/calibration.py` unless an existing
contract is extended without changing its compatibility guarantees. All
contracts are frozen dataclasses or equivalent immutable validated values and
must provide JSON-safe `to_dict()` serialization.

### 5.1 Horizon contract

The horizon has two axes:

```python
@dataclass(frozen=True)
class HorizonLabel:
    planned_critical_path_actions: int | None
    planned_critical_path_subgoals: int | None
    attempted_actions: int | None
    completed_actions: int | None
    attempted_subgoals: int | None
    completed_subgoals: int | None
    planned_source: Literal["mission_graph", "unknown"]
    realized_source: Literal["execution_trace", "unknown"]
    planned_valid: bool
    realized_valid: bool
```

`planned_critical_path_actions` and `planned_critical_path_subgoals` are
computed from the submitted Mission Graph before physical execution. They
describe task complexity. Realized fields are counted from execution trace
events and include retries/recovery according to one fixed event policy. A
runner must not derive either axis from `max_steps`, wall-clock timeout, or
the number of LLM calls.

The canonical complexity buckets are:

```text
H1     planned critical path: 1 action or 1 subgoal
H2-3   planned critical path: 2-3 actions or subgoals
H4-6   planned critical path: 4-6 actions or subgoals
H7+    planned critical path: 7 or more actions or subgoals
```

Reports show action- and subgoal-based buckets separately when they disagree.
Realized attempted/completed counts are diagnostic columns and do not replace
the planned bucket.

### 5.2 Feature snapshot and lineage

Calibration features are captured at the Arbiter decision boundary, after
candidate normalization/repair and before lease acquisition or physical
execution. The record includes:

```python
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
    collection_lane: Literal["physical", "rehearsal", "shadow"]
    features: Mapping[str, float | None]
    feature_status: Mapping[str, Literal["present", "unknown", "invalid"]]
    correlation_groups: Mapping[str, str]
    memory_skill_version: str
    robot_skill_version: str
    selection_probability: float | None = None
```

The feature snapshot is immutable. It must retain source scene/map versions,
provider versions, candidate rewrite metadata, and evidence references through
the existing `CandidateEvidence` objects. Future observations or evaluator
state cannot be joined into a pre-execution feature snapshot.

Unknown is represented by `None` plus `feature_status="unknown"`; it is not
converted to `0.0`. Invalid schema or provenance is a hard qualification
failure, not an unfavorable feature value.

### 5.3 Outcome record

```python
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
    failure_class: str | None
    horizon: HorizonLabel
    feature_snapshot: CandidateFeatureSnapshot
    dataset_split: Literal["train", "calibration", "test", "shadow", "unassigned"]
```

Tier A requires `execution_status="selected_executed"`, a conclusive
`task_success` value, a valid pre-execution feature snapshot, and an episode
identity that does not occur in another split. Tier B and Tier C cannot be
promoted to Tier A by normalization.

### 5.4 Prediction contract

```python
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
```

`rank_score` is the primary calibrated-Arbiter output for a non-abstained
prediction. It is `None` when calibration abstains; the fallback Arbiter owns
the effective score in that case. `success_probability` is non-null only for a
qualified family and a non-abstained prediction. An abstained prediction has a
stable machine-readable reason and must not be presented as a probability of
zero.

The first implementation defines `uncertainty` as the width of the Wilson 95%
interval for the empirical positive rate in the prediction's isotonic block.
It is normalized to `[0, 1]`; no eligible estimate uses `1.0`. The maximum
accepted width is selected from the calibration split, stored in the immutable
snapshot, and never tuned on the test split.

## 6. Evidence reduction and calibration

### 6.1 Layer 1: deterministic correlation control

Each feature declares one `correlation_group`. The initial group registry is:

| Group | Features | Rule |
| --- | --- | --- |
| `scene_grounding` | perception identity, visibility, freshness, grasp quality, reachability, and clearance derived from the same snapshot | Normalize available dimensions, compute a bounded group aggregate, and record coverage |
| `action_feasibility` | pre-execution static verifier and rehearsal signals | Prefer conclusive static verifier evidence; otherwise use rehearsal evidence; do not sum duplicate predictive evidence |
| `distribution_robustness` | OOD replay signals | Disabled in the initial active model; retain for shadow/audit |
| `cost_risk` | latency, recovery, collision, and other risk dimensions | Keep risk positive as a risk value and constrain its model coefficient to be non-positive |

The `CorrelationGroupReducer` computes each active group from non-unknown
features. Version 1 uses equal weights and a contribution cap of `1.0` for
`scene_grounding`; `action_feasibility` uses the declared static-verifier then
rehearsal precedence; `cost_risk` keeps its individual signed risk dimensions.
A group
with no known dimensions returns `unknown`, not zero. The reducer emits the
feature names, provenance, coverage, and cap used so an audit can reproduce
the result. Adding a feature requires assigning it to a group before it can
enter a model.

This deterministic layer is applied before fitting and at inference time. It
prevents static verifier and rehearsal predictions that describe the same
failure from being counted as independent successes. Dynamic post-execution
Verifier results are outcomes and can never enter the pre-execution feature
snapshot. The reducer also makes missingness behavior explicit.

### 6.2 Layer 2: constrained calibration

The initial calibrator is a constrained logistic model over reduced group
features, followed by a separately fitted isotonic calibration transform on
the calibration split. It is trained only on Tier A `task_success` labels.

Constraints:

- evidence/support coefficients are non-negative;
- latency, recovery, collision, and other safety-risk coefficients are
  non-positive;
- coefficients for disabled features are exactly zero;
- unknown group values use an explicit missingness indicator and cannot become
  an implicit zero-quality score;
- model and isotonic transforms are fit only on the training/calibration
  partitions and evaluated once on the locked test partition.

The initial active feature set excludes OOD. Therefore `ood_weight=0` is an
explicit model property, not a hidden omission. OOD support becomes a required
gate only when a later snapshot declares OOD an active feature. A physical
episode in an OOD split may still provide a Tier A `task_success` label if it
meets every Tier A contract, but the split identity and aggregate
`ood_success_rate` cannot enter the feature vector. Replay-only robustness
evidence remains Tier B/C shadow metadata and cannot affect rank or
probability.

Calibration is family-scoped. A family is the locked evaluation family ID
declared by the experiment manifest; P5.6 does not silently pool
`spatial-0`, `goal-1`, and `object-6` into one model. A global fallback may
serve an ineligible family, but it cannot publish a qualified probability for
that family.

## 7. Capability, qualification, and abstention gates

### 7.1 Capability gate

A family is capable for calibration collection only if a fixed ten-seed
diagnostic run has:

1. zero infrastructure-unknown cases;
2. typed failure provenance for every unsuccessful case;
3. at least 80% of cases reaching physical execution; and
4. at least one valid evaluator success.

Failure of this gate blocks calibrated online promotion for the family. It
does not block infrastructure development or diagnosis. A zero-success family
must be repaired or recollected before it can qualify.

### 7.2 Calibration eligibility gate

A family can fit and publish a qualified calibrator only with at least 20
independent Tier A physical outcomes, including at least five positive and
five negative `task_success` labels. The split is by episode and locked
provenance, not by individual candidate rows from the same episode. A family
that does not meet this gate must abstain and use fixed-weight fallback.

The gate is checked at snapshot build time and again at activation time.

### 7.3 Runtime abstention conditions

The calibrated path abstains for any of the following:

- family capability or calibration coverage is insufficient;
- stale scene/map/candidate evidence;
- feature schema, Memory Skill, or Robot Skill version mismatch;
- candidate fingerprint mismatch after normalization or repair;
- required active feature is missing or invalid;
- OOD support failure when OOD is active in the snapshot;
- uncertainty above threshold;
- top-two rank margin below threshold;
- evidence conflict that violates the group policy;
- calibration inference timeout or snapshot load failure.

Abstention uses this fallback route:

```text
safety hard gates
  -> qualified calibrated ranking
  -> fixed-weight evidence Arbiter
  -> deterministic evidence tie-break
  -> confidence fallback only when no candidate evidence exists
```

The fallback never bypasses stale checks, schema validation, Geometry gates,
Verifier gates, ActionLease, or single-owner execution. The emitted decision
records whether the reason was `evidence_tie_break` or
`confidence_fallback`; these are not interchangeable.

## 8. Calibration snapshot lifecycle

`CalibrationSnapshot` is immutable and content-addressed:

```python
@dataclass(frozen=True)
class CalibrationSnapshot:
    snapshot_id: str
    content_sha256: str
    feature_schema_version: str
    dataset_id: str
    train_split_id: str
    calibration_split_id: str
    test_split_id: str
    family_ids: tuple[str, ...]
    model_type: Literal["constrained_logistic_isotonic"]
    coefficients: Mapping[str, float]
    intercept: float
    isotonic_x: tuple[float, ...]
    isotonic_y: tuple[float, ...]
    isotonic_interval_widths: tuple[float, ...]
    abstention_thresholds: Mapping[str, float]
    group_policy_version: str
    calibration_method_version: str
    memory_skill_version: str
    robot_skill_version: str
    ood_feature_enabled: bool
    metrics: Mapping[str, float]
    qualification: Mapping[str, object]
    created_at_utc: str
```

The digest is computed over canonical content with `content_sha256` omitted.
The snapshot registry provides:

```python
publish(snapshot) -> snapshot_id
activate(snapshot_id, expected_previous_id=None) -> None
active(family_id) -> CalibrationSnapshot | None
pin(episode_id, snapshot_id) -> None
rollback(family_id, snapshot_id) -> None
```

Activation is atomic and validates all version and qualification fields.
Each episode pins one snapshot at initialization; background fitting cannot
mutate the active pointer or change a snapshot already pinned by an episode.
Rollback is explicit, auditable, and itself creates an activation event. A
failed load or digest mismatch causes an abstention and fixed-weight fallback,
not a partially loaded model.

## 9. Evaluation and promotion gates

P5.6 reports ranking and calibration separately from physical task success.
For Brier comparison, the baseline is the existing fixed-weight Arbiter score
passed through a frozen monotonic probability mapping fit on the same training
and calibration partitions; raw fixed-weight scores are never treated as
probabilities. The calibrated model and baseline are compared once on the same
locked test partition:

- Brier score improvement of at least 10% relative to the baseline mapping;
- Expected Calibration Error (ECE) at most `0.10`;
- per-family and horizon-bucket coverage, uncertainty, and abstention rates;
- no safety hard-gate disagreement;
- ranking metrics and tie-rate distribution;
- no leakage-audit violations.

The shadow Arbiter gate requires:

- zero hard-gate disagreement;
- calibration inference P95 at most 5 ms, measured separately from perception,
  geometry, and LLM latency;
- eligible calibrated coverage at least 50% of evaluated decisions;
- complete prediction and abstention reason artifacts.

The bounded online canary requires at least 20 matched physical episodes for
an eligible family. It is a safety and operability gate only; it is not a
downstream improvement claim. Any hard-gate disagreement, snapshot integrity
failure, unexplained schema mismatch, or budget violation aborts the canary
and rolls back the snapshot.

No global online calibrated Arbiter is enabled while any target family is
still ineligible. A family-scoped canary may run only for families that pass
both capability and calibration gates, with fixed-weight fallback for all
others.

## 10. Data splits and leakage prevention

The dataset builder creates immutable train, calibration, test, and shadow
partitions. The unit of splitting is an episode lineage group containing its
episode ID, seed, layout pair, candidate artifact, and all retries. Related
ID/OOD pairs and retries that share one lineage group are assigned to the same
partition. They cannot cross train, calibration, and test boundaries.

The builder rejects duplicate episode IDs, candidate fingerprint collisions
across forbidden partitions, future-state features, evaluator-derived
pre-execution fields, missing provenance, and a Tier B/C row marked as a Tier A
label. Every dataset records a canonical manifest digest and the versions of
the feature schema, Memory Skill, Robot Skill, prompts, environment, and code.

## 11. P5.6 implementation order

```text
P5.6.0  Diagnose and recollect spatial-0/goal-1 capability
P5.6.1  Add horizon, outcome-label, and version-lineage contracts
P5.6.2  Build three-tier dataset and leakage-safe split audit
P5.6.3  Implement deterministic correlation-group reduction
P5.6.4  Implement constrained logistic and isotonic calibration
P5.6.5  Implement immutable snapshot registry and episode pinning
P5.6.6  Add offline metrics, ablations, and qualification reports
P5.6.7  Add shadow Arbiter integration with abstention/fallback
P5.6.8  Recollect eligible families and run bounded online canary
P5.6.9  Run formal matched evaluation and write the Phase 6 handoff
```

P5.6.0 and P5.6.1--P5.6.6 may proceed in parallel when their interfaces are
stable. P5.6.7 must consume only immutable snapshots. P5.6.8 is blocked per
family until both qualification gates pass. P5.6.9 must report all eligible
and ineligible families separately.

## 12. Module boundaries

The first implementation should keep the following responsibilities separate:

| Module | Responsibility |
| --- | --- |
| `capmas/contracts/calibration.py` | Immutable contracts, validation, and serialization |
| `capmas/evaluation/labels.py` | Tier assignment, outcome normalization, and horizon extraction |
| `capmas/evaluation/dataset.py` | Lineage-aware dataset construction and leakage audit |
| `capmas/evaluation/correlation.py` | Correlation-group registry and deterministic reduction |
| `capmas/evaluation/calibration.py` | Constrained fitting, isotonic transform, prediction, and abstention |
| `capmas/evaluation/snapshot_registry.py` | Content-addressed publish, atomic activation, pin, rollback |
| `capmas/evaluation/calibrated_arbiter.py` | Safety gates, qualified ranking, fallback, and decision artifacts |
| `scripts/run_p56_capability.py` | Per-family diagnosis and recollection artifacts |
| `scripts/run_p56_offline.py` | Offline fit, metrics, ablations, and qualification report |
| `scripts/run_p56_shadow.py` | Shadow comparison without physical execution |
| `scripts/run_p56_canary.py` | Bounded eligible-family online canary |

These modules must not move physical execution into calibration, mutate the
active Memory/Robot Skill registries, or import evaluator state into the
realistic evidence path.

## 13. Testing and acceptance

### Contract tests

- reject negative or inconsistent horizon counts;
- preserve `None` for unknown outcomes and unselected candidates;
- reject Tier B/C masquerading as Tier A;
- validate canonical snapshot digests and schema/version bindings;
- serialize and deserialize all contracts without secret material.

### Calibration tests

- deterministic group reduction never double-counts a correlation group;
- unknown dimensions produce missingness/abstention, not zero-quality scores;
- coefficient constraints are enforced;
- disabled OOD weight remains exactly zero;
- train/calibration/test separation is preserved;
- isotonic calibration is fit only on its declared split;
- confidence and uncertainty outputs are reproducible for a frozen snapshot.

### Runtime tests

- stale evidence and schema mismatch fall back before selection;
- calibration cannot bypass a safety hard gate;
- low margin and high uncertainty abstain;
- `evidence_tie_break` and `confidence_fallback` remain distinct;
- episode pinning gives stable results while a new snapshot is published;
- atomic activation and explicit rollback are auditable;
- calibrated inference timeout leaves the physical Executor untouched.

### Empirical acceptance

The P5.6 code gate requires focused tests, the full regression suite,
`compileall`, `git diff --check`, and artifact manifest verification. The
empirical gate additionally requires capability and calibration eligibility for
each claimed family, offline Brier/ECE targets, shadow gates, and a bounded
canary. If a family fails capability or coverage, the report must say
`ineligible` and cannot publish a qualified probability for it.

## 14. Explicit handoff to later phases

P5.6 does not activate OOD weighting by default. A later OOD feature proposal
must supply independent counterfactual multi-layout support, a revised feature
schema, a new snapshot, and a new calibration/ablation report.

Adaptive topology, Memory Skill evolution, Robot Skill evolution, and Phase 6
controller learning consume P5.6 artifacts only through immutable versioned
interfaces. None may modify a pinned episode or retroactively relabel its
features.
