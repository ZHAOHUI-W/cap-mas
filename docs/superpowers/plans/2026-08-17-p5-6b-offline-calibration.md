# P5.6B Offline Calibration Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, leakage-audited, object-6-only offline
calibration pipeline that produces reduced evidence, a constrained logistic
model, PAVA calibration, and immutable Phase 5 artifacts without changing the
runtime Arbiter or physical execution.

**Architecture:** `correlation.py` reduces only decision-time feature snapshots
into auditable group dimensions. `calibration.py` fits pure-Python constrained
logistic and PAVA transforms. `offline.py` validates the existing dataset,
creates the locked exact-quota lineage partition, orchestrates fit/prediction,
and emits a JSON-safe report. The CLI performs input reconstruction and Phase 5
artifact persistence only; it must never import a LIBERO environment or invoke
an Executor.

**Tech Stack:** Python 3.10+, standard library only (`dataclasses`, `hashlib`,
`json`, `math`, `statistics`, `pathlib`), pytest, existing CAP-MAS contracts,
and `Phase5RunDirectory`.

## Global Constraints

- Add no NumPy, SciPy, pandas, scikit-learn, or other runtime dependency.
- Accept only object-6 Tier A physical selected-execution outcomes as labels.
- Preserve `unknown` as `None` plus explicit status; never encode it as a
  feature value of `0.0`.
- Keep OOD disabled: no OOD reduced feature, coefficient, or label use.
- Dynamic verifier and post-action/evaluator state must never enter an input
  vector.
- Lock exact lineage quotas at `12 train / 4 calibration / 4 test` using salt
  `p56b-object6-split-v1`; never select or change the salt from metrics.
- Fit logistic only on train, PAVA only on calibration, and use test only for
  frozen-model prediction/reporting.
- Emit fail-closed `fit_rejected_*` or `prediction_abstained_*` reasons.
- Do not instantiate `CalibrationSnapshot`, modify the Arbiter, run shadow or
  canary selection, or call a physical Executor.
- Each experiment must use `Phase5RunDirectory`, preserve `logs/runner.log`,
  finalize its manifest, and pass `verify_phase5_manifest.py`.

---

## File Structure

| Path | Responsibility |
| --- | --- |
| `capmas/evaluation/correlation.py` | Immutable reduced-feature values and deterministic correlation-group reduction. |
| `capmas/evaluation/calibration.py` | Pure-Python constrained logistic fitting, PAVA, Wilson uncertainty, metrics, and offline prediction. |
| `capmas/evaluation/offline.py` | Exact-quota lineage partition, source-manifest validation, fit orchestration, and JSON-safe report. |
| `capmas/evaluation/__init__.py` | Public exports for the new offline-only API. |
| `scripts/run_p56_offline.py` | File I/O, source collection reconstruction, Phase 5 output creation, and manifest verification. |
| `tests/test_p56_correlation.py` | Unit tests for reduction rules and unknown/error behavior. |
| `tests/test_p56_calibration.py` | Unit tests for optimizer constraints, PAVA, Wilson widths, and metrics. |
| `tests/test_p56_offline.py` | Split leakage and end-to-end in-memory offline orchestration tests. |
| `tests/test_run_p56_offline.py` | CLI artifact and no-runtime-side-effect integration tests. |
| `docs/implementation-roadmap.md` | P5.6B implementation status and follow-on boundary. |
| `docs/phase5-evidence-evolution.md` | Offline calibration artifact and no-online-effect record. |
| `docs/experiments.md` | Reproducible local command and required run artifacts. |

### Task 1: Deterministic Correlation-Group Reduction

**Files:**

- Create: `capmas/evaluation/correlation.py`
- Create: `tests/test_p56_correlation.py`
- Modify: `capmas/evaluation/__init__.py`

**Interfaces:**

- Consumes: `CandidateFeatureSnapshot` and `FEATURE_GROUPS_V1`.
- Produces: `ReducedDimension`, `ReducedFeatureVector`,
  `reduce_feature_snapshot(snapshot) -> ReducedFeatureVector`.
- Depends on no model, dataset split, evaluator, Arbiter, or environment.

- [ ] **Step 1: Write the failing reduction tests**

```python
def test_reducer_prefers_static_verifier_over_rehearsal() -> None:
    snapshot = _snapshot(
        static_verifier_pass_rate=0.25,
        static_verifier_coverage=1.0,
        rehearsal_success_rate=1.0,
    )
    reduced = reduce_feature_snapshot(snapshot)
    feasibility = reduced.dimension("action_feasibility")
    assert feasibility.value == 0.25
    assert feasibility.sources == ("static_verifier_pass_rate",)


def test_reducer_preserves_unknown_optional_group_without_zero_imputation() -> None:
    reduced = reduce_feature_snapshot(_snapshot(rehearsal_success_rate=1.0))
    scene = reduced.dimension("scene_grounding")
    assert scene.value is None
    assert scene.status == "unknown"
    assert scene.coverage == 0.0
```

Add cases for equal-weight scene aggregation, static verifier coverage `0.0`
falling back to rehearsal, a missing required feasibility value, invalid source
status, out-of-range support/risk values, latency/recovery transforms, and
deterministic `to_dict()` ordering.

- [ ] **Step 2: Run the reduction tests to verify failure**

Run: `pytest tests/test_p56_correlation.py -q`

Expected: collection/import failure because `capmas.evaluation.correlation`
does not exist.

- [ ] **Step 3: Implement the minimal reduction module**

```python
@dataclass(frozen=True)
class ReducedDimension:
    name: str
    value: float | None
    status: Literal["present", "unknown"]
    sources: tuple[str, ...]
    coverage: float


@dataclass(frozen=True)
class ReducedFeatureVector:
    episode_id: str
    candidate_id: str
    candidate_fingerprint: str
    family_id: str
    feature_schema_version: str
    dimensions: tuple[ReducedDimension, ...]
    policy_version: str = "p56b.reduction.v1"
```

Implement `ReducedFeatureVector.dimension(name)` by requiring exactly one
matching dimension and `to_dict()` by emitting sorted JSON-safe primitive
fields. Implement `reduce_feature_snapshot(snapshot)` with the policy helpers
below and return dimensions in this fixed order: `scene_grounding`,
`action_feasibility`, `collision_risk`, `expected_latency_risk`, and
`recovery_cost_risk`.

Implement exact V1 policies:

```python
def _action_feasibility(features: Mapping[str, float | None], statuses: Mapping[str, str]) -> ReducedDimension:
    if statuses["static_verifier_coverage"] == "present" and features["static_verifier_coverage"] > 0.0:
        return _present("action_feasibility", features["static_verifier_pass_rate"], ("static_verifier_pass_rate",))
    return _dimension_from_optional("action_feasibility", features["rehearsal_success_rate"], statuses["rehearsal_success_rate"], ("rehearsal_success_rate",))


def _latency_risk(milliseconds: float) -> float:
    return 1.0 - math.exp(-milliseconds / 1000.0)


def _recovery_risk(cost: float) -> float:
    return 1.0 - math.exp(-cost)
```

Reject a declared `invalid` source status, a status/value mismatch, any support
or collision value outside `[0, 1]`, and negative latency/recovery values.
Keep all optional all-unknown dimensions as `value=None`; do not produce an OOD
dimension. Export the API from `capmas.evaluation`.

- [ ] **Step 4: Run focused tests and static checks**

Run: `pytest tests/test_p56_correlation.py -q`

Expected: all new reduction tests pass.

Run: `ruff check capmas/evaluation/correlation.py tests/test_p56_correlation.py`

Expected: exit code `0` for touched files.

- [ ] **Step 5: Commit the independently testable reducer**

```bash
git add capmas/evaluation/correlation.py capmas/evaluation/__init__.py tests/test_p56_correlation.py
git commit -m "feat: add P5.6B evidence reduction"
```

### Task 2: Locked Exact-Quota Offline Partition

**Files:**

- Create: `capmas/evaluation/offline.py`
- Create: `tests/test_p56_offline.py`
- Modify: `capmas/evaluation/__init__.py`

**Interfaces:**

- Consumes: an already audit-passing `CalibrationDatasetManifest` and
  `ReducedFeatureVector` from Task 1.
- Produces: `ExactQuotaSplitConfig`, `OfflineExample`, and
  `partition_tier_a_outcomes(manifest, config) -> tuple[OfflineExample, ...]`.
- Does not mutate `CalibrationOutcome.dataset_split`; the source manifest's
  legacy/default split is only a provenance audit and is not reused for P5.6B.

- [ ] **Step 1: Write the failing partition tests**

```python
def test_exact_quota_partition_is_stable_and_keeps_all_lineages_together() -> None:
    manifest = _twenty_row_manifest()
    config = ExactQuotaSplitConfig.object6_v1()

    first = partition_tier_a_outcomes(manifest, config)
    second = partition_tier_a_outcomes(manifest, config)

    assert first == second
    assert Counter(row.dataset_split for row in first) == {
        "train": 12,
        "calibration": 4,
        "test": 4,
    }
    assert {row.dataset_split for row in first if row.episode_id == "episode-1"} == {"train"}


def test_partition_rejects_wrong_family_and_duplicate_tier_a_lineage() -> None:
    with pytest.raises(ValueError, match="object-6"):
        partition_tier_a_outcomes(_manifest(family_id="goal-1"), ExactQuotaSplitConfig.object6_v1())
```

Build the synthetic manifest with ten positive and ten negative Tier A rows,
plus optional Tier C rows. Add cases for a failed `audit_calibration_dataset`,
20-row quota mismatch, missing lineage, duplicate Tier A selected outcome in a
lineage, and `train`/`calibration` partitions missing either class.

- [ ] **Step 2: Run the partition tests to verify failure**

Run: `pytest tests/test_p56_offline.py -q`

Expected: import failure because `capmas.evaluation.offline` does not exist.

- [ ] **Step 3: Implement immutable configuration and partitioning**

```python
@dataclass(frozen=True)
class ExactQuotaSplitConfig:
    family_id: str
    split_salt: str
    train_count: int
    calibration_count: int
    test_count: int
    splitter_version: str = "p56b.exact_quota_sha256.v1"

    @classmethod
    def object6_v1(cls) -> "ExactQuotaSplitConfig":
        return cls("object-6", "p56b-object6-split-v1", 12, 4, 4)


@dataclass(frozen=True)
class OfflineExample:
    outcome: CalibrationOutcome
    dataset_split: Literal["train", "calibration", "test"]
    lineage_group_id: str
    reduced: ReducedFeatureVector | None = None


def partition_tier_a_outcomes(
    manifest: CalibrationDatasetManifest,
    config: ExactQuotaSplitConfig,
) -> tuple[OfflineExample, ...]:
    audit = audit_calibration_dataset(manifest)
    assert_dataset_eligible(audit)
    return _partition_validated_tier_a_rows(manifest, config)
```

Call `assert_dataset_eligible(audit_calibration_dataset(manifest))` first.
Filter only qualifying object-6 Tier A selected physical rows. Join every row
to exactly one lineage. Sort unique group IDs with:

```python
key = lambda group_id: hashlib.sha256(
    f"{config.split_salt}:{group_id}".encode("utf-8")
).hexdigest()
```

Require exactly `train_count + calibration_count + test_count` groups, allocate
in sorted order, and verify both labels appear in the train and calibration
partitions. Provide `to_dict()` for config and examples with source lineage
identity; preserve the original `CalibrationOutcome` unchanged. Export the
partition API from `capmas.evaluation`.

- [ ] **Step 4: Run partition tests and regression guards**

Run: `pytest tests/test_p56_offline.py -q`

Expected: all partition tests pass.

Run: `pytest tests/test_p56_dataset.py tests/test_p56_contracts.py -q`

Expected: existing split audit and immutable contract tests still pass.

- [ ] **Step 5: Commit the offline split boundary**

```bash
git add capmas/evaluation/offline.py capmas/evaluation/__init__.py tests/test_p56_offline.py
git commit -m "feat: add P5.6B exact offline partition"
```

### Task 3: Constrained Logistic, PAVA, and Offline Metrics

**Files:**

- Create: `capmas/evaluation/calibration.py`
- Create: `tests/test_p56_calibration.py`
- Modify: `capmas/evaluation/__init__.py`

**Interfaces:**

- Consumes: Task 1 `ReducedFeatureVector` values and Task 2 labelled
  `OfflineExample` values.
- Produces: `ConstrainedLogisticModel`, `IsotonicBlock`,
  `IsotonicCalibration`, `fit_constrained_logistic`, `fit_isotonic`,
  `predict_offline`, `brier_score`, and `expected_calibration_error`.
- Never opens files, partitions examples, calls an Arbiter, or accesses a
  simulator.

- [ ] **Step 1: Write failing model, calibration, and metric tests**

```python
def test_projected_fit_keeps_support_and_risk_weights_nonnegative() -> None:
    model = fit_constrained_logistic(_train_examples())

    assert all(weight >= 0.0 for weight in model.support_weights.values())
    assert all(weight >= 0.0 for weight in model.risk_weights.values())
    assert all(weight >= 0.0 for weight in model.missing_penalties.values())


def test_pava_merges_descending_empirical_rates() -> None:
    calibration = fit_isotonic(
        _model(),
        _examples_with_scores([(0.1, True), (0.2, False), (0.3, True)]),
    )

    assert [(block.lower, block.upper, block.probability) for block in calibration.blocks] == [
        (0.1, 0.2, 0.5),
        (0.3, 0.3, 1.0),
    ]
```

Add tests for deterministic refitting, smoothed intercept, wrong split input,
only-one-class rejection, non-finite vectors, no OOD coefficient, all required
evidence missing producing an abstained `CalibrationPrediction`, fixed ten-bin
ECE, and a Wilson width in `[0, 1]`.

- [ ] **Step 2: Run the model tests to verify failure**

Run: `pytest tests/test_p56_calibration.py -q`

Expected: import failure because `capmas.evaluation.calibration` does not exist.

- [ ] **Step 3: Implement the pure-Python model and transform**

```python
@dataclass(frozen=True)
class ConstrainedLogisticModel:
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


@dataclass(frozen=True)
class IsotonicBlock:
    lower: float
    upper: float
    probability: float
    sample_count: int
    positive_count: int
    uncertainty: float


@dataclass(frozen=True)
class IsotonicCalibration:
    calibration_version: str
    feature_schema_version: str
    blocks: tuple[IsotonicBlock, ...]
```

Implement `ConstrainedLogisticModel.raw_probability(vector)` with a numerically
stable sigmoid and its signed contribution policy. Implement
`IsotonicCalibration.calibrate(raw_probability)` by selecting the containing
PAVA block, with the first/last block covering scores outside the observed
range. Both values provide deterministic sorted `to_dict()` methods.

The module exports these exact call signatures:
`fit_constrained_logistic(examples: Sequence[OfflineExample]) ->
ConstrainedLogisticModel`, `fit_isotonic(model: ConstrainedLogisticModel,
examples: Sequence[OfflineExample]) -> IsotonicCalibration`, and
`predict_offline(model: ConstrainedLogisticModel, isotonic:
IsotonicCalibration, example: OfflineExample) -> CalibrationPrediction`.

Accept only `train` examples in logistic fitting and only `calibration`
examples in PAVA fitting. For each update compute binary-cross-entropy plus
`0.01 * sum(weight * weight)`, use the fixed learning-rate schedule from the
approved spec, then project support/risk/missing parameters using
`max(0.0, value)`. The score must add support contributions and subtract risk
or unknown-optional penalties. A required feasibility unknown returns an
abstained prediction with `success_probability=None`.

Implement PAVA by sorting `(raw_probability, label)` pairs, combining tied
scores before adjacent-rate pooling, and repeatedly merging blocks while the
left rate exceeds the right rate. Calculate Wilson width with
`z = 1.959963984540054`; retain the raw width because it is already bounded.
Use fixed ten equal-width bins for ECE and report `None` for an empty metric
input. Successful offline predictions must set `snapshot_id=None` and
`eligible_family=False`; they are not runtime-qualified. Export the API from
`capmas.evaluation`.

- [ ] **Step 4: Run focused tests and style checks**

Run: `pytest tests/test_p56_calibration.py -q`

Expected: all model and PAVA tests pass.

Run: `ruff check capmas/evaluation/calibration.py tests/test_p56_calibration.py`

Expected: exit code `0` for touched files.

- [ ] **Step 5: Commit model mathematics independently**

```bash
git add capmas/evaluation/calibration.py capmas/evaluation/__init__.py tests/test_p56_calibration.py
git commit -m "feat: add P5.6B constrained calibration"
```

### Task 4: Leakage-Safe Offline Orchestration and Report

**Files:**

- Modify: `capmas/evaluation/offline.py`
- Modify: `tests/test_p56_offline.py`
- Modify: `capmas/evaluation/__init__.py`

**Interfaces:**

- Consumes: audit-passing source manifest, `ExactQuotaSplitConfig`, Task 1
  reducer, and Task 3 fit/prediction functions.
- Produces: `OfflineCalibrationReport` and
  `run_offline_calibration(manifest, config) -> OfflineCalibrationReport`.
- Does not own JSON file I/O; the caller serializes `report.to_dict()`.

- [ ] **Step 1: Add failing orchestration tests**

```python
def test_offline_run_uses_only_train_for_fit_and_calibration_for_pava(monkeypatch) -> None:
    observed: dict[str, tuple[str, ...]] = {}
    monkeypatch.setattr(offline, "fit_constrained_logistic", _capture_fit(observed))
    monkeypatch.setattr(offline, "fit_isotonic", _capture_pava(observed))

    report = run_offline_calibration(_twenty_row_manifest(), ExactQuotaSplitConfig.object6_v1())

    assert observed["fit"] == ("train",) * 12
    assert observed["pava"] == ("calibration",) * 4
    assert report.split_counts == {"train": 12, "calibration": 4, "test": 4}
    assert report.online_effect is False


def test_offline_run_has_no_runtime_effect() -> None:
    report = run_offline_calibration(_twenty_row_manifest(), ExactQuotaSplitConfig.object6_v1())
    assert report.model is not None
    assert report.online_effect is False
    assert report.model.model_version.startswith("p56b.")
```

Add tests for source audit failure, a required-evidence abstention being counted
without changing labels, `test` predictions appearing only after model/PAVA
fit, deterministic report digests, and no model when the train or calibration
class gate fails.

- [ ] **Step 2: Run orchestration tests to verify failure**

Run: `pytest tests/test_p56_offline.py -q`

Expected: failure because `run_offline_calibration` and
`OfflineCalibrationReport` are not implemented.

- [ ] **Step 3: Implement orchestration and report serialization**

```python
@dataclass(frozen=True)
class OfflineCalibrationReport:
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
```

Implement `OfflineCalibrationReport.to_dict()` with a canonical payload and
`report_sha256` computed from the same payload with that field blank. Export
`run_offline_calibration(manifest: CalibrationDatasetManifest, config:
ExactQuotaSplitConfig) -> OfflineCalibrationReport`.

The function must execute in this order:

1. Audit the source manifest and call `partition_tier_a_outcomes`.
2. Reduce each partitioned snapshot exactly once.
3. If any selected row lacks required feasibility evidence, retain it in the
   report as an abstention and exclude it only from fitting; do not change its
   physical label or partition.
4. Fit the logistic model from eligible `train` rows only. On a `fit_rejected_*`
   error, return a report with no model/PAVA and abstained predictions for all
   eligible rows.
5. Fit PAVA using eligible `calibration` rows only.
6. Predict every eligible row under its frozen model/PAVA. Emit Brier and ECE
   with a `test_` prefix only from the test predictions; report train and
   calibration diagnostics separately if desired, never as selection criteria.
7. Build a canonical digest from `to_dict()` with its digest field omitted and
   set `online_effect=False` unconditionally.

Do not instantiate any snapshot registry or call the runtime. Export the
orchestrator and report types from `capmas.evaluation`.

- [ ] **Step 4: Run orchestration and related focused tests**

Run: `pytest tests/test_p56_correlation.py tests/test_p56_calibration.py tests/test_p56_offline.py -q`

Expected: all tests pass and no test imports CAP-X or LIBERO.

- [ ] **Step 5: Commit the offline-only pipeline**

```bash
git add capmas/evaluation/offline.py capmas/evaluation/__init__.py tests/test_p56_offline.py
git commit -m "feat: add P5.6B offline calibration report"
```

### Task 5: Reproducible CLI, Artifacts, and Documentation

**Files:**

- Create: `scripts/run_p56_offline.py`
- Create: `tests/test_run_p56_offline.py`
- Modify: `docs/implementation-roadmap.md`
- Modify: `docs/phase5-evidence-evolution.md`
- Modify: `docs/experiments.md`

**Interfaces:**

- Consumes: one or more P5.6A suite directories containing
  `results/outcomes.json` and `results/lineages.json`.
- Produces: a `P5.6.4_offline_calibration` `Phase5RunDirectory` with source
  dataset, locked split, reduction, model/PAVA, predictions, report, log, and
  verified manifest artifacts.
- Calls only Task 4's `run_offline_calibration`; it must not import a CAP-X
  factory, an LLM client, a simulator, or any runtime Arbiter class.

- [ ] **Step 1: Write failing CLI integration tests**

```python
def test_offline_cli_writes_complete_verified_phase5_run(tmp_path, capsys) -> None:
    first, second = _write_p56a_collection_suites(tmp_path, total_rows=20)

    assert main([
        "--collection-run", str(first),
        "--collection-run", str(second),
        "--output-root", str(tmp_path / "outputs"),
        "--run-id", "unit",
    ]) == 0

    run_dir = Path(capsys.readouterr().out.strip())
    assert (run_dir / "logs/runner.log").is_file()
    assert (run_dir / "artifacts/source_dataset_manifest.json").is_file()
    assert (run_dir / "results/offline_calibration_report.json").is_file()
    assert inspect_manifest(run_dir)["verified"] is True
```

Add cases for missing collection files, invalid JSON/contract payload, duplicate
episode IDs across collection runs, an ineligible source manifest, and source
files proving that the CLI never imports `libero`, `capx`, or an Arbiter module.

- [ ] **Step 2: Run CLI tests to verify failure**

Run: `pytest tests/test_run_p56_offline.py -q`

Expected: import failure because `scripts/run_p56_offline.py` does not exist.

- [ ] **Step 3: Implement the CLI and artifact lifecycle**

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run P5.6B offline calibration.")
    parser.add_argument("--collection-run", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", default="p56b-object6-offline")
    return parser


```

Export `load_collection_rows(paths: Sequence[str]) ->
tuple[tuple[CalibrationOutcome, ...], tuple[CalibrationLineage, ...]]` and
`main(argv: list[str] | None = None) -> int` with the strict decoding and
artifact behavior defined below.

For each input directory, load and strictly decode `results/outcomes.json` with
`CalibrationOutcome.from_dict` and `results/lineages.json` with
`CalibrationLineage.from_dict`. Reject duplicate episode IDs and lineages before
constructing a source dataset via `build_calibration_dataset`; use the existing
`assign_lineage_splits` with salt `p56b-source-audit-v1` only for the existing
source-manifest audit. Then call Task 4 with
`ExactQuotaSplitConfig.object6_v1()`.

Create `Phase5RunDirectory.create(output_root, "P5.6.4_offline_calibration",
run_id)` before processing. Write:

```text
run_config.json
logs/runner.log
artifacts/source_dataset_manifest.json
artifacts/exact_quota_split.json
artifacts/reduced_features.json
artifacts/constrained_logistic_model.json
artifacts/isotonic_calibration.json
results/predictions.json
results/offline_calibration_report.json
```

Use `None` artifacts for a fail-closed report rather than inventing a model.
Finalize the manifest, call `verify_and_record_manifest(run_dir.path)`, print
only the resulting run-directory path, and return nonzero for a malformed or
ineligible input. Keep the runner log secret-free.

- [ ] **Step 4: Update the Phase 5 documents**

Add P5.6B to the roadmap as "offline foundation implemented; no online
activation". Document the exact input/output contract and abstention behavior
in `phase5-evidence-evolution.md`. Add this reproducible real-data command to
`experiments.md`:

```bash
python scripts/run_p56_offline.py \
  --collection-run .worktrees/p56a-data-foundation/outputs/phase5/P5.6.2a_object6_collection/20260814_022145_suite_85dd4d7d \
  --collection-run .worktrees/p56a-data-foundation/outputs/phase5/P5.6.2a_object6_collection/20260817_070047_suite_c966d81c \
  --output-root outputs/phase5 \
  --run-id p56b-object6-offline
```

State that this command requires no GPU, LLM endpoint, CAP-X environment, or
robot execution and may only make functional/reproducibility claims.

- [ ] **Step 5: Run the full code and real-artifact verification sequence**

Run:

```bash
pytest tests/test_p56_correlation.py tests/test_p56_calibration.py tests/test_p56_offline.py tests/test_run_p56_offline.py -q
pytest -q
python -m compileall -q capmas scripts
ruff check capmas/evaluation/correlation.py capmas/evaluation/calibration.py capmas/evaluation/offline.py scripts/run_p56_offline.py tests/test_p56_correlation.py tests/test_p56_calibration.py tests/test_p56_offline.py tests/test_run_p56_offline.py
git diff --check
python scripts/run_p56_offline.py --collection-run .worktrees/p56a-data-foundation/outputs/phase5/P5.6.2a_object6_collection/20260814_022145_suite_85dd4d7d --collection-run .worktrees/p56a-data-foundation/outputs/phase5/P5.6.2a_object6_collection/20260817_070047_suite_c966d81c --output-root outputs/phase5 --run-id p56b-object6-offline
```

Expected: all tests and touched-file Ruff checks pass; the real run produces a
new self-contained Phase 5 directory with `online_effect=false`; its manifest
verification report has `verified=true`. Do not report Brier/ECE as a
significant improvement.

- [ ] **Step 6: Commit the CLI, docs, and verified offline run metadata**

```bash
git add scripts/run_p56_offline.py tests/test_run_p56_offline.py docs/implementation-roadmap.md docs/phase5-evidence-evolution.md docs/experiments.md
git commit -m "feat: run P5.6B offline calibration"
```

Do not commit generated `outputs/` unless the repository's existing experiment
artifact policy explicitly tracks that exact run directory. Preserve the output
directory on disk and record its absolute or workspace-relative path in the
final implementation report.
