# P5.6A Data Foundation and Capability Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the leakage-safe P5.6 data foundation: immutable calibration contracts, verified planned/realized horizon telemetry, decision-time feature snapshots, typed physical outcomes, lineage-safe datasets, read-only family diagnostics, and pre-registered object-6 collection blocks.

**Architecture:** Add one versioned contract module and keep extraction, normalization, dataset auditing, capability diagnosis, and historical compatibility auditing in separate evaluation modules. Extend the graph interpreter with an append-only event stream and expose evidence-enriched candidates at the existing rehearsal-Arbiter boundary so feature snapshots are captured before physical execution. Reuse the existing P5.3.1 runner, CAP-X factory, single physical Executor, and `Phase5RunDirectory`; P5.6A records data but does not fit a calibrator or alter candidate selection.

**Tech Stack:** Python 3.10+, frozen dataclasses, standard-library `json`, `hashlib`, `pathlib`, `collections`, existing CAP-MAS graph/runtime contracts, CAP-X `.venv-libero`, LIBERO, pytest, ruff.

## Global Constraints

- Follow [`docs/superpowers/specs/2026-08-11-p5-6-evidence-calibration-design.md`](../specs/2026-08-11-p5-6-evidence-calibration-design.md) exactly.
- P5.6.0 is read-only: it must not modify prompts, task maps, skill arguments, registries, candidate artifacts, or runtime configuration.
- P5.3.2 task-family capability repair is outside this plan and requires its own approved design and implementation plan.
- P5.6A must not fit coefficients, activate a calibration snapshot, change Arbiter ranking, or add a second physical Executor.
- Capture candidate features after normalization/evidence attachment and before lease acquisition or physical execution.
- Never reconstruct a pre-execution feature from evaluator output, execution traces, or an after-action scene.
- Only static verifier results may enter feature schema `p56.feature.v1`; dynamic verifier results are outcome evidence.
- `CandidateEvidence.ood_success_rate` is excluded from feature schema `p56.feature.v1`; initial OOD weight remains zero.
- Preserve `None` plus explicit `unknown` status; never coerce missing evidence to `0.0`.
- Count planned task horizon by action-bearing subgraphs on the successful critical path; checkpoint-only subgraphs do not increase the H bucket.
- Count every runtime re-entry/retry as another realized attempt; completed counts require a successful completion event.
- Every experiment and audit gets a fresh `Phase5RunDirectory` containing logs, results, evidence, artifacts, and a verified manifest.
- Use `CUDA_VISIBLE_DEVICES=5`, `max_workers=1`, and at most one physical execution per collection case.
- Do not store API keys, authorization headers, provider headers, or other secrets in source, logs, manifests, or artifacts.
- Add no SciPy, scikit-learn, NumPy, pandas, or other runtime dependency in P5.6A.
- Schema constants are `p56.feature.v1`, `p56.horizon.v1`, `p56.dataset.v1`, `p56.capability.v1`, and `p56.collection.v1`.

## Scope Boundary

This plan closes P5.6.0, P5.6.1, P5.6.2, and the infrastructure portion of P5.6.2a. P5.6B will own correlation reduction, constrained logistic fitting, PAVA isotonic calibration, offline metrics, and immutable calibration snapshots. P5.6C will own calibrated shadow arbitration, abstention/fallback integration, bounded canary execution, and formal matched evaluation.

---

### Task 1: Add immutable calibration and collection contracts

**Files:**
- Create: `capmas/contracts/calibration.py`
- Modify: `capmas/contracts/__init__.py`
- Create: `tests/test_p56_contracts.py`

**Interfaces:**
- Produces `HorizonLabel`, `CandidateFeatureSnapshot`, `CalibrationOutcome`, `CalibrationPrediction`, and `CalibrationCollectionContext`.
- Produces `horizon_bucket(label) -> Literal["H1", "H2-3", "H4-6", "H7+", "N/A"]`.
- Every public contract provides `to_dict() -> dict[str, object]`; `HorizonLabel`, `CandidateFeatureSnapshot`, and `CalibrationOutcome` also provide strict `from_dict()` class methods.

- [ ] **Step 1: Write failing validation and round-trip tests**

```python
from dataclasses import replace

import pytest

from capmas.contracts.calibration import (
    CalibrationOutcome,
    CandidateFeatureSnapshot,
    HorizonLabel,
    horizon_bucket,
)


def _horizon() -> HorizonLabel:
    return HorizonLabel(
        planned_critical_path_actions=2,
        planned_critical_path_subgoals=2,
        planned_checkpoint_subgraphs=1,
        attempted_actions=2,
        completed_actions=2,
        attempted_subgoals=2,
        completed_subgoals=2,
        attempted_checkpoints=3,
        completed_checkpoints=3,
        planned_source="mission_graph",
        realized_source="execution_trace",
        planned_valid=True,
        realized_valid=True,
    )


def _snapshot() -> CandidateFeatureSnapshot:
    return CandidateFeatureSnapshot(
        episode_id="object-6-seed11",
        episode_epoch=1,
        family_id="object-6",
        candidate_id="candidate-a",
        candidate_fingerprint="a" * 64,
        scene_version=1,
        map_version=None,
        feature_schema_version="p56.feature.v1",
        captured_at_ns=10,
        collection_lane="physical",
        features={"scene_freshness": None, "rehearsal_success_rate": 1.0},
        feature_status={"scene_freshness": "unknown", "rehearsal_success_rate": "present"},
        correlation_groups={
            "scene_freshness": "scene_grounding",
            "rehearsal_success_rate": "action_feasibility",
        },
        memory_skill_version="memory-frozen-v1",
        robot_skill_version="robot-frozen-v1",
        evidence_refs=("rehearsal://candidate-a/11",),
        evidence_providers={"rehearsal": "libero_process_rehearsal"},
        rewrite_metadata={"changed": False},
    )


def test_contracts_round_trip_without_converting_unknown_to_zero() -> None:
    outcome = CalibrationOutcome(
        episode_id="object-6-seed11",
        family_id="object-6",
        candidate_id="candidate-a",
        candidate_fingerprint="a" * 64,
        tier="A",
        execution_status="selected_executed",
        task_success=True,
        graph_completed=True,
        verifier_success=True,
        rehearsal_success=None,
        failure_class=None,
        horizon=_horizon(),
        feature_snapshot=_snapshot(),
        dataset_split="unassigned",
    )

    restored = CalibrationOutcome.from_dict(outcome.to_dict())

    assert restored == outcome
    assert restored.feature_snapshot.features["scene_freshness"] is None
    assert horizon_bucket(restored.horizon) == "H2-3"


def test_horizon_rejects_completed_count_above_attempted_count() -> None:
    with pytest.raises(ValueError, match="completed_actions"):
        replace(_horizon(), attempted_actions=1, completed_actions=2)


def test_tier_a_requires_selected_execution_and_physical_label() -> None:
    with pytest.raises(ValueError, match="Tier A"):
        CalibrationOutcome(
            episode_id="episode",
            family_id="object-6",
            candidate_id="candidate-a",
            candidate_fingerprint="a" * 64,
            tier="A",
            execution_status="not_selected",
            task_success=False,
            graph_completed=None,
            verifier_success=None,
            rehearsal_success=None,
            failure_class=None,
            horizon=_horizon(),
            feature_snapshot=_snapshot(),
            dataset_split="unassigned",
        )


def test_tier_b_uses_rehearsal_label_and_never_physical_label() -> None:
    with pytest.raises(ValueError, match="Tier B"):
        CalibrationOutcome(
            episode_id="episode",
            family_id="object-6",
            candidate_id="candidate-a",
            candidate_fingerprint="a" * 64,
            tier="B",
            execution_status="not_selected",
            task_success=True,
            graph_completed=None,
            verifier_success=None,
            rehearsal_success=True,
            failure_class=None,
            horizon=_horizon(),
            feature_snapshot=replace(_snapshot(), collection_lane="rehearsal"),
            dataset_split="shadow",
        )
```

- [ ] **Step 2: Run the contract tests and verify the missing-module failure**

Run: `pytest -q tests/test_p56_contracts.py`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'capmas.contracts.calibration'`.

- [ ] **Step 3: Implement contracts, deep mapping normalization, and strict codecs**

Use these exact public shapes:

```python
FEATURE_SCHEMA_VERSION = "p56.feature.v1"
HORIZON_SCHEMA_VERSION = "p56.horizon.v1"
DATASET_SCHEMA_VERSION = "p56.dataset.v1"
CAPABILITY_SCHEMA_VERSION = "p56.capability.v1"
COLLECTION_SCHEMA_VERSION = "p56.collection.v1"

ExecutionStatus = Literal[
    "selected_executed",
    "selected_not_started",
    "not_selected",
    "rejected_safety",
    "rejected_schema",
    "stale",
    "unknown",
]
DatasetSplit = Literal["train", "calibration", "test", "shadow", "unassigned"]
FeatureStatus = Literal["present", "unknown", "invalid"]
CollectionLane = Literal["physical", "rehearsal", "shadow"]


@dataclass(frozen=True)
class HorizonLabel:
    planned_critical_path_actions: int | None
    planned_critical_path_subgoals: int | None
    planned_checkpoint_subgraphs: int | None
    attempted_actions: int | None
    completed_actions: int | None
    attempted_subgoals: int | None
    completed_subgoals: int | None
    attempted_checkpoints: int | None
    completed_checkpoints: int | None
    planned_source: Literal["mission_graph", "unknown"]
    realized_source: Literal["execution_trace", "unknown"]
    planned_valid: bool
    realized_valid: bool


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
    collection_lane: CollectionLane
    features: Mapping[str, float | None]
    feature_status: Mapping[str, FeatureStatus]
    correlation_groups: Mapping[str, str]
    memory_skill_version: str
    robot_skill_version: str
    selection_probability: float | None = None
    evidence_refs: tuple[str, ...] = ()
    evidence_providers: Mapping[str, str] = field(default_factory=dict)
    rewrite_metadata: Mapping[str, object] = field(default_factory=dict)


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
    rehearsal_success: bool | None
    failure_class: str | None
    horizon: HorizonLabel
    feature_snapshot: CandidateFeatureSnapshot
    dataset_split: DatasetSplit


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


@dataclass(frozen=True)
class CalibrationCollectionContext:
    episode_id: str
    episode_epoch: int
    family_id: str
    feature_schema_version: str
    memory_skill_version: str
    robot_skill_version: str
    collection_lane: CollectionLane = "physical"
```

Normalize mapping inputs with `MappingProxyType(dict(value))` in `__post_init__`; emit ordinary sorted dictionaries in `to_dict()`. Reject non-finite feature values, mismatched feature/status/group key sets, empty versions and identities, negative counts/timestamps/versions, invalid SHA-256 fingerprints, and `completed > attempted`. Enforce these tier rules: Tier A is only `selected_executed` with a conclusive `task_success`; Tier B has `task_success=None` and a conclusive `rehearsal_success`; Tier C has both labels `None`. Export all constants and public contracts from `capmas/contracts/__init__.py`.

- [ ] **Step 4: Run focused tests and static checks**

Run: `pytest -q tests/test_p56_contracts.py && ruff check capmas/contracts/calibration.py tests/test_p56_contracts.py`

Expected: all tests pass and ruff exits zero.

- [ ] **Step 5: Commit the contract boundary**

```bash
git add capmas/contracts/calibration.py capmas/contracts/__init__.py tests/test_p56_contracts.py
git commit -m "feat: add P5.6 calibration contracts"
```

### Task 2: Add graph execution events and verified horizon extraction

**Files:**
- Modify: `capmas/contracts/trace.py`
- Modify: `capmas/contracts/__init__.py`
- Modify: `capmas/runtime/graph_interpreter.py`
- Modify: `scripts/run_libero_p53_online.py`
- Create: `capmas/evaluation/labels.py`
- Modify: `capmas/evaluation/__init__.py`
- Create: `tests/test_p56_horizon.py`
- Modify: `tests/test_graph_runtime.py`
- Modify: `tests/test_libero_p53_online.py`

**Interfaces:**
- Produces `GraphExecutionEvent(sequence, kind, subgraph_id, node_id, node_type, attempt, outcome, occurred_at_ns)`.
- Adds trailing `events: tuple[GraphExecutionEvent, ...] = ()` to `GraphExecutionResult`.
- Produces `planned_horizon(graph) -> HorizonLabel`, `realized_horizon(graph, events) -> HorizonLabel`, and `extract_horizon(graph, events) -> HorizonLabel`.
- Adds `graph_events` and serialized `horizon` to the existing physical result payload without changing success semantics.

- [ ] **Step 1: Write failing event and horizon tests**

```python
def test_interpreter_records_action_and_checkpoint_events_in_order() -> None:
    graph = _action_then_checkpoint_mission()
    result = FixedGraphInterpreter(_SuccessfulScheduler()).run(graph, _scene())

    assert [(event.kind, event.node_type) for event in result.events] == [
        ("subgraph_started", None),
        ("node_started", "action"),
        ("node_completed", "action"),
        ("node_started", "checkpoint"),
        ("node_completed", "checkpoint"),
        ("subgraph_completed", None),
    ]
    assert [event.sequence for event in result.events] == list(range(6))


def test_checkpoint_only_subgraph_does_not_inflate_planned_bucket() -> None:
    graph = _two_action_subgraphs_then_checkpoint_only_mission()
    label = planned_horizon(graph)

    assert label.planned_critical_path_actions == 2
    assert label.planned_critical_path_subgoals == 2
    assert label.planned_checkpoint_subgraphs == 1
    assert horizon_bucket(label) == "H2-3"


def test_realized_horizon_counts_reentry_as_an_attempt() -> None:
    graph = _single_action_loop_mission(max_visits=2)
    events = (
        _event(0, "subgraph_started", "pick", attempt=1),
        _event(1, "node_started", "pick", "grasp", "action", attempt=1),
        _event(2, "node_failed", "pick", "grasp", "action", attempt=1),
        _event(3, "subgraph_failed", "pick", attempt=1),
        _event(4, "subgraph_started", "pick", attempt=2),
        _event(5, "node_started", "pick", "grasp", "action", attempt=2),
        _event(6, "node_completed", "pick", "grasp", "action", attempt=2),
        _event(7, "subgraph_completed", "pick", attempt=2),
    )

    label = extract_horizon(graph, events)

    assert label.attempted_actions == 2
    assert label.completed_actions == 1
    assert label.attempted_subgoals == 2
    assert label.completed_subgoals == 1


def test_max_steps_is_not_used_as_horizon() -> None:
    graph = _two_action_subgraphs_then_checkpoint_only_mission()
    result = FixedGraphInterpreter(_SuccessfulScheduler(), max_steps=32).run(graph, _scene())

    assert extract_horizon(graph, result.events).planned_critical_path_subgoals == 2
```

- [ ] **Step 2: Run focused tests and verify missing event/horizon symbols**

Run: `pytest -q tests/test_p56_horizon.py tests/test_graph_runtime.py`

Expected: FAIL because `GraphExecutionEvent`, `GraphExecutionResult.events`, and horizon extractors are absent.

- [ ] **Step 3: Implement the event contract and append-only interpreter telemetry**

Add this trailing-compatible contract to `capmas/contracts/trace.py`:

```python
GraphEventKind = Literal[
    "subgraph_started",
    "subgraph_completed",
    "subgraph_failed",
    "node_started",
    "node_completed",
    "node_failed",
]


@dataclass(frozen=True)
class GraphExecutionEvent:
    sequence: int
    kind: GraphEventKind
    subgraph_id: str
    node_id: str | None
    node_type: Literal["action", "checkpoint", "router"] | None
    attempt: int
    outcome: str | None
    occurred_at_ns: int
```

Add `clock: Callable[[], int] = time.time_ns` to the end of `FixedGraphInterpreter.__init__` and add `events` after `next_subgraph` in `GraphExecutionResult`. Emit `subgraph_started` before each `_run_subgraph` call; emit `node_started` before evaluating every node; emit `node_completed` only for outcome `success`, otherwise `node_failed`; emit one subgraph terminal event for every return path. Sequence is `len(events)`, and `attempt` is the current visit count for that node or subgraph. Pass `tuple(events)` by keyword in every `GraphExecutionResult` construction so current positional constructors remain valid.

- [ ] **Step 4: Implement deterministic planned and realized horizon extraction**

In `capmas/evaluation/labels.py`, classify a subgraph as action-bearing when any node has `node_type == "action"`. For planned horizon, enumerate simple successful mission paths from `entry_subgraph` to `success_subgraphs`, following only edges whose condition is `None`, `success`, `completed`, `action_complete`, or `action_completed`; ignore a back-edge to a subgraph already on the path. Select the path with the lexicographically largest `(action_bearing_subgraphs, action_nodes, checkpoint_only_subgraphs, path_ids)` tuple. If no successful path exists, return unknown planned fields and `planned_valid=False`.

For realized horizon, count every `node_started/action` and `node_completed/action` event, every `subgraph_started` and `subgraph_completed` event whose graph subgraph is action-bearing, and every checkpoint node started/completed event. This deliberately counts retries and recovery re-entry as additional attempts. Merge planned and realized halves in `extract_horizon`; do not inspect `max_steps`, elapsed time, skill traces, or LLM calls.

- [ ] **Step 5: Persist graph events and horizon in physical result artifacts**

Extend `_physical_result_payload` with a trailing keyword-only
`graph: MissionGraph | None = None` argument. Physical execution always passes
the grounded graph; direct legacy helper callers receive an unknown horizon.
Add this payload when `graph is not None`:

```python
"graph_events": [
    {
        "sequence": event.sequence,
        "kind": event.kind,
        "subgraph_id": event.subgraph_id,
        "node_id": event.node_id,
        "node_type": event.node_type,
        "attempt": event.attempt,
        "outcome": event.outcome,
        "occurred_at_ns": event.occurred_at_ns,
    }
    for event in result.events
],
"horizon": extract_horizon(graph, result.events).to_dict(),
```

Pass the grounded graph from `_build_live_executor.execute`. Update the two
direct helper tests to pass their graph fixture when asserting horizon output.
Keep `trace_count`, success, evaluator, failure, scene diagnostics, and layout
fields unchanged.

- [ ] **Step 6: Run runtime, horizon, and existing online tests**

Run: `pytest -q tests/test_p56_horizon.py tests/test_graph_runtime.py tests/test_libero_p53_online.py tests/test_libero_p55_ood.py`

Expected: all tests pass; existing positional `GraphExecutionResult` tests remain compatible.

- [ ] **Step 7: Commit graph telemetry**

```bash
git add capmas/contracts/trace.py capmas/contracts/__init__.py capmas/runtime/graph_interpreter.py capmas/evaluation/labels.py capmas/evaluation/__init__.py scripts/run_libero_p53_online.py tests/test_p56_horizon.py tests/test_graph_runtime.py
git commit -m "feat: record verified graph horizon telemetry"
```

### Task 3: Capture leakage-safe candidate feature snapshots at the Arbiter boundary

**Files:**
- Create: `capmas/evaluation/feature_snapshots.py`
- Modify: `capmas/evaluation/__init__.py`
- Modify: `capmas/evaluation/online_rehearsal.py`
- Modify: `scripts/run_libero_p53_online.py`
- Create: `tests/test_p56_feature_snapshots.py`
- Modify: `tests/test_online_rehearsal.py`
- Modify: `tests/test_libero_p53_online.py`

**Interfaces:**
- Produces `FEATURE_GROUPS_V1` and `capture_feature_snapshot(candidate, context, *, map_version=None, clock=time.time_ns) -> CandidateFeatureSnapshot`.
- Adds trailing `evidence_candidates: tuple[GraphCandidate, ...] = ()` to `RehearsalArbitrationReport`.
- Adds optional `calibration_context: CalibrationCollectionContext | None = None` to `run_online_experiment`.
- Adds trailing `feature_snapshots: tuple[CandidateFeatureSnapshot, ...] = ()` to `OnlineSelectionOutcome`.
- Adds trailing `decision_completed_at_ns: int | None = None` and `physical_execution_started_at_ns: int | None = None` to `OnlineSelectionOutcome`.

- [ ] **Step 1: Write failing snapshot projection and timing tests**

```python
def test_feature_snapshot_uses_static_verifier_and_excludes_dynamic_and_ood() -> None:
    candidate = _candidate_with_perception_geometry_rehearsal_and_mixed_verifier()
    snapshot = capture_feature_snapshot(candidate, _collection_context(), clock=lambda: 50)

    assert snapshot.features["static_verifier_pass_rate"] == 1.0
    assert snapshot.features["static_verifier_coverage"] == 1.0
    assert "dynamic_verifier_pass_rate" not in snapshot.features
    assert "ood_success_rate" not in snapshot.features
    assert snapshot.captured_at_ns == 50


def test_feature_snapshot_preserves_unknown_dimensions() -> None:
    snapshot = capture_feature_snapshot(_candidate_without_evidence(), _collection_context())

    assert snapshot.features["scene_freshness"] is None
    assert snapshot.feature_status["scene_freshness"] == "unknown"
    assert snapshot.correlation_groups["scene_freshness"] == "scene_grounding"


def test_snapshot_rejects_scene_or_fingerprint_mismatch() -> None:
    candidate = _candidate_with_stale_evidence()

    with pytest.raises(ValueError, match="scene version"):
        capture_feature_snapshot(candidate, _collection_context())


def test_online_runner_writes_snapshots_before_physical_executor(tmp_path) -> None:
    observed = []

    def executor(candidate, graph):
        snapshots = list(tmp_path.rglob("evidence/calibration_feature_snapshots.json"))
        observed.append(bool(snapshots))
        return _physical_payload()

    outcome = run_online_experiment(
        **_online_kwargs(tmp_path),
        calibration_context=_collection_context(),
        physical_executor=executor,
    )

    assert observed == [True]
    assert outcome.feature_snapshots
```

- [ ] **Step 2: Run snapshot tests and verify failure**

Run: `pytest -q tests/test_p56_feature_snapshots.py tests/test_online_rehearsal.py tests/test_libero_p53_online.py`

Expected: FAIL because the snapshot builder and report fields are absent.

- [ ] **Step 3: Implement feature schema v1 projection**

Define exactly these keys and groups:

```python
FEATURE_GROUPS_V1 = {
    "scene_freshness": "scene_grounding",
    "scene_confidence": "scene_grounding",
    "target_visibility": "scene_grounding",
    "track_confidence": "scene_grounding",
    "identity_confidence": "scene_grounding",
    "pose_reliability": "scene_grounding",
    "grasp_quality": "scene_grounding",
    "reachability": "scene_grounding",
    "clearance": "scene_grounding",
    "static_verifier_pass_rate": "action_feasibility",
    "static_verifier_coverage": "action_feasibility",
    "rehearsal_success_rate": "action_feasibility",
    "collision_risk": "cost_risk",
    "expected_latency_ms": "cost_risk",
    "recovery_cost": "cost_risk",
}
```

Read perception and geometry only when their metric is declared available. Recompute static verifier pass rate and coverage from `VerifierEvidence.static_results`; never read its aggregate `pass_rate` when `dynamic_results` are present. Treat a declared-but-missing typed evidence object as invalid and raise. Copy evidence refs, providers, and `candidate.rewrite_report` into snapshot lineage. Validate the effective `subgraph_fingerprint`, evidence scene version, geometry fingerprint, geometry map version, and verifier fingerprint before constructing a snapshot.

- [ ] **Step 4: Preserve evidence-enriched candidates in all rehearsal report paths**

Add `evidence_candidates` as the final report field. Disabled/provider-missing/provider-error paths return the original `live_candidates`; successful evidence collection returns `tuple(enriched)`. Shadow mode still executes the baseline selection, and online-bounded mode still executes the existing evidence-aware selection. No score or selection behavior changes in this task.

- [ ] **Step 5: Add opt-in pre-execution persistence to the online runner**

After the final `select_with_rehearsal` call, record
`decision_completed_at_ns = time.time_ns()`. Before calculating or invoking
`physical_candidate_id`, capture one snapshot per
`report.evidence_candidates` when `calibration_context` is present. Write
`evidence/calibration_feature_snapshots.json` immediately. Immediately before
calling a non-null physical executor, record
`physical_execution_started_at_ns = time.time_ns()`. Persist both timestamps,
snapshot count, and schema version in `run_config.json`, `summary.json`, and
`logs/runner.log`. When context is absent, preserve current snapshot behavior
and return an empty snapshot tuple, while the two timing fields remain valid
general provenance.

- [ ] **Step 6: Run focused and regression tests**

Run: `pytest -q tests/test_p56_feature_snapshots.py tests/test_online_rehearsal.py tests/test_libero_p53_online.py tests/test_phase5_rehearsal_arbiter.py tests/test_libero_p55_ood.py`

Expected: all tests pass; P5.3/P5.5 selection winners and modes are unchanged.

- [ ] **Step 7: Commit decision-time capture**

```bash
git add capmas/evaluation/feature_snapshots.py capmas/evaluation/__init__.py capmas/evaluation/online_rehearsal.py scripts/run_libero_p53_online.py tests/test_p56_feature_snapshots.py tests/test_online_rehearsal.py tests/test_libero_p53_online.py
git commit -m "feat: capture P5.6 decision feature snapshots"
```

### Task 4: Normalize outcomes and build a lineage-safe three-tier dataset

**Files:**
- Modify: `capmas/contracts/calibration.py`
- Create: `capmas/evaluation/dataset.py`
- Modify: `capmas/evaluation/__init__.py`
- Create: `tests/test_p56_dataset.py`

**Interfaces:**
- Adds contract types `CalibrationLineage` and `CalibrationDatasetManifest`, plus evaluation types `LeakageFinding` and `DatasetAudit`.
- Produces `normalize_physical_outcomes(snapshots, *, selected_candidate_id, execution_started, task_success, graph_completed, verifier_success, failure_class, horizon, rehearsal_labels=None, rejection_codes=None) -> tuple[CalibrationOutcome, ...]`.
- Produces `assign_lineage_splits(lineages, *, salt, train_fraction=0.6, calibration_fraction=0.2) -> dict[str, DatasetSplit]`.
- Produces `build_calibration_dataset(outcomes, lineages, *, split_assignments, memory_skill_version, robot_skill_version, prompt_version, environment_version, code_revision, split_salt) -> CalibrationDatasetManifest`.
- Produces `audit_calibration_dataset(manifest) -> DatasetAudit` and `assert_dataset_eligible(audit) -> None`.

- [ ] **Step 1: Write failing tier, split, and leakage tests**

```python
def test_selected_candidate_is_tier_a_and_unselected_candidates_are_tier_c() -> None:
    outcomes = normalize_physical_outcomes(
        snapshots=(_snapshot("a"), _snapshot("b")),
        selected_candidate_id="a",
        execution_started=True,
        task_success=False,
        graph_completed=True,
        verifier_success=True,
        failure_class="task_failure",
        horizon=_horizon(),
    )

    assert [(row.candidate_id, row.tier, row.task_success) for row in outcomes] == [
        ("a", "A", False),
        ("b", "C", None),
    ]


def test_related_id_ood_and_retry_lineage_stays_in_one_split() -> None:
    lineages = (
        _lineage("id-1", group="pair-1"),
        _lineage("ood-1", group="pair-1"),
        _lineage("retry-1", group="pair-1"),
    )

    assignments = assign_lineage_splits(lineages, salt="frozen-split-v1")

    assert len(set(assignments.values())) == 1


def test_dataset_rejects_future_state_feature_timestamp() -> None:
    outcome = _outcome(snapshot_captured_at_ns=200)
    lineage = _lineage("episode", decision_boundary_ns=100)
    manifest = _manifest((outcome,), (lineage,))

    audit = audit_calibration_dataset(manifest)

    assert audit.passed is False
    assert "FUTURE_STATE_FEATURE" in {finding.code for finding in audit.findings}


def test_dataset_rejects_tier_b_or_c_as_supervised_label() -> None:
    manifest = _manifest((_tier_b_with_dataset_label(),), (_lineage("episode"),))

    with pytest.raises(ValueError, match="Tier B"):
        assert_dataset_eligible(audit_calibration_dataset(manifest))
```

- [ ] **Step 2: Run dataset tests and verify missing interfaces**

Run: `pytest -q tests/test_p56_dataset.py`

Expected: FAIL because dataset contracts and builders are absent.

- [ ] **Step 3: Add exact lineage and manifest contracts**

```python
@dataclass(frozen=True)
class CalibrationLineage:
    episode_id: str
    lineage_group_id: str
    seed: int
    split_identity: Literal["id", "ood", "native"]
    layout_pair_id: str | None
    retry_of_episode_id: str | None
    candidate_artifact_sha256: str
    decision_boundary_ns: int
    evaluator_observed_at_ns: int | None


@dataclass(frozen=True)
class CalibrationDatasetManifest:
    dataset_id: str
    dataset_schema_version: str
    feature_schema_version: str
    outcomes: tuple[CalibrationOutcome, ...]
    lineages: tuple[CalibrationLineage, ...]
    memory_skill_version: str
    robot_skill_version: str
    prompt_version: str
    environment_version: str
    code_revision: str
    split_salt: str
    manifest_sha256: str = ""


@dataclass(frozen=True)
class LeakageFinding:
    code: str
    episode_ids: tuple[str, ...]
    detail: str


@dataclass(frozen=True)
class DatasetAudit:
    passed: bool
    findings: tuple[LeakageFinding, ...]
    tier_counts: Mapping[str, int]
    split_counts: Mapping[str, int]
```

The manifest digest omits `manifest_sha256` and hashes canonical ASCII JSON with sorted keys and compact separators. Dataset IDs use the literal `sha256:` prefix followed by the 64-character manifest digest.

- [ ] **Step 4: Implement deterministic splitting, normalization, and fail-closed audit**

Use SHA-256 of `f"{salt}:{lineage_group_id}"` mapped into `[0, 1)` for split assignment. Values below `train_fraction` are train, values below `train_fraction + calibration_fraction` are calibration, and the remainder are test. Reject ratios outside `(0, 1)` or whose sum is at least `1.0`.

The audit must reject duplicate episode IDs with conflicting lineage, one lineage group across multiple splits, the same `(episode_id, candidate_fingerprint)` in multiple supervised splits, feature capture after `decision_boundary_ns`, evaluator-derived feature names, missing versions/provenance, mismatched snapshot/outcome identity, Tier B/C physical labels, Tier A without selected execution, and any dynamic-verifier/OOD aggregate key in feature schema v1. Independent episodes may reuse the same candidate fingerprint; that alone is not leakage.

`normalize_physical_outcomes` applies these exact status rules: the selected
candidate is `selected_executed` only when `execution_started=True`; otherwise
it is `selected_not_started` and receives no physical label. Candidate IDs in
`rejection_codes` map `STALE_SCENE`/`STALE_EVIDENCE` to `stale`, geometry or
safety hard-gate codes to `rejected_safety`, and schema/graph validation codes
to `rejected_schema`. Remaining candidates are `not_selected`. A conclusive
rehearsal label creates Tier B only for a rehearsal-lane snapshot; all other
unlabeled rows are Tier C.

- [ ] **Step 5: Run dataset and contract tests**

Run: `pytest -q tests/test_p56_dataset.py tests/test_p56_contracts.py`

Expected: all tests pass with deterministic manifest digests across repeated construction.

- [ ] **Step 6: Commit dataset construction**

```bash
git add capmas/contracts/calibration.py capmas/evaluation/dataset.py capmas/evaluation/__init__.py tests/test_p56_dataset.py
git commit -m "feat: build leakage-safe P5.6 datasets"
```

### Task 5: Implement read-only family capability diagnosis and P5.3.2 handoff

**Files:**
- Create: `capmas/evaluation/capability.py`
- Modify: `capmas/evaluation/__init__.py`
- Create: `scripts/run_p56_capability.py`
- Create: `tests/test_p56_capability.py`

**Interfaces:**
- Produces `CapabilityCase`, `CapabilityDiagnosticReport`, and `TaskFamilyRepairHandoff`.
- Produces `CapabilityRunResult(run_dir, reports, handoffs)` for the artifact-writing wrapper.
- Produces `load_p55_capability_cases(suite_dir, family_id, *, split="id") -> tuple[CapabilityCase, ...]`.
- Produces `diagnose_family_capability(cases, *, family_id, source_manifest_sha256) -> tuple[CapabilityDiagnosticReport, TaskFamilyRepairHandoff | None]`.
- Produces `run_capability_diagnosis(*, suite_dir, families, output_root, split="id") -> CapabilityRunResult`.
- CLI accepts `--suite-dir`, repeatable `--family`, `--split id`, and `--output-root`.

- [ ] **Step 1: Write failing capability-gate and read-only tests**

```python
def test_capability_gate_requires_execution_reach_and_one_success() -> None:
    cases = tuple(
        _capability_case(seed, reached=seed <= 8, success=seed == 1)
        for seed in range(1, 11)
    )

    report, handoff = diagnose_family_capability(
        cases,
        family_id="object-6",
        source_manifest_sha256="a" * 64,
    )

    assert report.eligible is True
    assert report.execution_reach_rate == 0.8
    assert handoff is None


def test_zero_success_family_emits_typed_p53_2_handoff() -> None:
    cases = tuple(
        _capability_case(seed, reached=True, success=False, failure="POSTCONDITION_FAILED")
        for seed in range(1, 11)
    )

    report, handoff = diagnose_family_capability(
        cases,
        family_id="spatial-0",
        source_manifest_sha256="b" * 64,
    )

    assert report.eligible is False
    assert "NO_EVALUATOR_SUCCESS" in report.gate_failures
    assert handoff is not None
    assert handoff.package == "P5.3.2 Task-Family Capability Repair"


def test_capability_cli_does_not_modify_source_suite(tmp_path) -> None:
    suite = _write_p55_suite_fixture(tmp_path / "source")
    before = _tree_digest(suite)

    result = run_capability_diagnosis(
        suite_dir=suite,
        families=("spatial-0",),
        output_root=tmp_path / "outputs",
    )

    assert _tree_digest(suite) == before
    assert (result.run_dir / "artifacts" / "p53_2_spatial-0.json").exists()
```

- [ ] **Step 2: Run capability tests and verify failure**

Run: `pytest -q tests/test_p56_capability.py`

Expected: FAIL because the capability module and script are absent.

- [ ] **Step 3: Implement typed gate computation and ownership hints**

`CapabilityCase` records case ID, family, seed, split, physical-execution reach, evaluator success, infrastructure-unknown state, failure class, and evidence refs. Require exactly ten unique seeds for a formal report. The four gate failures are `INFRASTRUCTURE_UNKNOWN`, `UNTYPED_FAILURE`, `EXECUTION_REACH_BELOW_0_80`, and `NO_EVALUATOR_SUCCESS`.

Use these exact contracts:

```python
@dataclass(frozen=True)
class CapabilityCase:
    case_id: str
    family_id: str
    seed: int
    split: Literal["id", "ood"]
    reached_physical_execution: bool
    evaluator_success: bool | None
    infrastructure_unknown: bool
    failure_class: str | None
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class CapabilityDiagnosticReport:
    schema_version: str
    family_id: str
    source_manifest_sha256: str
    case_count: int
    physical_execution_count: int
    execution_reach_rate: float
    evaluator_success_count: int
    infrastructure_unknown_count: int
    typed_failure_count: int
    failure_histogram: Mapping[str, int]
    representative_evidence_refs: tuple[str, ...]
    eligible: bool
    gate_failures: tuple[str, ...]


@dataclass(frozen=True)
class TaskFamilyRepairHandoff:
    package: Literal["P5.3.2 Task-Family Capability Repair"]
    family_id: str
    source_manifest_sha256: str
    suspected_owner: str
    failure_histogram: Mapping[str, int]
    representative_evidence_refs: tuple[str, ...]
    acceptance_test: str


@dataclass(frozen=True)
class CapabilityRunResult:
    run_dir: Path
    reports: tuple[CapabilityDiagnosticReport, ...]
    handoffs: tuple[TaskFamilyRepairHandoff, ...]
```

Map infrastructure failures to owner `runtime_infrastructure`, precondition failures to `perception_or_contract`, postcondition failures to `verification_or_robot_skill`, and ordinary task failures to `task_mapping_or_motion`. A failed family handoff records its histogram, representative evidence references, suspected owner, and this acceptance test verbatim: `rerun the same frozen ten-seed capability manifest with zero infrastructure unknowns, at least 80% physical execution reach, typed provenance for every failure, and at least one evaluator success`.

- [ ] **Step 4: Implement the read-only loader and artifact writer**

Scan only `case.json`, `summary.json`, and `evidence/ood_replay.json` under the supplied suite. Do not import CAP-X, start API servers, call an LLM, or write beneath `suite_dir`. Write one fresh run directory with `run_config.json`, `results/capability.json`, one artifact named with `f"artifacts/p53_2_{family_id}.json"` per failed family, `summary.md`, `logs/runner.log`, and `manifest.json`.

- [ ] **Step 5: Run focused tests and CLI help smoke**

Run: `pytest -q tests/test_p56_capability.py && python scripts/run_p56_capability.py --help >/dev/null`

Expected: all tests pass and help exits zero without importing simulator dependencies.

- [ ] **Step 6: Run the frozen ten-seed diagnosis for all three families**

Run:

```bash
python scripts/run_p56_capability.py \
  --suite-dir outputs/phase5/P5.5_matched_provenance_10seed_retry2_20260807/P5.5_frozen_ood_replay/20260807_024842_suite_20674432 \
  --family spatial-0 \
  --family goal-1 \
  --family object-6 \
  --split id \
  --output-root outputs/phase5
```

Expected: command examines exactly ten ID seeds per family, leaves the source
suite byte-identical, emits failed-family P5.3.2 handoffs for `spatial-0` and
`goal-1`, reports `object-6` separately, and creates a fresh manifest-verified
`P5.6.0_capability_diagnosis` run directory. Eligibility is computed from the
retained case evidence and is not hard-coded in the script.

- [ ] **Step 7: Commit read-only diagnosis**

```bash
git add capmas/evaluation/capability.py capmas/evaluation/__init__.py scripts/run_p56_capability.py tests/test_p56_capability.py
git commit -m "feat: add read-only P5.6 capability diagnosis"
```

### Task 6: Audit historical object-6 outcomes without future-state backfill

**Files:**
- Create: `capmas/evaluation/history_audit.py`
- Modify: `capmas/evaluation/__init__.py`
- Create: `scripts/audit_p56_history.py`
- Create: `tests/test_p56_history_audit.py`

**Interfaces:**
- Produces `HistoricalRowDecision` and `HistoricalCompatibilityAudit`.
- Produces `audit_p55_history(suite_dir, *, family_id) -> HistoricalCompatibilityAudit`.
- CLI accepts `--suite-dir`, `--family`, and `--output-root`.

- [ ] **Step 1: Write failing compatibility tests**

```python
def test_history_audit_rejects_missing_preexecution_snapshot(tmp_path) -> None:
    suite = _historical_case(
        tmp_path,
        include_selection=True,
        include_evaluator=True,
        include_feature_snapshot=False,
        include_horizon=False,
    )

    audit = audit_p55_history(suite, family_id="object-6")

    assert audit.admissible_tier_a_count == 0
    assert audit.rejection_counts["MISSING_PREEXECUTION_FEATURE_SNAPSHOT"] == 1
    assert audit.rejection_counts["MISSING_HORIZON_LINEAGE"] == 1


def test_history_audit_never_uses_after_scene_as_features(tmp_path) -> None:
    suite = _historical_case(
        tmp_path,
        include_selection=True,
        include_evaluator=True,
        include_feature_snapshot=False,
        include_horizon=True,
        include_after_scene=True,
    )

    decision = audit_p55_history(suite, family_id="object-6").rows[0]

    assert decision.admissible is False
    assert "MISSING_PREEXECUTION_FEATURE_SNAPSHOT" in decision.reasons


def test_history_audit_accepts_only_timestamp_ordered_native_record(tmp_path) -> None:
    suite = _native_p56_case(
        tmp_path,
        feature_captured_at_ns=100,
        execution_started_at_ns=200,
        evaluator_observed_at_ns=300,
    )

    audit = audit_p55_history(suite, family_id="object-6")

    assert audit.admissible_tier_a_count == 1
    assert audit.rows[0].admissible is True
```

- [ ] **Step 2: Run history tests and verify failure**

Run: `pytest -q tests/test_p56_history_audit.py`

Expected: FAIL because historical audit interfaces are absent.

- [ ] **Step 3: Implement explicit admissibility checks**

For each family case, require all of: candidate ID and fingerprint, a selection event, `evidence/calibration_feature_snapshots.json`, matching scene/candidate identity, snapshot capture timestamp no later than physical execution start, conclusive evaluator outcome, `graph_events`, serialized horizon with valid planned and realized sources, Memory Skill and Robot Skill versions, and a candidate artifact digest. Emit all applicable rejection reasons rather than stopping at the first. Never read `scene_diagnostics.after`, `physical_after`, evaluator state, or dynamic verifier output as a feature source.

Use these exact audit contracts:

```python
@dataclass(frozen=True)
class HistoricalRowDecision:
    case_id: str
    family_id: str
    candidate_id: str | None
    candidate_fingerprint: str | None
    admissible: bool
    reasons: tuple[str, ...]
    source_refs: tuple[str, ...]


@dataclass(frozen=True)
class HistoricalCompatibilityAudit:
    family_id: str
    source_suite: str
    source_manifest_sha256: str
    examined_count: int
    admissible_tier_a_count: int
    rejected_count: int
    rejection_counts: Mapping[str, int]
    rows: tuple[HistoricalRowDecision, ...]
```

- [ ] **Step 4: Add a read-only audit CLI with isolated output**

Write `results/history_audit.json`, `results/admissible_rows.json`, `summary.md`, and `logs/runner.log` under a new `P5.6.2a_object6_history_audit` run directory. `admissible_rows.json` contains only rows that pass every check. Finalize and verify the artifact manifest.

- [ ] **Step 5: Run focused tests and audit the retained P5.5 object-6 suite**

Run:

```bash
python scripts/audit_p56_history.py \
  --suite-dir outputs/phase5/P5.5_matched_provenance_10seed_retry2_20260807/P5.5_frozen_ood_replay/20260807_024842_suite_20674432 \
  --family object-6 \
  --output-root outputs/phase5
```

Expected: command exits zero, examines 20 object-6 physical outcomes, reports them as inadmissible because retained P5.5 artifacts lack native P5.6 pre-execution feature snapshots and graph-event horizon lineage, and writes a fresh manifest-verified audit directory. The exact admissible count is read from the generated audit rather than hard-coded into source.

- [ ] **Step 6: Commit historical compatibility auditing**

```bash
git add capmas/evaluation/history_audit.py capmas/evaluation/__init__.py scripts/audit_p56_history.py tests/test_p56_history_audit.py
git commit -m "feat: audit historical P5.6 data compatibility"
```

### Task 7: Generate and run pre-registered object-6 collection blocks

**Files:**
- Modify: `capmas/contracts/calibration.py`
- Create: `scripts/create_p56_object6_manifests.py`
- Create: `scripts/run_libero_p56_collect.py`
- Create: `tests/test_libero_p56_collection.py`
- Create after running generator: `configs/phase5/p56_object6_id_seeds_11_20.json`
- Create after running generator: `configs/phase5/p56_object6_id_seeds_21_30.json`

**Interfaces:**
- Produces strict `CalibrationCollectionCase` and `CalibrationCollectionManifest` codecs and canonical digest helpers.
- Produces `create_object6_manifests(project_root) -> tuple[CalibrationCollectionManifest, CalibrationCollectionManifest]`.
- Produces `run_collection(manifest, *, output_root, run_config, online_runner=run_online_experiment, executor_factory=_build_live_executor) -> CollectionSuiteReport`.
- Produces `summarize_collection(suite_dirs, *, history_audit=None) -> CollectionEligibilityReport`.
- CLI run mode accepts `--manifest`, `--output-root`, `--gpu`, `--max-workers`, `--timeout-s`, `--max-restarts`, `--max-steps`, and `--fail-fast`; summary mode accepts repeatable `--summarize-suite` plus optional `--history-audit`.

- [ ] **Step 1: Write failing manifest and fake-runner tests**

```python
def test_object6_manifests_are_disjoint_complete_fixed_blocks() -> None:
    first, second = create_object6_manifests(ROOT)

    assert tuple(case.seed for case in first.cases) == tuple(range(11, 21))
    assert tuple(case.seed for case in second.cases) == tuple(range(21, 31))
    assert set(case.seed for case in first.cases).isdisjoint(case.seed for case in second.cases)
    assert all(case.split_identity == "id" for case in (*first.cases, *second.cases))
    assert all(case.family_id == "object-6" for case in (*first.cases, *second.cases))


def test_collection_manifest_round_trip_preserves_digest() -> None:
    first, _ = create_object6_manifests(ROOT)
    restored = CalibrationCollectionManifest.from_dict(first.to_dict())

    assert restored == first
    assert collection_manifest_sha256(restored) == first.manifest_sha256


def test_collection_captures_all_candidates_but_labels_only_selected(tmp_path) -> None:
    manifest = _collection_manifest(seeds=(11,))

    def fake_runner(**kwargs):
        assert kwargs["calibration_context"].episode_id == "object-6-id-seed11"
        return _online_outcome_with_two_snapshots(selected="candidate-a", success=True)

    report = run_collection(
        manifest,
        output_root=tmp_path,
        run_config=CollectionRunConfig(max_workers=1, gpu="5"),
        online_runner=fake_runner,
        executor_factory=lambda **kwargs: object(),
    )

    assert report.completed_cases == 1
    rows = json.loads((report.suite_dir / "results" / "outcomes.json").read_text())
    assert [(row["candidate_id"], row["tier"], row["task_success"]) for row in rows] == [
        ("candidate-a", "A", True),
        ("candidate-b", "C", None),
    ]


def test_collection_runs_whole_block_without_adaptive_stopping(tmp_path) -> None:
    calls = []

    def fake_runner(**kwargs):
        calls.append(kwargs["seed"])
        return _online_outcome(success=kwargs["seed"] == 11)

    run_collection(
        _collection_manifest(seeds=(11, 12, 13)),
        output_root=tmp_path,
        run_config=CollectionRunConfig(),
        online_runner=fake_runner,
        executor_factory=lambda **kwargs: object(),
    )

    assert calls == [11, 12, 13]
```

- [ ] **Step 2: Run collection tests and verify failure**

Run: `pytest -q tests/test_libero_p56_collection.py`

Expected: FAIL because collection codecs, manifest generator, and runner are absent.

- [ ] **Step 3: Implement immutable collection manifest generation**

Every generated case uses:

```python
config_path = "configs/phase5/capx_libero_object_6_nonprivileged.yaml"
candidate_artifact = "outputs/phase5/P5.5_real_layout_assets_20260803/candidates/object_6.json"
family_id = "object-6"
task_id = "libero_object_6"
object_name = "butter"
target_name = "basket"
split_identity = "id"
layout_family = "native-object-6"
```

Add these exact collection contracts and runner records:

```python
@dataclass(frozen=True)
class CalibrationCollectionCase:
    case_id: str
    lineage_group_id: str
    family_id: str
    task_id: str
    seed: int
    split_identity: Literal["id"]
    config_path: str
    config_sha256: str
    candidate_artifact: str
    candidate_artifact_sha256: str
    object_name: str
    target_name: str
    layout_family: str
    layout_variant: Mapping[str, object]


@dataclass(frozen=True)
class CalibrationCollectionManifest:
    manifest_id: str
    schema_version: str
    cases: tuple[CalibrationCollectionCase, ...]
    feature_schema_version: str
    memory_skill_version: str
    robot_skill_version: str
    prompt_version: str
    environment_version: str
    code_revision: str
    manifest_sha256: str = ""


@dataclass(frozen=True)
class CollectionRunConfig:
    max_workers: int = 1
    timeout_s: float = 360.0
    max_restarts: int = 0
    max_steps: int = 32
    gpu: str = "5"
    fail_fast: bool = False


@dataclass(frozen=True)
class CollectionCaseResult:
    case_id: str
    seed: int
    status: Literal["completed", "failed"]
    case_dir: Path
    outcomes: tuple[CalibrationOutcome, ...]
    error: str | None = None


@dataclass(frozen=True)
class CollectionSuiteReport:
    suite_dir: Path
    cases: tuple[CollectionCaseResult, ...]
    completed_cases: int
    failed_cases: int
    tier_a_count: int
    positive_count: int
    negative_count: int
    eligible_20_5_5: bool


@dataclass(frozen=True)
class CollectionEligibilityReport:
    source_suites: tuple[str, ...]
    history_audit: str | None
    admissible_tier_a_count: int
    positive_count: int
    negative_count: int
    eligible_20_5_5: bool
```

Resolve paths before hashing, store their SHA-256 digests, use `lineage_group_id == case_id`, set zero-delta native layout transforms for `butter_1_main` and `basket_1_main`, and freeze Memory Skill, Robot Skill, prompt, environment, feature-schema, and code-revision versions in each manifest. Generator output must be byte-identical across repeated runs from the same revision except for no timestamp field; the canonical manifest digest is content-addressed.

- [ ] **Step 4: Implement the bounded collection runner**

Use `Phase5RunDirectory` for the suite and every case. For each manifest case,
load and digest-check the frozen candidate artifact and construct context with
exact keyword arguments:

```python
CalibrationCollectionContext(
    episode_id=case.case_id,
    episode_epoch=1,
    family_id=case.family_id,
    feature_schema_version=manifest.feature_schema_version,
    memory_skill_version=manifest.memory_skill_version,
    robot_skill_version=manifest.robot_skill_version,
    collection_lane="physical",
)
```

Call `run_online_experiment` in `online_bounded` mode with cache disabled, one
selection, `max_workers=1`, and one physical executor. Convert the selected
snapshot into Tier A only when physical execution started and evaluator
success is conclusive; keep unselected snapshots Tier C. Persist per-case
snapshots, physical payload, horizon, outcomes, failure provenance, and logs
even when a case fails. Do not stop based on success/failure balance; only
`--fail-fast` may stop on an infrastructure exception.

- [ ] **Step 5: Generate and verify both committed manifests**

Run:

```bash
python scripts/create_p56_object6_manifests.py --project-root .
python scripts/create_p56_object6_manifests.py --project-root . --check
```

Expected: first command writes both JSON manifests; second exits zero after byte and digest verification. Seed blocks are exactly 11-20 and 21-30 with no overlap.

- [ ] **Step 6: Run focused collection tests and existing online tests**

Run: `pytest -q tests/test_libero_p56_collection.py tests/test_libero_p53_online.py tests/test_libero_p55_ood.py`

Expected: all tests pass and fake-runner tests confirm no adaptive per-outcome stopping.

- [ ] **Step 7: Run block 11-20 only when the history audit fails the 20/5/5 gate**

Run:

```bash
CUDA_VISIBLE_DEVICES=5 .venv-libero/bin/python scripts/run_libero_p56_collect.py \
  --manifest configs/phase5/p56_object6_id_seeds_11_20.json \
  --output-root outputs/phase5 \
  --gpu 5 \
  --max-workers 1 \
  --timeout-s 360 \
  --max-restarts 0 \
  --max-steps 32
```

Expected: one fresh suite directory with ten completed-or-typed-failure case directories, one physical execution at most per case, decision-time feature snapshots, horizon events, normalized Tier A/C rows, logs, and a valid manifest. Evaluate the complete block only after seed 20 finishes.

- [ ] **Step 8: Gate the second fixed block without outcome-adaptive selection**

Resolve the completed suite and history-audit artifacts deterministically, then
summarize them together:

```bash
BLOCK_11_20_SUITE_DIR=$(find outputs/phase5/P5.6.2a_object6_collection -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
HISTORY_AUDIT=$(find outputs/phase5/P5.6.2a_object6_history_audit -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
python scripts/run_libero_p56_collect.py \
  --summarize-suite "$BLOCK_11_20_SUITE_DIR" \
  --history-audit "$HISTORY_AUDIT"
```

Inspect the generated `results/eligibility.json`. If total admissible object-6
Tier A rows remain below 20 or either label class remains below five, run the
already committed 21-30 manifest with the same command and settings. If the
first block closes 20/5/5, record block 21-30 as
`not_run_gate_already_met`; do not choose individual seeds based on observed
outcomes.

If block 21-30 runs, summarize both suite directories with two
`--summarize-suite` arguments and the same history audit. Reject duplicate
case IDs or lineage groups across summary inputs.

- [ ] **Step 9: Commit collection infrastructure and frozen manifests**

```bash
git add capmas/contracts/calibration.py scripts/create_p56_object6_manifests.py scripts/run_libero_p56_collect.py tests/test_libero_p56_collection.py configs/phase5/p56_object6_id_seeds_11_20.json configs/phase5/p56_object6_id_seeds_21_30.json
git commit -m "feat: add pre-registered P5.6 object collection"
```

### Task 8: Synchronize P5.6A documentation and run the code gate

**Files:**
- Modify: `docs/implementation-roadmap.md`
- Modify: `docs/phase5-evidence-evolution.md`
- Modify: `docs/experiments.md`
- Modify: `docs/glossary.md`
- Modify: `tests/test_phase5_docs.py`

**Interfaces:**
- Records P5.6A completion state without claiming calibration or downstream success improvement.
- Records capability results separately for `spatial-0`, `goal-1`, and `object-6`.
- Records historical compatibility count, executed collection blocks, label balance, H buckets, and artifact paths.

- [ ] **Step 1: Write failing documentation assertions**

```python
def test_phase5_docs_record_p56a_data_boundary() -> None:
    phase5 = (ROOT / "docs/phase5-evidence-evolution.md").read_text(encoding="utf-8")
    roadmap = (ROOT / "docs/implementation-roadmap.md").read_text(encoding="utf-8")
    experiments = (ROOT / "docs/experiments.md").read_text(encoding="utf-8")
    glossary = (ROOT / "docs/glossary.md").read_text(encoding="utf-8")

    for document in (phase5, roadmap, experiments):
        assert "P5.6A data foundation" in document
        assert "p56.feature.v1" in document
        assert "P5.3.2 Task-Family Capability Repair" in document
        assert "max_steps=32 is not a horizon" in document
    assert "Tier A" in glossary
    assert "decision-time feature snapshot" in glossary
```

- [ ] **Step 2: Run the docs test and verify failure**

Run: `pytest -q tests/test_phase5_docs.py::test_phase5_docs_record_p56a_data_boundary`

Expected: FAIL until status and experiment evidence are added.

- [ ] **Step 3: Update status, experiment provenance, and phase boundaries**

Document the exact code/test commit, capability artifact directories, history-audit directory, collection suite directories, case counts, positive/negative label counts, infrastructure-unknown count, and manifest verification. Mark P5.6.0/1/2 complete only when their code and artifact gates pass. Mark P5.6.2a complete only when object-6 has at least 20 admissible Tier A outcomes with at least five positives and five negatives; otherwise mark it blocked by data coverage. State that P5.6B/C remain open and no calibrated score, probability, active snapshot, online success improvement, or all-family claim exists yet.

- [ ] **Step 4: Run focused tests, full regression, compilation, lint, and diff checks**

Run:

```bash
pytest -q tests/test_p56_contracts.py tests/test_p56_horizon.py tests/test_p56_feature_snapshots.py tests/test_p56_dataset.py tests/test_p56_capability.py tests/test_p56_history_audit.py tests/test_libero_p56_collection.py tests/test_phase5_docs.py
pytest -q
python -m compileall -q capmas scripts
ruff check capmas scripts tests
git diff --check
```

Expected: focused and full suites pass, compileall and ruff exit zero, and `git diff --check` emits no output.

- [ ] **Step 5: Verify every new experiment manifest**

For each P5.6 capability, history-audit, and collection run directory, recompute every file SHA-256 listed in `manifest.json` and compare file size and digest. Write the verification result into that run's `results/manifest_verification.json`, regenerate `manifest.json`, and verify once more. Expected: zero missing files, zero size mismatches, and zero digest mismatches.

- [ ] **Step 6: Commit synchronized status and evidence**

```bash
git add docs/implementation-roadmap.md docs/phase5-evidence-evolution.md docs/experiments.md docs/glossary.md tests/test_phase5_docs.py
git commit -m "docs: record P5.6A data foundation status"
```

## Final Acceptance Checklist

- [ ] P5.6 contracts round-trip with unknown values preserved and no secrets.
- [ ] Graph execution events cover action, checkpoint, retry, recovery, and terminal paths.
- [ ] Frozen P5.5 three-family graphs classify as H2-3; `max_steps=32` never enters a horizon field.
- [ ] Feature snapshots are written before physical execution and exclude dynamic verifier and OOD aggregate leakage.
- [ ] Tier A labels exist only for selected, physically executed candidates with conclusive evaluator outcomes.
- [ ] ID/OOD pairs and retries cannot cross train/calibration/test partitions.
- [ ] P5.6.0 diagnosis leaves its source suite byte-identical and emits typed P5.3.2 handoffs for failed families.
- [ ] Historical object-6 records are accepted only when native decision-time feature and horizon lineage are present.
- [ ] Seed blocks 11-20 and 21-30 are immutable, disjoint, ID-only, and evaluated only as complete blocks.
- [ ] Every run has a separate log/artifact directory and a verified manifest.
- [ ] Full regression, compileall, ruff, and `git diff --check` pass.
- [ ] Documentation distinguishes P5.6A data readiness from P5.6B calibration and P5.6C online promotion.
