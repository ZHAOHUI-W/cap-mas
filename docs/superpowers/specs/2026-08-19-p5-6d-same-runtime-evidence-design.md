# P5.6D Same-Runtime Pre-Execution Evidence Design

## 1. Decision

P5.6D replaces the legacy calibration collection path that constructed a
synthetic empty `SceneSnapshot` before arbitration and independently created a
physical CAP-X/LIBERO runtime only after selection. It introduces one
`LiveLiberoEvidenceSession` per physical episode. The session owns the
environment from reset through the single permitted physical execution.

The decision-time evidence path is:

```text
CAP-X/LIBERO reset
  -> version-0 initial snapshot
  -> observe and commit version-1 snapshot
  -> perception + static verifier + candidate geometry evidence
  -> isolated rehearsal evidence
  -> Arbiter selects one candidate
  -> same session executes and verifies that candidate
```

No second environment may supply a P5.6 physical feature snapshot. Dynamic
verification, final scene observations, evaluator results, execution traces,
and recovery outcomes remain labels or diagnostics and are never added to the
decision-time feature snapshot.

## 2. Root Cause Addressed

The admissible seed-11--30 collections used
`scripts/run_libero_p53_online.py` with a synthetic scene containing no object
tracks, RGB-D artifacts, local map, or static predicate observations. Its only
available pre-decision feature was the isolated rehearsal result. Consequently
every `scene_grounding` and cost-risk dimension was `unknown`, train design
rank was two (intercept plus action feasibility), and the fitted calibrated
model could not improve on the fixed-weight/PAVA baseline.

This is an evidence-collection defect, not evidence that the constrained
logistic optimizer is unstable. The old collection and offline reports remain
immutable audit artifacts and are not mutated or relabelled.

## 3. Scope

### In scope

- A `LiveLiberoEvidenceSession` that owns one CAP-X/LIBERO runtime.
- A version-1 decision snapshot built from CAP-X RGB-D/object APIs after
  `RuntimeOrchestrator.start_episode()`.
- Candidate-bound perception and static verifier evidence from
  `libero_candidate_evidence()`.
- Candidate-bound, side-effect-free geometry evidence from
  `candidate_geometry_evidence()` and `ReferenceMotionPreview` using the
  session World Model's local map.
- Merging that evidence with isolated rehearsal evidence before Arbiter
  selection, without replacing either provenance lane.
- Reusing the same session to execute the selected graph once.
- P5.6 collection provenance that states whether a collection used
  `same_runtime` or legacy `rehearsal_only` mode.
- Unit/integration coverage for scene identity, evidence timing, feature
  availability, session closure, and old-path compatibility.

### Out of scope

- Changing the calibration reducer, optimizer, PAVA constants, qualification
  thresholds, feature schema, baseline weights, or fixed 12/4/4 split.
- Backfilling legacy rows or mixing old and new collection families.
- Activating a `CalibrationSnapshot`, Shadow Arbiter, canary, or online
  calibrated probability.
- Using privileged simulator object state as a feature in the physical lane.
- TSDF, semantic adapters, OOD feature activation, task-policy repair, or
  multi-task pooling.

## 4. Session Contract

`capmas.evaluation.libero_evidence_session` owns the live implementation and
exposes a small protocol used by the generic online runner:

```python
class PreExecutionEvidenceSession(Protocol):
    def start(self) -> SceneSnapshot: ...
    def candidate_evidence(self, candidate: GraphCandidate) -> CandidateEvidence: ...
    def execute(self, candidate: GraphCandidate, graph: MissionGraph) -> object: ...
    def close(self) -> None: ...
```

`start()` is single-use. It builds CAP-X resources, attaches the reference
RGB-D World Model enricher, resets with the frozen seed/layout, starts the
runtime at scene version 0, calls `backend.observe()`, and commits exactly one
version-1 decision snapshot. The returned scene must have the same version as
every submitted candidate; mismatch is fail-closed before any evidence or
execution occurs.

`candidate_evidence()` is valid only after `start()`. It rejects candidates
whose parent scene version differs from the session decision scene. It first
collects `libero_candidate_evidence()` from the committed snapshot and then
attaches a `GeometryEvidence` generated before the supplied monotonic deadline.
Geometry failures return typed `unknown` evidence and do not substitute values
from later observations. The merged envelope declares `perception`, `verifier`,
and `geometry` when those typed products exist; refs and provider metadata are
retained.

`execute()` is single-use. It grounds the submitted Mission Graph from the
same decision snapshot, uses the session's `RuntimeOrchestrator` and
`FixedGraphInterpreter`, then returns the normal physical-result diagnostics.
It cannot reset, observe, or rebuild an independent environment before the
first action. `close()` is idempotent and stops the backend even if selection
or execution raises.

## 5. Online Runner Integration

`run_online_experiment()` receives an optional
`evidence_session: PreExecutionEvidenceSession`. It keeps the synthetic scene
and caller-provided `physical_executor` behavior unchanged when no session is
given, preserving prior P5.3 tests and historical experiments.

With a session:

1. Call `start()` before the first baseline arbitration.
2. Enrich every typed candidate with `candidate_evidence()`.
3. Run the existing isolated rehearsal provider and merge its evidence onto
   each already enriched candidate.
4. Capture `CandidateFeatureSnapshot` values from the merged candidates.
5. Set `decision_completed_at_ns` only after snapshots are written.
6. Invoke `evidence_session.execute()` for the live winner.
7. Always call `evidence_session.close()`.

The physical result continues to use the established payload shape. The run
artifacts additionally record `decision_scene_version`, evidence mode,
per-candidate available metrics, and World Model diagnostics. They do not
persist raw RGB-D bytes in the calibration report.

## 6. P5.6 Collection Integration

`CollectionRunConfig` gains an `evidence_mode` field:

- `same_runtime`: required for all new P5.6D qualification collection;
- `rehearsal_only`: retained only to replay or inspect historical behavior and
  cannot be used to replace the P5.6D data requirement.

The CLI defaults to `same_runtime`; it creates a session factory for every
case, starts API servers once per suite as before, and permits one physical
execution per case. Existing programmatic tests that omit a session factory
continue through the legacy injection seam.

P5.6D will pre-register fresh native object-6 seed blocks before physical
collection. The current seed-11--30 feature-poor rows remain a failed
qualification attempt. New data must be evaluated only within a new immutable
dataset manifest and must satisfy all existing Tier-A lineage, timing, and
20/5/5 gates before another offline fit is considered.

The one-case transport gate uses `p56.collection.v2` with the signed
`collection_purpose="transport_smoke"`. Legacy `p56.collection.v1` manifests
retain their byte-identical implicit `qualification` purpose. A collection
summary preserves a smoke's physical execution diagnostics but excludes its
Tier-A row from the admissible calibration count. This prevents a smoke from
being accidentally reused as one of the 20/5/5 rows.

## 7. Safety and Leakage Invariants

- Every `feature_snapshot.captured_at_ns <= decision_boundary_ns <=
  physical_execution_started_at_ns`.
- All perception, verifier, and geometry evidence matches the candidate
  fingerprint and version-1 decision scene.
- The session does not expose CAP-X/LIBERO privileged evaluator state to
  feature extraction.
- A `GeometryEvidence` timeout or World Model failure produces explicit
  `unknown`, not `0.0`, and does not block execution by itself.
- Rehearsal remains isolated from the live session and has no `ActionLease` or
  live Executor access.
- The Arbiter still owns final selection; the session cannot submit an action
  itself.
- A session start/evidence/selection error closes the session and creates no
  Tier-A outcome.

## 8. Acceptance Criteria

1. A fake session test proves real-session mode calls `start`, collects
   evidence for all candidates before feature capture, executes only the live
   winner, and closes exactly once.
2. A session test proves mismatch between the live decision scene and candidate
   parent version fails before physical execution.
3. A feature snapshot from session-provided evidence contains non-unknown
   perception/geometry/verifier fields whenever the corresponding typed
   provider reports them.
4. Legacy `run_online_experiment()` callers without a session retain synthetic
   scene behavior and existing result artifacts.
5. P5.6 collection records `evidence_mode` and propagates its session factory
   to the online runner.
6. Focused tests, full `pytest -q`, `compileall`, touched-file Ruff, and
   `git diff --check` pass before a live smoke.
7. A new live single-seed smoke verifies the manifest, records real
   decision-time evidence, and has signed `transport_smoke` purpose; it is a
   transport gate only, not a calibration qualification or success-rate claim.
