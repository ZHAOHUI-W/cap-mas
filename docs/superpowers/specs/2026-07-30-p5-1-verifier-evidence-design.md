# P5.1 Verifier Evidence Design

## Status

Written spec for user review. The design has been approved; this package follows the
P5 evidence sequence in docs/phase5-evidence-evolution.md and depends on the
P5.0 scene/version contracts and the rolling Observable Verifier.

## Goal

Turn predicate verification into candidate-specific, scene-versioned evidence
that can be audited and consumed by the existing Arbiter without changing the
current ranking policy. Static evidence is available before arbitration;
dynamic postcondition evidence is produced only after execution and is
available to the next planning/recovery cycle.

The implementation also closes the existing predicate semantic bug where
object_in_gripper(obj) incorrectly requires a closed gripper.

## Non-goals

- Do not change the Arbiter weights or introduce learned evidence weighting.
- Do not use post-execution results to select the action that already ran.
- Do not add a new perception model, simulator rollout, TSDF backend, or OOD
  evaluator in P5.1.
- Do not require an external LLM/API call for verifier evidence.
- Do not remove the legacy scalar CandidateEvidence.verifier_pass_rate.

## Existing Contracts and Boundaries

The existing public seams remain the source of truth:

- PredicateBasedVerifier.evaluate_predicates() evaluates observable facts
  against one immutable SceneSnapshot.
- PredicateBasedVerifier.approve() checks action preconditions before
  execution.
- PredicateBasedVerifier.commit() checks expected postconditions against the
  after-action snapshot and returns VerificationResult.
- GraphCandidate.subgraph is the effective, normalized candidate. Evidence
  fingerprints are computed from this field, never from raw_subgraph.
- CandidateArbiter.select() remains the only selector and the physical
  Executor remains the only action executor.

P5.1 does not invent graph-level safety fields. The current
SubgraphNodeSpec has preconditions and postconditions, while ActionContract
has safety_invariants with a default empty tuple. Static candidate evidence
therefore evaluates only selected preconditions supplied by the caller.
Postconditions, checkpoint predicates, and any future safety predicates are
dynamic evidence unless an explicit future adapter declares them safe for
pre-execution evaluation.

## Design Alternatives

### Alternative A: Extend only scalar fields

Add more numbers to CandidateEvidence, such as a verifier rate and unknown
rate. This is simple but loses per-predicate provenance, cannot represent
three-state outcomes cleanly, and makes cross-candidate debugging difficult.

### Alternative B: Reuse VerificationResult directly

Attach VerificationResult to CandidateEvidence. This reuses an existing type
but binds evidence to an execution contract_id, mixes precondition and
postcondition timing, and does not identify the effective candidate graph.

### Alternative C: Typed verifier evidence with legacy scalar projection

Add immutable per-predicate evidence and a summary object, then project its
pass_rate to the existing scalar field. This preserves current Arbiter and
artifact consumers while making source identity, scene version, unknown state,
and static/dynamic timing explicit. P5.1 uses this alternative.

## Data Model

Create capmas/verification/evidence.py with the following immutable types.

~~~
from typing import Literal

VerifierPhase = Literal["static", "dynamic"]
VerifierStatus = Literal["pass", "fail", "unknown"]


@dataclass(frozen=True)
class VerifierPredicateEvidence:
    predicate: str
    phase: VerifierPhase
    status: VerifierStatus
    confidence: float | None
    reason: str | None
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class VerifierEvidence:
    candidate_fingerprint: str
    scene_version: int
    pass_rate: float
    coverage: float
    provider: str
    captured_at_ns: int
    static_results: tuple[VerifierPredicateEvidence, ...] = ()
    dynamic_results: tuple[VerifierPredicateEvidence, ...] = ()
    source_verification: str | None = None
~~~

Validation rules:

- Predicate names and provider are non-empty.
- candidate_fingerprint is non-empty, scene_version and captured_at_ns are
  non-negative.
- confidence, when present, is in [0, 1].
- pass_rate and coverage are in [0, 1].
- A pass or fail result requires a non-None confidence; an unknown result may
  have confidence=None.
- A result's phase must match the tuple containing it.
- Duplicate predicate names within one phase are rejected so that rates have a
  deterministic denominator.

The types expose deterministic JSON-compatible to_dict() methods. Tuple
fields are emitted as lists, and no live SceneSnapshot, simulator handle, or
in-memory artifact object is embedded in the result. from_dict() is not a
P5.1 requirement; artifacts are written by existing experiment serializers and
future replay code can use the stable dictionary shape.

## Summary Semantics

The summary is calculated independently for each evidence-producing call:

~~~
determined = count(status in {pass, fail})
passed = count(status == pass)
total = count(all results in the selected evidence set)
pass_rate = passed / determined if determined > 0 else 0.0
coverage = determined / total if total > 0 else 0.0
~~~

For a VerifierEvidence containing both static and dynamic results, the
summary uses the union of both phases. Static collection normally contains
only static results; dynamic conversion normally contains only dynamic
results. Unknown results never count as failures in pass_rate, but low
coverage must remain visible to callers and artifacts.

An empty result set has pass_rate=0.0 and coverage=0.0. It is unavailable
evidence, not a successful verifier result.

## Static Evidence Collection

Implement:

~~~
def collect_static_verifier_evidence(
    candidate: GraphCandidate,
    scene: SceneSnapshot,
    verifier: PredicateBasedVerifier,
    *,
    predicate_selector: Callable[[str], bool] | None = None,
    provider: str = "predicate_verifier.static",
    clock: Callable[[], int] = time.time_ns,
) -> VerifierEvidence: ...
~~~

The function:

1. Computes subgraph_fingerprint(candidate.subgraph).
2. Rejects the candidate with EvidenceCompatibilityError if
   candidate.parent_scene_version != scene.scene_version.
3. Collects predicates from action-node preconditions in stable node and
   declaration order, removes duplicates, and applies predicate_selector when
   supplied.
4. Evaluates the selected predicates through the public
   verifier.evaluate_predicates() seam.
5. Converts each PredicateReport into a static
   VerifierPredicateEvidence.
6. Returns evidence bound to the effective candidate fingerprint and source
   scene version.

If no selector is supplied, all action-node preconditions are considered
static. The LIBERO adapter passes compile_time_preconditions semantics so that
downstream state facts such as object_in_gripper, object_at_target, and
gripper state are deferred until their runtime dispatch point. A static
collection must not inspect node postconditions or validating checkpoint
predicates.

## Dynamic Evidence Conversion

Implement:

~~~
def verifier_evidence_from_result(
    candidate_fingerprint: str,
    result: VerificationResult,
    *,
    provider: str = "predicate_verifier.dynamic",
    clock: Callable[[], int] = time.time_ns,
) -> VerifierEvidence: ...
~~~

The converter maps every PredicateReport in result.predicate_results to a
dynamic result. result.checked_scene_version is the source scene version and
result.contract_id is recorded as source_verification. No candidate graph is
reconstructed from the contract.

Report classification is deterministic:

- passed=True becomes pass.
- A missing observation or unsupported observable fact becomes unknown,
  including reasons such as unknown predicate, unknown observable predicate,
  track not found, object track not found, pose is unavailable, not observed,
  and unavailable.
- A measured predicate contradiction becomes fail, including gripper state
  threshold failures, measured distance threshold failures, stale scene age,
  malformed predicate arguments, and explicit safety/precondition failures.

This classification is implemented in one helper with explicit reason tokens;
callers must not duplicate ad-hoc string checks. Unknown classification is
conservative: an unavailable measurement is not treated as a physical failure,
but it still lowers coverage.

## CandidateEvidence Compatibility

Extend CandidateEvidence with:

~~~
verifier: VerifierEvidence | None = None
~~~

Compatibility rules:

- Existing constructors that provide only scalar fields remain valid.
- If verifier is present, verifier_pass_rate must equal verifier.pass_rate,
  scene_version must equal verifier.scene_version when both are present, and
  provider/captured_at_ns must not contradict the typed evidence when supplied.
- When a typed verifier is present, verifier in available_metrics requires
  verifier.coverage > 0.0. The legacy scalar path may still declare verifier
  in available_metrics without a typed verifier.
- A typed verifier with zero coverage may be attached for diagnostics, but the
  caller must omit verifier from available_metrics; the Arbiter then treats
  the typed dimension as unavailable instead of as a zero-quality score.
- A helper attach_verifier_evidence(base, verifier) returns a new
  CandidateEvidence using dataclasses.replace, sets the scalar projection,
  propagates scene/provider/timestamp fields, and adds verifier only when
  coverage is positive. It never mutates the source object.

The existing libero_candidate_evidence() provider will compose its current
perception evidence with static verifier evidence. It will use the effective
candidate, current scene, and LIBERO's compile-time predicate filtering. This
is the first online path in which the Arbiter can consume real
candidate-bound verifier evidence; no LLM call is involved.

## Arbiter and Cache Integration

CandidateArbiter._evidence_gate() will validate typed verifier evidence in
addition to the existing scene and geometry checks:

- verifier.scene_version == scene.scene_version;
- verifier.candidate_fingerprint == subgraph_fingerprint(candidate.subgraph);
- the scalar projection is consistent with the typed summary.

Any mismatch rejects the candidate as STALE_EVIDENCE before scoring. The
existing P5.4 VersionedEvidenceCache stores the enclosing CandidateEvidence;
no new cache key is introduced. This preserves the candidate-fingerprint plus
scene-version cache contract.

Dynamic evidence is not attached to a candidate that has already been
selected. Runtime integration stores it in the execution trace/artifact and
passes it to the next rolling cycle's context, recovery selector, memory
controller, or rehearsal adapter. A later candidate may receive it only after
being regenerated for the new scene version.

## Predicate Semantics Fix

Update PredicateBasedVerifier._evaluate() as follows:

- object_in_gripper(obj_id) checks that the object track and end-effector
  position are available and their distance is within
  object_gripper_distance_threshold_m. It does not read gripper opening.
- gripper_closed() independently checks gripper opening against
  gripper_closed_threshold.
- object_held(obj_id) is the strict composite predicate: object is in the
  gripper region and gripper is closed.
- object_near_gripper(obj_id) keeps its existing distance-only semantics.

Tests that intended the old strict behavior must use object_held(obj_id) or
explicitly combine object_in_gripper(obj_id) and gripper_closed().

## Error Handling and Provenance

- Candidate/scene version mismatches raise EvidenceCompatibilityError and are
  converted by the existing candidate proposal boundary into a candidate
  diagnostic; they must not silently produce evidence for the wrong scene.
- Verifier exceptions for an individual predicate become an unknown result
  with the exception text in reason; one bad predicate must not discard other
  predicate results.
- Fingerprints always refer to the normalized/effective subgraph.
- Evidence references remain URI strings and are copied immutably into the
  typed result.
- Provider name and capture timestamp are recorded for every non-empty or
  diagnostic result.

## Testing and Acceptance Gates

Tests must use public seams and cover:

1. Typed evidence validation, three-state semantics, summary calculation,
   duplicate rejection, and deterministic dictionary output.
2. Static collection from candidate preconditions, selector filtering,
   candidate fingerprint binding, and scene-version rejection.
3. Dynamic conversion of pass/fail/unknown reports and preservation of
   contract/scene provenance.
4. CandidateEvidence scalar compatibility, typed-field consistency, and
   immutable attachment.
5. Arbiter rejection of stale/cross-candidate verifier evidence and scoring
   through the existing verifier_pass_rate projection.
6. Regression coverage proving object_in_gripper passes with an open gripper
   when the object is in the end-effector region, while object_held requires
   closure.
7. Existing full test suite remains green.

The P5.1 implementation gate is closed only when the focused verifier/evidence
tests and the full repository test suite pass. The separate empirical gate is
a CUDA-visible LIBERO smoke run followed by a matched multi-seed run; each run
uses a new experiment directory containing logs/, results/, summary.json,
summary.md, and manifest.json. Those experiments validate runtime effect, not
type-contract completeness.

## Explicitly Deferred

- Candidate-conditioned geometry and grasp evidence remain P5.2.
- Process rehearsal remains P5.3.
- Evidence cache remains P5.4.
- OOD replay and calibration remain P5.5/P5.6.
- Learned verifier confidence or RL-based verifier updates remain later work.
