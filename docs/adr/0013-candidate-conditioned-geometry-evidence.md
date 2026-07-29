# ADR-0013: Candidate-Conditioned Geometry and Grasp Evidence

## Status

Accepted

## Context

P3.2 perception evidence is mostly a property of the current
`SceneSnapshot`. Candidates that refer to the same tracks can therefore
receive identical evidence even when their grasp pose, approach, clearance, or
motion plan differs. A constant confidence fallback cannot establish that one
candidate is physically more feasible than another.

The first P5.2 proposal placed geometry fields directly in
`PerceptionEvidence` and assumed Policy-generated `approach` and `standoff`
skill arguments. That conflicts with the current CAP-X skill registry, whose
LIBERO seam uses registered calls such as `sample_grasp_pose(object_name,
use_multiview)` and `goto_pose(position, quaternion_wxyz, z_approach)`.

## Decision

1. Geometry is a separate `GeometryEvidence` object under `CandidateEvidence`,
   not an extension of `PerceptionEvidence`.
2. Each geometry dimension uses `pass`, `fail`, or `unknown` semantics. Unknown
   never becomes zero and never becomes pass.
3. An action node may carry typed `MotionIntent` metadata. The intent is not a
   Robot Skill and is canonicalized and checked against registered skill calls
   by `CandidateNormalizer`.
4. Candidate-specific evidence is computed from the effective normalized
   candidate fingerprint, one immutable `SceneSnapshot`, and one frozen map
   view. Any grounding or repair that changes executable motion intent forces
   evidence recomputation.
5. Preview uses a side-effect-free `MotionPreviewBackend`. The reference
   implementation performs conservative workspace, pose, map-corridor, and
   clearance checks. A CAP-X planner adapter may add read-only IK and collision
   queries. Preview never calls physical Robot Skills or obtains `ActionLease`.
6. `reachability.fail` and `collision_risk.fail` may be hard gates. Unknown
   dimensions are excluded from soft scoring and are recorded with a reason.
7. P5.2 uses fixed transparent geometry weights. Weight learning and evidence
   correlation calibration are deferred to P5.6.
8. Geometry preview runs before lease acquisition under a 50 ms global proposal
   wave deadline. Candidate-level timeouts return unknown and cannot block the
   high-frequency World Model.
9. Primary experiments use sensor-derived realistic `SceneSnapshot` data.
   Privileged LIBERO poses are allowed only in a separately labelled diagnostic
   ablation and cannot enter primary success statistics.

## Consequences

Positive:

- Arbiter can distinguish candidates that differ in actual motion intent.
- Geometry provenance and scene versions support safe evidence cache invalidation.
- Candidate-specific evidence survives later evidence calibration without
  conflating scene perception and action feasibility.
- The existing CAP-X Robot Skill API and single physical Executor remain
  unchanged.
- Preview failures are fail-closed without turning unavailable information
  into artificial low confidence.

Negative:

- The graph contract and strict decoder gain a typed optional intent field.
- A complete reachability/collision score requires a read-only planner backend;
  the reference backend may return unknown for those dimensions.
- Candidate evidence computation adds pre-lease latency and must be carefully
  bounded and measured.
- Geometry evidence alone does not establish downstream task success; verifier
  and rehearsal evidence remain necessary.

## Alternatives rejected

- Adding geometry fields to `PerceptionEvidence`.
- Passing unregistered `approach` or `standoff` arguments to CAP-X skills.
- Calling physical `goto_pose` as a feasibility probe.
- Treating `None` or timeout as a zero score.
- Using LIBERO evaluator or ground-truth object state in the primary evidence
  path.
