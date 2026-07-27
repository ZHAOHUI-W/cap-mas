# CAP-MAS P3.2 Evidence Precedence and Arbitration Closure

## Problem Statement

The first P3.2 slice attaches SceneSnapshot-backed perception evidence, but
scheduler-created candidates still inherit a synthetic `confidence=0.5`. The
Arbiter also does not distinguish an evidence tie from an evidence-free
confidence fallback, and evidence is not explicitly bound to the scene version
that produced it. These ambiguities make multi-Policy artifacts difficult to
interpret.

## Solution

Close P3.2 with a bounded contract update: scheduler confidence becomes
optional and fallback-only; evidence-mode scores exclude it; candidate evidence
records provider and source scene version; stale evidence is rejected; and
arbitration records explicit basis labels. Defer dynamic geometry, rehearsal,
OOD, and evidence calibration to Phase 5.

## User Stories

1. As an experimenter, I want scheduler candidates to omit synthetic
   confidence, so that artifacts do not present a constant as model quality.
2. As an Arbiter, I want evidence-mode scores to exclude legacy confidence, so
   that candidate-specific evidence determines quality ranking.
3. As an experimenter, I want evidence ties labeled separately from confidence
   fallback, so that a deterministic baseline is not reported as evidence-based
   selection.
4. As a runtime, I want evidence tied to a SceneSnapshot version, so that stale
   perception cannot influence a newer arbitration decision.
5. As a CAP-X researcher, I want the existing single physical Executor and
   skill API boundary unchanged, so that P3.2 remains comparable with CAP-X.
6. As a maintainer, I want dynamic rehearsal and OOD work isolated in a later
   phase, so that P3.2 remains fast and its ablations remain interpretable.

## Implementation Decisions

- `GraphCandidate.confidence` is optional. Existing explicit confidence values
  remain accepted for legacy tests and fallback ablations.
- `CandidateEvidence` carries `scene_version`, provider identity, and capture
  timestamp as provenance.
- Evidence mode excludes `GraphCandidate.confidence` from score breakdowns.
- Missing evidence in an otherwise evidence-enabled arbitration is rejected;
  all-evidence-free arbitration is labeled `confidence_fallback`.
- Equal evidence scores are labeled `evidence_tie_break` and still use stable
  duration/resource/id ordering.
- No CAP-X worker, rehearsal rollout, OOD suite, or dynamic geometry scorer is
  added in this closure.

## Testing Decisions

- Test the public `CandidateArbiter.select()` seam for confidence exclusion,
  stale evidence rejection, and basis labels.
- Test the scheduler compile seam to confirm default candidates have no
  confidence value and use fallback labeling when no evidence provider exists.
- Test the LIBERO provider emits scene-version and provider provenance.
- Reuse existing scheduler, graph-runtime, LIBERO evidence, and serialization
  tests; run the full suite and compile checks before closing the issue.

## Out of Scope

- Candidate-specific dynamic verifier rollouts.
- Action-conditioned geometry, collision, and grasp-quality scoring.
- CAP-X/MuJoCo process rehearsal and worker respawn.
- Frozen OOD replay, confidence intervals, and leakage controls.
- Learned Arbiter calibration or RL.
- Physical parallel execution.

## Further Notes

Phase 5 owns the deferred evidence stack. Historical artifacts with
`confidence=0.5` remain valid regression records but are not closure-semantics
quality results.
