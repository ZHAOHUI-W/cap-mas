# ADR-0010: Evidence Precedence and Deferred Candidate Rehearsal

## Status

Accepted

## Context

P3.2 initially assigned every scheduler-created candidate a default confidence
of `0.5`. The value did not come from the Policy model or from execution
evidence. At the same time, the first LIBERO evidence provider measured mostly
SceneSnapshot properties, so two candidates referring to the same tracks could
receive identical perception scores. Treating either case as a quality ranking
would overstate the multi-Policy contribution.

Expensive candidate-specific evidence requires geometry analysis, dynamic
postcondition evaluation, isolated CAP-X rehearsal, and frozen OOD replay.
Those operations have different latency, reproducibility, and simulator-state
requirements from the P3.2 online proposal path.

## Decision

1. Scheduler-created `GraphCandidate.confidence` is optional and defaults to
   absent. It is retained only for explicit legacy/fallback ablations.
2. When candidate evidence is present, the Arbiter excludes legacy confidence
   from the quality score and ranks only declared evidence dimensions.
3. Candidate evidence records its provider and source `SceneSnapshot` version.
   Evidence with a mismatched scene version is rejected before arbitration.
4. Arbitration artifacts use separate basis labels:
   `evidence_score`, `evidence_tie_break`, `confidence_fallback`, and
   `hard_gate`.
5. P3.2 does not synthesize verifier, geometry, rehearsal, or OOD scores.
   Candidate-specific dynamic evidence, process rehearsal, OOD replay,
   evidence caching, and calibration are Phase 5 work.
6. The single physical Executor and ActionLease remain unchanged. Rehearsal is
   an offline/asynchronous evidence producer and cannot access live actuator
   authority.

## Consequences

Positive:

- A default constant cannot masquerade as model confidence.
- Evidence-free baselines are explicitly separated from evidence rankings.
- Stale evidence cannot silently influence arbitration.
- P3.2 remains bounded and compatible with the current CAP-X runner.
- Later rehearsal/OOD work can be evaluated as an isolated phase.

Negative:

- P3.2 may still produce `evidence_tie_break` when candidates have identical
  observable scene facts.
- Candidate quality is incomplete until Phase 5 adds action-conditioned and
  rollout evidence.
- Existing artifacts containing `confidence=0.5` remain historical baselines
  and must not be compared as if they used the closure semantics.
