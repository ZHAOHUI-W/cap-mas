# Phase 5 Evidence Evolution Handoff

## Purpose

This document records the optimizations intentionally deferred from the small
P3.2 closure. P3.2 establishes provenance, perception-only evidence, explicit
Arbiter basis labels, and a CAP-X-compatible online path. Phase 5 may add
candidate-specific evidence only after those artifacts are stable.

The boundary is important: a `confidence_fallback` or perception-only
`evidence_score` is a valid baseline, but it is not evidence that a candidate
will succeed under its proposed action. Phase 5 is where action-conditioned
quality and generalization evidence are introduced and evaluated.

## Deferred work packages

| Package | Scope | Required dependency | Exit evidence |
| --- | --- | --- | --- |
| P5.1 verifier evidence | Static coverage plus post-execution, candidate-specific predicate outcomes | Observable Verifier and rolling scene refresh | Per-candidate pass/unknown/fail trace with source scene version |
| P5.2 geometry and grasp | Reachability, grasp quality, clearance, collision risk, and target-pose feasibility conditioned on the candidate graph | Stable 3D scene/object-pose alignment | Reproducible geometry breakdown; no privileged evaluator in the online path |
| P5.3 process rehearsal | CAP-X/LIBERO rollouts in spawned simulator workers with reset, timeout, respawn, and matched seeds | Serializable graph and skill contracts | Candidate-level rehearsal result with seed, checkpoint, failure, and latency |
| P5.4 evidence cache | Cache evidence by candidate fingerprint and source `SceneSnapshot.scene_version`; invalidate on refresh | P5.1/P5.2 provider contracts | No stale evidence reaches Arbiter; cache hit/miss and invalidation are logged |
| P5.5 OOD replay | Frozen layout/object/task variants, confidence intervals, and leakage controls | Reproducible rehearsal and locked evaluation split | ID/OOD success gap and uncertainty intervals without training/eval leakage |
| P5.6 calibration | Correct for correlated perception, verifier, rehearsal, and OOD signals | At least two independent evidence sources | Ablation showing calibrated evidence does not double-count the same fact |

## Sequencing

Implement the packages in this order:

```text
P5.1 verifier evidence
        ↓
P5.2 geometry/grasp evidence
        ↓
P5.3 isolated process rehearsal
        ↓
P5.4 versioned asynchronous cache
        ↓
P5.5 frozen OOD replay
        ↓
P5.6 evidence calibration
```

P5.1 and P5.2 may share the same immutable `SceneSnapshot`, but neither may
evaluate a downstream postcondition against the initial scene. P5.3 must run
outside the live `ActionLease` and physical Executor. P5.4 must reject any
evidence whose scene version or candidate fingerprint no longer matches the
arbitration request. P5.5 must remain evaluation-only and use frozen seeds and
task variants. P5.6 is required before evidence weights are learned or used to
claim a downstream success-rate improvement.

## Explicit non-goals for Phase 5

- No direct LLM call in the high-frequency perception/control loop.
- No simulator rehearsal process may acquire the live robot lease.
- No physical parallel execution unless the graph declares disjoint resources
  and an explicit join/checkpoint.
- No replacement of CAP-X skill APIs; the adapter remains the comparison seam.
- No simultaneous Memory Skill and Robot Skill evolution. The existing order is
  memory first, then robot skill promotion in a later phase.

## Phase 5 evaluation matrix

Every increment should compare the same task seeds and report:

- downstream task success and success by horizon bucket;
- candidate validity, rejection, and arbitration basis distribution;
- raw versus normalized candidate diversity;
- evidence latency, cache hit rate, stale invalidation, and timeout rate;
- recovery count and human intervention count;
- ID/OOD success gap once the frozen OOD suite is enabled.

The minimum acceptance condition is improvement on the locked downstream suite
without a regression in CAP-X parity, scene freshness deadline, or the
single-owner actuator invariant. A successful rehearsal score alone is not a
terminal task reward.

## Current implementation boundary

`capmas/evaluation/rehearsal.py` already defines a serializable spawned-process
boundary and unit tests for it. That module is preparatory infrastructure only:
it is not wired into the P3.2 online candidate provider, does not provide
LIBERO reset/respawn semantics, and does not produce OOD or calibrated evidence.
