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

## Phase 5 structure

Phase 5 starts with `P5.0` infrastructure completion and then develops two
lanes against the same immutable scene, trace, version, and provenance
contracts:

```text
P5.0 TSDF + semantic adapter foundation
        ├── Evidence Lane: P5.1 -> P5.2 -> P5.3 -> P5.4 -> P5.5 -> P5.6
        └── Memory Lane: working memory -> failure extraction -> hard cases
                          -> rules Memory Skills -> replay/promotion/rollback
```

Memory Lane is rules-based in Phase 5. Memory Controller RL belongs to Phase 6,
and Robot Skill evolution remains a later sequential phase.

## Deferred work packages

| Package | Scope | Required dependency | Exit evidence |
| --- | --- | --- | --- |
| P5.0 perception foundation | TSDF local-map backend, semantic adapter protocol/reference backend, asynchronous queues, realistic/diagnostic privilege separation | P4.5 process World Model gate | P5.0 latency/fallback gate and versioned sensor evidence |
| P5.1 verifier evidence | Static coverage plus post-execution, candidate-specific predicate outcomes | Observable Verifier and rolling scene refresh | Per-candidate pass/unknown/fail trace with source scene version |
| P5.2 geometry and grasp | Candidate-conditioned `MotionIntent`, read-only motion preview, reachability, grasp quality, clearance, collision risk, and target-pose feasibility | P5.0 map/adapter contracts and stable 3D scene/object-pose alignment | Distinct candidate geometry breakdown, hard-gate behavior, 50 ms wave budget, no privileged evaluator in primary path |
| P5.3 process rehearsal | CAP-X/LIBERO rollouts in spawned simulator workers with reset, timeout, respawn, and matched seeds | Serializable graph and skill contracts | Candidate-level rehearsal result with seed, checkpoint, failure, and latency |
| P5.4 evidence cache | Cache evidence by candidate fingerprint and source `SceneSnapshot.scene_version`; invalidate on refresh | P5.1/P5.2 provider contracts | No stale evidence reaches Arbiter; cache hit/miss and invalidation are logged |
| P5.5 OOD replay | Frozen layout/object/task variants, confidence intervals, and leakage controls | Reproducible rehearsal and locked evaluation split | ID/OOD success gap and uncertainty intervals without training/eval leakage |
| P5.6 calibration | Correct for correlated perception, verifier, rehearsal, and OOD signals | At least two independent evidence sources | Ablation showing calibrated evidence does not double-count the same fact |

## Sequencing

Implement the evidence packages in this order after P5.0:

```text
P5.0 foundation
        ↓
P5.1 verifier evidence
        ↓
P5.2 candidate-conditioned geometry/grasp evidence
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

After P5.0, Memory Lane may be developed in parallel, but its active Memory
Bank and rules controller must not change the evidence lane's locked baseline.
Evidence and memory are integrated only after their independent replay gates
pass.

### P5.2 accepted design boundary

P5.2 stores geometry separately from `PerceptionEvidence` in a typed
`GeometryEvidence` object. Its dimensions use `pass/fail/unknown` semantics;
unknown is never converted to zero. Action nodes may carry a typed
`MotionIntent`, which `CandidateNormalizer` canonicalizes and checks against
registered CAP-X skill calls. Unregistered `approach` or `standoff` skill
arguments are invalid.

The geometry provider consumes the effective normalized candidate fingerprint,
one immutable `SceneSnapshot`, and one frozen map view. It uses a side-effect-
free `MotionPreviewBackend` and runs before ActionLease under a 50 ms global
proposal-wave deadline. Candidate timeouts return unknown. Reachability and
collision failures may be hard gates; scoring uses fixed transparent weights
until P5.6 calibration. The primary mode is sensor-derived realistic state;
privileged LIBERO poses are diagnostic-only and are never pooled with primary
success statistics. The detailed contract is in
[`2026-07-28-phase5-2-candidate-conditioned-geometry-evidence.md`](superpowers/specs/2026-07-28-phase5-2-candidate-conditioned-geometry-evidence.md)
and ADR-0013.

The code-level P5.2 increment is available in `capmas/graph/normalizer.py`,
`capmas/perception/motion_preview.py`, and
`capmas/perception/geometry_evidence.py`. The existing B3-LLM runner accepts
`--geometry-mode disabled|shadow|online_bounded`; online mode attaches the
candidate-conditioned geometry component to Arbiter input, while shadow mode
computes and records it without changing selection. The pilot driver is
`scripts/run_libero_p52_geometry.py` and creates one independent artifact
directory for every mode/seed pair.

The strict LLM wire schema also exposes `motion_intent` on every graph node.
This is required for live Policy candidates to carry typed approach, standoff,
object-track, and target-track information; without it, the decoder reduced
every candidate to the same object-only intent. LIBERO scene grounding now
rebases a place intent's target pose to the effective grounded `goto_pose` while
preserving the candidate's approach and standoff fields. Geometry evidence is
therefore bound to the post-rewrite fingerprint rather than rejecting a valid
candidate as an intent/call conflict.

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

Every Phase 5 experiment writes to a new run-scoped directory and never
overwrites a prior run:

```text
outputs/phase5/<experiment_name>/<timestamp>_<run_id>/
  run_config.json  manifest.json  summary.json  summary.md
  logs/  results/  traces/  evidence/  artifacts/
```

Failed runs retain their failure artifact and complete log. API keys and
Authorization headers are never persisted.

### P5.2 pilot status (2026-07-29)

The endpoint-backed five-seed pilot completed all 15 independent runs using
CUDA device 5 and the CAP-X LIBERO Spatial-0 environment:

| mode | evaluator successes | geometry evidence | privileged state |
| --- | ---: | ---: | ---: |
| `geometry_disabled` | 2/5 | disabled | 0/5 |
| `geometry_shadow` | 2/5 | recorded, not selected | 0/5 |
| `geometry_online_bounded` | 2/5 | 4 candidate records/run | 0/5 |

All run directories contain their own configuration, manifest, summary,
episode result, and complete runner log. The online geometry provider stayed
well below the declared deadline (observed per-candidate P95 was approximately
0.152--0.183 ms), and all four candidate fingerprints per run were distinct.
The endpoint smoke after the schema and grounding fixes reached physical
execution and completed with `evaluator_success=true`.

This is not yet a full P5.2 evidence-closure result. In the realistic B3-LLM
path the reference preview still receives `local_map=None`, so the measured
geometry breakdown is `reachability=pass` while grasp quality, clearance, and
collision risk remain `unknown`. The online Arbiter records `evidence_score`,
but the observed winner is separated by perception/strategy weights rather
than by a distinct geometry score. The next blocking increment is therefore
P5.0 local-map/semantic geometry transport into the B3-LLM evidence provider;
P5.3 process rehearsal must wait until a realistic run can produce at least
one candidate-specific measurable geometry difference.

The minimum acceptance condition is improvement on the locked downstream suite
without a regression in CAP-X parity, scene freshness deadline, or the
single-owner actuator invariant. A successful rehearsal score alone is not a
terminal task reward.

## Current implementation boundary

`capmas/evaluation/rehearsal.py` already defines a serializable spawned-process
boundary and unit tests for it. That module is preparatory infrastructure only:
it is not wired into the P3.2 online candidate provider, does not provide
LIBERO reset/respawn semantics, and does not produce OOD or calibrated evidence.
