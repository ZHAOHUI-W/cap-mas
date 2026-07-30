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

P5.0 is split into two gates. `P5.0-contract` closes the map-backend factory,
fail-closed fallback semantics, and scene/candidate version compatibility
needed by later evidence workers. `P5.0-runtime` remains open until a real
TSDF backend and a real semantic adapter can publish versioned results.
TSDF and real semantic adapters remain open.
`P5.3 may proceed` after the contract gate using the locked SparseVoxel
baseline; this does not make the TSDF or semantic runtime claims complete.

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

Implement the evidence packages in this order after P5.0-contract:

```text
P5.0-contract foundation
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

This 15-run pilot remains the historical baseline for the pre-transport path:
the B3-LLM provider supplied `local_map=None`, so the measured geometry
breakdown was only `reachability=pass` and the online winner was driven by
perception/strategy weights rather than a distinct geometry score.

The local-map transport closure was then validated in a fresh real endpoint
run at
`outputs/phase5/P5.2_live_map_fix_20260729/B3-LLM/20260729_085300_5d0d420c-4302-41b3-82e1-2f0e7d59f055/`.
That run used the same CAP-X Spatial-0 task, CUDA device 5, realistic sensor
mode, and `geometry_depth_subsample=16`. It completed physical execution with
`evaluator_success=true`, `map_version=4`, `processed_observations=4`, and no
World Model error. All four candidate geometry records used
`map_backend=local_map`, stayed below the 50 ms budget (23.83 ms maximum in
this run), and produced candidate-specific clearance scores (0.4713 versus
0.3604). The Arbiter recorded `selection_basis=evidence_score` and included a
non-zero geometry component in both action-subgoal score breakdowns.

The matched post-transport five-seed pilot was subsequently completed at
`outputs/phase5/P5.2_geometry_evidence_posttransport_20260729/` with
`geometry_depth_subsample=16`, CUDA device 5, realistic CAP-X RGB-D input, and
one immutable artifact directory per mode/seed pair:

| mode | evaluator successes | geometry records/run | map version | processed observations | observed geometry latency | privileged state |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `geometry_disabled` | 2/5 | 0 | 0 | 0 | n/a | 0/5 |
| `geometry_shadow` | 4/5 | 4 | 3--4 | 3--4 | 23.70--24.28 ms | 0/5 |
| `geometry_online_bounded` | 4/5 | 4 | 3--4 | 3--4 | 23.70--24.14 ms | 0/5 |

All 15 manifests and all 75 manifest-listed files passed size and SHA-256
verification. No World Model error was recorded in the geometry-enabled runs.
The geometry records were genuinely candidate-conditioned: each enabled run
contained two distinct clearance values (global observed range
0.3604--0.4895), while collision risk and reachability were measurable. The
reference backend still reports `grasp_quality=unknown` because surface
normal/contact estimation is intentionally not implemented.

The downstream result is promising but not attributable to online geometry in
this pilot: both `geometry_shadow` and `geometry_online_bounded` reached 4/5,
while `geometry_disabled` reached 2/5. However, every one of the 10 enabled
action-subgoal arbitration decisions used `selection_basis=evidence_tie_break`
and had tied candidate score breakdowns. Thus the online Arbiter consumed
geometry evidence, but did not yet produce a unique geometry-driven winner;
the 4/5 result must not be reported as a causal geometry improvement. This is
an evaluation limitation, not a transport failure.

Therefore the local-map transport blocker and the P5.2 five-seed execution gate
are closed.
The P5.2 causal selection-quality gate remains open because candidate scores
still tie at arbitration time, and the grasp-quality adapter also remains
open. P5.3 isolated process rehearsal can proceed as the next engineering
increment, but it must preserve this matched baseline and must not claim that
geometry improves downstream success until rehearsal evidence and a non-tied
candidate-selection analysis are available.

The minimum acceptance condition is improvement on the locked downstream suite
without a regression in CAP-X parity, scene freshness deadline, or the
single-owner actuator invariant. A successful rehearsal score alone is not a
terminal task reward.

## Current implementation boundary

`P5.0-contract` is implemented: `map_factory.py` provides an explicit
SparseVoxel backend factory with fail-closed unsupported-TSDF behavior, and
`evidence_contracts.py` rejects stale scene or candidate evidence.

P5.3 implementation is now available in `capmas/evaluation/libero_rehearsal.py`
and `scripts/run_libero_p53_rehearsal.py`. It provides a pickle-safe CAP-X
worker boundary, serialized graph reset/execute, checkpoint and failure
metadata, bounded worker respawn, version-bound rehearsal evidence, and one
artifact directory per seed. Rehearsal evidence is shadow by default; the
explicit Arbiter attachment helper must be enabled by the caller before it can
affect selection. The real CAP-X/LIBERO matched two-candidate, five-seed gate
is closed at
`outputs/phase5/P5.3_process_rehearsal_matched_fix_20260730/`: `policy-0:0`
reached 0/5 evaluator successes and `policy-1:safety:1` reached 2/5, with
candidate outcomes differing on seeds 1 and 5. The remaining failures were
classified as `postcondition_failure` and every seed has a separate log and
manifest.

The gate establishes candidate-specific process evidence, not a causal
downstream improvement. The identity-closure increment now preserves the raw
full-graph fingerprint and derives the local Arbiter identity from an explicit
`arbiter_subgraph_id`. Graph-scoped `RehearsalEvidence` carries the mapped
`arbiter_fingerprint`; `merge_rehearsal_evidence()` rejects missing, stale, or
mismatched mappings rather than silently attaching evidence to a different
candidate.

The same increment adds the pure
`capmas.evaluation.shadow_arbiter.run_shadow_arbitration()` API. It runs the
baseline Arbiter on the original candidates, runs a hypothetical evidence-aware
selection on copied candidates, and reports both winners, selection bases,
score breakdowns, and mapping rejections. A changed shadow winner is
diagnostic only: `physical_execution_required` is always false, and the
baseline candidate remains the live selection. Missing rehearsal evidence
leaves a candidate unchanged rather than assigning it a zero score. The
graph-to-subgraph identity gate and pure shadow-Arbiter gate are now closed at
the code/test level; the causal online-selection gate remains open.

The remaining P5.3 evaluation work is intentionally separate: expand beyond
the five-seed one-task pilot, compare a controlled online-selection mode, and
measure downstream success. Ten-plus seeds, multiple tasks, physical online
promotion, and the P5.4 versioned evidence cache are not closed by this
increment.

TSDF, real semantic adapters, OOD replay, and calibration remain outside this
implementation increment.

### P5.4 evidence cache status

The P5.4 code increment is now available in
`capmas/evaluation/evidence_cache.py`. `VersionedEvidenceCache` is a bounded,
thread-safe, process-local LRU keyed by the canonical local candidate
fingerprint and source `SceneSnapshot.scene_version`. A newer scene version
invalidates older entries; stale reads return unavailable and record a
`stale_rejection` event. Exact hits, misses, stores, invalidations, evictions,
and bounded event history are exposed through immutable statistics. Candidate
attachment uses `dataclasses.replace` and never mutates the live candidate.

This closes the P5.4 contract/code gate only. The cache is not persistent or
cross-process, is not implicitly wired into physical execution, and has not
yet produced a real multi-task hit-rate or downstream success experiment.
Those measurements remain a separate run-scoped evaluation task.
