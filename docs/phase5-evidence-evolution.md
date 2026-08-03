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

### P5.1 implementation status

The P5.1 code slice is implemented. VerifierEvidence and
VerifierPredicateEvidence provide immutable, candidate-specific evidence
with pass, fail, and unknown states, deterministic pass-rate/coverage
summaries, source provider, capture time, candidate fingerprint, and source
SceneSnapshot.scene_version.

Static evidence is collected before arbitration only from action-node
preconditions selected as safe by the caller. The LIBERO provider currently
selects compile-time facts such as track_exists:*; dynamic state facts,
postconditions, and validating checkpoints are not evaluated against the
initial scene. The provider composes typed verifier evidence with existing
perception evidence while preserving the legacy scalar
CandidateEvidence.verifier_pass_rate projection used by the Arbiter.

Dynamic evidence is converted from post-execution VerificationResult and
contains the checked scene version and contract provenance. It is not used to
select the action that already executed; it is reserved for the next rolling
cycle, Recovery, Memory, and later rehearsal/cache consumers. The Arbiter
rejects typed verifier evidence whose candidate fingerprint or scene version
does not match the effective candidate/current scene.

The runtime publication boundary is now closed. FixedGraphInterpreter adds the
effective subgraph fingerprint, subgraph id, and node id to every action trace;
the B3-LLM runner publishes `scheduler_metrics.verifier_evidence` and
`evidence/verifier.json`. Phase 5 summary serialization preserves structured
dataclass traces, and the manifest covers the verifier artifact. Dynamic
artifacts retain the original trace index, so replay can prove that evidence
was produced after the corresponding action.

The predicate contract is explicit: object_in_gripper(obj) checks object/EE
distance and pose only, gripper_closed() checks opening independently, and
object_held(obj) is the strict composite predicate. The code gate is covered
by focused tests and the full repository suite. The 2026-07-30 CAP-X/LIBERO
pilot used CUDA_VISIBLE_DEVICES=5, gpt-5.5, and the staged two-policy path:
one seed-1 smoke plus matched seeds 2--5. All five run directories contain
the required logs, results, summary, manifest, and verifier artifact; all
dynamic records pass identity/version validation. Evaluator success was 2/5
and graph completion was 1/5. Dynamic evidence captured both successful and
failed postconditions, including a seed where the external evaluator reported
success while the observable placement predicate failed.

The typed static evidence objects were emitted for all five runs, but all ten
selected-candidate static collections had zero coverage because the generated
graphs did not declare a compile-time `track_exists:*` precondition. Therefore
the runtime publication gate is closed, while the meaningful static-coverage
empirical gate remains open. Empty coverage is not counted as a successful
verifier measurement.

### P5.1 typed condition-default closure

The follow-up condition-default slice adds a deterministic
`SkillConditionEnricher` at both the staged decoder boundary and the scheduler
candidate boundary. Registered CAP-X skills declare additive defaults:
`goto_pose`, `sample_grasp_pose`, and `lift_after_grasp` publish
`scene_fresh(2000)`, `close_gripper` publishes `gripper_closed()`, and
`open_gripper` publishes `gripper_open()`. Explicit LLM predicates remain
authoritative, and task predicates such as `object_in_gripper` and
`object_at_target` are never invented from a generic skill call.

`balanced` and `efficient` profiles require only the current-source freshness
precondition, while `safety` and `robust` additionally require
`track_exists:<id>` when an explicit motion-intent or typed object argument
resolves uniquely to a current scene track. The local response schema permits
an empty postcondition list so the runtime can fill safe skill defaults before
GraphValidator runs; the final enriched action contract still requires an
observable postcondition. LIBERO static verifier selection now includes
`scene_fresh(...)` alongside `track_exists:*` and `object_visible:*`, while
future object/gripper state remains dynamic.

This closes the software path for meaningful static evidence when the LLM
omits generic skill postconditions. The fresh matched LIBERO rerun below
measures positive static coverage while reporting evaluator success and graph
completion independently; this mechanism alone makes no downstream success
rate claim.

### P5.1 empirical condition-default rerun (2026-07-31)

The post-fix matched pilot used the non-privileged CAP-X Spatial-0 config,
CUDA device 5, gpt-5.5, staged ready-wave scheduling, two Policy strategies
(`balanced,safety`), fixed-graph execution, and disabled geometry evidence.
Each invocation created a new run directory under
`outputs/phase5/P5.1_condition_defaults_20260731/`.

| seed | evaluator success | graph completed | classification |
| --- | --- | --- | --- |
| 1 | true | true | normal success |
| 2 | false | false | task failure after normal execution |
| 3 | true | false | evaluator success with incomplete graph convergence |
| 4 | n/a | n/a | both Policy calls exhausted the 60 s upstream LLM read deadline |
| 5 | false | false | task failure after normal execution |

Across the matched set, evaluator success was `2/5` and graph completion was
`1/5`. Four normal runs produced eight static evidence records with coverage
`1.0` for every selected collection and seven dynamic verifier records. The
static freshness predicate often failed because static evidence is collected
before the LLM compilation interval; execution then performs the separate
scene refresh/rebase boundary. This is a measurement-timing limitation, not
evidence that the typed defaults were absent. The timeout run retained its
failure artifact, manifest, and full log but correctly has no verifier file
because arbitration was never reached.

This closes the P5.1 condition-default and evidence-publication experiment
gate. It does not claim a downstream success-rate improvement, and it leaves
LLM endpoint availability, latency budgeting, and graph convergence as later
optimization work.

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

### P5.3.1 online rehearsal-Arbiter status (2026-07-31)

The online selection seam is implemented in
`capmas/evaluation/online_rehearsal.py`. `disabled` never calls a provider;
`shadow` records a hypothetical evidence-aware result while keeping the
baseline result live; `online_bounded` promotes the evidence-aware result only
when it contains a selected candidate and otherwise records an explicit
fallback. The provider is batch-scoped and has no `ActionLease` or physical
executor access. Every attached result passes the existing candidate
fingerprint, graph-to-subgraph mapping, and scene-version checks. Missing,
stale, and mismatched evidence remains unavailable rather than becoming a
zero score.

`LLMGraphScheduler` routes legacy, staged serial, ready-wave, and rolling
frontier candidate selection through this one seam. Each
`LLMGraphCompileResult` exposes `rehearsal_reports`; the existing
`arbitrations` field remains the only live result consumed by execution.
`scripts/run_libero_p53_online.py` reuses the isolated CAP-X/LIBERO rehearsal
worker, writes a new run-scoped directory with rehearsal and selection
artifacts, and invokes the physical executor at most once.

The code gate is closed by focused tests for disabled/shadow/online behavior,
provider failure fallback, identity rejection, report serialization, the
single-execution driver, and timed-out worker termination. The matched
endpoint-backed evaluation is also complete in two suites:
`outputs/phase5/P5.3.1_matched_spatial0_20260731/` for seeds 1--5 and
`outputs/phase5/P5.3.1_matched_spatial0_seeds6_10_20260731/` for seeds 6--10.
It used the same two candidates, CAP-X config, scene version, object/target
names, and reset seeds 1--10 for `disabled` and `online_bounded`, with one
rehearsal worker to avoid invalid depth frames from same-GPU process
contention.

All ten pairs completed and each mode performed exactly one physical execution
per seed. The baseline achieved `0/10`; online achieved `2/10`, for a matched
delta of `+2/10`. Rehearsal evidence for both candidates attached in all ten
online runs with zero fingerprint, graph-to-subgraph, or scene-version
rejections. Seeds 1 and 5 changed the winner from baseline policy-0 to
policy-1 and both online episodes passed the CAP-X evaluator. Seeds 2--4 and
6--10 retained the baseline winner via `evidence_tie_break` and failed
downstream.
This closes the matched single-task gate and establishes the physical baseline
control, but it is not yet a multi-task or statistically powered causal claim.
Multiple tasks, larger seed sets, and confidence intervals remain open.
`TSDF`, semantic adapters, persistent evidence cache, OOD replay,
calibration, and adaptive topology remain outside this increment.

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

### P5.4 isolated evaluation implementation (2026-07-31)

The isolated evaluation driver is `scripts/run_p54_evidence_cache.py`. It
compares `cache_disabled` and `cache_enabled` on an identical deterministic
trace containing first misses, repeated exact-version requests, a scene
refresh, a stale probe, and a distinct candidate fingerprint. The two modes
are stored in separate run directories under
`outputs/phase5/P5.4_cache_evaluation/` and include cache trace, metrics,
bounded event history, logs, and SHA-256 manifests. Provider failures retain
partial artifacts and redact `sk-*` tokens from failure output.

This evaluation is intentionally process-local and has no LLM, CAP-X, LIBERO,
or physical execution path. Its result can close the isolated cache hit-rate
contract only; it does not establish a downstream success-rate improvement or
replace the later online and multi-task evaluation. The seed-1 run was stored
under
`outputs/phase5/P5.4_cache_evaluation/20260731_112755_cache_disabled_seed1/`
and
`outputs/phase5/P5.4_cache_evaluation/20260731_112755_cache_enabled_seed1/`.
The control used 9 provider calls; the enabled lane used 5 and recorded 3
hits, 2 invalidations, 1 stale rejection, 5 stores, and no stale attachment.
Its final scene version was 2 with 3 entries, and both manifests passed
SHA-256 verification. These are isolated cache metrics, not a downstream
success-rate claim.

### P5.4 online cache seam implementation (2026-08-03)

The cache is now available at the online arbitration seam. `select_with_rehearsal`
accepts an optional `VersionedEvidenceCache`; `LLMGraphScheduler` forwards its
long-lived cache to every legacy, staged, ready-wave, and rolling selection.
The provider wrapper first advances the current scene version, reuses only an
identity-checked `RehearsalEvidence` for the exact local candidate fingerprint
and scene version, and sends only misses to the rehearsal provider. A refresh
invalidates older entries before arbitration. Missing, stale, or mismatched
evidence remains unavailable and is still handled by the existing Arbiter
fallback rules.

`scripts/run_libero_p53_online.py` exposes `--cache-mode disabled|enabled` and
`--selection-repeats N`; it writes `results/cache_events.json` plus cache
statistics into `results/selection.json`. The default is disabled, preserving
the P5.3.1 control. With repeated selection in one episode, the provider is
called only for the first request, cache hits serve later requests, and the
physical executor remains single-owner and is called once. Focused tests cover
repeated same-scene requests, refresh invalidation, scheduler forwarding, and
run-scoped artifact publication.

This closes the P5.4 online integration/code gate, not the multi-seed empirical
gate. A single P5.3.1 decision has no repeated request and therefore cannot
produce a cache hit; the real repeated-selection smoke below supplies the
single-episode comparison. Cache state must not be shared across different
reset seeds unless the provider evidence contract is explicitly seed-
independent.

### P5.4 real CAP-X repeated-selection smoke (2026-08-03)

The first real runner-level paired smoke used the CAP-X `.venv-libero`
interpreter, CUDA device 5, Spatial-0, reset seed 1, the same two graph
candidates, and `selection_repeats=2`. The cache-enabled run is retained at
`outputs/phase5/P5.4_online_cache_smoke_20260803_venv/`; the matched disabled
run is at
`outputs/phase5/P5.4_online_cache_smoke_20260803_disabled_venv/`. Both runs
performed exactly one physical execution and both manifests passed SHA-256
verification.

| mode | rehearsal records | provider calls | cache hits | total selection latency | physical executions | evaluator success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cache_enabled` | 2 | 1 | 2 | 341.4 s | 1 | false |
| `cache_disabled` | 4 | 2 | 0 | 683.8 s | 1 | false |

This closes the single-episode real cache/latency smoke gate: repeated
same-scene arbitration reuses candidate evidence and does not duplicate
physical execution. It does not close the multi-seed/multi-task cache gate or
claim downstream task improvement; the task itself failed in both matched
runs. The run-scoped `run_config.json`, `results/selection.json`,
`summary.json`, and `logs/runner.log` now expose the same cumulative
`provider_call_count` (enabled `1`, disabled `2`) for auditability. The failed
system-Python attempt is retained separately with a bounded failure artifact
and was caused by the incompatible interpreter-level LIBERO
import, not by the cache seam.

### P5.4 matched multi-seed evaluation implementation

The multi-seed driver is `scripts/run_libero_p54_matched.py`. For every task
and reset seed it runs two independent `online_bounded` lanes with identical
candidate and scene inputs: `cache_disabled` and `cache_enabled`. Each lane
gets a separate child artifact directory and a fresh process-local cache;
cache state is never shared across seeds or between the control and enabled
lane. The physical executor remains single-owner and is invoked at most once
per lane.

The suite emits one pair directory per task/seed under
`outputs/phase5/P5.4_matched_online_cache/<suite>/pairs/`, retaining child run
references, pair-level logs, failure artifacts, and manifests. The aggregate
reports provider calls, exact cache hits, selection latency, physical
execution counts, evaluator success, and paired downstream outcomes. A
positive cache-call reduction or latency reduction is reported separately from
task success and is not sufficient to claim a downstream improvement.

The matched gate requires all of the following: identical candidate artifact
hashes within each pair, no cross-seed cache reuse, complete control/enabled
trace pairs, enabled cache hits or fewer provider calls, no duplicate physical
execution, and independent success reporting for both lanes. Multi-task
coverage and statistical confidence intervals remain open until the driver is
run on the locked task suite.

### P5.4 matched online cache result (2026-08-03)

The first real matched run completed with the CAP-X `.venv-libero` interpreter
on `CUDA_VISIBLE_DEVICES=5`, LIBERO Spatial-0, seeds 1--5,
`selection_repeats=2`, `max_workers=1`, and `timeout_s=360`. It is retained at
`outputs/phase5/P5.4_matched_online_cache_20260803/P5.4_matched_online_cache/20260803_054828_suite_d64ab784/`.
All five pairs completed and all five used the same candidate artifact,
scene version, and physical-execution budget in the two lanes:

| lane | provider calls | rehearsal records | cache hits | physical executions | evaluator successes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cache_disabled` | 10 | 20 | n/a | 5 | 2/5 |
| `cache_enabled` | 5 | 10 | 10 | 5 | 2/5 |

Enabled mode reduced provider calls by 50% and total measured selection
latency from 1792.70 s to 899.77 s (49.77%), while preserving one physical
execution per lane and producing two exact hits per pair. The same winner was
used by both lanes for every seed. Seeds 1 and 5 selected
`policy-1:safety:1` through `evidence_score` and succeeded; seeds 2--4
selected `policy-0:0` through `evidence_tie_break` and failed. The paired
success result is `2/5` versus `2/5`, so this run demonstrates cache reuse and
latency reduction only; it does not demonstrate a downstream success-rate
improvement or a causal change in Arbiter selection.

The suite manifest contains 89 entries, and all suite, pair, and lane
manifests passed SHA-256 and size verification. The single-task five-seed
cache-efficiency gate is closed. Multi-task coverage, larger statistical
power, persistent/cross-process cache evaluation, and downstream success
claims remain open.
