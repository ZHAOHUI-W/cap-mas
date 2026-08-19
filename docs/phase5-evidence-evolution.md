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
distance and pose only, gripper_closed() checks the commanded gripper fraction
when CAP-X exposes it and otherwise falls back to physical opening, and
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

### P5.5 frozen OOD replay implementation (2026-08-03)

P5.5 now has immutable `OODCase`/`OODSplitManifest` contracts, explicit
`pair_id` propagation through `OODReplayEvidence`, fail-closed family/digest
validation, and a deterministic leakage audit. The dependency-free report
computes ID/OOD Wilson intervals, paired success deltas, exact McNemar tests,
bootstrap OOD gaps, failure classes, selection bases, and infrastructure
unknown counts. Unknown evaluator or verifier outcomes remain `None`.

The case-scoped driver is `scripts/run_libero_p55_ood.py`. It validates the
frozen candidate artifact before starting any case, reuses the existing
CAP-X/LIBERO `run_online_experiment` and physical executor factory, runs the
initial correctness lane with `max_workers=1`, and retains independent case
manifests, logs, failure artifacts, and SHA-256 file manifests. Every
`OODReplayEvidence` record is forced to `shadow_only=true` (shadow-only); it cannot change
the live winner, prompt, Memory Bank, Robot Skill Registry, or Arbiter policy.

The smoke manifest is `configs/phase5/p55_ood_smoke.json`. Its layout-OOD
membership is explicitly curated metadata and currently has no perturbation
generator, so the smoke validates replay, provenance, pairing, leakage control,
and artifact retention only. It is not a physical OOD success-rate claim.
The real layout-variant pilot below supplies the first formal measurement
with five paired seeds and multiple task/layout families. P5.6 calibration
still remains deferred until the measurement result and Arbiter selection
behavior are separately interpreted.

### P5.5 real CAP-X replay smoke (2026-08-03)

The first real smoke completed with the CAP-X `.venv-libero` interpreter on
`CUDA_VISIBLE_DEVICES=5`, `max_workers=1`, `max_restarts=0`, and one physical
execution per case. The retained suite is
`outputs/phase5/P5.5_ood_replay_20260803_smoke1/P5.5_frozen_ood_replay/20260803_090001_suite_fa2ccc0e/`.
Both the ID and manually labeled layout-OOD case completed without runner
failure; both selected the same candidate through `evidence_tie_break` and
both reported evaluator success `false` and graph completion `false`.

| split | cases | evaluator success | graph completed | physical executions | unknowns |
| --- | ---: | ---: | ---: | ---: | ---: |
| ID | 1 | 0/1 | 0/1 | 1 | 0 |
| layout-OOD label | 1 | 0/1 | 0/1 | 1 | 0 |

The suite and both case manifests passed SHA-256 and file size checks, and every replay
evidence record retained `shadow_only=true`, the explicit pair id, and the
candidate fingerprint. This smoke closes the real runner/provenance/retention
gate only. Because the two cases intentionally use the same CAP-X config and
candidate artifact and the fixture has no perturbation generator, the result
is not a physical OOD gap or a downstream success-rate claim. A real pilot
must replace the placeholder with distinct layout variants before increasing
the number of seeds or task families.

### P5.5 real layout-variant five-seed multi-family pilot (2026-08-04)

The real layout implementation is in `capmas/evaluation/layout_variants.py`.
It applies auditable MuJoCo free-joint translations after the CAP-X reset and
refreshes the environment observation cache before the first `SceneSnapshot`.
The frozen manifest is generated by `scripts/create_p55_real_layout_manifest.py`
and retained at
`outputs/phase5/P5.5_real_layout_assets_20260803/p55_real_layout_3family_5seed.json`.
It contains 30 cases, 15 matched ID/OOD pairs, five seeds, and three task
families: `spatial-0`, `goal-1`, and `object-6`.

The real-layout smoke completed first at
`outputs/phase5/P5.5_real_layout_smoke_20260803/` with six cases and zero
runner failures. The formal five-seed pilot then used the same frozen
manifest, the CAP-X `.venv-libero` interpreter, CUDA device 5,
`cache_mode=disabled`, `max_restarts=0`, `max_steps=32`, and
`max_workers=2`. The retained suite is
`outputs/phase5/P5.5_real_layout_pilot_20260803/P5.5_frozen_ood_replay/20260803_113118_suite_b4bbc31b/`.
Each case has an independent log, evidence directory, case manifest, and
failure artifact boundary.

| family | ID success | OOD success | pairs | mean reported latency |
| --- | ---: | ---: | ---: | ---: |
| `spatial-0` | 0/5 | 0/5 | 5 | 75.92 s |
| `goal-1` | 0/5 | 0/5 | 5 | 73.63 s |
| `object-6` | 0/5 | 0/5 | 5 | 74.33 s |
| **all** | **0/15** | **0/15** | **15** | **74.63 s** |

All 30 cases completed at the runner level with zero infrastructure
unknowns. All 30 reported `evaluator_success=false`,
`graph_completed=false`, `verifier_success=null`,
`failure_class=task_failure`, zero recovery count, and zero human
intervention. The ID and OOD Wilson estimates are both 0.0 with a 95%
upper bound of 0.2039. The paired result is 15/15 ties, paired delta 0,
and exact McNemar `p=1.0`; this is a zero-success measurement, not evidence
that the translated layouts preserve task performance.

The layout provenance gate did pass: all 15 OOD pairs have distinct ID/OOD
state fingerprints, all 30 layout reports have `applied=true`, and all case
manifest SHA-256 and size checks pass. Every replay record remains
`shadow_only=true`. The Arbiter used `selection_basis=evidence_tie_break`
in all 30 cases, so P5.5 does not establish a causal evidence-driven
selection improvement. This pilot was subsequently expanded to the formal
ten-seed gate below; P5.6 calibration, active OOD weighting, and downstream
success claims remain deferred.

### P5.5 real layout-variant ten-seed formal gate (2026-08-05)

The formal gate reused the frozen real-layout protocol and expanded the
manifest to seeds 1--10 for the same three task families:
`spatial-0`, `goal-1`, and `object-6`. The manifest contains 60 cases and 30
matched ID/OOD pairs. ID uses the native reset layout; OOD applies the
audited deterministic free-joint translation and records a distinct layout
state fingerprint. The formal manifest is
`outputs/phase5/P5.5_real_layout_assets_20260803/p55_real_layout_3family_10seed.json`
with canonical manifest digest (SHA-256 excluding the self-digest field)
`5aeff85dae764c72fe9c0b1f3a0a07f4070e95247baea1b8d93f15311ea72141`.

The retained suite is
`outputs/phase5/P5.5_real_layout_formal_20260804/P5.5_frozen_ood_replay/20260804_014522_suite_dda9defe/`.
It used the CAP-X `.venv-libero` interpreter on CUDA device 5,
`max_workers=2`, `max_restarts=0`, `max_steps=32`,
`timeout_s=360`, `selection_repeats=1`, and disabled evidence cache. Every
case has an independent log, evidence directory, case manifest, and failure
artifact boundary. All 60 cases completed at the runner level and no
`failure.json` was produced.

This retained suite is now marked diagnostic-only. A retrospective audit of
its 120 rehearsal candidate records found 117 depth-initialization failures
and three candidates that reached the first skill path. The two-worker replay
shared CUDA device 5 and allowed EGL/model initialization to contend. The
pre-diagnostic adapter also had no explicit physical failure class and mapped
an incomplete graph to `task_failure`; those facts invalidate its success rate
as a task-performance result.

| family | ID success | OOD success | pairs | mean reported latency |
| --- | ---: | ---: | ---: | ---: |
| `spatial-0` | 0/10 | 0/10 | 10 | 74.99 s |
| `goal-1` | 0/10 | 0/10 | 10 | 78.34 s |
| `object-6` | 0/10 | 0/10 | 10 | 82.86 s |
| **all** | **0/30** | **0/30** | **30** | **78.73 s** |

The ID and OOD Wilson estimates are both `0.0` with a 95% upper bound of
`0.1135`. The paired result is 30/30 ties, paired delta `0`, and exact
McNemar `p=1.0`. All 60 cases are classified as `task_failure`; there are
zero infrastructure unknowns, zero recoveries, and zero human
interventions. Reported latency has mean `78.73 s`, median `72.68 s`, and
range `55.18--211.06 s` across all cases.

This suite does not provide a valid horizon-bucket result. The P5.5 case
evidence records do not contain executed subgoal counts or a
`horizon_bucket`; `max_steps=32` is an execution budget, not an observed
horizon. Because all 60 episodes had `graph_completed=false`, no realized
H2/H4/H6 stratification can be inferred from this artifact. The runner
protocol must record verified subgoal count before a horizon-stability claim
can be made.

The formal provenance gate passed: all 30 matched pairs have distinct ID/OOD
layout fingerprints, all 60 case manifests pass digest/size verification,
and every replay record has `shadow_only=true`. Selection used
`selection_basis=evidence_tie_break` in all 60 cases. Therefore P5.5 closes
the ten-seed measurement, pairing, layout, and artifact-retention gates, but
it does not demonstrate downstream success, OOD generalization, or causal
Arbiter improvement. Calibration, correlated-signal correction, active OOD
weighting, and any learned Arbiter weighting are deferred to P5.6.

### P5.5 failure-diagnostics correction (2026-08-05)

The corrected runner serializes CAP-X rehearsal on one configured CUDA device
(`max_workers=1`) and classifies depth setup failures as `reset_failure`.
Physical graph failures now retain their class, reason, node, subgraph, and
trace metadata. The aggregation boundary treats `reset_failure`,
`worker_crash`, `timeout`, and `infrastructure_unknown` as non-evaluable
(`evaluator_success=null`); only explicit graph/task failures or a completed
graph with a failed evaluator are counted as task failures. Each completed
case writes `evidence/rehearsal_failure_summary.json` with candidate-level
failure provenance. The old two-worker formal suite must be rerun with this
protocol before any ID/OOD success or horizon claim is made.

The corrected single-worker matched smoke completed at
`outputs/phase5/P5.5_failure_diag_smoke_20260805/P5.5_frozen_ood_replay/20260805_032606_suite_d733ad12/`.
It used the three-family, one-seed real-layout manifest on CUDA device 5 with
`max_workers=1`, `max_restarts=0`, and disabled cache. All six cases completed
with zero runner failures and zero infrastructure unknowns. The 12 rehearsal
candidate attempts were classified as `skill_failure`, and all six physical
executions retained explicit `EXECUTION_ERROR` metadata with node, subgraph,
and trace context. ID and OOD evaluator success were both `0/3`.

This closes the single-GPU isolation and failure-provenance smoke gate, but it
is not a downstream quality claim: it uses one seed, all physical candidates
failed during execution, and all selections used `evidence_tie_break`. A new
five-seed run is permitted by this infrastructure gate and must be reported
separately from the invalidated two-worker formal suite.

### P5.5 execution-grounding smoke (2026-08-05)

The next smoke moved LIBERO graph grounding to the execution reset boundary:
both the online physical executor and isolated rehearsal worker now call
`ground_libero_mission_graph()` after reset and after the refreshed
`SceneSnapshot` is available. The rewrite updates scene-dependent target poses
and leaves candidate fingerprints unchanged. The six-case real-layout smoke
used the one-seed, three-family manifest
`outputs/phase5/P5.5_real_layout_assets_20260803/p55_real_layout_3family_1seed.json`
on CUDA device 5 with the CAP-X Python 3.10 environment, one worker, zero
restarts, `max_steps=32`, and disabled cache. Results are retained at
`outputs/phase5/P5.5_grounding_smoke_venv_20260805/P5.5_frozen_ood_replay/20260805_085352_suite_fd89ecef/`.

| split | cases | evaluator success | infrastructure unknowns |
| --- | ---: | ---: | ---: |
| ID | 3 | 1/3 | 0 |
| layout-OOD | 3 | 0/3 | 0 |

The suite completed all six cases with no runner or infrastructure failures.
Five cases ended in explicit `POSTCONDITION_FAILED` task failures. The ID/OOD
Wilson estimates are `0.3333` (95% CI `[0.0615, 0.7923]`) and `0.0` (95% CI
`[0.0, 0.5615]`), respectively; the paired table has one ID-only success and
two ties. Arbiter selection used `evidence_score` once and
`evidence_tie_break` five times. This closes the execution-grounding smoke
check but not the P5.5 multi-seed or OOD-quality gate.

Grounding is observable in the physical trace: the spatial placement target
changed from `x=0.72409` in native layout to `x=0.81099` in the translated
layout, consistent with the recorded layout translation. The remaining failure
signatures are `object_at_target` after OOD spatial release,
`object_in_gripper` in the goal family, and `gripper_closed` in the object
family. These are retained as follow-up grasp/coordinate and task-mapping
diagnostics rather than being relabeled as infrastructure failures. The
regression suite passed `421` tests and Python compilation succeeded.

### P5.5 gripper-state semantic correction (2026-08-05)

The object-family grounding probe isolated the remaining `gripper_closed()`
failure. CAP-X's `robot_cartesian_pos[-1]` reflects measured finger opening,
so a held object leaves it at approximately `0.486` even though the low-level
controller's commanded fraction is `0.0` (closed). CAP-MAS now propagates that
optional command as `ObservationBundle.robot_state["gripper_commanded_fraction"]`
from the CAP-X factory, and the verifier prefers it for `gripper_open()` and
`gripper_closed()`. Snapshots without the field retain the legacy
`gripper_opening` fallback. `object_in_gripper()` remains purely geometric;
`object_held()` uses the same closure-state preference as its strict composite
check.

A fresh CUDA-5 grounded probe at
`outputs/phase5/P5.5_grasp_probe_object6_commanded_20260805/20260805_101645_c98dd434/`
physically lifted the butter from approximately `z=0.0087` to `z=0.1234`.
The final reports were `object_in_gripper(butter)=passed` and
`gripper_closed()=passed`, while `task_completed=false` because the probe
intentionally stops after the pick/lift checkpoint. This closes the verifier
semantic regression at the pick checkpoint; it is not a full-task success
claim or an OOD result.

### P5.5 target-pose verified object-6 online closure (2026-08-06)

The placement regression was isolated to target geometry under partial basket
occlusion. The prior path used the semantic body-center pose as the release
target, while the physical evaluator accepted the placement; the verifier then
used the occluded target pose and rejected the episode with
`object_at_target` distance `0.1369 m`. CAP-MAS now derives a safe placement
pose from the clipped target point cloud, applies top release clearance, and
uses robust XY plus semantic Z when checking the final relation. The grounded
placement sequence is explicitly approach, descent, release, and retreat.

The real online closure run is retained at
`outputs/phase5/P5.5_target_pose_verified_object6_online_20260806/P5.3.1_online_rehearsal_arbiter/20260806_052654_seed1_d1b5f0d1/`.
It used CUDA device 5, `max_workers=1`, zero restarts, `max_steps=32`, one
selection repeat, and disabled cache. Both candidate rehearsal attempts
passed, with latencies `122301.288867 ms` and `91026.945827 ms`; the single
online provider call took `215962.848232 ms`. The Arbiter attached both
candidates, used `evidence_tie_break`, and executed only the selected winner.

The physical boundary passed all required checks:

| check | result |
| --- | --- |
| candidate rehearsal | 2/2 success |
| online selection | `evidence_tie_break` |
| physical executions | 1 |
| graph completion | `true` |
| LIBERO evaluator | `true` |
| final success | `true` |

This closes the target-pose/verifier regression and the full object-6 online
smoke. The evidence tie-break did not change the confidence baseline winner,
so the run is not a causal candidate-selection gain. It also remains a
one-seed, one-family validation and does not replace the required matched
multi-seed ID/OOD evaluation. The original run predates placement provenance
in the scene debug projection; the apparent `null` came from querying a missing
JSON field, not from a confirmed provider-side fallback.

That artifact contract is now implemented. `ObjectTrack`, snapshot JSON, and
the online scene debug payload preserve `placement_pose_source` and
`placement_pose_reason`. The CAP-X factory emits `geometry_pointcloud` on a
valid estimate and `semantic_pose_fallback` with an explicit exception,
payload, or point-cloud reason otherwise. This closes the silent-null software
gap; the real-capture gate remains open until a fresh object-6 run records the
new fields.

The fresh real capture at
`outputs/phase5/P5.5_placement_provenance_object6_20260806/P5.3.1_online_rehearsal_arbiter/20260806_064328_seed1_5168c2e2/`
closed that gate. The basket track carried `geometry_pointcloud` provenance
before and after execution, both placement poses were non-null, and both
reasons were `null`. The physical graph, LIBERO evaluator, and final success
all passed. Fallback source/reason behavior is covered by regression tests and
can now be measured in future multi-seed runs without changing execution.

### P5.5 matched provenance five-seed status (2026-08-06)

The corrected matched suite at
`outputs/phase5/P5.5_matched_provenance_5seed_20260806/P5.5_frozen_ood_replay/20260806_091429_suite_e169a480/`
completed 30/30 cases on CUDA device 5 with one worker, no infrastructure
unknowns, and geometry-derived target placement provenance. ID evaluator
success was `3/15`; OOD evaluator success was `5/15`. Graph and known
verifier success were `2/15` for ID and `4/15` for OOD. There were two
OOD-only paired successes, 13 ties, and exact McNemar `p=0.5`; the observed
gap is not statistically persuasive and does not support an OOD improvement
claim.

The original aggregate's 24 `POSTCONDITION_FAILED` entries are graph failure
provenance, not 24 downstream failures. Reclassification gives 22 physical
task failures and 2 verifier false negatives: `id-object-6-seed4` and
`ood-object-6-seed2` both passed the LIBERO evaluator while the internal
point-distance placement predicate failed. The immutable offline correction
is retained at
`outputs/phase5/P5.5_matched_provenance_5seed_report_correction_20260807/P5.5_offline_reaggregation/20260807_013832_suite_e169a480/`.

Arbitration used `evidence_tie_break` for 28 cases and `evidence_score` for
two. This validates the matched execution and reporting path but not causal
candidate-selection benefit. The formal P5.5 gate remains a corrected
ten-seed run across all three families; this five-seed result is a pilot.

### P5.5 corrected matched-provenance ten-seed gate (2026-08-07)

The corrected formal suite is retained at
`outputs/phase5/P5.5_matched_provenance_10seed_retry2_20260807/P5.5_frozen_ood_replay/20260807_024842_suite_20674432/`.
It used the frozen 60-case, 30-pair, three-family manifest with canonical
SHA-256
`5aeff85dae764c72fe9c0b1f3a0a07f4070e95247baea1b8d93f15311ea72141`.
The CAP-X `.venv-libero` run used CUDA device 5, `max_workers=1`,
`max_restarts=0`, `max_steps=32`, `timeout_s=360`, one selection repeat, and
disabled cache. All 60 cases completed with no case-level failure artifact or
infrastructure-unknown record.

| family | ID evaluator | OOD evaluator | ID graph/verifier | OOD graph/verifier | mean latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| `spatial-0` | 0/10 | 0/10 | 0/10 | 0/10 | 190.98 s |
| `goal-1` | 0/10 | 0/10 | 0/10 | 0/10 | 165.10 s |
| `object-6` | 4/10 | 10/10 | 4/10 | 8/10 | 223.35 s |
| **all** | **4/30** | **10/30** | **4/30** | **8/30** | **193.15 s** |

ID evaluator success was 13.3% with Wilson 95% CI `[0.0531, 0.2968]`; OOD
success was 33.3% with CI `[0.1923, 0.5122]`. The estimated `ID - OOD` gap was
`-0.2` with deterministic paired-bootstrap CI `[-0.3667, -0.0667]`. There
were zero ID-only successes, six OOD-only successes, and 24 ties; exact
McNemar `p=0.03125`. The split-specific mean/median latency was
`172.29/171.49 s` for ID and `214.00/208.44 s` for OOD. Overall latency had
median `175.14 s` and range `137.98--323.84 s`. All 60 cases recorded zero
recoveries and zero human interventions, one provider call, and no cache hit.

The corrected failure taxonomy records 48 graph-level
`POSTCONDITION_FAILED` outcomes, of which 46 were physical task failures. The
two remaining cases, `ood-object-6-seed1` and `ood-object-6-seed2`, passed the
LIBERO evaluator while the point-distance verifier rejected the final state;
they are verifier false negatives. Arbitration used `evidence_tie_break` 57
times and `evidence_score` three times, with no `confidence_fallback`.

All 30 pairs had distinct, non-null ID/OOD layout fingerprints; every record
was `shadow_only=true`. Independent reaggregation exactly matched the retained
aggregate, the suite manifest's 907 entries and all 60 case manifests passed
size/SHA-256 verification, and no API secret was found in the suite. This
closes the corrected P5.5 measurement, pairing, provenance, and artifact gate,
not a causal evidence-selection or general OOD-robustness claim. The observed
gap is confined to an easier `object-6` OOD layout, while two families remain
at zero success.

No evidence record contains realized executed horizon, so this suite cannot
support horizon buckets or horizon-stability analysis. `max_steps=32` remains
an execution budget. P5.6 must add a verified horizon field and calibrate
correlated verifier, geometry, rehearsal, and OOD signals before any active
weighting or causal Arbiter claim.

## P5.3.2 task-family capability repair boundary

P5.3.2 is an independent work package triggered by a read-only P5.6.0
diagnosis. It owns task-goal mapping, Policy prompt, typed skill arguments,
placement/release parameters, and other task-completion fixes for
`spatial-0` and `goal-1`. It cannot add calibration weights or probability
logic. It requires a separate design and implementation plan and closes only
when the affected family passes the fixed ten-seed capability gate with zero
infrastructure unknowns, typed failures, at least 80% physical-execution reach,
and at least one evaluator success.

P5.3.2 failure does not block P5.6 contracts, object-6 offline calibration, or
eligible-family shadow evaluation. It blocks calibrated promotion for that
family and any all-family performance claim.

## P5.6 accepted design boundary

P5.6 is a family-scoped, qualified calibration increment. The online primary
output is a candidate `rank_score`; `success_probability` is emitted only
after capability and data gates pass. The corrected P5.5 suite does not provide
a directly eligible P5.6 dataset: `spatial-0` and `goal-1` have zero evaluator
successes, while the 20 `object-6` physical outcomes predate the required
pre-execution feature-snapshot, horizon, and calibration-lineage contracts.
They remain a smoke/provenance source and must be recollected or normalized
only through a separately validated compatibility audit.

The implementation route is parallel but gated. P5.6.0 is diagnostic-only and
emits a `P5.3.2 Task-Family Capability Repair` package for each zero-success
family. P5.3.2 owns task mapping, prompt, skill argument, and motion-parameter
repairs; it is not calibration work and blocks only that family's promotion.
Meanwhile P5.6.1--P5.6.6 build horizon/label contracts, three-tier datasets,
leakage audits, correlation-group reduction, constrained logistic/isotonic
fitting, and immutable snapshots. P5.6.7 adds shadow arbitration; P5.6.8
canary promotion is allowed only for eligible families; P5.6.9 performs the
formal matched evaluation.

The object-6 result contains 20 physical outcomes, not 14 samples: 14 are
positive and six are negative. This meets the numerical 20/5/5 gate but not
automatically the new lineage contract. A read-only compatibility audit must
prove that each admitted row has pre-execution features, candidate/scene
identity, selection, evaluator outcome, and reconstructable horizon without
future-state leakage. If the audit leaves too few rows, fresh collection uses
pre-registered ID seed blocks 11-20 and then 21-30. No per-outcome stopping or
cross-family pooling is allowed. This compatibility audit and fixed-block
collection is P5.6.2a and must complete before a qualified object-6 offline
fit; P5.6.8 does not collect training labels during canary execution.

The initial active feature set excludes OOD (`ood_weight=0`). Tier A contains
only conclusive physical selected-candidate evaluator outcomes; Tier B is
isolated rehearsal; Tier C is unlabeled evidence. Unselected candidates are
not physical failures. Unknown evidence remains explicit. Planned horizon is
bucketed by action-bearing subgraphs; checkpoint-only subgraphs, skill calls,
and `max_steps` do not increase the task bucket. A single action-bearing
subgraph is H1 and empty higher buckets are `N/A`. The frozen P5.5 candidates
have two action-bearing subgraphs and one checkpoint-only subgraph, so they
are H2-3. Realized actions/subgoals/checkpoints remain separate diagnostics.

The deterministic correlation-group reducer prevents verifier/rehearsal or
other correlated signals from being summed as independent votes. The
calibrator enforces non-negative support coefficients and non-positive
latency/recovery/collision risk coefficients. Calibration abstention falls
back to safety hard gates, qualified calibration, fixed-weight evidence
ranking, deterministic evidence tie-break, and confidence fallback only when
no candidate evidence exists. A content-addressed snapshot is atomically
activated, pinned per episode, and explicitly rollbackable.

Offline qualification requires at least 20 Tier A outcomes per family, with
at least five positive and five negative labels, plus the fixed ten-seed
capability gate. Offline targets are Brier improvement >=10% and ECE <=0.10.
Shadow requires zero hard-gate disagreement, calibration inference P95 <=5 ms,
and eligible coverage >=50%. The bounded canary requires at least 20 matched
physical episodes and supports only a safety/operability claim. Full details
are in [the P5.6 design](superpowers/specs/2026-08-11-p5-6-evidence-calibration-design.md)
and [ADR-0014](adr/0014-calibrated-evidence-and-snapshot-activation.md).

## P5.6A data foundation status (2026-08-19)

P5.6A closes the contracts, execution telemetry, feature-snapshot, dataset,
capability-diagnosis, compatibility-audit, and fixed-block collection gates.
Its feature contract is `p56.feature.v1`. A decision-time feature snapshot is
captured before lease acquisition and physical execution; dynamic verification
and after-action scene state remain outcome evidence. `max_steps=32 is not a horizon`:
planned and realized horizon are recorded separately from the action
budget.

The capability artifact
`outputs/phase5/P5.6.0_capability_diagnosis/20260813_072308_capability_c9df3f4b/`
records `spatial-0=0/10`, `goal-1=0/10`, and `object-6=4/10` evaluator
successes, with no infrastructure unknowns. The first two families remain
blocked on `P5.3.2 Task-Family Capability Repair`; that repair must not alter
P5.6A records or add a learned score.

Historical object-6 data was rejected by the compatibility audit at
`outputs/phase5/P5.6.2a_object6_history_audit/20260813_103434_history_a6bc49b1/`
with zero admissible Tier A rows. The recollected immutable ID suites
`20260818_090102_suite_63248cf1` (seeds 11-20) and
`20260818_095350_suite_50dc9bd3` (seeds 21-30) under
`outputs/phase5/P5.6.2a_object6_collection/` each completed 10 cases with
five positive and five negative Tier A labels. Their combined 20 Tier A rows,
10 positive and 10 negative, close the family-scoped 20/5/5 collection gate.
Both suites pass the Phase 5 manifest verifier. Every selected Tier-A row
also satisfies the corrected decision-time contract
`feature_snapshot.captured_at_ns <= lineage.decision_boundary_ns`; the
observed minimum decision slack is positive in both blocks. The earlier
future-state-invalid suites remain retained for audit and are not reused.
The suite directories retain `results/manifest_verification.json`.

This closes the data-collection gate but does not activate a calibrator.
P5.6B retains correlation reduction, constrained fitting, isotonic calibration,
and immutable snapshots; P5.6C retains calibrated shadow arbitration,
abstention/fallback integration, and bounded canary evaluation. There is no
active `success_probability`, no new Arbiter ranking, and no claim of
downstream task-success improvement.

## P5.6C fit stability and real offline calibration status (2026-08-19)

The offline calibration implementation now has a V2 constrained-logistic
stability layer. It derives rank and availability diagnostics from train rows
only, fixes zero-variance non-intercept columns at zero, records the final
projected-KKT infinity norm, and requires that norm together with the fixed
loss-delta condition before reporting convergence. A candidate whose reduced
availability differs from an `all_present` or `all_unknown` train dimension
abstains offline; `mixed` dimensions accept either availability state.

The real-data offline run is
`outputs/phase5/P5.6.4_offline_calibration/20260819_012907_p56b-object6-offline/`.
Its manifest is verified with no missing files, digest mismatches, size
mismatches, or untracked files. The locked splitter produced 12 train, 4
calibration, and 4 test lineage groups. The V2 constrained-logistic fit
converged at iteration 3,730 with final loss `0.41435925` and projected-KKT
infinity norm `9.9696e-9` (tolerance `1e-8`). Because the recollected rows
currently expose only `rehearsal_success_rate` as a present reduced feature,
the train design correctly freezes all scene/risk/missingness coefficients and
learns only the intercept and action-feasibility support weight.

The frozen test report is descriptive only: Brier score is `0.1111` and ECE
is `0.1667`. ECE therefore does not meet the predeclared `<= 0.10` offline
target, and this run does not establish a calibrated-quality gate closure or
a downstream success-rate improvement. The calibration split has only four
rows, so the PAVA blocks have wide Wilson uncertainty (`0.7308` and `0.7935`)
and must not be interpreted as a production-quality probability estimate.

This is not a calibrated shadow-Arbiter or canary result. The report's
predictions are explicitly offline-only (`online_effect=false` and
`eligible_family=false`); no `CalibrationSnapshot` was published, no active
evidence weight was changed, and no physical Executor or runtime Arbiter used
the fitted model. P5.6B/C implementation and real-data fit gates are closed;
the offline qualification gate remains open because ECE failed and a fixed
weighted baseline comparison has not yet been produced. P5.6.5--P5.6.9 remain
blocked on that qualification decision and subsequent shadow safety checks.
