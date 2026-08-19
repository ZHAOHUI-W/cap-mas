# Implementation Roadmap

## Phase 0 — Project and parity harness

Status: the dependency-light foundation slice and CAP-MAS-only LIBERO B0
integration slice are implemented; real execution still depends on a working
CAP-X LIBERO environment.

- Create package layout and configuration system.
- Implement CAP-X adapter and capx_legacy runner.
- Load CAP-X YAML and construct its low-level LIBERO environment directly.
- Reuse CAP-X's registered API factory and bind allowlisted functions as typed skills.
- Emit a CAP-MAS-only `EpisodeRunResult` and JSON episode artifact.
- Reproduce one LIBERO task with identical initial-state seeds.
- Store normalized traces for both systems.
- Foundation slice: core contracts, in-memory state/lease runtime, typed-skill
  seam, CAP-X adapter protocols, artifactized observations, memory proposal
  store, LLM boundary, and reward boundary.

Exit criterion: CAP-MAS can run B0 and produce a CAP-MAS episode artifact.
The direct YAML/API factory, observable LIBERO verifier, skill-output chaining,
and episode runner satisfy the code-level portion; the real simulator smoke
run remains the next environment gate.

## Phase 1 — Contracts and state store

- Implement SceneSnapshot, ActionContract, VerificationResult, ActionLease, and failure enums.
- Add monotonic scene versions and episode epochs.
- Add static schema and resource validation.

Exit criterion: stale contracts are rejected deterministically.

## Phase 2 — Typed executor and verifier

- Wrap CAP-X APIs as typed skills.
- Remove direct env and arbitrary import access from the main runtime.
- Implement bounded execution and cancellation.
- Add precondition, invariant, and postcondition evaluators.

Exit criterion: B2 isolates contract benefit from multi-agent scheduling and
commits only when observable postconditions pass. Built-in predicates include
`object_in_gripper(obj_id)`, `object_at_target(obj_id, target_id)`,
`gripper_open()`, `gripper_closed()`, and `scene_fresh(threshold_ms)`.

## Phase 2.5 — Single-agent multi-cycle contract loop

- Run one bounded `ActionContract` per subgoal or local action chunk.
- Re-plan from the latest committed `SceneSnapshot` after every cycle.
- Retain failed traces and invoke bounded deterministic recovery contracts.
- Terminate on task-level observable goals or explicit cycle/recovery budgets.
- Keep evaluator-only completion outside `AgentContext`.

Exit criterion: LIBERO task 0 completes through multiple contracts with a
CAP-MAS-only artifact, and induced execution/postcondition failures recover or
terminate with an explicit reason. This phase is the bridge before the fixed
multi-agent scheduler.

## Phase 3 — Multi-agent scheduler

- [x] Define `MissionGraph` and `SubgraphSpec` as the shared typed planning space.
- [x] Add `GraphValidator` for graph structure, port types, checkpoints, reachability,
  and exclusive-resource conflicts.
- [x] Add a compatibility lowering from one graph action node to `ActionContract`.
- [x] Implement strict `schema_version=1` graph serialization and unknown-field rejection.
- [x] Add explicit `LoopSpec` budgets and reject unbounded cycles.
- [x] Implement the deterministic `FixedGraphInterpreter` through `Scheduler.dispatch()`.
- [x] Implement typed `ArtifactStore`, `EventBus`, and deterministic `CandidateArbiter`.
- [x] Resolve typed local/mission dataflow and reject unbound or unreachable mission inputs.
- [x] Publish `FailureArtifact` through the artifact/event boundary and constrain recovery
  to explicit matching graph edges.
- [x] Rank candidates using verifier, rehearsal, OOD, latency, and recovery-cost evidence.
- [x] Add provider-independent `LLMMissionManager` and `LLMGraphPolicyAgent` seams.
- [x] Add CAP-X-compatible HTTP client and stable Manager/Policy prompt builders.
- [x] Add bounded read-only Policy fan-out with one `CandidateArbiter` and one physical Executor seam.
- [x] Add staged graph protocol: compact `MissionTopology` Manager output plus direct local graph envelopes.
- [x] Preserve a `legacy|staged` runner switch for matched protocol ablations.
- [ ] Add LLM Recovery and Monitor roles with deadline-aware fallback.
- [x] Add provider-independent strict `MissionGraphDecoder` before graph execution.
- [x] Add a bounded spawned-process rehearsal boundary with serializable jobs/results.
- [ ] Add CAP-X/LIBERO worker factories, worker respawn, and asynchronous SceneSnapshot streaming.
- [x] Keep a fixed graph before adaptive topology.

The implemented graph foundation is described in
[docs/graph-as-policy.md](graph-as-policy.md). It is a deterministic scheduler
foundation with a provider-independent LLM proposal path. LLM Recovery/Monitor
roles, CAP-X worker factories, worker respawn, asynchronous world modeling, and
adaptive topology remain open work.

Exit criterion: B3 runs with matched model budgets. The deterministic LIBERO
entry point `scripts/run_libero_b3.py` has now passed the seed-1 smoke test; it
reuses the CAP-X YAML/API factory,
the observable verifier, and the single physical scheduler seam. The next
sub-phase is to replace only graph proposal generation with typed LLM agents,
while preserving this B3 execution path as the regression baseline.

### Phase 3.1 — LLM-driven fixed graph and read-only fan-out

Status: the provider boundary, prompt schema, strict decoding, bounded Policy
fan-out, deterministic arbitration, staged graph protocol, LIBERO runner, and
P3.1a run/call observability are implemented. A strict staged P3.1b endpoint
run and a heterogeneous `balanced,safety` P3.1c LIBERO run both completed with
`evaluator_success=True`; matched multi-seed evaluation and the P3.1b
`max_workers=1` versus `max_workers=2` measurement remain open.

- [x] Implement `CAPXCompatibleLLMClient` with deadline and bounded retry policy.
- [x] Build typed Manager and local Policy prompts from `SceneSnapshot`.
- [x] Compile Manager graph plus local Policy candidates through `GraphValidator`.
- [x] Preserve one physical Executor and `ActionLease` owner after arbitration.
- [x] Add `scripts/run_libero_b3_llm.py` for endpoint-backed LIBERO execution.
- [x] Add `MissionTopology -> local SubgraphArtifact -> MissionGraph` staged compilation.
- [x] Add strict topology and direct local-subgraph decoders with scene-version checks.
- [x] Record non-secret `ExperimentRunConfig` and per-request `LLMCallTrace` in endpoint artifacts.
- [ ] Run matched `max_workers=1` versus `max_workers=2` candidate fan-out A/B.
- [x] Add dependency-aware Policy proposal waves across independent subgoals.
- [x] Add heterogeneous Policy roles and evidence-backed Arbiter selection seams.
- [x] Expose Policy strategy profiles, record them in `ExperimentRunConfig`, and
  mark evidence-free Arbiter tie-breaks.
- [x] Complete a real staged P3.1c heterogeneous-profile LIBERO smoke with
  `balanced,safety` and preserve its artifact/log pair.
- [x] Add a rolling graph runner that executes one verified subgraph and
  recompiles from the latest scene.
- [x] Add direct ready-frontier Policy replanning with one fixed Manager
  topology and no discarded full suffix.
- [ ] Add LLM Recovery and Monitor roles after proposal-path measurements.
- [ ] Add matched endpoint-backed B3-LLM versus CAP-X trials.

Exit criterion: B3-LLM completes the same LIBERO task with matched model,
request, and time budgets under both `legacy` and `staged` protocols while
preserving B3 deterministic success and failure artifacts as regression
baselines. P3.1a additionally requires run configuration and LLM-call traces
for every endpoint-backed artifact. The next measurement gate is the
`max_workers=1` versus `max_workers=2` staged fan-out latency and success
comparison.

### Phase 3.2 — Policy specialization and non-collapsing candidate evidence

Status: the P3.2 closure is implemented. Strategy profiles are carried in
Policy payloads and candidate artifacts; raw/normalized candidate views and
rewrite fingerprints are retained; the LIBERO runner attaches a
SceneSnapshot-backed perception provider; confidence is fallback-only; evidence
is scene-version-bound; and Arbiter decisions expose explicit score-basis
labels. Candidate-specific dynamic evidence, simulator rehearsal, and OOD
evidence are deferred to Phase 5.

- [x] Add typed balanced/safety/robust/efficient StrategyProfile contracts.
- [x] Preserve raw and normalized candidate views with rewrite fingerprints.
- [x] Add structured PerceptionEvidence and explicit available evidence metrics.
- [x] Add strategy-aware Arbiter scoring and perception hard gates.
- [x] Attach the read-only LIBERO SceneSnapshot evidence provider.
- [x] Add prompt/profile and candidate/artifact regression seams.
- [x] Make scheduler candidate confidence optional and fallback-only.
- [x] Distinguish evidence score, evidence tie-break, confidence fallback, and
  evidence hard-gate outcomes.
- [x] Bind candidate evidence to its source SceneSnapshot version and provider.
- [ ] Run matched endpoint-backed P3.2 multi-seed trials and report raw versus
  normalized candidate diversity.

Exit criterion: an endpoint-backed artifact contains non-null scene-bound
perception evidence, optional/null scheduler confidence, an explicit
arbitration basis, and raw/normalized fingerprints sufficient to identify
candidate collapse. A `confidence_fallback` result is a regression baseline,
not evidence of quality improvement.

### P3.2 runtime safety closure

Status: implemented as a safety closure after the endpoint-backed P3.2 failure
analysis. The graph is now checked for state-flow provenance before execution;
runtime verifier rejection is a typed recovery result rather than an uncaught
exception; and failure artifacts retain the selected graph and predicate-level
diagnostics when the LIBERO runner fails after compilation.

- [x] Validate node preconditions against initial SceneSnapshot facts and all
  mandatory normal-path predecessor postconditions.
- [x] Reject joins where a dynamic fact is established on only one predecessor
  path.
- [x] Return structured `CycleResult(rejected=True, reason=...)` without
  acquiring a lease or executing skills.
- [x] Route failure-class transitions through the declared recovery edge and
  preserve predicate reports in `FailureArtifact.metadata`.
- [x] Persist graph, arbitration, proposal failures, scene, and partial result
  in endpoint failure JSON artifacts.
- [x] Preserve compile-time typed-skill rejection context, including candidate,
  node, skill, raw/normalized args, signature, and missing/unexpected keys.

The `scene_fresh(threshold_ms)` refresh/rebase path remains an execution-boundary
concern. The current rolling runner refreshes before dispatch when its callback
is enabled; adaptive perception caching and action-conditioned evidence remain
Phase 5 work. P3.3 now uses the fixed-topology ready-frontier path, while full
adaptive topology edits remain Phase 8 work.

### Phase 3.3 — Fixed-topology rolling replanning

Status: implemented at the runtime and local-regression level. The Manager is
called once per staged rolling episode; each cycle compiles only the selected
frontier subgraph, rebases it when scene freshness advances, and follows the
fixed topology's explicit success/failure edge. The LIBERO endpoint comparison
and matched full-graph baseline remain experiment work.

- [x] Reuse one `MissionTopology` across rolling cycles.
- [x] Compile one ready frontier and preserve the single physical executor.
- [x] Route success and failure through typed topology transitions.
- [x] Fail closed on missing, ambiguous, or dangling transitions.
- [x] Rebase graph and topology scene envelopes before dispatch refreshes.
- [x] Record planning scope, frontier ids, compile latency, and Manager calls.
- [x] Add recovery, terminal, and scene-refresh regression tests.
- [x] Reject normal success back-edges that would re-enter a committed rolling
  subgraph.
- [x] Complete one endpoint-backed real LIBERO rolling success trial and retain
  its JSON and log artifacts.
- [ ] Run matched endpoint-backed rolling versus fixed-graph LIBERO trials.

Exit criterion: local tests prove fixed-topology rolling semantics and artifacts
contain auditable frontier/scene metrics. Downstream success-rate claims require
matched endpoint-backed trials; adaptive topology remains Phase 8.

## Phase 4 — Real-time world model

Status: the dependency-light reference contracts, replay path, sparse voxel
map, deterministic tracker, semantic trigger queue, observation/committed state
views, thread/process runtimes, CAP-X streaming adapter, and B5 runner are
implemented. The live CAP-X observation gate has passed in thread mode and
the P4.5 process gate now passes with real LIBERO RGB-D, shared `.npy`
artifacts, and CAP-X object-pose measurements carried in the JSON envelope.

- [x] Implement timestamped replay and bounded sensor synchronization.
- [x] Add fast FK/camera-pose and dependency-light depth-to-world geometry seams.
- [x] Add incremental local sparse voxel map; TSDF remains disabled.
- [x] Add known-object tracking and confidence-aware prediction.
- [x] Add asynchronous deterministic semantic triggers and bounded queue.
- [x] Separate observation snapshots from committed action state.
- [x] Add thread runtime and process runtime with bounded restart/fallback.
- [x] Add file ArtifactRef bridge and strict observation/snapshot envelopes.
- [x] Add CAP-X streaming metadata adapter and processing-latency metrics.
- [x] Add `scripts/run_libero_b5.py` replay/live benchmark with JSON/log artifacts.
- [x] Run live CAP-X B5 thread mode with declared deadline/freshness targets.
- [x] Run live CAP-X B5 process mode with shared file-backed capture artifacts.
- [ ] Add TSDF backend and semantic model adapters.

Exit criterion: the reference/live-thread and P4.5 process B5 gates meet the
predeclared deadline and freshness targets under their declared artifact-root
and depth-stride profiles. The accepted P4.5 design and implementation
checklist are in
[`docs/superpowers/plans/2026-07-28-phase4-5-process-world-model.md`](superpowers/plans/2026-07-28-phase4-5-process-world-model.md)
and [ADR-0012](adr/0012-phase4-5-cross-process-artifacts.md).

## Phase 5 — Memory, rehearsal, and evidence evolution

The P3.2 deferred-work handoff, dependencies, and acceptance gates are listed
in [`phase5-evidence-evolution.md`](phase5-evidence-evolution.md). Phase 5 is
not part of the P3.2 closure and must not be enabled implicitly by the current
SceneSnapshot-only online provider.

- Implement Episode Working Memory, Experience Memory, and the hard-case buffer.
- Add Memory Skill contracts, provenance, conflict handling, and TTL policy.
- Implement a rules-based Memory Controller and Memory Executor first.
- Add trace replay and memory-only regression/OOD tests.
- Add candidate-specific static verifier coverage and dynamic postcondition
  evidence without evaluating future postconditions against the initial scene.
- Add action-conditioned geometry evidence for grasp quality, reachability,
  clearance, and collision risk.
- Add CAP-X/LIBERO process rehearsal workers with deterministic state reset,
  matched seeds, worker respawn, timeout, and checkpoint-level results.
- Add asynchronous evidence caching keyed by candidate fingerprint and source
  SceneSnapshot version; invalidate stale evidence after rolling refresh.
- Add frozen OOD replay suites with confidence intervals and leakage controls.
- Calibrate correlated perception, verifier, rehearsal, and OOD evidence before
  using it as a learned Arbiter signal.

P5.1 implementation status (2026-07-30): typed
VerifierEvidence/VerifierPredicateEvidence, static precondition coverage,
dynamic post-execution conversion, LIBERO provider composition, and Arbiter
candidate/scene freshness gates are implemented. Legacy scalar
verifier_pass_rate construction remains compatible. object_in_gripper now
means object inside the gripper region; object_held retains the
closed-gripper requirement. The software gate is covered by focused
regression tests. The runtime publication boundary is also implemented:
action traces carry candidate identity and the runner writes structured
static/dynamic verifier evidence under each run directory. The CUDA-visible
2026-07-30 pilot produced five valid run-scoped artifacts with evaluator
success 2/5 and graph completion 1/5. Static evidence objects were present,
but all had zero coverage because no generated action declared a safe
compile-time precondition; the meaningful static-coverage empirical gate
remains open.

P5.1 condition-default update (2026-07-31): registered typed skills now carry
additive default predicates through `SkillConditionEnricher`. The staged
decoder and scheduler both enrich action nodes before validation, so an empty
LLM postcondition list can receive `scene_fresh(2000)` or a gripper-specific
default without weakening GraphValidator. `balanced`/`efficient` profiles use
freshness only; `safety`/`robust` add uniquely resolved `track_exists:*`
preconditions. `scene_fresh(...)` is now compile-time-safe for static evidence.
The software coverage path is closed. The fresh matched LIBERO rerun below
confirms positive static coverage, and the result is reported separately from
downstream success-rate measurement.

Empirical condition-default rerun (2026-07-31): the matched non-privileged
LIBERO Spatial-0 pilot used CUDA device 5, gpt-5.5, staged ready-wave
proposal, two Policies (`balanced,safety`), and fixed-graph execution. Seeds
1--5 reached evaluator success `2/5` and graph completion `1/5`; seed 4 was a
bounded infrastructure failure because both Policy requests timed out at the
upstream endpoint. The four normal runs emitted positive static evidence
coverage and seven dynamic records. Static freshness failures are expected
in this measurement because evidence is captured before compilation finishes;
the pre-dispatch scene refresh protects physical execution but does not
rewrite the earlier static measurement. The condition-default software and
empirical evidence-publication gates are closed. Downstream success-rate gain,
LLM transport robustness, and graph-convergence improvement remain open.

Implementation status (2026-07-29): P5.2 contracts, strict response-schema
support for typed `motion_intent`, grounding-aware intent rebasing,
side-effect-free reference geometry preview, Arbiter geometry gates, scheduler
timeout handling, and run-scoped artifacts are implemented and covered by the
local test suite. The CAP-X LIBERO environment is usable through
`cap-x/.venv-libero`; the endpoint-backed five-seed pilot completed all 15
mode/seed runs with independent logs and artifacts. Each mode reached 2/5
`evaluator_success` and the online mode did not yet improve downstream task
success over the disabled baseline.

The first post-transport real run now uses the CAP-X RGB-D observation to
update a live `SparseVoxelMap` before candidate evidence. It completed with
`evaluator_success=true`, `map_version=4`, four candidate records, measurable
candidate-specific clearance, and a non-zero geometry term in the Arbiter
score breakdown. The matched post-transport five-seed pilot then completed all
15 mode/seed runs: `geometry_disabled` reached 2/5,
`geometry_shadow` 4/5, and `geometry_online_bounded` 4/5. Every enabled run
used `map_backend=local_map`, processed 3--4 observations, stayed within the
50 ms geometry budget, and used no privileged state. Each enabled run also
contained two distinct clearance scores, but all 10 enabled action-subgoal
decisions selected via `evidence_tie_break`; geometry did not yet yield a
unique candidate ranking. The reference provider still leaves grasp quality
unknown. P5.2 is therefore code-complete, transport-closed, and pilot-closed,
but its causal selection-quality gate remains open. P5.3 isolated process
rehearsal is the next engineering phase and must preserve the matched
evidence/version baseline.

Exit criterion: memory updates and candidate evidence are reproducible,
attributable, versioned, and cannot change active robot execution; rehearsal
and OOD evidence improve candidate selection without introducing regression on
the locked suite.

P5.0 contract-level closure is the prerequisite for the next isolated
rehearsal increment. It freezes the active SparseVoxel baseline and strict
candidate/scene version checks; TSDF and real semantic adapters remain open
runtime work and must not be implied by P5.2 completion. P5.3 may proceed with
the contract gate closed while preserving the matched P5.2 baseline.

P5.3 code and process-gate status (2026-07-30): the serializable rehearsal
contract, CAP-X worker boundary, bounded respawn, version-bound evidence
conversion, shadow Arbiter attachment, and seed-scoped artifact driver are
implemented and pass the local suite. The real matched two-candidate,
five-seed LIBERO gate is closed at
`outputs/phase5/P5.3_process_rehearsal_matched_fix_20260730/`. Candidate
`policy-0:0` reached 0/5 evaluator successes and `policy-1:safety:1` reached
2/5; the candidates differed on seeds 1 and 5, with the remaining failures
classified as `postcondition_failure`. This demonstrates candidate-specific
process evidence, not an online selection or downstream improvement claim.

The real input artifact uses a graph-scoped fingerprint. The P5.3 identity
closure now preserves that source hash and requires an explicit
`arbiter_subgraph_id` plus a derived local subgraph fingerprint before
attaching evidence to a `GraphCandidate`. A pure shadow-Arbiter API compares
baseline and evidence-enriched hypothetical decisions without accessing a
backend, lease, or executor; mapped evidence failures remain unavailable.
This closes the graph/subgraph identity and shadow-Arbiter code gates, but it
does not promote the shadow winner to physical execution or establish a
downstream success-rate gain. Ten-plus seeds, multiple tasks, online physical
selection, TSDF, real semantic adapters, evidence caching, OOD replay, and
calibration remain open follow-up work.

P5.3.1 code status (2026-07-31):
`capmas/evaluation/online_rehearsal.py` adds the batch provider seam and typed
baseline/evidence-aware/live report. `LLMGraphScheduler` uses it consistently
for legacy, staged, ready-wave, and rolling-frontier selection. The scheduler
default remains `disabled`; shadow mode never changes the live arbitration,
and `online_bounded` is fail-closed with explicit baseline fallback. The
driver `scripts/run_libero_p53_online.py` reuses the isolated rehearsal worker,
records identity/version rejections and provider latency, and performs no
more than one physical execution. The runner also retains failure artifacts
when live execution raises, gives cap-x's LIBERO robosuite fork priority over
the generic robosuite checkout, and terminates timed-out rehearsal workers
before the live executor starts. The first real endpoint-backed
`online_bounded` smoke completed on LIBERO spatial task 0 with two candidates:
the baseline `confidence_fallback` winner was policy-0, while rehearsal
evidence selected policy-1 and the single live execution completed with
`evaluator_success=true`. The smoke artifact is retained under
`outputs/phase5/P5.3.1_real_smoke_20260731_cleanfix/`.

The matched downstream evaluation is now complete in two independent suites:
`outputs/phase5/P5.3.1_matched_spatial0_20260731/` for seeds 1--5 and
`outputs/phase5/P5.3.1_matched_spatial0_seeds6_10_20260731/` for seeds 6--10.
Using the same two candidates and reset seeds 1--10 for both modes, with
`CUDA_VISIBLE_DEVICES=5`, `max_workers=1`, and `timeout_s=360`, all ten pairs
completed. The disabled baseline was `0/10`; `online_bounded` was `2/10`, a
matched delta of `+2/10`. Both candidates' rehearsal evidence attached in all
ten online decisions with zero identity/version rejections. Seeds 1 and 5
changed the live winner from policy-0 to policy-1 and both passed the CAP-X
evaluator; seeds 2--4 and 6--10 retained policy-0 through
`evidence_tie_break` and failed. This closes the single-task matched gate and confirms that the baseline
physical control is present. Multiple tasks, larger seed sets, and confidence
intervals are still required before a general causal claim.

P5.4 code status (2026-07-30): `capmas/evaluation/evidence_cache.py` provides
the versioned, thread-safe process-local LRU specified by the Phase 5 handoff.
It keys entries by canonical local candidate fingerprint and source scene
version, invalidates old entries after refresh, rejects missing or mismatched
scene metadata, and exposes cache statistics/events for run artifacts. The
candidate attach helper is opt-in and returns immutable candidate copies; it
does not alter Arbiter weights or physical execution. The P5.4 code gate is
closed, while persistent/cross-process caching, real hit-rate measurement, and
downstream causal evaluation remain open.

P5.4 isolated evaluation implementation (2026-07-31):
`scripts/run_p54_evidence_cache.py` compares `cache_disabled` and
`cache_enabled` using the same deterministic request trace. It records exact
hits, invalidations, stale rejections, provider calls, bounded cache events,
failure artifacts, and per-lane manifests under
`outputs/phase5/P5.4_cache_evaluation/`. This is an isolated local cache
evaluation; it does not call an LLM, start CAP-X/LIBERO, execute a robot, or
establish a downstream success-rate improvement. The formal run artifact and
observed metrics are recorded at
`outputs/phase5/P5.4_cache_evaluation/20260731_112755_cache_disabled_seed1/`
and
`outputs/phase5/P5.4_cache_evaluation/20260731_112755_cache_enabled_seed1/`.
The control made 9 provider calls and the enabled lane made 5, with 3 exact
hits, 5 stores, 2 invalidations, 1 stale rejection, final scene version 2,
and zero stale attachments. Both manifests passed SHA-256 verification. This
isolated result does not establish downstream success-rate improvement.

P5.4 online cache seam status (2026-08-03): `select_with_rehearsal` and
`LLMGraphScheduler` now accept an opt-in process-local `VersionedEvidenceCache`.
The cache is keyed by effective local candidate fingerprint and current scene
version, rejects stale or mismatched rehearsal evidence, and forwards only
uncached candidates to the provider. The LIBERO online runner exposes
`--cache-mode disabled|enabled` and `--selection-repeats N`, and persists cache
events and statistics. Repeated requests in one episode reuse the cache while
the physical executor remains single-owner and executes once. This closes the
online integration/code gate while the empirical repeated-rolling hit-rate/
latency gate remains open; the historical one-decision episodes can only
produce stores, not hits.

Real P5.4 repeated-selection smoke (2026-08-03): using the CAP-X
`.venv-libero` environment on CUDA device 5, Spatial-0 seed 1, and the same
two candidates, `selection_repeats=2` reduced rehearsal records from 4 to 2,
provider calls from 2 to 1, recorded two cache hits, reduced total selection
latency from approximately 683.8 s to 341.4 s, and kept physical execution at
exactly one in both modes. The run-scoped configuration, selection result,
summary, history, and runner log all expose the cumulative provider call
count. Both evaluator results were false, so this closes only the real
single-episode cache/latency smoke gate; multi-task, multi-seed, and
downstream-success evaluation remain open.

The next matched evaluation is implemented by
`scripts/run_libero_p54_matched.py`. It runs cache-disabled and cache-enabled
`online_bounded` lanes independently for every task/seed pair, with
`selection_repeats >= 2`, fresh process-local caches, separate child artifact
directories, and one physical execution maximum per lane. Its suite aggregate
separates provider-call reduction, cache hits, selection latency, physical
execution, and evaluator success. The multi-seed/multi-task empirical gate is
still open until this driver is run on the locked task suite; cache efficiency
alone is not a downstream success-rate claim.

The first real matched cache run completed on 2026-08-03 at
`outputs/phase5/P5.4_matched_online_cache_20260803/P5.4_matched_online_cache/20260803_054828_suite_d64ab784/`.
For LIBERO Spatial-0 seeds 1--5, all five pairs completed and all manifests
passed SHA-256/size verification. The disabled lane made 10 provider calls
and the enabled lane made 5, with 10 exact cache hits in the enabled lane;
total selection latency was 1792.70 s versus 899.77 s. Both lanes executed
once per seed and both achieved evaluator success on seeds 1 and 5 only,
giving `2/5` in each condition. The same candidate was selected in every pair,
so this closes the single-task five-seed cache-efficiency gate but not a
downstream success-rate or multi-task generalization gate.

P5.5 frozen OOD replay code status (2026-08-03): `capmas/evaluation/ood.py`
defines immutable case/split/evidence contracts and explicit `pair_id`
provenance. `capmas/evaluation/ood_statistics.py` provides Wilson intervals,
paired deltas, exact McNemar tests, bootstrap OOD gaps, and unknown/failure
accounting. `scripts/run_libero_p55_ood.py` performs preflight digest and
leakage validation, then runs independent ID/OOD cases through the existing
CAP-X online path with shadow-only evidence and retained manifests/logs.

The initial fixture is `configs/phase5/p55_ood_smoke.json`. Its layout-OOD
label is a manually curated smoke membership without a physical perturbation
generator; therefore it is a replay/provenance gate, not a measured OOD gap.
The runner does not alter active Arbiter selection or promote Memory/Robot
Skills. The real paired smoke and five-seed, multi-family pilot are now
completed and documented below with confidence intervals. P5.6 calibration is
required before OOD evidence can become an active selection signal.

P5.5 real smoke result (2026-08-03): the suite at
`outputs/phase5/P5.5_ood_replay_20260803_smoke1/P5.5_frozen_ood_replay/20260803_090001_suite_fa2ccc0e/`
completed one ID and one manually labeled layout-OOD case with no runner
failures. Both cases executed one physical candidate and both had evaluator
success `0/1`; the paired result was a tie at `0/0`. Suite and case artifact
manifests passed SHA-256 and size checks and evidence remained shadow-only. Since both
cases used the same physical config and candidate artifact, this closes only
the real replay/provenance gate, not the physical OOD-generalization gate.

P5.5 real layout-variant pilot result (2026-08-04):
`outputs/phase5/P5.5_real_layout_pilot_20260803/P5.5_frozen_ood_replay/20260803_113118_suite_b4bbc31b/`
completed all 30 cases across `spatial-0`, `goal-1`, and `object-6`, with
five matched seeds per family. The ID and real layout-OOD results were both
`0/15`, with zero infrastructure unknowns, 15 paired ties, paired delta 0,
and exact McNemar `p=1.0`. All cases retained `shadow_only=true`, all 15
pairs had distinct layout state fingerprints, and all artifact manifests
passed digest/size verification. This closes the five-seed multi-family
measurement gate only; the formal ten-seed gate is recorded below. Active OOD
calibration and downstream success-rate claims remain open. The Arbiter used
`evidence_tie_break` in all 30 cases, so no causal selection improvement is
claimed.

P5.5 real layout-variant ten-seed formal gate result (2026-08-05): the
frozen manifest
`outputs/phase5/P5.5_real_layout_assets_20260803/p55_real_layout_3family_10seed.json`
(canonical manifest SHA-256, excluding the self-digest field:
`5aeff85dae764c72fe9c0b1f3a0a07f4070e95247baea1b8d93f15311ea72141`)
was evaluated in
`outputs/phase5/P5.5_real_layout_formal_20260804/P5.5_frozen_ood_replay/20260804_014522_suite_dda9defe/`.
It contains 60 cases, 30 matched pairs, three families, and seeds 1--10.
The CAP-X `.venv-libero` run used CUDA device 5, two workers, no restarts,
`max_steps=32`, `timeout_s=360`, one selection repeat, and disabled cache.
All 60 cases completed at the runner level with zero infrastructure
unknowns; all 60 were `task_failure` with zero recovery and intervention.
ID and OOD success were both `0/30`, both Wilson estimates were `0.0` with
95% upper bound `0.1135`, paired delta was `0`, there were 30 paired ties,
and exact McNemar `p=1.0`. Mean/median/range reported latency was
`78.73/72.68/55.18--211.06 s`. Horizon buckets were not available because
the case artifact records no realized subgoal count; `max_steps=32` is only a
budget. All 30 pairs had distinct layout fingerprints, all manifests passed
digest/size checks, and `selection_basis=evidence_tie_break` was used in all
60 cases. This closes the P5.5 measurement/provenance gate only; it is not a
downstream success or causal Arbiter result. P5.6 calibration, correlated
signal correction, active OOD weighting, and learned Arbiter weighting remain
deferred.

P5.5 execution-grounding smoke result (2026-08-05): after reset-time graph
grounding was added to both the live executor and isolated rehearsal worker,
the six-case three-family, one-seed real-layout smoke completed at
`outputs/phase5/P5.5_grounding_smoke_venv_20260805/P5.5_frozen_ood_replay/20260805_085352_suite_fd89ecef/`.
It used CUDA device 5, one worker, zero restarts, `max_steps=32`, and disabled
cache. ID success was `1/3`; layout-OOD success was `0/3`; infrastructure
unknowns were `0`. The five failures were explicit
`POSTCONDITION_FAILED` outcomes. The native/OOD spatial placement trace moved
from `x=0.72409` to `x=0.81099`, matching the direction of the audited layout
translation, so execution grounding is confirmed. This is a one-seed
grounding smoke only; it does not close multi-seed OOD quality or causal
Arbiter gates. Remaining failures concern grasp/coordinate postconditions and
task-to-graph mapping. Full regression verification was `421 passed` with
compileall success.

P5.5 gripper-state semantic correction (2026-08-05): CAP-X measured finger
opening and commanded gripper fraction are now represented separately. The
CAP-MAS CAP-X adapter publishes optional
`robot_state["gripper_commanded_fraction"]` from the low-level environment;
the verifier uses that signal for `gripper_open()` and `gripper_closed()` and
falls back to `gripper_opening` for legacy snapshots. A fresh object-6
grounded probe physically lifted butter and passed both
`object_in_gripper(butter)` and `gripper_closed()`. The probe stopped before
placement, so this closes only the pick-checkpoint semantic issue.

### P5.5 target-pose verified object-6 online closure (2026-08-06)

Placement grounding now separates a container's semantic body-center pose from
the safe release pose. The implementation uses the clipped point-cloud center,
adds top release clearance for basket-like targets, and grounds placement as
pre-place approach, target descent, release, and retreat. The placement
postcondition uses robust target XY while retaining semantic target Z, which
prevents partial target occlusion from producing a false verifier failure.

The disabled-mode physical smoke completed object-6, and the subsequent full
online rehearsal-Arbiter smoke completed at
`outputs/phase5/P5.5_target_pose_verified_object6_online_20260806/P5.3.1_online_rehearsal_arbiter/20260806_052654_seed1_d1b5f0d1/`.
The online run used CUDA device 5, one worker, zero restarts, 32 maximum
steps, one selection repeat, and disabled cache. Both isolated candidates
completed rehearsal, the Arbiter attached both and selected the first
candidate with `evidence_tie_break`, and exactly one physical execution was
performed. The final boundary reported `completed=true`,
`evaluator_success=true`, and `success=true` with no failure artifact.

The pre-fix online failure was caused by a verifier/placement-frame mismatch,
not by failure to execute the robot action: a partially occluded basket
produced an erroneous semantic target pose and `object_at_target` reported a
`0.1369 m` distance. This run closes the object-6 target-pose regression and
the full online smoke. It is still a one-seed, one-family smoke and does not
close the P5.5 multi-seed OOD success-rate gate or establish causal Arbiter
improvement. The retained online artifact has a `null` final basket
`placement_pose_wxyz_xyz` only when queried as a missing JSON field: that run
predates placement provenance in `_scene_debug_payload`, so it did not prove a
provider-side null.

P5.5 placement observability implementation (2026-08-06): `ObjectTrack` now
retains `placement_pose_source` and `placement_pose_reason`. The CAP-X adapter
marks successful estimates as `geometry_pointcloud`, while callback errors,
unexpected payloads, and invalid/insufficient point clouds become explicit
`semantic_pose_fallback` records. Runtime behavior remains unchanged. Unit and
serialization coverage is complete; a fresh real object-6 capture is the
remaining validation gate.

P5.5 placement provenance real capture (2026-08-06): the fresh CUDA-5 run at
`outputs/phase5/P5.5_placement_provenance_object6_20260806/P5.3.1_online_rehearsal_arbiter/20260806_064328_seed1_5168c2e2/`
recorded basket `placement_pose_source=geometry_pointcloud` both before and
after physical execution, with non-null placement poses and no fallback
reason. The object-6 mission again completed with evaluator success. This
closes the placement provenance software and real-capture gates; matched
multi-seed ID/OOD evaluation remains separate.

P5.5 matched provenance five-seed pilot (2026-08-06): the corrected
single-worker suite at
`outputs/phase5/P5.5_matched_provenance_5seed_20260806/P5.5_frozen_ood_replay/20260806_091429_suite_e169a480/`
completed all 30 cases with zero infrastructure unknowns. Evaluator success
was ID `3/15` and OOD `5/15`; graph/verifier success was ID `2/15` and OOD
`4/15`. The paired result had two OOD-only successes and McNemar `p=0.5`, so
no OOD improvement is claimed. Selection used `evidence_tie_break` 28 times
and `evidence_score` twice.

The reporting audit separates 24 graph-level `POSTCONDITION_FAILED` records
into 22 physical task failures and 2 verifier false negatives. The latter are
`id-object-6-seed4` and `ood-object-6-seed2`, where LIBERO accepted the
placement but the point-distance verifier rejected it. The corrected offline
report is retained under
`outputs/phase5/P5.5_matched_provenance_5seed_report_correction_20260807/P5.5_offline_reaggregation/20260807_013832_suite_e169a480/`.
P5.5 remains open until a corrected ten-seed, three-family run is complete;
the five-seed pilot does not close the formal gate or demonstrate causal
Arbiter improvement.

P5.5 corrected matched-provenance ten-seed formal gate (2026-08-07): the
single-worker suite at
`outputs/phase5/P5.5_matched_provenance_10seed_retry2_20260807/P5.5_frozen_ood_replay/20260807_024842_suite_20674432/`
completed all 60 cases and 30 matched pairs across `spatial-0`, `goal-1`, and
`object-6`. It used the frozen manifest digest
`5aeff85dae764c72fe9c0b1f3a0a07f4070e95247baea1b8d93f15311ea72141`,
CUDA device 5, one worker, no restarts, `max_steps=32`, `timeout_s=360`, one
selection repeat, and disabled cache. There were zero case failures, zero
infrastructure unknowns, zero recoveries, and zero human interventions. All
30 pairs had distinct ID/OOD layout fingerprints, all 60 records remained
shadow-only, and the suite manifest's 907 entries plus all 60 case manifests
passed size and SHA-256 verification.

Evaluator success was ID `4/30` and OOD `10/30`; graph/verifier success was
ID `4/30` and OOD `8/30`. The paired table contains zero ID-only successes,
six OOD-only successes, and 24 ties, with exact McNemar `p=0.03125`. The
observed difference comes entirely from `object-6` (`4/10` versus `10/10`);
both `spatial-0` and `goal-1` remained `0/10` in each split. Of 48 graph-level
`POSTCONDITION_FAILED` outcomes, 46 were physical task failures and two were
verifier false negatives (`ood-object-6-seed1` and `seed2`). Selection used
`evidence_tie_break` 57 times and `evidence_score` three times.

This closes the corrected P5.5 ten-seed measurement, pairing, provenance, and
artifact-retention gate. It does not establish causal Arbiter improvement or
general OOD robustness: P5.5 is shadow-only, the OOD layout was easier for one
task family, and 57/60 selections were ties. Realized horizon is still absent
from `OODReplayEvidence`; `max_steps=32` is only a budget. Adding a verified
executed-horizon field and calibrated, actively weighted evidence is the P5.6
handoff before any horizon-stability or causal selection claim.

### Phase 5.6 - Qualified evidence calibration

P5.6 proceeds on two parallel but gated lanes. Calibration infrastructure,
horizon/label contracts, leakage-safe datasets, correlation control, and
offline fitting may be built while `spatial-0` and `goal-1` capability failures
are diagnosed. `object-6` is a pipeline smoke family only and cannot define a
global model. No family receives an online calibrated probability until it
passes both gates below.

P5.6.0 is diagnostic-only. It emits typed root-cause artifacts and an
independent `P5.3.2 Task-Family Capability Repair` work package. P5.3.2 owns
goal-task mapping, prompt, skill-argument, and placement/release parameter
changes plus the replacement ten-seed capability run. Its failure blocks only
the affected family's promotion and an all-family claim; it does not block
object-6 offline calibration or family-scoped shadow evaluation.

The capability gate requires a fixed ten-seed diagnostic run with zero
infrastructure unknowns, typed failures, at least 80% physical-execution
reachability, and at least one valid evaluator success. The calibration gate
requires at least 20 independent Tier A physical outcomes for that family,
including at least five positive and five negative `task_success` labels.
Ineligible families abstain and use the fixed-weight Arbiter fallback.

#### P5.6A data foundation status (2026-08-19)

P5.6.0, P5.6.1, P5.6.2, and the collection portion of P5.6.2a are complete
as a data foundation, not as an active calibration feature. The immutable
contracts use `p56.feature.v1`; each selected candidate has a decision-time
feature snapshot, a typed physical outcome, lineage, and planned/realized
horizon telemetry. `max_steps=32 is not a horizon`; it is an execution budget
and never determines an H bucket.

The read-only capability diagnosis at
`outputs/phase5/P5.6.0_capability_diagnosis/20260813_072308_capability_c9df3f4b/`
found zero infrastructure unknowns and 10/10 physical execution reach for all
three families. `spatial-0` and `goal-1` each had 0/10 evaluator successes and
are explicitly handed to `P5.3.2 Task-Family Capability Repair`. `object-6`
had 4/10 evaluator successes and is eligible only as a family-scoped
calibration data source.

The historical compatibility audit at
`outputs/phase5/P5.6.2a_object6_history_audit/20260813_103434_history_a6bc49b1/`
accepted zero historical rows. Two complete, disjoint, frozen object-6 ID
blocks were therefore executed: seeds 11-20 at
`outputs/phase5/P5.6.2a_object6_collection/20260818_090102_suite_63248cf1/`
and seeds 21-30 at
`outputs/phase5/P5.6.2a_object6_collection/20260818_095350_suite_50dc9bd3/`.
Each block completed all 10 cases with 5 positive and 5 negative Tier A
outcomes. Together they provide 20 Tier A outcomes, 10 positive and 10
negative, closing the 20/5/5 data gate without adaptive seed selection.
Both collection manifests verify cleanly. The selected Tier-A feature
snapshots are decision-time snapshots: every row satisfies
`captured_at_ns <= decision_boundary_ns`. The earlier invalid collection
directories with future-state features remain retained for audit and are not
used by calibration. Each suite retains `results/manifest_verification.json`.

The P5.6B/C implementation and real-data offline fit are now complete, but
the offline qualification gate remains open. The corrected baseline run
`outputs/phase5/P5.6.4_offline_calibration/20260819_014928_p56b-object6-offline-baseline-v2/`
has a verified manifest and frozen predictions paired by test-row lineage.
The calibrated model scores Brier `0.02778` and ECE `0.08333`; the
fixed-weight/PAVA baseline scores Brier `0.00826` and ECE `0.04545`.
Therefore ECE passes but calibrated Brier improvement is `-2.3611`, below
the required `>= 0.10`. No calibration snapshot, active evidence weight, or
calibrated Arbiter selection is enabled. P5.3.2 remains a separate
task-completion repair, and no all-family or downstream-success claim is
justified.

The current object-6 suite contains 20 physical outcomes: 14 positive and six
negative. It meets the numerical 20/5/5 gate, but the records predate P5.6
feature-snapshot and horizon lineage and therefore require a read-only
compatibility audit. If admissible rows are insufficient, collection uses
pre-registered ID seed blocks 11-20 and then, only after block-level review,
21-30. It never performs outcome-adaptive single-seed stopping or cross-family
pooling.

P5.6 buckets planned horizon by action-bearing subgraphs on the critical path.
Checkpoint-only verification subgraphs, skill-call counts, and `max_steps` do
not inflate the task bucket. A single action-bearing subgraph is H1; empty
H2-3/H4-6/H7+ buckets are `N/A`. The frozen P5.5 candidates each have two
action-bearing subgraphs plus one checkpoint-only subgraph and are H2-3.
Realized attempted/completed actions, subgoals, and checkpoints remain separate
diagnostics. Physical task success is the only primary calibration label;
graph completion, verifier success, rehearsal outcomes, OOD metadata, and
horizon remain separate diagnostics or shadow inputs.

The initial active feature set has `ood_weight=0`. A deterministic
correlation-group reducer precedes constrained logistic plus isotonic
calibration. Support coefficients must be non-negative; latency, recovery,
collision, and safety-risk coefficients must be non-positive. Unknown evidence
is explicit and cannot become zero quality. Snapshots are immutable and
content-addressed, episodes pin one snapshot, activation is atomic, and
rollback is explicit.

Implementation order:

```text
P5.6.0 diagnostic-only capability audit and P5.3.2 handoff
P5.6.1 horizon, label, and lineage contracts
P5.6.2 three-tier dataset and leakage audit
P5.6.2a object-6 compatibility audit and fixed-block collection
P5.6.3 deterministic correlation-group reduction
P5.6.4 constrained logistic/isotonic calibration
P5.6.5 immutable snapshot registry and episode pinning
P5.6.6 offline metrics and ablations
P5.6.7 shadow Arbiter with abstention/fallback
P5.6.8 eligible-family bounded canary
P5.6.9 formal matched evaluation and Phase 6 handoff
```

Offline targets are Brier improvement of at least 10% against the fixed-weight
baseline mapping and ECE at most `0.10`. The shadow gate requires zero safety
hard-gate disagreement, calibration inference P95 at most 5 ms, and eligible
coverage at least 50%. The bounded canary requires at least 20 matched
physical episodes and is safety/operability evidence only, not a success-rate
improvement claim. See
[`P5.6 design`](superpowers/specs/2026-08-11-p5-6-evidence-calibration-design.md)
and [ADR-0014](adr/0014-calibrated-evidence-and-snapshot-activation.md).

#### P5.6C fit stability and baseline comparison status (2026-08-19)

P5.6C adds an offline-only `p56b.constrained_logistic.v2` fitter. It records
train-design rank and availability, freezes constant non-intercept columns,
requires both loss-delta and projected-KKT convergence, and abstains when a
scoring vector violates the train availability signature. It changes neither
the evidence reducer, the locked `12/4/4` split, OOD policy, runtime Arbiter,
nor physical execution.

The real recollection and offline run are now available and independently
verified. The train design has rank 2 of 10 columns because all optional
scene/risk dimensions are unknown and only action feasibility varies. The
fit's projected-KKT residual is below tolerance, so the model is a valid
offline artifact rather than a loss-only false-convergence result. The PAVA
calibration and test prediction artifacts are also present. The corrected test
pairing handles repeated candidate IDs without overwriting labels. On the
locked `12/4/4` split, the calibrated model obtains Brier `0.02778` and ECE
`0.08333`; the fixed-weight/PAVA baseline obtains Brier `0.00826` and ECE
`0.04545`. Thus ECE passes, but calibrated Brier improvement is `-2.3611`,
below the required `>= 0.10`, and `offline_qualification_passed=false`.
P5.6C implementation and fit stability are complete, while calibrated
activation, shadow arbitration, canary promotion, and causal evaluation
remain blocked. The fitted artifact remains offline-only and must not change
runtime Arbiter selection.

## Phase 6 — Memory Controller learning

- Add verified terminal and intermediate learning-return calculation.
- Train the Stage 1 offline selector from archived traces.
- Add Stage 2 short-horizon simulation adaptation with the Robot Skill Registry
  frozen.
- Add Stage 3 long-horizon constrained PPO/GRPO as an offline/asynchronous
  training path with snapshot rollback.

Exit criterion: the learned controller beats the rules baseline without a
regression on the locked suite or control deadline.

## Phase 7 — Robot Skill evolution

- Implement quarantine registry and immutable skill versions.
- Extract candidate compositions from traces.
- Add shadow validation, regression tests, and OOD boundary tests.
- Keep the promoted Memory Skill Bank and controller frozen while candidates
  are evaluated.
- Add safe-boundary and later-episode activation modes.

Exit criterion: Robot Skill evolution improves targeted OOD cases without
locked-suite regression and without relying on simultaneous memory updates.

## Phase 8 — Adaptive topology and research evaluation

- Add event-triggered role activation and sparse edge edits.
- Run the full ablation matrix.
- Freeze test seeds and evaluation budgets.
- Produce failure analysis and paper-ready plots.
