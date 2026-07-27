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
implemented. The live CAP-X observation gate has passed in thread mode with
real LIBERO RGB-D and CAP-X object-pose measurements. Process-mode replay and
IPC tests pass; a live process-mode run remains deferred until CAP-X capture
artifacts are emitted directly into the shared file store.

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
- [ ] Run live CAP-X B5 process mode with shared file-backed capture artifacts.
- [ ] Add TSDF backend and semantic model adapters.

Exit criterion: the reference/live-thread B5 gate meets the predeclared
control deadline and freshness targets; process-mode live capture is a
follow-up integration gate.

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

Exit criterion: memory updates and candidate evidence are reproducible,
attributable, versioned, and cannot change active robot execution; rehearsal
and OOD evidence improve candidate selection without introducing regression on
the locked suite.

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
