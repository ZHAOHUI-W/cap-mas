# Staged Graph Protocol

## Motivation

The legacy P3.1 protocol asks the Manager to produce a complete `MissionGraph`.
Each local Policy Agent then returns another complete graph wrapper and the
runtime extracts one `SubgraphSpec`. This is safe, but it repeats mission
fields, topology, and the full nested graph schema on every request. That
repetition increases prompt size, output length, provider validation cost, and
timeout probability.

The staged protocol separates the two planning decisions:

```text
Stage 1: MissionTopology
  Manager -> mission identity, subgoals, dependencies, terminals, bounded loops

Stage 2: local SubgraphArtifact
  Policy Agent -> one versioned SubgraphSpec for one Manager subgoal

Stage 3: server assembly
  topology + selected local graphs -> MissionGraph
  -> GraphValidator -> CandidateArbiter -> FixedGraphInterpreter
```

The change optimizes the LLM boundary only. It does not create an additional
actuator path and does not allow a Policy Agent to acquire `ActionLease`.

## Wire contracts

`MissionTopology` is versioned independently from the executable graph and
contains mission identity, scene version, `TopologySubgoal` records,
dependencies, terminal subgraphs, mission edges, optional typed bindings, and
explicit finite `LoopSpec` records. A topology subgoal contains no executable
skill calls or local node details. It declares `execution_kind`: a
`physical_action` subgoal requires Policy proposals, while a `checkpoint_only`
subgoal contains only observable scene predicates and is compiled
deterministically.

The local Policy response uses this strict envelope:

```json
{
  "schema_version": 1,
  "subgraph": { "subgraph_id": "...", "subgoal_id": "...", "nodes": [] }
}
```

The local envelope intentionally has no `mission_id`, full mission edge list,
or other subgraph candidates. `LocalSubgraphDecoder` requires the requested
IDs, validates the local graph, and rejects unknown fields. The scheduler then
checks that every topology success predicate is declared by a candidate node or
validating checkpoint.

## Runtime protocol

1. `LLMTopologyManager` requests topology against scene version `v`.
2. `MissionTopologyDecoder` rejects malformed, stale, mismatched, or cyclic
   topology before any Policy request is sent.
3. The scheduler creates one typed `topology_subgoal` artifact per subgoal.
4. `checkpoint_only` subgoals are lowered to deterministic checkpoint graphs
   without calling a Policy Agent. Registered local Policy Agents produce
   read-only candidates concurrently for `physical_action` subgoals.
5. `LocalSubgraphDecoder`, `GraphValidator`, and `CandidateArbiter` reject
   malformed, stale, wrong-ID, unsafe, or low-ranked candidates.
6. Selected local graphs are assembled in Manager order into one `MissionGraph`.
7. The existing full graph validator and skill registry validator run again.
8. Only the single selected graph reaches the existing executor, lease manager,
   observable verifier, and recovery boundary.

No physical action starts if topology decoding, local candidate generation, or
assembly fails. Checkpoint nodes are evaluated against the committed
`SceneSnapshot` by the runtime checkpoint evaluator; a failed predicate follows
the checkpoint failure edge and cannot be treated as a successful no-op. This
preserves the same fail-closed behavior as legacy P3.1.

### Proposal scheduling modes

The runner exposes two proposal modes:

- `subgoal_serial`: each subgoal fans out its Policy candidates, arbitrates
  them, and then advances to the next subgoal.
- `ready_wave`: the scheduler computes the dependency-ready frontier and fans
  out all read-only Policy requests in that frontier through one bounded worker
  pool. The mode changes proposal latency only; it does not execute robot
  actions in parallel.

`ready_wave` is safe only for candidates grounded in the same immutable
`SceneSnapshot`. A candidate depending on an action output or later scene
version must be revalidated or regenerated before dispatch. Compile results
report `proposal_mode`, `proposal_waves`, and `compile_latency_ms` for matched
experiments.

### Compile-time scene refresh and graph rebase

The scene used for proposal generation is immutable, but a long Manager plus
Policy compile can make its `scene_fresh(threshold_ms)` predicate invalid by
dispatch time. The runner therefore performs a fail-closed boundary before
physical execution:

1. call `RobotBackend.observe()` after compilation;
2. commit the observation with `InMemoryStateStore.compare_and_commit()`;
3. call `LLMGraphScheduler.rebase_graph()` with the new scene version;
4. rerun scene-dependent LIBERO grounding, including target-pose grounding;
5. validate the rebased graph and only then invoke `FixedGraphInterpreter`.

This keeps freshness verification meaningful instead of removing the
predicate to accommodate LLM latency. Rolling execution uses the same refresh
hook before each dispatch cycle. If a state commit races or a scene rewriter
changes graph identity, execution stops with a failure artifact.

## Legacy comparison and ablations

The runner supports both protocols:

```bash
# compact staged protocol (default)
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. ../cap-x/.venv-libero/bin/python \
  scripts/run_libero_b3_llm.py \
  --config-path ../cap-x/env_configs/libero/franka_libero_spatial_0_privileged.yaml \
  --api-base "$CAPMAS_LLM_API_BASE" --model gpt-5.5 \
  --graph-protocol staged --skip-api-servers

# full-graph P3.1 ablation
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. ../cap-x/.venv-libero/bin/python \
  scripts/run_libero_b3_llm.py \
  --config-path ../cap-x/env_configs/libero/franka_libero_spatial_0_privileged.yaml \
  --api-base "$CAPMAS_LLM_API_BASE" --model gpt-5.5 \
  --graph-protocol legacy --skip-api-servers
```

To compare serial and dependency-ready proposal compilation while retaining the
same physical executor:

```bash
... scripts/run_libero_b3_llm.py ... --graph-protocol staged \
  --proposal-mode subgoal_serial
... scripts/run_libero_b3_llm.py ... --graph-protocol staged \
  --proposal-mode ready_wave
```

To activate rolling execution for the same staged planner:

```bash
... scripts/run_libero_b3_llm.py ... --graph-protocol staged \
  --execution-mode rolling --max-steps 32
```

Rolling artifacts use a distinct baseline name and contain all compile results
under `result.compilations`, so fixed-graph and rolling execution must not be
pooled as one protocol.

| Variable | Legacy | Staged |
| --- | --- | --- |
| Manager output | complete `MissionGraph` | `MissionTopology` |
| Policy output | full graph wrapper | one local graph envelope |
| Policy fan-out | concurrent | concurrent |
| Static validation | local and assembled graph | local and assembled graph |
| Physical executor | one | one |
| CAP-X skills/API | same | same |

The expected benefit is lower request size and lower model-side planning
coupling. Staged planning adds a strict boundary and can have more total calls
than a Manager-only ablation; measure total wall time, request latency,
input/output tokens, rejection rate, and downstream success rather than
assuming a latency improvement.

Policy strategy diversity is an explicit ablation, not an implicit property of
the agent name. The runner accepts `--policy-strategies balanced,safety` (or a
single strategy repeated for all agents). Strategy guidance and a typed
`strategy_profile` are included in the Policy request, and the strategy is
included in the agent identity recorded in the candidate and LLM call trace.
The scheduler evidence provider can attach typed perception, verifier,
rehearsal, OOD, latency, and recovery-cost evidence before Arbiter selection.
The LIBERO runner now attaches a read-only SceneSnapshot-backed perception
provider; rehearsal and OOD remain unavailable until their later providers are
enabled. Evidence is bound to the source `SceneSnapshot.scene_version` and
stale evidence is rejected before selection. When no evidence is attached,
`ArbitrationResult` records `selection_basis=confidence_fallback`; equal
evidence scores are recorded as `selection_basis=evidence_tie_break`.
`ExperimentRunConfig.policy_strategies` records the ordered
strategy profile for every Policy index, so `balanced` remains distinguishable
from explicit heterogeneous profiles even when its agent name is the compact
`policy-0` form.

Pure scene verification is excluded from the Policy latency budget. The Manager
must emit `execution_kind: "checkpoint_only"` for goals such as track
existence or scene freshness. The scheduler records a
`deterministic_checkpoint` arbitration with producer `checkpoint_compiler`,
preserving graph artifacts and failure routing while removing otherwise
redundant Policy requests. The LIBERO runner binds
`FixedGraphInterpreter.checkpoint_evaluator` to
`LiberoObservableVerifier.goal_satisfied`; bypassing Policy does not bypass
predicate verification.

Strict OpenAI-compatible gateways require `additionalProperties: false` on
every object, which conflicts with an open-ended runtime `SkillCall.args`
dictionary. CAP-MAS therefore derives the union of parameter names from the
registered typed skills, emits those keys in the provider schema, and uses
`null` for parameters unused by a particular call. The local graph decoder
removes those null placeholders before CAP-X execution, so the runtime API
continues to receive the original compact argument dictionary. Unknown
parameter names remain rejected by the typed skill registry.

For gateways with an incompatible structured-output implementation,
`--no-provider-structured-output` remains available and the client retains a
narrow automatic fallback for schema-compatibility 400 responses; local
decoders remain mandatory. Gateway 504/read-timeout failures are not treated
as valid plans and do not trigger robot execution.

The LIBERO execution adapter adds three runtime hardening rules after local
candidate validation:

- `sample_grasp_pose -> goto_pose -> close_gripper` is required for sampled
  grasps; `goto_pose` is rewritten to reference the sampled pose output and a
  bounded approach offset is supplied when omitted.
- A placement `goto_pose` is grounded to the current target track in the
  `SceneSnapshot`, so approximate LLM coordinates cannot move the object away
  from the target.
- Dynamic predicates such as `scene_fresh`, `object_in_gripper`, and gripper
  state are checked at dispatch time. Only stable track/visibility facts are
  checked while compiling candidates.
- The scheduler retains each raw Policy subgraph alongside the normalized
  grounding/repair result and records stable fingerprints in the arbitration
  artifact. This makes normalization-induced candidate collapse measurable.
- If typed-skill validation rejects every candidate, the scheduler failure
  carries the candidate id, node id, skill id/version, call index, normalized
  args, raw args when available, expected callable signature, and missing or
  unexpected argument names. The runner copies these records into the
  `.failure.json` artifact before re-raising, so a compile-time failure is
  diagnosable without replaying the LLM response.

Topology assembly normalizes dependency edges to the success outcome and uses
the declared subgoal order to resolve provider-added duplicate success edges.
Unresolvable ambiguity is rejected by the static validators before execution.
Failure-edge recovery targets are additionally checked against the source's
normal success ancestry. A recovery target cannot depend on the failed source
or another subgraph that is not guaranteed to have committed before the
failure; such topology is rejected with
`RECOVERY_DEPENDS_ON_UNCOMMITTED`.

## Implementation map

- `capmas/contracts/staged.py`: `MissionTopology` and `TopologySubgoal`;
- `capmas/graph/staged.py`: topology schema and static validator;
- `capmas/llm/staged_decoder.py`: strict topology/local graph decoders;
- `capmas/llm/prompts.py`: compact stage-specific schemas and prompts;
- `capmas/agents/manager.py`: `LLMTopologyManager`;
- `capmas/agents/policy.py`: `LLMStagedGraphPolicyAgent`;
- `capmas/runtime/llm_scheduler.py`: `compile_staged()` and protocol switch;
- `scripts/run_libero_b3_llm.py`: `--graph-protocol staged|legacy` and
  `--proposal-mode subgoal_serial|ready_wave`.
- `capmas/evaluation/rehearsal.py`: spawned-process `RehearsalJob` /
  `RehearsalResult` boundary for offline candidate evidence.

P3.2 adds typed strategy profiles and SceneSnapshot-backed perception evidence;
the scheduler confidence is optional and fallback-only, and evidence records
its provider and source scene version. Dynamic geometry, candidate-specific
postcondition scoring, process rehearsal, OOD replay, and evidence calibration
are deliberately deferred to Phase 5. This protocol does not yet make robot
execution parallel or distributed. The only online parallel work remains
read-only Policy inference. The implemented
`ProcessRehearsalPool` is an offline-only spawned process boundary; it cannot
access the live robot lease or backend. CAP-X worker factories, LLM
Recovery/Monitor roles, streaming scene updates, and adaptive topology edits
remain separate phases.
`RollingGraphRunner` now provides the first closed-loop
execution seam: it executes one verified subgraph, refreshes the committed
`SceneSnapshot`, compiles only the current ready frontier, and follows the
fixed topology's explicit success/failure edge. A changed or missing
next-subgraph ID fails closed. The Manager is called once per episode in this
mode; subsequent Policy calls are frontier-scoped. See
[`p3-3-rolling-replan.md`](p3-3-rolling-replan.md) for the state machine and
acceptance tests. Rolling uses the same nullable/semantic edge normalization
as topology assembly, and rejects stale or cross-episode refresh snapshots.
Adaptive topology edits remain a later phase.
