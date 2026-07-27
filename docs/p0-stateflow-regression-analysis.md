# P0 Stateflow Endpoint Failure Analysis

Date: 2026-07-23

## Scope

This note explains why the recent staged LIBERO runs did not reach the robot
executor, and separates code-path changes from provider failures. The relevant
boundary is:

```text
Manager topology -> Policy proposal -> grounding/repair -> typed skill validation
  -> Arbiter -> FixedGraphInterpreter -> CAP-X executor
```

The failed historical runs stopped before the Arbiter and therefore could not
have moved the robot.

## Evidence

The earlier P0 artifacts reported:

- `staged_gpt55_p0_stateflow_20260723.log`: both Policy candidates were
  rejected as `invalid arguments for goto_pose`.
- `staged_gpt55_p0_stateflow_retry1_20260723.log`: one candidate was rejected
  for `goto_pose` and the other for `sample_grasp_pose`.
- Both artifacts had `graph: null`, empty arbitration results, and no execution
  result. This is a compile-time rejection, not a motion or verifier failure.

Those artifacts were created before candidate-level validation diagnostics were
persisted, so they cannot identify the exact candidate, node, or argument map.

The diagnostic path now records those fields in `PolicyProposalFailure` and
copies them into the runner's `.failure.json` artifact even when compilation
raises `LLMGraphScheduleError`.

## Comparison With Successful Runs

The successful P3.2 artifact used the same Manager and Policy schema hashes as
the P0 failure. Its normalized action calls use the expected CAP-X forms:

```text
sample_grasp_pose(object_name=...)
goto_pose(position=..., quaternion_wxyz=..., z_approach=...)
close_gripper()
```

The current code was also replayed against that known-good graph using the
current `repair_libero_grasp_subgraph -> ground_libero_grasp_subgraph ->
SkillRegistry.validate_contract` path. Both executable subgraphs passed typed
validation without entering the interpreter. Therefore the current grounding
code is not a universal regression against valid P3.2 candidates.

The post-success P0 state-flow changes do alter where invalid candidates are
rejected: `_filter_skill_candidates()` validates each normalized candidate
before arbitration. This is a stricter and safer admission point, but it does
not by itself turn a valid argument dictionary into an invalid one. It exposes
an invalid model candidate earlier and prevents any candidate from reaching the
single executor.

Other observed differences can increase the probability of malformed output but
are not proven root causes:

- P0 used `subgoal_serial`, 1536 output tokens, and fewer proposal retries;
  successful P3.2 used 2048 tokens and more retries.
- Manager-produced subgoal names differed between runs, indicating different
  model plans despite identical schema hashes.
- Grounding/repair can preserve an invalid extra key or fail to invent a
  missing required semantic argument. The new raw/normalized argument fields
  are required to decide this per candidate.

## Independent Provider Failure

A fresh replay with the new diagnostics reached the Manager successfully, then
both Policy requests for `sg_scene_verify` timed out at the 60-second deadline.
Its artifact contains two `LLMTransportError` proposal failures and no graph.
This run never reached typed skill validation, so it is an endpoint latency
failure, not evidence against the graph or robot runtime.

A second replay with a 120-second deadline was stopped after more than four
minutes without a Policy completion. Its log is retained, but it is not a
successful or parameter-level experiment.

## Current Root-Cause Classification

1. Historical P0 failure: invalid Policy candidate arguments at compile time;
   exact origin (raw model output versus postprocessing) was previously
   unobservable.
2. Fresh diagnostic replay: upstream LLM transport timeout.
3. Robot execution: never entered for either class of run; the executor is
   correctly fail-closed.

The next parameter-level replay should use a responsive endpoint and preserve
the new diagnostic artifact. Do not loosen `SkillRegistry.validate_contract()`
or inject default poses to force execution; that would remove the safety
boundary rather than fix the candidate-generation problem.

## Checkpoint-only timeout fix

The timeout investigation also exposed an avoidable protocol ambiguity. A pure
scene verification subgoal could still be dispatched to Policy Agents even
though the registered Policy skills were physical operations. The local graph
schema requires the `skill_calls` field, but it permits an empty array and a
`checkpoint` node; the schema does not require a non-empty action. The ambiguity
was in the topology and prompt semantics, where action-oriented instructions
could make a Policy Agent synthesize an unnecessary physical action.

Topology schema version 2 now requires every subgoal to declare
`execution_kind`:

- `physical_action`: generate, validate, arbitrate, and execute Policy
  candidates;
- `checkpoint_only`: compile a deterministic checkpoint graph and skip Policy
  inference entirely.

The deterministic graph still carries the declared success predicates, a
failure checkpoint edge, and an auditable arbitration record. In the LIBERO
runner, `FixedGraphInterpreter` evaluates checkpoint predicates against the
current `SceneSnapshot` through `LiberoObservableVerifier`; it does not mark a
checkpoint successful merely because its node was visited.

This fixes the architectural latency amplifier, but it does not explain the
historical invalid `goto_pose`/`sample_grasp_pose` failures in physical-action
subgoals. Those remain subject to raw-versus-normalized diagnostics and typed
skill validation. The upstream HTTP read timeout remains an independent
provider-latency failure and must still be measured with a responsive endpoint.
