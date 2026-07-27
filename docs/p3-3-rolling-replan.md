# P3.3 Fixed-Topology Rolling Replanning

## Scope

P3.3 adds closed-loop staged execution without adaptive topology edits. The
Manager creates one immutable `MissionTopology`; each cycle selects one
frontier subgraph, asks the local Policy agents for candidates, arbitrates one
candidate, executes it through the existing `FixedGraphInterpreter`, and then
uses the newly committed `SceneSnapshot` for the next cycle.

Adaptive edge edits, learned recovery, simulator rehearsal evidence, and
parallel physical execution remain later phases.

## Control flow

```text
Manager: MissionTopology (once)
        |
        v
ready frontier -> Policy candidates -> GraphValidator/SkillValidator
        -> Arbiter -> one-subgraph MissionGraph
        -> scene refresh/rebase -> FixedGraphInterpreter
        -> verified success/failure
             | success edge / failure edge
             v
        next frontier or terminal result
```

The ready-frontier compiler accepts exactly one subgraph per cycle in P3.3.
This keeps execution serial and makes the measurement boundary explicit;
parallel proposal waves remain available as a separate read-only mode.

## State and authority

- `MissionTopology.edges` is the authoritative source of the next
  `success`/`failure` transition in rolling mode. Nullable dependency edges
  and semantic failure labels use the same normalization as graph assembly.
- A missing transition is terminal for that outcome. An ambiguous or dangling
  transition fails closed with `RollingGraphError`.
- Only a successfully committed subgraph enters `completed_subgraphs` and can
  satisfy normal `depends_on` constraints. A recovery branch should therefore
  be reachable through an explicit `failure` edge and should not require the
  failed branch as a normal dependency. The topology validator enforces the
  stronger invariant that recovery dependencies must be on the failed source's
  normal success ancestry; otherwise the topology is rejected before Policy
  fan-out.
- Normal `success` edges must not point back to a subgraph on their own
  committed success ancestry. Rolling execution is commit-once for each
  subgraph, so a success back-edge would otherwise make the next frontier
  request an already completed subgraph. `TopologyValidator` rejects this as
  `SUCCESS_BACK_EDGE` before Policy fan-out or robot execution.
- The interpreter remains the sole physical executor; `ActionLease` and the
  existing CAP-X-compatible backend are unchanged.

## Scene freshness boundary

Policy generation starts from the latest committed scene passed to the
scheduler. If `scene_refresh` advances the scene before dispatch, the runner
calls `rebase_graph()` and updates the graph and topology scene envelope before
execution. Artifacts consequently distinguish:

- the scene version used to generate a candidate;
- the scene version used by the rebased graph at dispatch;
- the scene version committed after execution.

This is a safety boundary, not dynamic topology adaptation. Candidate-specific
rehearsal and action-conditioned geometry evidence remain Phase 5 work.

## Artifact metrics

Rolling LIBERO artifacts record `planning_mode`, `frontier_subgraphs`, one
`planning_scope` per compilation, total `manager_topology_calls`, compile
latencies, and proposal waves. A valid P3.3 run should show:

- `planning_mode = "ready_frontier"`;
- `planning_scope = "ready_frontier"` for every rolling compile;
- `manager_topology_calls = 1` for a complete episode;
- one Policy proposal/arbitration set per executed frontier;
- a scene version that advances monotonically through committed traces.

## Acceptance tests

- Manager is called once while subsequent cycles reuse the fixed topology.
- Policy receives only the current frontier, not the full unexecuted suffix.
- Success edges select the next frontier and success terminals stop completed.
- Failure edges route to an explicit recovery frontier.
- Missing or ambiguous transitions fail closed or return a typed terminal result.
- A failure target depending on an uncommitted subgraph is rejected as
  `RECOVERY_DEPENDS_ON_UNCOMMITTED` before physical execution.
- A normal success back-edge is rejected as `SUCCESS_BACK_EDGE` before
  physical execution.
- A pre-dispatch scene refresh rebases the graph and topology to the current
  scene version.
- Older or cross-episode refresh snapshots are rejected before dispatch.
- The physical executor is never called concurrently.

## Endpoint validation

On 2026-07-24, `retry10` completed the LIBERO spatial task
`libero_spatial_0` (`Place akita black bowl on plate`) through the real
CAP-X backend:

- `completed = true`, `evaluator_success = true`;
- four rolling frontiers: scene precheck, grasp, placement, and final verify;
- one Manager topology call and two physical ActionContract executions;
- grasp verification passed `object_in_gripper` and `gripper_closed`;
- placement verification passed `object_at_target` and `gripper_open`;
- Policy arbitration used SceneSnapshot perception evidence for both physical
  subgoals; the placement winner was the safety candidate with evidence score
  `0.55` versus `0.25` for balanced.

The run artifact and preserved log are:
`outputs/capmas_libero_b3_llm/p33_rolling_gpt55_20260724_retry10.json` and
`outputs/capmas_libero_b3_llm/p33_rolling_gpt55_20260724_retry10.log`.
This is an endpoint-backed success smoke trial, not yet a matched multi-seed
comparison against fixed-graph CAP-X.
