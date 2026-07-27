# CAP-MAS P2.5 Single-Agent Multi-Cycle Contract Loop

## Problem Statement

The current CAP-MAS LIBERO smoke path executes one large `ActionContract` that
contains the full seven-skill task plan. Although the runtime already validates
preconditions, executes typed skills, observes a new `SceneSnapshot`, and
checks observable postconditions, the policy does not re-plan from that new
state. A failed cycle stops the episode instead of invoking bounded recovery.

This prevents measurement of the properties needed before the fixed multi-agent
phase: subgoal-level replanning, observable state transitions, recovery from
execution or postcondition failures, and stable trace accumulation as the task
horizon grows.

## Solution

Add a reusable single-agent multi-cycle episode runner and a LIBERO task-0
multi-step policy. Each cycle proposes one bounded `ActionContract` for one
subgoal or action chunk. `RuntimeOrchestrator.run_cycle()` remains the atomic
execution seam: it owns validation, the action lease, execution, the after
observation, state commit, and contract-level postcondition verification.

The outer runner consumes the returned `CycleResult`, records the trace, checks
the task-level observable goal, and either asks the policy for the next
contract or invokes a bounded Recovery Agent. The evaluator-only CAP-X success
signal is reported at the end but is never placed in `AgentContext` or used to
choose the next action.

The first LIBERO policy uses explicit stages:

```text
OPEN_GRIPPER
  -> APPROACH_OBJECT
  -> CLOSE_AND_VERIFY_GRASP
  -> APPROACH_TARGET
  -> RELEASE_AND_VERIFY_PLACEMENT
  -> DONE
```

An action chunk may contain multiple typed skills when their intermediate
output is local to that contract, but physical history is never rewritten.

## User Stories

1. As a Policy Agent, I want to receive the latest committed `SceneSnapshot` before every cycle, so that each action is grounded in current state.
2. As a Policy Agent, I want to emit one bounded `ActionContract` per subgoal or action chunk, so that long plans do not become stale as soon as the robot moves.
3. As a runtime, I want each contract to carry the current parent scene version, so that stale plans are rejected before actuator access.
4. As an executor, I want each cycle to use the existing action lease, so that multi-cycle replanning does not weaken exclusive robot authority.
5. As a verifier, I want contract postconditions evaluated after every cycle, so that the next stage is entered only after observable evidence supports it.
6. As a mission controller, I want to distinguish subgoal completion from whole-task completion, so that an intermediate grasp does not falsely terminate a placement task.
7. As a LIBERO policy, I want `gripper_open` to complete the opening stage, so that later stages can assume an explicit gripper state.
8. As a LIBERO policy, I want `object_in_gripper(obj_id)` to complete the grasp stage, so that transport starts only after the object is visibly held.
9. As a LIBERO policy, I want `object_at_target(obj_id, target_id)` and `gripper_open` to complete the placement stage, so that task completion is based on observable placement and release.
10. As a runtime, I want a failed execution or postcondition to produce a retained `CycleResult`, so that recovery can reason from the failure rather than a hidden exception.
11. As a Recovery Agent, I want the failed trace, verification result, and latest scene, so that recovery proposes a new suffix plan without rewriting physical history.
12. As a Recovery Agent, I want bounded retry and recovery budgets, so that repeated failures terminate deterministically.
13. As a runtime, I want a missing recovery proposal to stop the episode with an explicit reason, so that failure is distinguishable from policy completion.
14. As an evaluator, I want all cycle traces aggregated into one `EpisodeTrace`, so that horizon-dependent reliability can be measured.
15. As an experimenter, I want the output to record committed cycles, recovery attempts, and stop reason, so that B0 and P2.5 can be compared fairly.
16. As an experimenter, I want the task loop to use the same CAP-X YAML and typed API registry as the smoke path, so that improvements are not caused by a changed robot backend.
17. As an experimenter, I want the single-agent multi-cycle loop to run before multi-agent scheduling, so that contract/replanning benefits are isolated from role coordination.
18. As a safety boundary, I want the agent-facing loop to use observable predicates rather than evaluator completion, so that success claims remain constrained by evidence.
19. As a maintainer, I want the existing one-cycle runner to remain available, so that the previous B0 smoke path stays a regression baseline.
20. As a maintainer, I want policy stage transitions to be deterministic in the first LIBERO implementation, so that failures can be reproduced before adding LLM variability.
21. As a test author, I want tests to exercise the public multi-cycle runner seam, so that implementation refactors do not require rewriting behavior tests.
22. As a researcher, I want induced execution and postcondition failures to trigger recovery in tests, so that the recovery path is not only documented.
23. As a researcher, I want stale contracts and wrong episode epochs to remain rejected across cycles, so that replanning does not bypass state isolation.
24. As a researcher, I want maximum-cycle and maximum-recovery termination to be tested, so that long-horizon loops cannot run indefinitely.
25. As a researcher, I want a successful multi-cycle LIBERO artifact, so that P2.5 is validated against a real simulator rather than only mocks.

## Implementation Decisions

- Add a reusable multi-cycle runner above `RuntimeOrchestrator.run_cycle()`.
  The runner owns episode lifecycle, policy invocation, cycle history, task-goal
  termination, recovery dispatch, and loop budgets. The orchestrator remains the
  atomic contract executor and is not duplicated.
- Keep the existing one-cycle runner and B0 script unchanged as the baseline.
  P2.5 receives a separate entry point and a separate deterministic LIBERO
  policy.
- Define a policy boundary that can see the current `AgentContext` and bounded
  cycle history. The history includes prior `ExecutionTrace` values, the latest
  `VerificationResult`, the current stage/subgoal, and recovery count; it does
  not include evaluator-only completion.
- Define a recovery boundary that receives the failed trace, failed
  verification, current context, and recovery budget, and returns either a new
  `ActionContract` or no proposal.
- Treat `CycleResult.after_scene` and the runtime state store's latest snapshot
  as the observation for the next cycle. Do not perform a duplicate observation
  between `run_cycle()` and the next policy call.
- Use explicit stage transitions for the first LIBERO policy. A committed
  contract advances the stage only when its declared postconditions pass. A
  failed contract keeps physical history immutable and routes to recovery.
- Use the existing deterministic observable predicates. The task-level terminal
  goal for LIBERO task 0 is `object_at_target(akita_black_bowl, plate)` together
  with `gripper_open`.
- Use a default 0.16m object-to-EE distance threshold for
  `object_in_gripper`, matching the current LIBERO TCP-to-object-center offset;
  keep the threshold configurable for other embodiments and object geometries.
- Permit an action contract to contain a small bounded skill chunk. Do not force
  one skill per cycle when a local output dependency is only meaningful inside
  that contract; do not put the entire seven-skill task in one contract.
- Add explicit loop limits: maximum cycles, maximum recovery attempts, and a
  per-stage retry limit. The loop must stop with a structured reason such as
  `task_goal_reached`, `max_cycles`, `recovery_exhausted`, `policy_finished`, or
  `cycle_failed`.
- Keep `backend.evaluator_success()` evaluator-only. It may be recorded in the
  final output and used for benchmark reporting, but it cannot be included in
  `AgentContext`, stage transitions, or recovery decisions.
- Preserve failure classification. Execution errors and failed postconditions
  remain distinct in the trace and are both recoverable when policy permits.
- Keep CAP-X resource construction unchanged: P2.5 reuses the existing YAML
  loader, low-level environment, registered API factories, typed skills,
  observable verifier, and object-pose alignment.
- The runner output extends the existing episode artifact with cycle/recovery
  accounting while retaining the existing `EpisodeTrace` and `ExecutionTrace`
  schema.

## Testing Decisions

- The primary test seam is the public multi-cycle runner. Tests supply a fake
  `RobotBackend`, typed skills, deterministic verifier, policy, and recovery
  collaborators, then assert the returned episode result and trace behavior.
- Tests verify behavior, not internal helper calls, private state, or call
  counts. A fake backend is used only at the public `RobotBackend` boundary.
- Add a successful multi-cycle test in which two or more contracts commit and
  the runner stops on the observable task goal.
- Add a postcondition-failure recovery test in which the first contract fails,
  Recovery returns a new contract, and the later contract succeeds without
  deleting the failed trace.
- Add an execution-failure recovery test with the same trace-retention
  requirement.
- Add a max-cycle and recovery-budget test that terminates with explicit stop
  reasons and no infinite loop.
- Add a policy-finished test for a policy returning no contract before the task
  goal is reached.
- Retain existing contract, verifier, state-store, lease, CAP-X adapter, and
  episode-runner tests as regression coverage.
- Add one real LIBERO integration smoke test or manual acceptance run using the
  existing CAP-X privileged configuration, the same seed, and the CAP-MAS-only
  output boundary.

## Out of Scope

- Fixed-graph multi-agent scheduling, event bus, or role-to-role communication.
- Adaptive topology or dynamic agent activation.
- Real-time incremental TSDF/world-model implementation.
- LLM-based policy or recovery generation; the first policy and recovery path
  are deterministic.
- Online RL, Memory Controller training, Memory Skill evolution, or Robot Skill
  evolution.
- CAP-X parity benchmark execution across the full task suite.
- Changes to CAP-X source code or its evaluator semantics.
- Servo-loop cancellation, distributed execution, or real-robot safety claims.

## Further Notes

This is a bridge milestone between P2 and the fixed multi-agent scheduler. It
should be reported separately from the existing B0 one-contract smoke result.
The experiment matrix should later compare B0, P2.5 single-agent multi-cycle,
and B3 fixed multi-agent under matched API, seed, skill, model, and time
budgets.
