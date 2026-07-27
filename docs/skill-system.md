# Robot Skill System and Self-Evolution

CAP-MAS distinguishes Robot Skills from Memory Skills. This document covers
Robot Skills that can cause physical state transitions. Memory Skill contracts
and their sequential evolution are specified in
[memory-skill-contracts.md](memory-skill-contracts.md) and
[memory-skill-evolution.md](memory-skill-evolution.md).

## 1. Skill abstraction

A robot primitive is a versioned state transition, not a prompt or a free-form code fragment.

~~~text
Skill = identity + typed arguments + preconditions + implementation
        + postconditions + safety invariants + failure modes + tests
~~~

Examples for the first LIBERO adapter include observation, object pose estimation, grasp-pose sampling, Cartesian motion, opening and closing the gripper, and homing. Their implementations may wrap CAP-X APIs, but their contracts must be explicit.

code_snippet_compose and error_attribution_repair are agent capabilities, not primitive robot skills.

## 2. Robot Skill registry states

~~~text
DISCOVERED -> QUARANTINED -> SHADOW_VALIDATED -> ACTIVE -> RETIRED
~~~

- DISCOVERED: extracted or synthesized candidate.
- QUARANTINED: cannot be called by active execution.
- SHADOW_VALIDATED: passed schema, sandbox, simulation, and contract tests.
- ACTIVE: approved immutable version available to Policy Agent.
- RETIRED: blocked for future use but retained for reproducibility.

## 3. Evolution loop

~~~text
successful or failed trace
  -> Critic attributes failure
  -> generate candidate composition or parameter repair
  -> static validation and sandbox test
  -> targeted boundary tests
  -> regression tests on prior tasks
  -> OOD validation
  -> activate at safe checkpoint or promote for later episodes
~~~

## 4. Activation policy

During the current episode, candidates may be used only in a shadow executor or after a subgoal checkpoint where the active registry is atomically switched. A candidate cannot redefine an already active skill during an irreversible action.

Permanent promotion requires no regression on the locked validation suite and a predeclared improvement threshold on targeted boundary cases. Thresholds are experimental parameters and must not be tuned on the final test set.

## 5. CAP-X comparison

CAP-X's evolving library extracts functions from successful trial code and promotes them based largely on occurrence. CAP-MAS retains this adapter as a baseline, then adds explicit contracts, versioning, quarantine, and regression gates. The comparison should measure not only skill count but also regression rate, transfer success, and recovery behavior. Robot Skill promotion is performed only after the Memory Skill phase has been frozen or completed for the comparison condition.
