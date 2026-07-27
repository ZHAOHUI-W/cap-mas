# Memory Skill Evolution and Sequential Robot Skill Evolution

## 1. Evolution invariant

Memory Skills and Robot Skills do not evolve simultaneously in the active
runtime. Evolution uses two frozen registries and an explicit outer-loop
protocol:

```text
(MemoryBank M_k, RobotRegistry R_k, Controller C_k)
        |
        | hold R_k fixed; evolve memory
        v
(M_(k+1), R_k, C_(k+1))
        |
        | hold M_(k+1) and C_(k+1) fixed; evolve robot skills
        v
(M_(k+1), R_(k+1), C_(k+1))
```

The active tuple changes atomically at a safe checkpoint or between episodes.
There is no mid-action registry update.

## 2. Phase A: evolve Memory Skills first

1. Freeze the active Robot Skill Registry and controller snapshot.
2. Capture traces, failure cases, recovery outcomes, and hard cases.
3. Retrieve relevant memories and train/evaluate the Memory Controller.
4. The Memory Designer refines an existing Memory Skill or proposes a new one.
5. Run replay, conflict, regression, OOD, and latency validation.
6. Promote the best Memory Skill Bank snapshot or roll back on regression.
7. Run additional exploration after a new Memory Skill is introduced.

The primary evidence is better diagnosis, recovery guidance, memory precision,
long-horizon success, and lower intervention under the same Robot Skill set.

## 3. Phase B: evolve Robot Skills second

1. Freeze the promoted Memory Skill Bank and Memory Controller.
2. Use memory-derived failure conditions and hard cases to generate Robot Skill
   candidates or parameter repairs.
3. Put candidates in the Robot Skill quarantine registry.
4. Run static contract checks, shadow execution, boundary tests, locked-suite
   regression, and OOD validation.
5. Promote only an immutable new Robot Skill version at a safe boundary.
6. Re-run memory and controller evaluation without changing their snapshots.

This ordering makes it possible to answer whether a robot-skill improvement was
enabled by better experience extraction rather than by an uncontrolled joint
update.

## 4. Automatic promotion and rollback

The promotion manager maintains:

```text
active_snapshot
candidate_snapshot
locked_regression_suite
hard_case_buffer
ood_suite
promotion_report
```

Promotion is allowed only when:

- no safety or contract invariant regresses;
- the locked-suite lower confidence bound does not fall below its baseline;
- the targeted hard-case improvement clears a predeclared threshold;
- OOD performance and latency remain within their declared budgets;
- all changed outputs retain provenance and version compatibility.

Otherwise the candidate is retired or kept quarantined and the last validated
snapshot remains active. The system records the failed promotion reason for
future Memory Skill selection.

## 5. Hard-case buffer

The buffer is a bounded, sliding collection of representative difficult cases.
Sampling priority combines failure severity, novelty, recurrence, uncertainty,
and under-covered task families. It must preserve rare safety and recovery
failures even when they are infrequent.

At each evolution cycle, cases are clustered by trace and failure signature;
representatives are selected for design and validation. The buffer is never
allowed to become a hidden test set: the locked test suite remains untouched
until final evaluation.

## 6. Why sequential evolution is preferred

Sequential evolution costs extra evaluation time and may delay a useful Robot
Skill repair. It provides cleaner attribution, smaller search space, easier
rollback, and lower risk of mutually compensating errors. Joint evolution is
retained only as a later ablation, with separate registries and snapshots even
there.
