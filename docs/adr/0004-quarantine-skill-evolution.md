# ADR-0004: Quarantine and Safe-Boundary Skill Evolution

## Status

Accepted for the initial research prototype.

## Context

Online skill generation can reduce human intervention, but changing active skill semantics during irreversible motion can destabilize an episode and invalidate comparisons.

## Decision

Generated skills enter a quarantine registry. They may be shadow-tested asynchronously and activated only at a subgoal checkpoint. Permanent promotion requires regression and OOD validation; later-episode activation is the safest default.

## Consequences

Positive: self-evolution remains auditable and does not alter active behavior mid-action.

Negative: the system may not repair a failure immediately, and candidate validation adds compute cost.
