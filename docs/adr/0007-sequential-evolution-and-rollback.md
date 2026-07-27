# ADR-0007: Sequential Evolution with Hard Cases and Rollback

## Status

Accepted.

## Context

Jointly changing the Memory Skill Bank and Robot Skill Registry can make an
improvement impossible to attribute and can allow one change to hide a
regression in the other. Long-horizon robot failures are also too sparse for
ordinary average-case replay to expose reliably.

## Decision

Evolve Memory Skills first while Robot Skills are frozen. After promotion,
freeze the Memory Skill Bank and controller while evolving Robot Skills. Use a
sliding hard-case buffer, locked regression/OOD suites, immutable snapshots,
automatic promotion gates, and rollback on regression.

## Consequences

Positive: controlled credit assignment, reproducible comparisons, reduced
human approval per episode, and safer active behavior.

Negative: slower total evolution and a possible lag before a cross-layer repair
becomes available. Joint evolution remains a later ablation, not the default.
