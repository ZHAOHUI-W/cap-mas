# ADR-0006: Separate Memory Skills from Robot Skills

## Status

Accepted.

## Context

Memory operations extract and revise experience, while Robot Skills cause
physical state transitions. Treating both as one evolving skill library makes
authority, safety, evaluation, and regression attribution ambiguous.

## Decision

CAP-MAS uses separate Memory Skill and Robot Skill namespaces, registries,
contracts, versions, and validation suites. Memory Skills may emit structured
facts or references to recovery/planning rules, but cannot execute actuators or
access the environment handle.

## Consequences

Positive: clearer safety boundaries, independent ablations, auditable memory
updates, and causal attribution of self-evolution gains.

Negative: duplicated registry and evaluation machinery, plus a delay between a
new memory rule and a physical-skill repair.
