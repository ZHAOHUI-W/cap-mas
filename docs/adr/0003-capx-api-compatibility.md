# ADR-0003: Preserve CAP-X API Compatibility

## Status

Accepted for the initial research prototype.

## Context

The project must attribute gains to coordination rather than to a different robot API or perception backend.

## Decision

Keep a CAP-X compatibility adapter and expose three modes: capx_legacy, capx_typed, and capmas_contract. All primary comparisons use the same CAP-X API backend where possible.

## Consequences

Positive: direct baselines and module-level ablations are possible.

Negative: the adapter must document semantic differences between permissive CAP-X execution and restricted typed execution.
