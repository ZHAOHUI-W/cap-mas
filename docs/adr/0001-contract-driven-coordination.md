# ADR-0001: Contract-Driven Multi-Agent Coordination

## Status

Accepted for the initial research prototype.

## Context

CAP-X uses a single-agent code-generation loop. Regenerated code can be issued after physical actions have already changed the environment, while the agent lacks a typed distinction between observed, predicted, and committed state.

## Decision

CAP-MAS agents communicate through versioned typed artifacts. A robot action requires a valid parent scene version, checked preconditions, a verifier approval, and an exclusive action lease. Completion requires observable postcondition verification.

## Consequences

Positive: stale actions, false completion, and uncontrolled role interference become measurable and rejectable. Recovery can replan from actual observations.

Negative: the runtime and schemas are more complex than CAP-X, and verification may reject valid but uncertain actions. The implementation must report rejection overhead and false negatives.

## Alternatives rejected

- Free-form agent chat with implicit state.
- Manager-only arbitration without independent verification.
- Treating LLM FINISH as task completion.
