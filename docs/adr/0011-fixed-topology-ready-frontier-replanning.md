# ADR 0011: Fixed-Topology Ready-Frontier Replanning

## Status

Accepted

## Context

Full staged compilation asks the Manager and local Policies to produce an
entire mission graph before any action executes. That makes long-horizon
suffixes vulnerable to stale scene state and spends LLM latency on actions
that may no longer be relevant after an early failure. Recompiling the whole
graph after every action avoids stale suffix execution but repeatedly pays the
global planning cost and can change topology unexpectedly.

## Decision

P3.3 keeps the Manager-produced `MissionTopology` fixed for one episode and
compiles only one dependency-ready frontier subgraph per rolling cycle. The
Policy stage remains scene-local and may produce multiple candidates; the
Arbiter selects one. The selected graph is executed only through the existing
interpreter and physical scheduler.

Rolling control flow resolves the next subgraph from the fixed topology's
explicit `success` or `failure` edge. It never infers a transition from list
order or from a newly generated suffix. Missing transitions are terminal;
ambiguous/dangling transitions fail closed.

When a pre-dispatch observation advances the scene version, the selected graph
is rebased before execution. This refresh does not authorize topology edits.
Adaptive topology changes are deferred to Phase 8.

## Consequences

Positive:

- Long-horizon actions are replanned from committed state without repeatedly
  asking the Manager to redesign the mission.
- Policy generation and Arbiter cost are paid only for the current frontier.
- Recovery follows typed, auditable topology edges.
- One physical executor and one lease owner remain unchanged.

Negative:

- A topology error persists for the episode and cannot be repaired online in
  P3.3.
- The first implementation serializes frontier execution; independent ready
  branches are not physically parallel.
- Candidate generation can still use a scene that is older than the dispatch
  refresh, so rebase must remain mandatory whenever freshness advances.

## Rejected alternatives

- Re-run the Manager after every action: higher latency and no fixed-topology
  audit boundary.
- Execute a complete LLM-generated suffix: unsafe under scene drift.
- Execute multiple robot branches concurrently: deferred until disjoint
  resources and explicit joins are implemented.

