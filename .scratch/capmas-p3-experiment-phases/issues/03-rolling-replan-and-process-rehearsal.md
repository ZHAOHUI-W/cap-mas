Status: ready-for-agent
Type: task
Labels: ready-for-agent

# Implement rolling replanning and isolated rehearsal boundaries

Implement P3.2/P3.3 from `spec.md`: execute one verified subgraph at a time,
recompile from the latest committed scene, reject suffix drift, and provide a
spawned-process rehearsal interface for offline candidate evaluation.

## Acceptance Criteria

- Rolling execution observes scene versions 0, 1, ... at separate planner
  calls.
- A subgraph is not executed twice merely because the graph was recompiled.
- A missing next-subgraph ID fails closed.
- Candidate grounding can use the current scene rather than an initial-scene
  closure.
- Process rehearsal accepts serializable jobs and returns deterministic result
  ordering with bounded worker and timeout settings.
- Rehearsal cannot access the live executor or ActionLease boundary.
