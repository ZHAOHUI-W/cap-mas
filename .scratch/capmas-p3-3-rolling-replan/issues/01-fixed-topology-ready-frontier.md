Status: completed
Type: task
Labels: ready-for-agent

# Implement fixed-topology ready-frontier rolling replanning

Use the latest committed scene to compile one staged subgraph per cycle while
preserving the single physical executor. Reuse the Manager topology, route
success/failure through explicit `MissionTopology.edges`, and record artifact
metrics that distinguish candidate generation from scene-refresh rebase.

## Acceptance Criteria

- Manager is called once and subsequent cycles reuse its topology.
- Policy candidates are generated only for the current frontier.
- A verified success selects the topology success edge or terminal success;
  execution failure selects the topology failure edge or returns a typed
  failure result.
- A missing success transition returns `no_next_subgraph`; ambiguous or
  dangling transitions fail closed.
- Scene refresh rebases both graph and topology scene envelopes before action
  dispatch.
- Rolling artifacts contain planning mode, planning scope, frontier ids,
  compile latency, and Manager topology-call count.
- Local tests and compile checks pass.
