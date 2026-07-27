# P3.3 Fixed-Topology Ready-Frontier Replanning

## Objective

Execute staged CAP-MAS missions as a verified closed loop: call the Manager
once, compile one Policy frontier at a time, refresh/rebase stale scene-bound
graphs before dispatch, and follow explicit topology success/failure edges.

## In scope

- `LLMGraphScheduler.compile_ready_frontier()` with topology reuse;
- `RollingGraphRunner` success, recovery, terminal, and scene-refresh paths;
- Manager call count and per-cycle planning-scope artifact metrics;
- one physical `FixedGraphInterpreter` and one ActionLease boundary;
- local tests and documentation for the fixed-topology protocol.

## Out of scope

- adaptive topology edits;
- parallel robot actions or distributed actuator ownership;
- rehearsal/OOD evidence and learned Recovery/Monitor roles.

## Acceptance

- Manager call count is one for a complete ready-frontier episode;
- each Policy proposal is scoped to the current frontier;
- success and failure edges select the declared next subgraph;
- missing transitions return a typed terminal result and ambiguous/dangling
  transitions fail closed;
- scene refresh updates graph/topology parent scene versions before dispatch;
- `PYTHONPATH=. pytest -q` and `python -m compileall -q capmas scripts tests`
  pass.
