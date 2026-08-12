# CAP-MAS P5.6A Task 2 Report

## Scope

Recorded append-only graph execution telemetry and extracted verified planned
and realized horizon labels. Physical result artifacts now include serialized
`graph_events` and `horizon` when the grounded mission graph is available.
Success semantics and the existing single physical executor path are unchanged.

## RED Evidence

1. `pytest -q tests/test_p56_horizon.py tests/test_graph_runtime.py`
   initially failed during collection with:
   `ModuleNotFoundError: No module named 'capmas.evaluation.labels'`.

2. After adding the labels module, the focused re-entry telemetry regression
   failed as expected:
   `tests/test_p56_horizon.py::test_interpreter_records_terminal_subgraph_attempts_after_recovery_reentry`
   reported the second `node_started/action` event with `attempt=1` rather
   than `attempt=2`. The cause was a node visit counter scoped to one
   `_run_subgraph` call.

## GREEN Evidence

- `pytest -q tests/test_p56_horizon.py tests/test_graph_runtime.py`
  - `22 passed in 0.36s` after the initial labels implementation.
- `pytest -q tests/test_p56_horizon.py::test_interpreter_records_terminal_subgraph_attempts_after_recovery_reentry`
  - `1 passed in 0.24s` after making telemetry node attempts mission-scoped.
- Brief Step 6:
  `pytest -q tests/test_p56_horizon.py tests/test_graph_runtime.py tests/test_libero_p53_online.py tests/test_libero_p55_ood.py`
  - `41 passed in 0.75s`.
- `ruff check capmas/contracts/trace.py capmas/contracts/__init__.py capmas/runtime/graph_interpreter.py capmas/evaluation/labels.py capmas/evaluation/__init__.py scripts/run_libero_p53_online.py tests/test_p56_horizon.py tests/test_graph_runtime.py tests/test_libero_p53_online.py`
  - Passed.
- `python -m compileall -q capmas scripts/run_libero_p53_online.py`
  - Passed.
- `git diff --check`
  - Passed.

## Notes

- Planned horizon follows only the brief's allowed successful edge conditions,
  ignores path-local back-edges, and never reads `max_steps`, elapsed time,
  skill traces, or LLM calls.
- Realized horizon counts event attempts, including retry and recovery re-entry.
