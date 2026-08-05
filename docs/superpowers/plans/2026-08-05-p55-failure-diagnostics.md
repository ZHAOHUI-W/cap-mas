# P5.5 Failure Diagnostics and Stable Replay Implementation Plan

> **For agentic workers:** Execute this plan task-by-task with tests after each task.

**Goal:** Prevent same-GPU CAP-X rehearsal contention and preserve infrastructure, reset, execution, and task failures through the P5.5 evaluation artifacts.

**Architecture:** P5.5 will serialize CAP-X rehearsal workers on a single configured GPU. The rehearsal boundary will classify depth/reset initialization failures explicitly and retain structured diagnostics. The physical executor will serialize `GraphExecutionResult.failure` into its result payload, and the OOD adapter will only assign `task_failure` when the physical run is a valid task evaluation.

**Tech Stack:** Python 3.12, dataclasses, pytest, CAP-X LIBERO, MuJoCo EGL.

## Global Constraints

- Preserve all existing uncommitted P5.5 files and artifacts.
- Keep CAP-X as the environment and API source of truth.
- Do not count renderer/reset infrastructure failures as task failures.
- Keep one physical execution per P5.5 case.
- Retain one log directory per experiment and case.

### Task 1: Serialize Same-GPU P5.5 Rehearsal

**Files:**
- Modify: `scripts/run_libero_p55_ood.py`
- Test: `tests/test_libero_p55_ood.py`

- [x] Add a regression test showing P5.5 rejects multi-worker execution when only one CUDA device is configured.
- [x] Run the focused test and verify it fails before the guard exists.
- [x] Add an explicit `max_workers == 1` validation with a diagnostic error.
- [x] Run the focused test and the existing OOD contract tests.

### Task 2: Preserve Rehearsal Reset/Render Failures

**Files:**
- Modify: `capmas/evaluation/libero_rehearsal.py`
- Test: `tests/test_libero_rehearsal.py`

- [x] Add a fake runtime test where reset raises `LIBERO depth initialization failed` and assert the result uses `reset_failure` with the original reason and step zero.
- [x] Run the focused test to establish the current `worker_crash` failure.
- [x] Classify depth/reset initialization exceptions as `RehearsalFailureClass.RESET_FAILURE` while preserving other worker exceptions as `WORKER_CRASH`.
- [x] Run the focused rehearsal tests.

### Task 3: Propagate Physical Graph Failures

**Files:**
- Modify: `scripts/run_libero_p53_online.py`
- Test: `tests/test_libero_p53_online.py`

- [x] Add a fake interpreter result test asserting physical execution output includes failure class, message, node/subgraph, and trace count.
- [x] Run the focused test to establish the missing failure metadata.
- [x] Serialize `GraphExecutionResult.failure` into the physical result and distinguish valid task failures from runtime/infrastructure failures.
- [x] Run the focused online-driver tests.

### Task 4: Correct P5.5 Aggregation and Verify a Matched Smoke

**Files:**
- Modify: `scripts/run_libero_p55_ood.py`
- Modify: `docs/experiments.md`
- Modify: `docs/phase5-evidence-evolution.md`
- Test: `tests/test_libero_p55_ood.py`

- [x] Add an adapter test asserting a physical `reset_failure` produces `evaluator_success=None` and `infrastructure_unknown`, not `task_failure`.
- [x] Run the focused test to establish the current incorrect fallback.
- [x] Implement failure-aware aggregation and persist rehearsal failure summaries in each case artifact.
- [x] Run the full unit suite (`416 passed`).
- [x] Run one matched ID/OOD smoke with CUDA device 5, `max_workers=1`, and a fresh output directory.
- [x] Confirm the smoke has no reset/render failures and its artifacts retain valid failure provenance; a larger seed run is now permitted but not started in this task.
