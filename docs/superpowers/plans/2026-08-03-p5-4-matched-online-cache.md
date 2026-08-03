# P5.4 Matched Online Cache Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a run-scoped matched multi-seed driver that compares repeated online rehearsal with the versioned cache disabled versus enabled.

**Architecture:** Reuse `run_online_experiment` as the only execution seam. The driver creates independent control and enabled runs for every task/seed, records provider calls, cache statistics, selection latency, physical execution count, and task outcomes, then emits paired and aggregate artifacts. Cache instances are created inside each run and are never shared across seeds or lanes.

**Tech Stack:** Python dataclasses, existing `Phase5RunDirectory`, CAP-X/LIBERO runner seam, pytest.

## Global Constraints

- Use `CUDA_VISIBLE_DEVICES=5` in real experiments unless explicitly overridden.
- Preserve every run directory and log; never overwrite prior artifacts.
- Do not persist API keys or authorization headers.
- Keep the physical executor single-owner and execute at most once per lane.
- Do not claim downstream success-rate improvement from cache latency or call reduction alone.

### Task 1: Expose online-run metrics

**Files:**
- Modify: `scripts/run_libero_p53_online.py`
- Test: `tests/test_libero_p53_online.py`

- [x] Add `provider_call_count` and cumulative `selection_latency_ms` to `OnlineSelectionOutcome` and publish both in run artifacts.
- [x] Add regression assertions for the public outcome and serialized artifacts.
- [x] Run the focused online runner tests.

### Task 2: Add the matched multi-seed driver

**Files:**
- Create: `scripts/run_libero_p54_matched.py`
- Create: `tests/test_libero_p54_matched.py`

- [x] Define `MatchedCacheTaskSpec`, `MatchedCachePairResult`, and `MatchedCacheEvaluationReport`.
- [x] Run `online_bounded` with `cache_mode=disabled` and `cache_mode=enabled`, both with the same candidates, seed, scene version, and `selection_repeats`.
- [x] Write suite, pair, and lane references without mutating child artifacts.
- [x] Aggregate provider calls, hits, misses, invalidations, latency, physical executions, evaluator success, and paired deltas.
- [x] Retain failed pairs and continue unless `--fail-fast` is set.
- [x] Add CLI support for task manifest or single task, seed list, selection repeats, output root, timeout, restart, max steps, GPU, and API-server skip.

### Task 3: Document and verify the evaluation gate

**Files:**
- Modify: `docs/experiments.md`
- Modify: `docs/phase5-evidence-evolution.md`
- Modify: `docs/implementation-roadmap.md`
- Modify: `tests/test_phase5_docs.py`

- [x] Record the matched driver contract and artifact layout.
- [x] State that the multi-seed cache gate requires same-trace pairs, no cross-seed cache reuse, reduced provider calls or positive hit rate, single physical execution, and independent downstream reporting.
- [x] Run focused tests, full suite, compile check, diff check, and artifact manifest validation.
