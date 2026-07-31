# P5.3.1 Online Rehearsal Arbiter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bounded scheduler seam that can use version-bound rehearsal evidence for live candidate selection while preserving a non-physical shadow mode.

**Architecture:** Keep `CandidateArbiter` as the only selection authority. Add a batch `RehearsalEvidenceProvider` seam and a typed report that retains baseline, evidence-aware, and live results. The scheduler attaches only evidence accepted by the existing fingerprint/scene gate; online mode selects once from the evidence-aware candidates, while shadow mode leaves the baseline result live.

**Tech Stack:** Python dataclasses, existing `CandidateArbiter`, `RehearsalEvidence`, `merge_rehearsal_evidence`, pytest, existing Phase 5 artifact serializer.

## Global Constraints

- Rehearsal providers are read-only and never receive an `ActionLease` or physical executor.
- `disabled`, `shadow`, and `online_bounded` are explicit modes; default remains `disabled`.
- Missing, stale, or mismatched evidence is unavailable and is never converted into a zero score.
- Existing CAP-X skill APIs and baseline Arbiter weights remain unchanged.
- Existing output directories and unrelated dirty worktree changes must not be reverted.
- No TSDF, semantic adapter, persistent cache, adaptive topology, or learned weight changes are included.

---

### Task 1: Add the typed online rehearsal decision seam

**Files:**
- Create: `capmas/evaluation/online_rehearsal.py`
- Test: `tests/test_online_rehearsal.py`

**Interfaces:**
- `RehearsalEvidenceProvider`
- `RehearsalArbitrationReport`
- `select_with_rehearsal(candidates, scene, arbiter, mode, provider)`

- [ ] **Step 1: Write the failing tests** for disabled mode not calling the provider, shadow mode keeping the baseline winner live, online mode using the evidence-aware winner, and the report exposing `would_change_selection`.
- [ ] **Step 2: Run** `pytest -q tests/test_online_rehearsal.py`; expect collection/import failure because the module does not exist.
- [ ] **Step 3: Implement** the report and helper. Compute baseline first, call the provider once for shadow/online, validate every result through `merge_rehearsal_evidence(..., include_in_arbiter=True)`, collect rejection strings, and use baseline fallback when online evidence cannot produce a selected candidate.
- [ ] **Step 4: Run** `pytest -q tests/test_online_rehearsal.py`; expect all focused tests to pass.

### Task 2: Integrate the seam into every LLM scheduler selection path

**Files:**
- Modify: `capmas/runtime/llm_scheduler.py`
- Modify: `tests/test_llm_scheduler.py`
- Modify: `tests/test_staged_protocol.py`
- Modify: `tests/test_rolling_scheduler.py`

**Interfaces:**
- Add constructor parameters `rehearsal_mode` and `rehearsal_evidence_provider`.
- Add `rehearsal_reports` to `LLMGraphCompileResult`.
- Route legacy, staged serial, ready-wave, and rolling-frontier arbitration through one private helper.

- [ ] **Step 1: Add failing scheduler tests** for an online provider changing the selected candidate and a shadow provider leaving the selected subgraph unchanged.
- [ ] **Step 2: Run** the focused scheduler tests and verify failure because the constructor/result do not expose the new seam.
- [ ] **Step 3: Implement** one `_select_candidates_with_rehearsal(...)` helper and replace direct Arbiter calls in all four selection paths. Store reports by subgraph id and keep `arbitrations` equal to the live result.
- [ ] **Step 4: Add a failure test** proving provider exceptions create a report fallback and do not abort baseline selection.
- [ ] **Step 5: Run** the focused scheduler, staged, and rolling tests.

### Task 3: Publish online-selection metrics in the B3-LLM artifact

**Files:**
- Modify: `scripts/run_libero_b3_llm.py`
- Modify: `capmas/contracts/experiment.py`
- Modify: `tests/test_phase5_artifacts.py`
- Modify: `tests/test_llm_scheduler.py`

**Interfaces:**
- Add a CLI `--rehearsal-mode` with choices `disabled`, `shadow`, `online_bounded`, defaulting to `disabled`.
- Serialize `rehearsal_reports` under `scheduler_metrics`.
- Record baseline winner, live winner, fallback reason, attached evidence ids, and provider latency without serializing provider secrets.

- [ ] **Step 1: Write failing serialization tests** for a compile result containing a report and a run config containing the selected mode.
- [ ] **Step 2: Implement** the config field and artifact serializer. Keep the CLI disabled unless a provider is explicitly configured; reject `online_bounded` without a provider instead of silently claiming online evidence.
- [ ] **Step 3: Run** focused artifact and scheduler tests.

### Task 4: Add a deterministic provider-backed integration smoke

**Files:**
- Create: `scripts/run_libero_p53_online.py`
- Test: `tests/test_libero_p53_online.py`
- Modify: `docs/experiments.md`
- Modify: `docs/phase5-evidence-evolution.md`
- Modify: `docs/implementation-roadmap.md`

**Interfaces:**
- The driver loads a serialized candidate artifact and uses the existing isolated CAP-X rehearsal worker to obtain evidence before one live Arbiter decision.
- The driver writes one independent Phase 5 run directory with `run_config.json`, `results/rehearsal.json`, `results/selection.json`, `summary.json`, `summary.md`, `manifest.json`, and `logs/runner.log`.

- [ ] **Step 1: Write failing driver tests** for disabled/shadow/online selection, exact candidate identity mapping, and one physical execution call.
- [ ] **Step 2: Implement** the driver by reusing `parse_candidate_mapping`, `build_rehearsal_jobs`, `run_with_respawn`, and `merge_rehearsal_evidence`; never execute both baseline and evidence-aware winners.
- [ ] **Step 3: Run** the driver tests with a fake rehearsal worker and fake physical executor.
- [ ] **Step 4: Run** one real CAP-X/LIBERO smoke from an existing real-LLM candidate artifact, using `CUDA_VISIBLE_DEVICES=5` and a new output root. Retain all logs and report whether the online winner differs from baseline.

### Task 5: Verify the increment and update the phase gate

**Files:**
- Modify: `docs/superpowers/plans/2026-07-31-p5-3-1-online-rehearsal-arbiter.md`
- Test: full repository suite

- [ ] **Step 1: Run** `pytest -q`.
- [ ] **Step 2: Run** `python -m compileall -q capmas scripts`.
- [ ] **Step 3: Run** `git diff --check` and scan the new output root for provider secrets.
- [ ] **Step 4: Record** exact smoke outcome, including evaluator success, graph completion, baseline/live winner, selection basis, rehearsal latency, and any fallback.
