# P5.3 Fingerprint Mapping and Shadow Arbiter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Map full-MissionGraph rehearsal evidence to the exact local subgraph Arbiter candidate and provide a non-physical shadow arbitration report.

**Architecture:** A typed `CandidateIdentity` preserves source graph provenance and derives one explicit target subgraph fingerprint from the normalized graph. Rehearsal payloads carry both scopes. A pure shadow-Arbiter function compares baseline and hypothetical evidence-enriched decisions without accessing a backend, lease, or executor.

**Tech Stack:** Existing typed graph serializer, dataclasses, `CandidateArbiter`, pytest, JSON artifact driver.

## Global Constraints

- Graph-scoped source fingerprints are computed from the raw source JSON mapping so legacy `call_index` artifacts retain identity.
- Subgraph-scoped Arbiter fingerprints are computed from the canonical typed `SubgraphSpec` serializer.
- The target subgraph is named explicitly by `arbiter_subgraph_id`; candidate-id parsing is forbidden.
- Shadow arbitration never mutates live candidates and never owns or invokes a physical backend, `ActionLease`, or executor.
- Missing, stale, or mismatched evidence remains unavailable and is never converted to zero-quality evidence.

### Task 1: Add candidate identity derivation

**Files:**
- Create: `capmas/evaluation/candidate_identity.py`
- Modify: `capmas/evaluation/__init__.py`
- Test: `tests/test_candidate_identity.py`

**Interfaces:**
- `CandidateIdentity`
- `candidate_identity_from_raw_graph(raw_graph, subgraph_id, scene_version)`
- `raw_graph_fingerprint(raw_graph)`

- [x] **Step 1: Write failing tests** for raw graph fingerprint stability, explicit subgraph lookup, missing-subgraph rejection, and legacy `call_index` preservation.
- [x] **Step 2: Run** `pytest -q tests/test_candidate_identity.py`; expected failure because the module and derivation function do not exist.
- [x] **Step 3: Implement** raw JSON hashing, typed `mission_graph_from_dict`, target lookup, and existing `subgraph_fingerprint` reuse.
- [x] **Step 4: Run** `pytest -q tests/test_candidate_identity.py`; expected all tests pass.

### Task 2: Carry dual identity through rehearsal artifacts

**Files:**
- Modify: `capmas/evaluation/rehearsal.py`
- Modify: `capmas/evaluation/libero_rehearsal.py`
- Modify: `capmas/evaluation/rehearsal_evidence.py`
- Modify: `scripts/run_libero_p53_rehearsal.py`
- Tests: `tests/test_rehearsal.py`, `tests/test_rehearsal_evidence.py`, `tests/test_libero_p53_rehearsal.py`

**Interfaces:**
- Add optional `fingerprint_scope`, `arbiter_subgraph_id`, and `arbiter_fingerprint` fields to job/result/evidence payloads.
- Extend `CandidateSpec` with optional `CandidateIdentity`.

- [x] **Step 1: Write failing tests** for graph-scoped candidate parsing, job/result round-trip of both fingerprints, and rejection when a declared target subgraph is missing.
- [x] **Step 2: Run** the focused tests and verify they fail on missing fields or mapping behavior.
- [x] **Step 3: Implement** identity derivation in `parse_candidate_mapping`, propagate fields through spawned jobs/results, and copy them into `RehearsalEvidence`.
- [x] **Step 4: Update** the sample matched candidate artifact with explicit `fingerprint_scope="graph"` and `arbiter_subgraph_id` fields without changing its graph payload.
- [x] **Step 5: Run** focused rehearsal and artifact tests; expected all pass.

### Task 3: Attach mapped evidence safely to local candidates

**Files:**
- Modify: `capmas/evaluation/rehearsal_arbiter.py`
- Tests: `tests/test_phase5_rehearsal_arbiter.py`

**Interfaces:**
- `merge_rehearsal_evidence(candidate, rehearsal, include_in_arbiter)` accepts the mapped `arbiter_fingerprint` for graph-scoped evidence.

- [x] **Step 1: Write failing tests** proving graph-scoped evidence with a matching mapped subgraph attaches, wrong mapped fingerprint rejects, and unmapped graph evidence rejects.
- [x] **Step 2: Run** the focused Arbiter tests and verify failure.
- [x] **Step 3: Implement** scope-aware matching while preserving the legacy subgraph fallback.
- [x] **Step 4: Run** `pytest -q tests/test_phase5_rehearsal_arbiter.py tests/test_phase5_evidence_contracts.py`; expected all pass.

### Task 4: Add pure shadow arbitration

**Files:**
- Create: `capmas/evaluation/shadow_arbiter.py`
- Modify: `capmas/evaluation/__init__.py`
- Test: `tests/test_shadow_arbiter.py`

**Interfaces:**
- `ShadowArbitrationReport`
- `run_shadow_arbitration(candidates, rehearsals, scene, arbiter=None)`

- [x] **Step 1: Write failing tests** for baseline preservation, changed hypothetical winner, `selection_basis` reporting, missing evidence, and `physical_execution_required=False`.
- [x] **Step 2: Run** `pytest -q tests/test_shadow_arbiter.py`; expected failure because the report/function do not exist.
- [x] **Step 3: Implement** baseline selection, copied evidence-enriched candidates, shadow selection, and report construction with no backend dependency.
- [x] **Step 4: Run** the shadow test and existing Arbiter tests; expected all pass.

### Task 5: Synchronize documentation and verify

**Files:**
- Modify: `docs/phase5-evidence-evolution.md`
- Modify: `docs/implementation-roadmap.md`
- Modify: `docs/experiments.md`
- Modify: `docs/superpowers/plans/2026-07-30-p5-3-libero-rehearsal.md`

- [x] **Step 1: Document** that the graph/subgraph identity gate and shadow Arbiter gate are closed only after the focused tests pass.
- [x] **Step 2: Document** that 10+ seeds, multiple tasks, online physical selection, and P5.4 cache remain open.
- [x] **Step 3: Run** `pytest -q`, `python -m compileall -q capmas scripts`, and `git diff --check`.
- [x] **Step 4: Verify** existing P5.3 run artifacts remain unchanged and readable.
