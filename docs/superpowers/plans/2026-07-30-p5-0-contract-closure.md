# P5.0 Contract-Level Foundation Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Close the P5.0 interfaces required by rehearsal without claiming that TSDF or learned semantic inference is implemented.

**Architecture:** Keep `SparseVoxelMap` as the active reference backend, add an explicit backend configuration/factory boundary, and make unsupported TSDF selection fail with a typed, observable error. Add version/fingerprint validation helpers for rehearsal evidence; do not change the live CAP-X execution path.

**Tech Stack:** Python 3, dataclasses, typing Protocols, pytest, existing `capmas.perception` and `capmas.evaluation` contracts.

## Global Constraints

- TSDF remains unavailable until a real TSDF integration/query implementation exists.
- Unsupported backend selection must fail closed; no silent SparseVoxel fallback.
- Rehearsal never acquires the live `ActionLease` or physical executor.
- Existing P5.2 artifacts and user changes must be preserved.
- Every behavior change starts with a failing test.

### Task 1: Add explicit map backend configuration and factory

**Files:**
- Create: `capmas/perception/map_factory.py`
- Modify: `capmas/perception/local_map.py`
- Test: `tests/test_phase5_map_factory.py`

**Interfaces:**
- `MapBackendConfig.from_mapping(config: Mapping[str, object]) -> MapBackendConfig`
- `build_local_map_backend(config: Mapping[str, object]) -> LocalMapBackend`
- `UnsupportedMapBackendError(ValueError)`
- `MapBackendConfig.backend`, `voxel_size_m`, `local_radius_m`, and `tsdf_enabled`

- [x] **Step 1: Write the failing tests** for sparse factory creation, unknown backend rejection, and TSDF rejection with an error containing the requested backend.
- [x] **Step 2: Run** `pytest tests/test_phase5_map_factory.py -q` and verify the new imports/factory fail.
- [x] **Step 3: Implement** the immutable config parser and factory using `SparseVoxelMap`; route validation through existing numeric checks and preserve the current fail-closed TSDF behavior.
- [x] **Step 4: Run** `pytest tests/test_phase5_map_factory.py -q` and verify pass.
- [x] **Step 5: Run** `pytest tests/test_phase4_local_map.py tests/test_phase5_map_factory.py -q`.

### Task 2: Add versioned candidate/evidence compatibility helpers

**Files:**
- Create: `capmas/evaluation/evidence_contracts.py`
- Test: `tests/test_phase5_evidence_contracts.py`

**Interfaces:**
- `EvidenceRequestContext(candidate_fingerprint: str, scene_version: int, map_version: int | None = None)`
- `EvidenceCompatibilityError(ValueError)`
- `assert_evidence_compatible(context, *, candidate_fingerprint, scene_version, map_version=None) -> None`

- [x] **Step 1: Write failing tests** for matching metadata, stale scene versions, mismatched candidate fingerprints, and optional map-version mismatch.
- [x] **Step 2: Run** the focused tests and confirm failure because the module does not exist.
- [x] **Step 3: Implement** the immutable context and strict equality checks. Unknown `map_version` is accepted only when the request did not require one.
- [x] **Step 4: Run** the focused tests and confirm pass.

### Task 3: Record the contract-level P5.0 boundary

**Files:**
- Modify: `docs/phase5-evidence-evolution.md`
- Modify: `docs/implementation-roadmap.md`
- Test: `tests/test_phase5_docs.py`

- [x] **Step 1: Write a failing documentation assertion** that the roadmap distinguishes `P5.0-contract` from the still-open TSDF/semantic runtime work and identifies P5.3 as the next implementation phase.
- [x] **Step 2: Run** the documentation test and verify failure.
- [x] **Step 3: Update** the two documents with the approved sequence and explicit non-claim language.
- [x] **Step 4: Run** the documentation test and `git diff --check`.

### Task 4: P5.0 contract gate

- [x] **Step 1: Run** `pytest tests/test_phase4_local_map.py tests/test_phase5_map_factory.py tests/test_phase5_evidence_contracts.py tests/test_phase5_docs.py -q`.
- [x] **Step 2: Run** `python -m compileall capmas scripts`.
- [x] **Step 3: Run** `git diff --check`.
- [x] **Step 4: Record** that P5.0 contract-level closure is complete while TSDF and real semantic adapters remain open.
