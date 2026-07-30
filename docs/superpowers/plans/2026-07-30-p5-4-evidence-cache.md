# P5.4 Versioned Evidence Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bounded, thread-safe, scene-versioned evidence cache that only
returns exact candidate evidence and fails closed on stale or mismatched data.

**Architecture:** `EvidenceCacheKey` identifies the effective local
subgraph fingerprint and source scene version. `VersionedEvidenceCache` owns a
process-local LRU protected by one lock, invalidates older entries when the
scene advances, and exposes immutable statistics/events. Candidate helpers
derive the canonical fingerprint and attach only an exact cache hit through a
copied `GraphCandidate`; the existing Arbiter remains the execution boundary.

**Tech Stack:** Python dataclasses, `OrderedDict`, `threading.RLock`, existing
`CandidateEvidence`, `GraphCandidate`, `SceneSnapshot`, and pytest.

## Global Constraints

- Cache keys use canonical local `subgraph_fingerprint`, never candidate IDs.
- Evidence must declare the exact non-negative `SceneSnapshot.scene_version`.
- Stale or mismatched evidence returns unavailable or raises a typed
  compatibility error; it is never converted to zero quality.
- The cache is in-process only in P5.4; no process-shared Python state or
  persistent cache is introduced.
- Cache operations do not call an LLM, semantic model, geometry backend,
  executor, lease manager, or robot API.
- Existing CAP-X and Arbiter behavior remains unchanged unless a caller
  explicitly invokes the new cache helpers.
- Do not modify or remove existing experiment artifacts.

---

### Task 1: Define cache contracts and public identity helpers

**Files:**
- Create: `capmas/evaluation/evidence_cache.py`
- Test: `tests/test_phase5_evidence_cache.py`

**Interfaces:**
- `EvidenceCacheKey(candidate_fingerprint: str, scene_version: int)`
- `EvidenceCacheStats(hits, misses, stale_rejections, stores, invalidations, evictions, current_scene_version, size)`
- `EvidenceCacheEvent(kind, candidate_fingerprint, scene_version)`
- `VersionedEvidenceCache(max_entries=256, event_limit=512)`

- [x] **Step 1: Write failing contract tests** for empty fingerprints, negative versions, invalid capacities, and immutable key/stats/event values.
- [x] **Step 2: Run** `pytest -q tests/test_phase5_evidence_cache.py`; expect collection or import failure because `evidence_cache.py` does not exist.
- [x] **Step 3: Implement** the frozen dataclasses and constructor validation without adding cache behavior beyond empty stats/events.
- [x] **Step 4: Run** the contract tests and verify they pass.

### Task 2: Implement exact lookup, write, and scene invalidation

**Files:**
- Modify: `capmas/evaluation/evidence_cache.py`
- Test: `tests/test_phase5_evidence_cache.py`

**Interfaces:**
- `VersionedEvidenceCache.advance_scene(scene_version: int) -> None`
- `VersionedEvidenceCache.get(key: EvidenceCacheKey) -> CandidateEvidence | None`
- `VersionedEvidenceCache.put(key: EvidenceCacheKey, evidence: CandidateEvidence) -> None`
- `VersionedEvidenceCache.stats() -> EvidenceCacheStats`
- `VersionedEvidenceCache.events() -> tuple[EvidenceCacheEvent, ...]`

- [x] **Step 1: Add failing tests** for exact hit, wrong-version miss, future-version miss, stale rejection after `advance_scene`, automatic invalidation on a newer write, and non-monotonic scene rejection.
- [x] **Step 2: Run** the focused tests and verify the new behavior fails.
- [x] **Step 3: Implement** lock-protected exact lookup/write semantics. Require `evidence.scene_version == key.scene_version`, advance on a newer write, remove older entries, and record hit/miss/stale/store/invalidation events.
- [x] **Step 4: Run** `pytest -q tests/test_phase5_evidence_cache.py -k 'hit or version or stale or scene'` and verify all focused tests pass.

### Task 3: Add bounded LRU behavior and candidate attachment

**Files:**
- Modify: `capmas/evaluation/evidence_cache.py`
- Test: `tests/test_phase5_evidence_cache.py`

**Interfaces:**
- `VersionedEvidenceCache.get_for_candidate(candidate: GraphCandidate, scene: SceneSnapshot) -> CandidateEvidence | None`
- `VersionedEvidenceCache.put_for_candidate(candidate: GraphCandidate, scene: SceneSnapshot, evidence: CandidateEvidence) -> None`
- `VersionedEvidenceCache.attach_for_candidate(candidate: GraphCandidate, scene: SceneSnapshot) -> GraphCandidate`
- `VersionedEvidenceCache.invalidate_candidate(candidate_fingerprint: str) -> None`

- [x] **Step 1: Add failing tests** for LRU eviction, candidate fingerprint derivation, stale candidate rejection, copied candidate attachment, input immutability, and explicit candidate invalidation.
- [x] **Step 2: Run** the focused candidate/cache tests and verify failure.
- [x] **Step 3: Implement** `OrderedDict` LRU updates, `subgraph_fingerprint` candidate helpers, `dataclasses.replace` attachment, and typed compatibility checks for stale parent scene or mismatched evidence scene.
- [x] **Step 4: Run** the complete cache test file and verify all tests pass.

### Task 4: Verify concurrent access and package exports

**Files:**
- Modify: `capmas/evaluation/evidence_cache.py`
- Modify: `capmas/evaluation/__init__.py`
- Test: `tests/test_phase5_evidence_cache.py`

**Interfaces:**
- `from capmas.evaluation import EvidenceCacheKey, EvidenceCacheStats, EvidenceCacheEvent, VersionedEvidenceCache`

- [x] **Step 1: Add a failing concurrent test** using `ThreadPoolExecutor` with simultaneous puts and gets on the same current scene version, asserting no exceptions and consistent stats.
- [x] **Step 2: Implement** the `RLock` critical sections and export all four cache interfaces from `capmas.evaluation`.
- [x] **Step 3: Run** `pytest -q tests/test_phase5_evidence_cache.py` and verify the concurrency test passes repeatedly.
- [x] **Step 4: Run** a star-import smoke test so the package `__all__` includes the new cache interfaces.

### Task 5: Synchronize Phase 5 documentation and perform verification

**Files:**
- Modify: `docs/phase5-evidence-evolution.md`
- Modify: `docs/implementation-roadmap.md`
- Modify: `docs/experiments.md`
- Modify: `docs/superpowers/plans/2026-07-30-p5-4-evidence-cache.md`
- Test: `tests/test_phase5_docs.py`

- [x] **Step 1: Document** P5.4 implementation status, exact key/version semantics, fail-closed stale behavior, and process-local scope.
- [x] **Step 2: Document** that no real cache hit-rate experiment or cross-process persistence gate is closed yet.
- [x] **Step 3: Run** `pytest -q`, `python -m compileall -q capmas scripts`, and `git diff --check`.
- [x] **Step 4: Verify** all existing P5.2/P5.3 manifests remain readable and SHA-256 valid.
