# P5.6D Same-Runtime Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` task-by-task.

**Goal:** Collect P5.6 decision-time evidence from the CAP-X/LIBERO runtime that later executes the selected candidate.

**Architecture:** A session owns reset, real-scene observation, evidence collection, and one winner execution. The generic online runner consumes the session before arbitration, while P5.6 collection records its provenance mode.

**Tech Stack:** Python 3.12, CAP-MAS, CAP-X/LIBERO, pytest, Ruff.

## Global Constraints

- Keep `p56.feature.v1`, 12/4/4 partitioning, calibration constants, and gates unchanged.
- Do not mutate or reuse historical feature-poor P5.6 rows as P5.6D data.
- Never join post-execution state into a decision-time feature snapshot.
- Use GPU 5 only for a live smoke. Never commit `outputs/`.

### Task 1: Same-Runtime Session

**Files:** create `capmas/evaluation/libero_evidence_session.py`; create `tests/test_libero_evidence_session.py`.

**Produces:** `PreExecutionEvidenceSession`, `LiveLiberoEvidenceSession`, and `LiveLiberoEvidenceSessionConfig`. The protocol methods are `start() -> SceneSnapshot`, `candidate_evidence(candidate) -> CandidateEvidence`, `execute(candidate, graph) -> object`, and `close() -> None`.

- [ ] Write failing tests:

```python
def test_session_rejects_candidate_from_another_scene() -> None:
    session = _session_with_fake_runtime(scene_version=1)
    session.start()
    with pytest.raises(ValueError, match="decision scene"):
        session.candidate_evidence(_candidate(parent_scene_version=2))

def test_session_merges_all_preexecution_evidence() -> None:
    session = _session_with_fake_runtime(scene_version=1)
    session.start()
    evidence = session.candidate_evidence(_candidate(parent_scene_version=1))
    assert {"perception", "verifier", "geometry"} <= set(evidence.available_metrics)
```

- [ ] Run `pytest tests/test_libero_evidence_session.py -q`; expect import failure.
- [ ] Implement the session. `start()` resets and commits exactly version 1. `candidate_evidence()` merges same-scene perception/static-verifier/geometry evidence. `execute()` uses the retained runtime exactly once. `close()` is idempotent.
- [ ] Run `pytest tests/test_libero_evidence_session.py -q`; expect PASS.
- [ ] Commit with `git add capmas/evaluation/libero_evidence_session.py tests/test_libero_evidence_session.py && git commit -m "feat: add same-runtime LIBERO evidence session"`.

### Task 2: Online Runner Integration

**Files:** modify `scripts/run_libero_p53_online.py`; modify `tests/test_libero_p53_online.py`.

**Consumes:** `PreExecutionEvidenceSession`.

**Produces:** `run_online_experiment(..., evidence_session=...)` with decision-time snapshots and deterministic cleanup.

- [ ] Write failing tests:

```python
def test_online_runner_collects_session_evidence_before_execution(tmp_path):
    session = FakeSession(scene=_scene(version=4))
    outcome = run_online_experiment(..., evidence_session=session)
    assert session.events == ["start", "evidence:candidate-a", "evidence:candidate-b", "execute:candidate-b", "close"]
    assert all(snapshot.features["scene_confidence"] is not None for snapshot in outcome.feature_snapshots)

def test_online_runner_closes_session_after_execution_error(tmp_path):
    session = FakeSession(execute_error=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        run_online_experiment(..., evidence_session=session)
    assert session.closed
```

- [ ] Run `pytest tests/test_libero_p53_online.py -q`; expect failure because no `evidence_session` argument exists.
- [ ] Start and validate the live scene before baseline arbitration, enrich each candidate before rehearsal, capture snapshots from merged candidates, execute the live winner through the session, and close in `finally`. Preserve the no-session path.
- [ ] Run `pytest tests/test_libero_p53_online.py -q`; expect PASS.
- [ ] Commit with `git add scripts/run_libero_p53_online.py tests/test_libero_p53_online.py && git commit -m "feat: collect online evidence from live session"`.

### Task 3: P5.6 Collection Provenance

**Files:** modify `scripts/run_libero_p56_collect.py`, `tests/test_libero_p56_collection.py`, `docs/phase5-evidence-evolution.md`, `docs/implementation-roadmap.md`, and `docs/experiments.md`.

**Produces:** `CollectionRunConfig.evidence_mode` and an injected session path for same-runtime collection.

- [ ] Write a failing test proving `same_runtime` creates a session and passes it as `evidence_session` to the online runner.
- [ ] Run `pytest tests/test_libero_p56_collection.py -q`; expect failure because collection cannot pass or persist evidence mode.
- [ ] Add `same_runtime` and legacy `rehearsal_only` modes. In same-runtime mode, do not create a second physical executor; persist mode in suite/case artifacts; default the CLI to `same_runtime`.
- [ ] Run `pytest tests/test_libero_p56_collection.py -q`; expect PASS.
- [ ] Commit with `git add scripts/run_libero_p56_collect.py tests/test_libero_p56_collection.py docs/phase5-evidence-evolution.md docs/implementation-roadmap.md docs/experiments.md && git commit -m "feat: collect P5.6 evidence in execution session"`.

### Task 4: Verification and Transport Smoke

**Files:** no source changes expected; create only untracked `outputs/phase5/P5.6D_same_runtime_collection/<timestamp>_*` artifacts.

- [ ] Run `pytest -q`, `python -m compileall -q capmas scripts`, touched-file Ruff, and `git diff --check`; all must exit zero.
- [x] Run the pre-registered seed-31 transport smoke using `CUDA_VISIBLE_DEVICES=5`, one worker, and no LLM service. The transport retry completed at `20260819_030619_suite_6d229e5c`.
- [x] Verify its directory with `python scripts/verify_phase5_manifest.py --run-dir <new-suite-dir>`; the retry manifest is verified.
- [x] Inspect decision-time coverage and provenance without claiming qualification or success improvement. The real v1 decision scene and two pre-decision feature snapshots were recorded; the selected physical graph later failed its placement freshness checkpoint.
- [ ] Commit source/tests/docs only with `git add capmas scripts tests docs configs/phase5/p56d_object6_id_seed_31.json && git commit -m "feat: record P5.6D same-runtime evidence transport gate"`.
