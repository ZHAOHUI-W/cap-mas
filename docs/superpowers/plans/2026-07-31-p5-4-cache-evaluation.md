# P5.4 Evidence Cache Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with checkpoints.

**Goal:** Add a deterministic, run-scoped P5.4 experiment that compares cache-disabled and cache-enabled evidence lookup while proving exact scene-version invalidation and provider-call reduction.

**Architecture:** Keep `VersionedEvidenceCache` unchanged and add a script-level evaluation seam. The driver builds typed CAP-MAS fixtures, runs one identical request trace through a disabled provider control and the public cache API, then writes one `Phase5RunDirectory` per mode plus paired comparison artifacts. No scheduler, Arbiter, LLM, CAP-X, or physical executor is touched.

**Tech Stack:** Python 3, dataclasses, existing CAP-MAS contracts, `VersionedEvidenceCache`, `Phase5RunDirectory`, pytest, JSON artifacts, SHA-256 manifests.

## Global Constraints

- Use the exact candidate fingerprint and source `SceneSnapshot.scene_version` as the cache identity.
- The cache-enabled lane must fail closed on stale evidence and must not call the provider for the stale probe.
- The cache-disabled lane must call the provider for every logical query, including the stale probe, and record `disabled`.
- Both lanes must execute the identical ordered request trace.
- Each lane must have a separate run-scoped artifact directory under `outputs/phase5/P5.4_cache_evaluation/`.
- Failure handling must preserve partial trace, `failure.json`, logs, and a final manifest before re-raising.
- Do not add persistent storage, cross-process state, LLM calls, CAP-X startup, LIBERO simulation, or physical execution.
- Every implementation slice follows red -> verify red -> green -> verify green.

---

### Task 1: Define the paired evaluation seam in tests

**Files:**
- Create: `tests/test_phase5_cache_evaluation.py`
- Test public seam: `scripts/run_p54_evidence_cache.py::run_cache_evaluation`

**Interfaces:**
- Consumes: `output_root: str | Path`, `seed: int`.
- Produces: `CacheEvaluationReport` with `control`, `enabled`, and paired
  comparison data; each mode exposes serializable metrics and trace entries.

- [ ] **Step 1: Write the failing test**

Add a test that imports the planned public function and asserts the required
paired behavior:

```python
def test_cache_evaluation_proves_hits_and_scene_invalidation(tmp_path):
    from scripts.run_p54_evidence_cache import run_cache_evaluation

    report = run_cache_evaluation(output_root=tmp_path, seed=1)

    assert report.control.metrics.provider_calls == 9
    assert report.enabled.metrics.provider_calls == 5
    assert report.enabled.metrics.hits == 3
    assert report.enabled.metrics.invalidations == 2
    assert report.enabled.metrics.stale_rejections == 1
    assert report.enabled.metrics.stale_attachments == 0
    assert report.provider_call_reduction == pytest.approx(4 / 9)
    assert report.same_trace is True
    assert report.enabled.metrics.current_scene_version == 2
```

The test must also assert that the two mode traces have the same ordered
`(operation, candidate_id, scene_version)` projection. Use the serialized
public result fields rather than private cache containers.

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest -q tests/test_phase5_cache_evaluation.py::test_cache_evaluation_proves_hits_and_scene_invalidation
```

Expected: FAIL because `scripts/run_p54_evidence_cache.py` and
`run_cache_evaluation` do not exist.

- [ ] **Step 3: Commit**

```bash
git add tests/test_phase5_cache_evaluation.py
git commit -m "test: define P5.4 cache evaluation seam"
```

### Task 2: Implement deterministic fixtures, provider, and trace runner

**Files:**
- Create: `scripts/run_p54_evidence_cache.py`
- Modify: none

**Interfaces:**
- `CacheMode = Literal["cache_disabled", "cache_enabled"]`
- `CacheTraceEntry`: frozen serializable record containing operation index,
  operation, candidate ID, fingerprint, requested scene version, cache result,
  provider-called flag, evidence scene version, and attached flag.
- `CacheModeMetrics`: frozen serializable record containing request count,
  provider calls, hits, misses, stale rejections, stores, invalidations,
  evictions, current scene version, size, disabled requests, and stale
  attachments.
- `CacheModeResult`: frozen record containing mode, metrics, trace, and the
  completed run directory `run_dir: Path`.
- `CacheEvaluationReport`: frozen record containing control result, enabled
  result, `provider_call_reduction: float`, and `same_trace: bool`.
- `run_cache_mode(output_root: str | Path, mode: CacheMode, seed: int = 1,
  provider: object | None = None, *, max_entries: int = 256,
  event_limit: int = 512) -> CacheModeResult`
- `run_cache_evaluation(output_root: str | Path, seed: int = 1) -> CacheEvaluationReport`
- `main(argv: Sequence[str] | None = None) -> int`

- [ ] **Step 1: Write the failing test**

Extend the test file with fixture-level behavior that cannot pass before the
driver exists:

```python
def test_cache_evaluation_emits_trace_and_manifests(tmp_path):
    from scripts.run_p54_evidence_cache import run_cache_evaluation

    report = run_cache_evaluation(output_root=tmp_path, seed=7)

    for result in (report.control, report.enabled):
        assert result.trace[0].operation == "publish"
        assert result.trace[-1].candidate_id == "candidate-c"
        assert result.run_dir.exists()
        assert (result.run_dir / "run_config.json").exists()
        assert (result.run_dir / "logs" / "runner.log").exists()
        assert (result.run_dir / "results" / "cache_trace.json").exists()
        assert (result.run_dir / "results" / "summary.json").exists()
        assert (result.run_dir / "summary.md").exists()
        assert (result.run_dir / "manifest.json").exists()

    assert report.control.run_dir != report.enabled.run_dir
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest -q tests/test_phase5_cache_evaluation.py
```

Expected: FAIL at import or missing result fields.

- [ ] **Step 3: Write the minimal implementation**

Implement the driver in these small units:

1. `_build_fixture(seed)` creates candidates A/B/C and scene snapshots v1/v2.
   A-v1/A-v2 and B-v1/B-v2 reuse their subgraph fingerprints while carrying
   the corresponding parent scene version. C-v2 has a distinct subgraph.
   Reuse the existing typed contract constructors from
   `capmas.contracts.action`, `capmas.contracts.core`,
   `capmas.contracts.graph`, `capmas.contracts.candidates`, and
   `capmas.contracts.scene`.
2. `_DeterministicProvider.call(candidate, scene)` increments `calls` and
   returns `CandidateEvidence` with `scene_version=scene.scene_version`,
   `provider="p5.4-deterministic"`, `captured_at_ns=scene.publish_timestamp_ns`,
   and a stable candidate-specific rehearsal score.
3. `_trace_spec()` returns the exact eleven-operation sequence from the P5.4
   spec: two publish operations, four v1 queries, one stale probe, and four v2
   queries.
4. `_run_mode(mode, ...)` executes that sequence. In enabled mode it calls
   `cache.advance_scene()` on publish, performs exact-key `get`, calls the
   provider only on a miss, and stores with `put`. The stale probe uses the
   old v1 key after publishing v2. In disabled mode it never constructs a
   cache and calls the provider for all nine query operations, recording
   `disabled`.
5. `_write_mode_artifacts(...)` creates a `Phase5RunDirectory`, writes
   `run_config.json`, `results/cache_trace.json`, `results/summary.json`, and
   `summary.md`, then finalizes the manifest. The summary contains explicit
   acceptance assertions and the paired control/enabled comparison.
6. `run_cache_evaluation()` runs both lanes with the same fixture and trace,
   computes `same_trace` and `(control_calls - enabled_calls) /
   control_calls`, writes paired comparison data into both mode directories,
   and re-finalizes both manifests.
7. On an exception, preserve the current run directory with
   `failure.json`, partial trace, `logs/runner.log`, and `manifest.json`, then
   re-raise. Do not catch and convert a failed acceptance assertion into a
   successful report.
8. `main()` accepts `--output-root`, `--seed`, and optional `--max-entries` /
   `--event-limit` bounds, prints both run paths and key metrics, and returns
   zero only after all acceptance assertions pass.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
pytest -q tests/test_phase5_cache_evaluation.py
```

Expected: all driver tests PASS. Also run the existing cache contract tests:

```bash
pytest -q tests/test_phase5_evidence_cache.py
```

- [ ] **Step 5: Commit**

```bash
git add scripts/run_p54_evidence_cache.py tests/test_phase5_cache_evaluation.py
git commit -m "feat: add P5.4 evidence cache evaluation driver"
```

### Task 3: Add failure-artifact and secret-redaction coverage

**Files:**
- Modify: `tests/test_phase5_cache_evaluation.py`
- Modify: `scripts/run_p54_evidence_cache.py`

**Interfaces:**
- The driver accepts an internal test seam `provider_factory` only through
  `run_cache_mode(...)`; the CLI and normal `run_cache_evaluation(...)` use
  `_DeterministicProvider`.
- Failure artifacts must be written through `Phase5RunDirectory` so the
  existing redaction and atomic-write behavior is preserved.

- [ ] **Step 1: Write the failing test**

Add a test with a provider that raises on its second call and assert that the
mode directory still contains partial evidence and failure metadata:

```python
def test_cache_mode_retains_partial_failure_artifacts(tmp_path):
    from scripts.run_p54_evidence_cache import run_cache_mode

    def _test_evidence(scene_version):
        from capmas.contracts.candidates import CandidateEvidence

        return CandidateEvidence(
            rehearsal_success_rate=0.5,
            available_metrics=("rehearsal",),
            scene_version=scene_version,
            provider="p5.4-test",
        )

    class FailingProvider:
        calls = 0

        def call(self, candidate, scene):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("provider failure")
            return _test_evidence(scene.scene_version)

    provider = FailingProvider()
    with pytest.raises(RuntimeError, match="provider failure"):
        run_cache_mode(
            output_root=tmp_path,
            mode="cache_enabled",
            seed=1,
            provider=provider,
        )

    run_dirs = list((tmp_path / "P5.4_cache_evaluation").iterdir())
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "failure.json").exists()
    assert (run_dirs[0] / "results" / "cache_trace.json").exists()
    assert (run_dirs[0] / "logs" / "runner.log").exists()
    assert (run_dirs[0] / "manifest.json").exists()
```

Add a second assertion that serialized run configuration and logs contain no
literal provider secret if a test provider exposes one in its diagnostic
metadata.

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest -q tests/test_phase5_cache_evaluation.py::test_cache_mode_retains_partial_failure_artifacts
```

Expected: FAIL because `run_cache_mode` does not yet expose the injected
provider seam or retain partial artifacts on provider failure.

- [ ] **Step 3: Write the minimal implementation**

Refactor the driver so `run_cache_mode(...)` owns one run directory and wraps
trace execution in `try/except BaseException`. On failure it writes:

```python
run_dir.write_json("failure.json", {
    "status": "failed",
    "stage": "cache_trace",
    "error_type": type(exc).__name__,
    "error": str(exc),
})
run_dir.write_json("results/cache_trace.json", [asdict(item) for item in trace])
run_dir.write_text("logs/runner.log", failure_log)
run_dir.finalize_manifest()
raise
```

Use `Phase5RunDirectory.write_json()` for all structured payloads; do not
write secrets or raw provider objects to the artifacts.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
pytest -q tests/test_phase5_cache_evaluation.py
```

Expected: all driver and failure-retention tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_p54_evidence_cache.py tests/test_phase5_cache_evaluation.py
git commit -m "test: retain P5.4 cache failure artifacts"
```

### Task 4: Document the code and experiment status

**Files:**
- Modify: `docs/experiments.md`
- Modify: `docs/phase5-evidence-evolution.md`
- Modify: `docs/implementation-roadmap.md`
- Modify: `tests/test_phase5_cache_evaluation.py`

**Interfaces:**
- Documentation consumes the generated `CacheEvaluationReport` artifact
  schema and must distinguish isolated cache evaluation from online physical
  selection.

- [ ] **Step 1: Write the failing documentation regression test**

Add a text test that requires the status documents to mention the driver,
both lanes, and the fact that no downstream claim is made:

```python
def test_phase5_cache_docs_name_the_isolated_evaluation():
    from pathlib import Path

    root = Path(__file__).parents[1]
    text = "\n".join(
        (root / name).read_text(encoding="utf-8")
        for name in (
            "docs/experiments.md",
            "docs/phase5-evidence-evolution.md",
            "docs/implementation-roadmap.md",
        )
    )
    assert "run_p54_evidence_cache.py" in text
    assert "cache_disabled" in text
    assert "cache_enabled" in text
    assert "does not establish" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest -q tests/test_phase5_cache_evaluation.py::test_phase5_cache_docs_name_the_isolated_evaluation
```

Expected: FAIL because the status documents still describe only the P5.4
code gate and not the new evaluation driver.

- [ ] **Step 3: Update the status documents**

Add a dated status entry that records the driver path, artifact layout,
observed control/enabled metrics after the real run, and the remaining
non-goals. Keep historical P5.3.1 results unchanged. Do not put a secret,
endpoint, or API key into any document or artifact.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
pytest -q tests/test_phase5_cache_evaluation.py
```

Expected: all focused tests PASS.

- [ ] **Step 5: Commit**

```bash
git add docs/experiments.md docs/phase5-evidence-evolution.md docs/implementation-roadmap.md tests/test_phase5_cache_evaluation.py
git commit -m "docs: record P5.4 cache evaluation status"
```

### Task 5: Run the isolated experiment and final verification

**Files:**
- Create by command: `outputs/phase5/P5.4_cache_evaluation/<run directories>/`
- No source modifications expected unless verification exposes a defect.

- [ ] **Step 1: Run the real local evaluation**

Run:

```bash
python scripts/run_p54_evidence_cache.py --output-root outputs/phase5 --seed 1
```

Expected: exit code 0, two distinct run directories, enabled provider calls
lower than disabled, and enabled hits/invalidation/stale rejection all
positive.

- [ ] **Step 2: Verify artifact integrity and secret absence**

Run:

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path("outputs/phase5/P5.4_cache_evaluation")
runs = sorted(path for path in root.iterdir() if path.is_dir())
assert len(runs) >= 2
for run in runs[-2:]:
    manifest = json.loads((run / "manifest.json").read_text())
    for entry in manifest["files"]:
        path = run / entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        assert "sk-" not in path.read_text(errors="ignore")
print("P5.4 artifacts verified")
PY
```

- [ ] **Step 3: Run the complete verification suite**

Run:

```bash
pytest -q
python -m compileall -q capmas scripts
git diff --check
```

Expected: all tests pass, compileall is silent, and diff check is clean.

- [ ] **Step 4: Update the plan and commit retained source changes**

Record the exact output paths and observed metrics in the status documents.
Commit only source, tests, and docs; retain generated output directories as
experiment artifacts according to repository convention.

```bash
git status --short
git add scripts/run_p54_evidence_cache.py tests/test_phase5_cache_evaluation.py docs/experiments.md docs/phase5-evidence-evolution.md docs/implementation-roadmap.md
git commit -m "feat: close P5.4 evidence cache evaluation"
```
