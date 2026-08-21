# P5.3.2.1 Diagnostic Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make P5.3.2 failures inspectable without changing policy, safety thresholds, formal manifests, or calibration state.

**Architecture:** A shared physical-result serializer becomes the single projection from `GraphExecutionResult` to persisted execution telemetry. The same-runtime LIBERO session and the legacy online runner both use it. A CAP-X monkey-patch probe, scoped to a diagnostic process, records raw and converted depth plus camera render parameters; a new diagnostic-only runner invokes one of three bounded modes: execute, depth, or preview.

**Tech Stack:** Python 3, pytest, CAP-MAS contracts, CAP-X/LIBERO, JSON artifact directories.

## Global Constraints

- P5.3.2 seeds 52--61 and `outputs/phase5/P5.3.2_object6_capability/20260820_081203_3c028a8f/` are immutable.
- Every diagnostic run creates a new `outputs/phase5/P5.3.2.1_diagnostics/` directory with logs and a manifest.
- The runner must mark every artifact `diagnostic_only=true` and `eligible_for_evaluation=false`.
- `depth` and `preview` modes submit zero physical actions. `execute` performs at most one submission.
- GPU 5 is used only by child processes created by the diagnostic runner. Do not terminate or restart external services.
- No change to Arbiter thresholds, candidate prompts, typed skill arguments, calibration, Shadow Arbiter, TSDF, or Memory Controller behavior is in scope.

---

### Task 1: Shared Physical Execution Payload

**Files:**
- Create: `capmas/evaluation/physical_payload.py`
- Modify: `capmas/evaluation/libero_evidence_session.py`
- Modify: `scripts/run_libero_p53_online.py`
- Test: `tests/test_physical_payload.py`
- Test: `tests/test_libero_evidence_session.py`

**Consumes:** `ExecutionTrace`, `VerificationResult`, `PredicateReport`, `FailureArtifact`, `GraphExecutionEvent`, and `SceneSnapshot`.

**Produces:** `physical_result_payload(result, evaluator_success, graph, scene_before, scene_after, ...) -> dict[str, object]` with detailed failure, traces, predicate reports, events, horizon, and scene projections.

- [x] **Step 1: Write failing payload tests**

```python
def test_physical_payload_preserves_predicate_reports_and_scene_transitions():
    payload = physical_result_payload(
        result,
        evaluator_success=False,
        graph=graph,
        scene_before=before,
        scene_after=after,
    )
    assert payload["traces"][0]["postcondition_result"]["predicate_results"][0]["reason"] == "gripper remained open"
    assert payload["scene_diagnostics"]["before"]["scene_version"] == 1
    assert payload["scene_diagnostics"]["after"]["scene_version"] == 2
```

- [x] **Step 2: Verify RED**

Run: `pytest -q tests/test_physical_payload.py`

Expected: import failure because `capmas.evaluation.physical_payload` does not exist.

- [x] **Step 3: Implement the smallest shared projection**

```python
def physical_result_payload(result, *, evaluator_success, graph=None, scene_before=None, scene_after=None, layout_report=None, scene_diagnostics=None):
    payload = {"completed": bool(getattr(result, "completed", False)), ...}
    payload["traces"] = [execution_trace_payload(trace) for trace in getattr(result, "traces", ())]
    if graph is not None:
        payload["graph_events"] = [graph_event_payload(event) for event in getattr(result, "events", ())]
    return payload
```

- [x] **Step 4: Rewire both call sites**

`LiveLiberoEvidenceSession._execute_live_graph()` supplies its retained decision scene and the latest committed scene. `run_libero_p53_online._physical_result_payload()` delegates to the shared helper while preserving its current public private-wrapper name for callers and tests.

- [x] **Step 5: Verify GREEN**

Run: `pytest -q tests/test_physical_payload.py tests/test_libero_evidence_session.py tests/test_libero_p53_online.py`

Expected: all selected tests pass.

### Task 2: Scoped CAP-X Depth Probe

**Files:**
- Create: `capmas/evaluation/libero_depth_probe.py`
- Test: `tests/test_libero_depth_probe.py`

**Consumes:** A CAP-X `FrankaLiberoEnv` instance, its current RGB-D observations, a depth converter, and MuJoCo camera metadata.

**Produces:** A context-scoped `installed_depth_probe()` that records raw/metric finite range, uniformity, conversion source, and render near/far values without modifying CAP-X source files.

- [x] **Step 1: Write failing pure-capture tests**

```python
def test_depth_probe_records_raw_metric_and_render_ranges():
    record = capture_depth_snapshot(environment, converter)
    assert record["cameras"]["agentview"]["raw"]["maximum"] == 1.0
    assert record["cameras"]["agentview"]["metric"]["maximum"] == 529.771
    assert record["render"]["znear"] == 0.01
```

- [x] **Step 2: Verify RED**

Run: `pytest -q tests/test_libero_depth_probe.py`

Expected: import failure because the probe module does not exist.

- [x] **Step 3: Implement pure capture and reversible installation**

The context manager imports CAP-X only when called, wraps `_depth_health_stats` for the current process, appends one record per check, and restores the original method in `finally`.

- [x] **Step 4: Verify GREEN**

Run: `pytest -q tests/test_libero_depth_probe.py`

Expected: all tests pass with no CAP-X/LIBERO import required by the pure capture test.

### Task 3: Diagnostic-Only Runner

**Files:**
- Create: `scripts/run_libero_p532_diagnostics.py`
- Test: `tests/test_p532_diagnostics.py`

**Consumes:** A CAP-X YAML, candidate artifact, seed, diagnostic mode, and `MOLMO_DEVICE=cpu` profile.

**Produces:** A new Phase 5 run directory containing `run_config.json`, `results/diagnostic.json`, per-mode data, logs, and a manifest.

- [x] **Step 1: Write failing runner tests**

```python
def test_preview_diagnostic_is_marked_ineligible_and_never_executes(tmp_path, monkeypatch):
    outcome = run_diagnostic(..., mode="preview", runner=fake_preview)
    config = json.loads((outcome.run_dir.path / "run_config.json").read_text())
    assert config["diagnostic_only"] is True
    assert config["physical_execution_limit"] == 0
```

- [x] **Step 2: Verify RED**

Run: `pytest -q tests/test_p532_diagnostics.py`

Expected: import failure because the diagnostic runner does not exist.

- [x] **Step 3: Implement mode boundaries**

`depth` instantiates only the low-level CAP-X environment while the scoped depth probe is installed. `preview` starts one session, writes candidate/program/segment diagnostics, and closes without execution. `execute` reuses `run_online_experiment` once and persists its richer physical result.

- [x] **Step 4: Verify GREEN**

Run: `pytest -q tests/test_p532_diagnostics.py tests/test_p532_manifest.py`

Expected: all tests pass; no test imports or starts a live simulator.

### Task 4: Documentation and Verification

**Files:**
- Modify: `docs/superpowers/specs/2026-08-20-p5-3-2-object6-effective-motion-design.md`
- Modify: `docs/implementation-roadmap.md`
- Modify: `docs/experiments.md`
- Modify: `docs/phase5-evidence-evolution.md`
- Test: `tests/test_phase5_docs.py`

**Consumes:** The P5.3.2 closed-gate result and the diagnostic-only contract.

**Produces:** Explicit P5.3.2.1 scope, artifact contract, and promotion rule: only a confirmed single root cause may enter a later repair plan.

- [x] **Step 1: Document non-evaluative lanes and promotion criteria**

State that 53 is evaluator/graph-disagreement reproduction, 54/58 are depth probes, and 57/59/60/61 are no-submit geometry probes. No historical or diagnostic artifact becomes calibration evidence.

- [x] **Step 2: Run static verification**

Run: `pytest -q tests/test_physical_payload.py tests/test_libero_depth_probe.py tests/test_p532_diagnostics.py tests/test_phase5_docs.py && ruff check capmas/evaluation/physical_payload.py capmas/evaluation/libero_depth_probe.py capmas/evaluation/libero_evidence_session.py scripts/run_libero_p53_online.py scripts/run_libero_p532_diagnostics.py && python -m compileall -q capmas scripts`

Expected: all selected tests and touched-file lint pass.

### Task 5: Isolated Live Diagnostics

**Files:**
- Create: new directories only under `outputs/phase5/P5.3.2.1_diagnostics/`

- [x] **Step 1: Run single-seed execution reproduction**

Run: `CUDA_VISIBLE_DEVICES=5 MOLMO_DEVICE=cpu python scripts/run_libero_p532_diagnostics.py --mode execute --seed 53 ...`

Expected: one physical submission maximum, rich predicate/trace/scene artifact.

- [x] **Step 2: Run reset-only depth probes**

Run once each for seeds 54 and 58 with `--mode depth`.

Expected: zero physical submissions and records of raw/metric depth and render parameters.

- [x] **Step 3: Run no-submit geometry preview**

Run: `--mode preview --seed 57`.

Expected: zero physical submissions with per-candidate, per-segment map/frame/occupancy/clearance data.

- [x] **Step 4: Analyze and stop**

Record the confirmed root cause or declare it unresolved. Do not change policy parameters or register another ten-seed block in this task.

### Diagnostic outcome

Seed 53 confirmed one verifier false negative: evaluator success was true,
but `object_at_target(butter,basket)` measured `0.0694 m` against the `0.06 m`
threshold. Seeds 54 and 58 had valid non-uniform depth after bare reset, so
the old uniform-depth fault was not reproduced. Seed 57 had distinct bound
programs but identical occupied grasp/lift/transfer geometry. The combined
evidence is mixed; no repair is promoted and the P5.3.2 gate remains closed.
