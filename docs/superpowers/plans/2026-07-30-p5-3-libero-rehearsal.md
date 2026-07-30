# P5.3 LIBERO Process Rehearsal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Run serializable CAP-X/LIBERO candidate rehearsals in isolated spawned workers and expose auditable candidate-level evidence to the Arbiter without touching live robot execution.

**Architecture:** The parent process serializes a candidate graph, source scene metadata, and a matched seed into a `RehearsalJob`. Each spawned worker builds a fresh CAP-X LIBERO environment, resets to the declared seed/state, executes only the candidate, emits checkpoint/failure/latency data, and exits. The parent converts results into version-bound evidence and writes one immutable artifact directory per run.

**Tech Stack:** Python multiprocessing spawn, existing CAP-X factory and skill registry, LIBERO simulator environment, existing `ProcessRehearsalPool` and `Phase5RunDirectory`, pytest.

## Global Constraints

- Worker processes must not receive or create the live robot backend, `ActionLease`, or physical executor.
- Every candidate gets the same task, initial state policy, seed set, timeout, and simulator configuration.
- Worker failure, timeout, and respawn are evidence outcomes, not silent candidate success.
- Rehearsal results are advisory evidence; they cannot directly mutate robot execution state.
- Every run writes a new directory containing config, logs, results, traces, evidence, and manifest.

### Task 1: Define serializable rehearsal payloads and result schema

**Files:**
- Modify: `capmas/evaluation/rehearsal.py`
- Test: `tests/test_rehearsal.py`

**Interfaces:**
- Extend `RehearsalJob` with `task_id`, `scene_version`, `candidate_fingerprint`, and `checkpoint_budget`.
- Extend `RehearsalResult` with `checkpoint_results`, `failure_step`, `failure_reason`, `scene_version`, and `candidate_fingerprint`.
- Add `RehearsalFailureClass` values for `invalid_graph`, `reset_failure`, `skill_failure`, `postcondition_failure`, `timeout`, and `worker_crash`.

- [x] **Step 1: Write failing tests** for round-trip serializability, required scene/fingerprint metadata, timeout result semantics, and deterministic result ordering.
- [x] **Step 2: Run** `pytest tests/test_rehearsal.py -q` and verify failure.
- [x] **Step 3: Implement** the minimal dataclass fields, validation, and bounded pool result conversion without changing existing callers' defaults.
- [x] **Step 4: Run** the focused tests and verify pass.

### Task 2: Implement a CAP-X/LIBERO worker boundary

**Files:**
- Create: `capmas/evaluation/libero_rehearsal.py`
- Test: `tests/test_libero_rehearsal.py`

**Interfaces:**
- `LiberoRehearsalConfig`
- `run_libero_rehearsal_job(job: RehearsalJob, config: LiberoRehearsalConfig) -> RehearsalResult`
- `LiberoRehearsalWorker(config: LiberoRehearsalConfig)` callable wrapper

- [x] **Step 1: Write failing tests** with a lightweight fake CAP-X factory for reset, one candidate execution, checkpoint capture, postcondition result, and worker failure classification.
- [x] **Step 2: Run** the focused tests and verify failure.
- [x] **Step 3: Implement** a pickle-safe config and worker that constructs a fresh environment inside the spawned process, executes the deserialized graph through the existing skill registry, records checkpoints, and always closes the environment.
- [x] **Step 4: Run** the focused tests and verify pass.

### Task 3: Add timeout/respawn and evidence conversion

**Files:**
- Modify: `capmas/evaluation/rehearsal.py`
- Create: `capmas/evaluation/rehearsal_evidence.py`
- Test: `tests/test_rehearsal_evidence.py`

**Interfaces:**
- `RehearsalEvidence`
- `rehearsal_result_to_evidence(result, request_context) -> RehearsalEvidence`
- `run_with_respawn(jobs, worker_factory, pool_config) -> tuple[RehearsalResult, ...]`

- [x] **Step 1: Write failing tests** for timeout conversion, crashed worker replacement up to a bounded retry count, stale scene rejection, and evidence serialization.
- [x] **Step 2: Run** the focused tests and verify failure.
- [x] **Step 3: Implement** bounded respawn and strict evidence compatibility checks using the P5.0 contract helper.
- [x] **Step 4: Run** the focused tests and verify pass.

### Task 4: Add the real LIBERO pilot driver

**Files:**
- Create: `scripts/run_libero_p53_rehearsal.py`
- Test: `tests/test_libero_p53_rehearsal.py`

**Interfaces:**
- CLI accepts `--config-path`, `--candidate-artifact`, `--seeds`, `--max-workers`, `--timeout-s`, `--gpu`, and `--output-root`.
- One run directory per seed/candidate batch under `outputs/phase5/P5.3_process_rehearsal/`.

- [x] **Step 1: Write failing tests** for CLI parsing, unique run directories, secret-free config artifacts, and preservation of failure logs.
- [x] **Step 2: Run** the focused tests and verify failure.
- [x] **Step 3: Implement** the driver using the existing CAP-X YAML/factory seam, `CUDA_VISIBLE_DEVICES`, `Phase5RunDirectory`, and complete manifest finalization.
- [x] **Step 4: Run** the focused tests and verify pass.

### Task 5: Integrate rehearsal evidence with the Arbiter in shadow mode

**Files:**
- Modify: existing staged scheduler/Arbiter integration module identified by the focused test import.
- Test: `tests/test_phase5_rehearsal_arbiter.py`
- Modify: `docs/phase5-evidence-evolution.md`

- [x] **Step 1: Write failing tests** proving rehearsal evidence changes only the shadow score first, records `selection_basis`, and never executes a second physical action.
- [x] **Step 2: Run** the focused test and verify failure.
- [x] **Step 3: Implement** optional evidence attachment keyed by candidate fingerprint and scene version; unavailable rehearsal remains unknown rather than zero.
- [x] **Step 4: Run** focused scheduler tests and update the P5.3 contract/status documentation.

### Task 6: Run the matched LIBERO gate

- [x] **Step 1: Run** the unit/integration suite for rehearsal and existing P5.2 tests.
- [x] **Step 2: Run** a real one-seed CAP-X/LIBERO smoke with `CUDA_VISIBLE_DEVICES=5`.
- [x] **Step 3: Verify** the smoke run has a separate log directory, manifest hashes, worker PID, seed, candidate fingerprint, checkpoint results, failure classification, timeout data, and latency.
- [x] **Step 4: Compare** rehearsal evidence availability, candidate diversity, downstream evaluator success, and CAP-X parity against the locked P5.2 baseline using matched seeds.

### Matched gate result (2026-07-30)

The real two-candidate LIBERO Spatial-0 rehearsal was rerun after adding
backward-compatible decoding for pre-canonical `call_index` skill-output
references. The matched five-seed artifact is:

`outputs/phase5/P5.3_process_rehearsal_matched_fix_20260730/`

Each seed has an independent run directory, complete runner log, result,
summary, and SHA-256 manifest. The two candidates were executed with the same
task, reset seed, CAP-X YAML, worker budget, and simulator services:

| candidate | evaluator successes | rate | failure class |
| --- | ---: | ---: | --- |
| `policy-0:0` | 0/5 | 0.00 | `postcondition_failure` |
| `policy-1:safety:1` | 2/5 | 0.40 | `postcondition_failure` on 3 seeds |

The candidate results differed on seeds 1 and 5, and all ten candidate runs
reached the physical CAP-X execution boundary. This closes the P5.3 process
rehearsal and candidate-discrimination gate. It is not a causal downstream
improvement claim: the run is an isolated evidence collection experiment and
does not select or execute a live robot action.

The serialized input is a full MissionGraph artifact whose fingerprint is
graph-scoped. The P5.3 identity closure preserves that source fingerprint and
requires an explicit `arbiter_subgraph_id` plus a derived local fingerprint
before attaching evidence to a `GraphCandidate`. The pure shadow-Arbiter path
now compares the baseline with an evidence-enriched hypothetical selection,
records mapping rejections, and keeps `physical_execution_required=false`.
This closes the identity and shadow code gates without promoting a shadow
winner to live physical execution. Ten-plus seeds, multiple tasks, online
physical selection, and P5.4 cache remain follow-up work.
