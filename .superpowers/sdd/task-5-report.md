# CAP-MAS P5.6A Task 5 TDD/Implementation Report

## Scope

Implemented read-only family capability diagnosis and typed P5.3.2 handoffs.

Allowed touched files:

- `capmas/evaluation/capability.py`
- `capmas/evaluation/__init__.py`
- `scripts/run_p56_capability.py`
- `tests/test_p56_capability.py`
- `.superpowers/sdd/task-5-report.md` (ignored report)

No CAP-X imports, simulator calls, API server startup, LLM calls, credential use, family repair, calibration fitting, or online activation were added.

## RED

Created `tests/test_p56_capability.py` before production files.

Command:

```bash
pytest -q tests/test_p56_capability.py
```

Observed expected failure before production implementation:

```text
ModuleNotFoundError: No module named 'capmas.evaluation.capability'
```

Exit code: 2.

## GREEN

Implemented:

- Frozen dataclass contracts: `CapabilityCase`, `CapabilityDiagnosticReport`, `TaskFamilyRepairHandoff`, `CapabilityRunResult`.
- Formal ten-unique-seed diagnosis gate with gate failures:
  - `INFRASTRUCTURE_UNKNOWN`
  - `UNTYPED_FAILURE`
  - `EXECUTION_REACH_BELOW_0_80`
  - `NO_EVALUATOR_SUCCESS`
- P5.3.2 typed handoff package:
  - package: `P5.3.2 Task-Family Capability Repair`
  - acceptance test: `rerun the same frozen ten-seed capability manifest with zero infrastructure unknowns, at least 80% physical execution reach, typed provenance for every failure, and at least one evaluator success`
- Owner mapping:
  - infrastructure failures -> `runtime_infrastructure`
  - precondition failures -> `perception_or_contract`
  - postcondition failures -> `verification_or_robot_skill`
  - ordinary task failures -> `task_mapping_or_motion`
- Read-only P5.5 loader that resolves exact per-case relative paths:
  - `case.json`
  - case-level `summary.json`
  - `evidence/ood_replay.json`
- Fresh `Phase5RunDirectory` artifact writer for:
  - `run_config.json`
  - `results/capability.json`
  - `artifacts/p53_2_{family_id}.json`
  - `summary.md`
  - `logs/runner.log`
  - `manifest.json`
- CLI: `scripts/run_p56_capability.py`.

Focused command:

```bash
pytest -q tests/test_p56_capability.py && python scripts/run_p56_capability.py --help >/dev/null
```

Observed:

```text
5 passed in 0.25s
```

Exit code: 0.

## Real frozen diagnosis

Source suite:

```text
outputs/phase5/P5.5_matched_provenance_10seed_retry2_20260807/P5.5_frozen_ood_replay/20260807_024842_suite_20674432
```

Command:

```bash
python scripts/run_p56_capability.py \
  --suite-dir outputs/phase5/P5.5_matched_provenance_10seed_retry2_20260807/P5.5_frozen_ood_replay/20260807_024842_suite_20674432 \
  --family spatial-0 \
  --family goal-1 \
  --family object-6 \
  --split id \
  --output-root outputs/phase5
```

Observed:

```text
outputs/phase5/P5.6.0_capability_diagnosis/20260813_070751_capability_cc632b7a
spatial-0: cases=10 physical_execution=10/10 evaluator_success=0/10 eligible=False gate_failures=NO_EVALUATOR_SUCCESS
goal-1: cases=10 physical_execution=10/10 evaluator_success=0/10 eligible=False gate_failures=NO_EVALUATOR_SUCCESS
object-6: cases=10 physical_execution=10/10 evaluator_success=4/10 eligible=True gate_failures=-
handoffs=spatial-0,goal-1
```

Run directory:

```text
outputs/phase5/P5.6.0_capability_diagnosis/20260813_070751_capability_cc632b7a
```

Source manifest SHA propagated in `run_config.json`, `results/capability.json`, and both handoffs:

```text
5aeff85dae764c72fe9c0b1f3a0a07f4070e95247baea1b8d93f15311ea72141
```

## Source byte identity proof

Deterministic digest method: SHA-256 over sorted file relative paths plus file bytes.

Before real command:

```text
32ec24898071c892c36e4296cfb65a3c5bd7d5a705818a82cc022f3c9429ba26
1028 files
```

After real command:

```text
32ec24898071c892c36e4296cfb65a3c5bd7d5a705818a82cc022f3c9429ba26
1028 files
```

Result: source suite remained byte-identical.

## Generated artifact/manifest verification

Run manifest checked all entries for existence, size, and SHA-256 digest.

Observed:

```text
manifest_entries 6
missing []
mismatch []
paths artifacts/p53_2_goal-1.json,artifacts/p53_2_spatial-0.json,logs/runner.log,results/capability.json,run_config.json,summary.md
```

## Final verification

Commands:

```bash
pytest -q tests/test_p56_capability.py && python scripts/run_p56_capability.py --help >/dev/null
python -m ruff check capmas/evaluation/capability.py capmas/evaluation/__init__.py scripts/run_p56_capability.py tests/test_p56_capability.py
python -m compileall -q capmas/evaluation/capability.py capmas/evaluation/__init__.py scripts/run_p56_capability.py tests/test_p56_capability.py
git diff --check -- capmas/evaluation/capability.py capmas/evaluation/__init__.py scripts/run_p56_capability.py tests/test_p56_capability.py
```

Observed:

```text
5 passed in 0.25s
All checks passed!
compileall exit code 0
git diff --check exit code 0
```

## Review notes

Actual subagent spawn/wait tooling was not available in this Codex tool surface, so I preserved the strict SDD checkpoints in this report and performed implementation/review/verification in separate phases in-session.

Concern: `outputs/phase5/P5.6.0_capability_diagnosis/` appears untracked in this worktree rather than ignored by `git check-ignore`; it was deliberately not staged or committed.

## Independent review fixes for P5.6A Task 5

Implemented all five independent review findings via strict RED -> GREEN TDD.

### RED

Added focused regressions in `tests/test_p56_capability.py` and updated the existing eligible mixed-result test so all false evaluator cases have typed failure classes.

Command:

```bash
pytest -q tests/test_p56_capability.py
```

Observed expected failures before implementation:

```text
5 failed, 6 passed in 0.48s
```

Failing regressions covered:

- `UNTYPED_FAILURE` on any `evaluator_success is False` case with missing `failure_class`, even when the family also has successes.
- Loader rejection when case-level summary `primary_winner` mismatches evidence `candidate_id` after either side claims selection.
- Conservative coherent no-execution behavior with `primary_winner` absent and `candidate_id` equal to `unselected`.
- Symlink-aware rejection of `output_root` equal to or nested under `suite_dir` before run directory creation.
- Failed-run `run_config.json` rewrite to `status=failed` with safe `error_type`/`error`, failed log, and finalized manifest.
- Duplicate family argument rejection before run directory creation.

### GREEN

Minimal production changes in `capmas/evaluation/capability.py`:

- Removed the zero-success condition around `UNTYPED_FAILURE`.
- Added duplicate-family and symlink-aware source/output containment preflight checks before creating run directories.
- Added fail-closed winner/candidate validation at the loader seam.
- Rewrote failed `run_config.json` after post-allocation exceptions and finalized failed artifacts.

Focused GREEN:

```bash
pytest -q tests/test_p56_capability.py
```

Observed:

```text
11 passed in 0.39s
```

### Required verification

Commands:

```bash
pytest -q tests/test_p56_capability.py && python scripts/run_p56_capability.py --help >/dev/null
python -m ruff check capmas/evaluation/capability.py tests/test_p56_capability.py
python -m compileall -q capmas/evaluation/capability.py tests/test_p56_capability.py
git diff --check -- capmas/evaluation/capability.py tests/test_p56_capability.py
```

Observed:

```text
11 passed in 0.36s
All checks passed!
compileall exit code 0
git diff --check exit code 0
```

### Real frozen diagnosis rerun

Source suite:

```text
outputs/phase5/P5.5_matched_provenance_10seed_retry2_20260807/P5.5_frozen_ood_replay/20260807_024842_suite_20674432
```

Source byte digest before:

```text
32ec24898071c892c36e4296cfb65a3c5bd7d5a705818a82cc022f3c9429ba26
1028 files
```

Command:

```bash
python scripts/run_p56_capability.py \
  --suite-dir outputs/phase5/P5.5_matched_provenance_10seed_retry2_20260807/P5.5_frozen_ood_replay/20260807_024842_suite_20674432 \
  --family spatial-0 \
  --family goal-1 \
  --family object-6 \
  --split id \
  --output-root outputs/phase5
```

Observed:

```text
outputs/phase5/P5.6.0_capability_diagnosis/20260813_072308_capability_c9df3f4b
spatial-0: cases=10 physical_execution=10/10 evaluator_success=0/10 eligible=False gate_failures=NO_EVALUATOR_SUCCESS
goal-1: cases=10 physical_execution=10/10 evaluator_success=0/10 eligible=False gate_failures=NO_EVALUATOR_SUCCESS
object-6: cases=10 physical_execution=10/10 evaluator_success=4/10 eligible=True gate_failures=-
handoffs=spatial-0,goal-1
```

Source byte digest after:

```text
32ec24898071c892c36e4296cfb65a3c5bd7d5a705818a82cc022f3c9429ba26
1028 files
```

Manifest verification:

```text
manifest_entries 6
missing []
mismatch []
paths artifacts/p53_2_goal-1.json,artifacts/p53_2_spatial-0.json,logs/runner.log,results/capability.json,run_config.json,summary.md
```

Report counts:

```text
report spatial-0 10 10 0 False NO_EVALUATOR_SUCCESS
report goal-1 10 10 0 False NO_EVALUATOR_SUCCESS
report object-6 10 10 4 True -
handoffs spatial-0,goal-1
```
