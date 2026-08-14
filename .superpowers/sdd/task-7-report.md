# CAP-MAS P5.6A Task 7 report

## Scope

Implemented pre-registered object-6 physical collection infrastructure and fixed ID manifests only. I did not run a real LIBERO collection.

Changed intended paths:

- `capmas/contracts/calibration.py`
- `scripts/create_p56_object6_manifests.py`
- `scripts/run_libero_p56_collect.py`
- `tests/test_libero_p56_collection.py`
- `configs/phase5/p56_object6_id_seeds_11_20.json`
- `configs/phase5/p56_object6_id_seeds_21_30.json`
- `.superpowers/sdd/task-7-report.md`

## RED

Command:

```bash
pytest -q tests/test_libero_p56_collection.py
```

Output:

```text
==================================== ERRORS ====================================
_____________ ERROR collecting tests/test_libero_p56_collection.py _____________
ImportError while importing test module '/data/MLLM/wzh/agent/paper/infiAgent/workspace/cap-mas/.worktrees/p56a-data-foundation/tests/test_libero_p56_collection.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/data/envs/miniforge3/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/test_libero_p56_collection.py:12: in <module>
    from capmas.contracts.calibration import (
E   ImportError: cannot import name 'CalibrationCollectionCase' from 'capmas.contracts.calibration' (/data/MLLM/wzh/agent/paper/infiAgent/workspace/cap-mas/.worktrees/p56a-data-foundation/capmas/contracts/calibration.py)
=========================== short test summary info ============================
ERROR tests/test_libero_p56_collection.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.40s
```

## GREEN implementation summary

- Added strict immutable `CalibrationCollectionCase` and `CalibrationCollectionManifest` contracts with strict field decoders, JSON-safe frozen mappings, canonical payload, and content-addressed SHA-256 helpers.
- Added byte-stable object-6 manifest generator with fixed ID seed blocks 11-20 and 21-30, resolved path SHA-256s, zero-delta native layout transforms, no timestamps, and `--check` byte/digest validation.
- Added P5.6 collection runner with lazy P5.3 live imports, preflight config/candidate digest enforcement, one-worker/one-selection/cache-disabled online bounded calls, physical collection context propagation, per-suite and per-case `Phase5RunDirectory` allocation, decision-time snapshots, physical payload, graph events, horizon evidence, normalized Tier A/C outcomes, typed failure artifacts, and non-adaptive full-block execution unless `fail_fast`.
- Added collection summary mode combining suite rows with optional history audit rows, duplicate case/lineage rejection, and 20/5/5 gate reporting without selecting or running block 21-30.

## Generated manifests

Command:

```bash
python scripts/create_p56_object6_manifests.py --project-root .
```

Output:

```text
wrote /data/MLLM/wzh/agent/paper/infiAgent/workspace/cap-mas/.worktrees/p56a-data-foundation/configs/phase5/p56_object6_id_seeds_11_20.json sha256=7104dee4da0a59e3aa4f3dbb11f65924c2b3242423d895aff3c9fe355282726b
wrote /data/MLLM/wzh/agent/paper/infiAgent/workspace/cap-mas/.worktrees/p56a-data-foundation/configs/phase5/p56_object6_id_seeds_21_30.json sha256=97ba76e1918fb4f087ce522d79599be6da8b1333026e0292da7cb807b3f6ca1b
```

Manifest details:

- `p56_object6_id_seeds_11_20.json`: seeds 11-20, `manifest_sha256=7104dee4da0a59e3aa4f3dbb11f65924c2b3242423d895aff3c9fe355282726b`, no timestamp fields.
- `p56_object6_id_seeds_21_30.json`: seeds 21-30, `manifest_sha256=97ba76e1918fb4f087ce522d79599be6da8b1333026e0292da7cb807b3f6ca1b`, no timestamp fields.

## Verification

Command:

```bash
python scripts/create_p56_object6_manifests.py --project-root . --check
```

Output:

```text
validated /data/MLLM/wzh/agent/paper/infiAgent/workspace/cap-mas/.worktrees/p56a-data-foundation/configs/phase5/p56_object6_id_seeds_11_20.json sha256=7104dee4da0a59e3aa4f3dbb11f65924c2b3242423d895aff3c9fe355282726b
validated /data/MLLM/wzh/agent/paper/infiAgent/workspace/cap-mas/.worktrees/p56a-data-foundation/configs/phase5/p56_object6_id_seeds_21_30.json sha256=97ba76e1918fb4f087ce522d79599be6da8b1333026e0292da7cb807b3f6ca1b
```

Command:

```bash
pytest -q tests/test_libero_p56_collection.py tests/test_libero_p53_online.py tests/test_libero_p55_ood.py
```

Output:

```text
.............................                                            [100%]
29 passed in 3.13s
```

Command:

```bash
python scripts/create_p56_object6_manifests.py --help
```

Output:

```text
usage: create_p56_object6_manifests.py [-h] [--project-root PROJECT_ROOT]
                                       [--output-dir OUTPUT_DIR] [--check]

Create byte-stable CAP-MAS P5.6 object-6 collection manifests.

options:
  -h, --help            show this help message and exit
  --project-root PROJECT_ROOT
  --output-dir OUTPUT_DIR
  --check
```

Command:

```bash
python scripts/run_libero_p56_collect.py --help
```

Output:

```text
usage: run_libero_p56_collect.py [-h] [--manifest MANIFEST]
                                 [--output-root OUTPUT_ROOT] [--gpu GPU]
                                 [--max-workers MAX_WORKERS]
                                 [--timeout-s TIMEOUT_S]
                                 [--max-restarts MAX_RESTARTS]
                                 [--max-steps MAX_STEPS] [--fail-fast]
                                 [--summarize-suite SUMMARIZE_SUITE]
                                 [--history-audit HISTORY_AUDIT]

Run or summarize CAP-MAS P5.6 object-6 physical collection suites.

options:
  -h, --help            show this help message and exit
  --manifest MANIFEST
  --output-root OUTPUT_ROOT
  --gpu GPU
  --max-workers MAX_WORKERS
  --timeout-s TIMEOUT_S
  --max-restarts MAX_RESTARTS
  --max-steps MAX_STEPS
  --fail-fast
  --summarize-suite SUMMARIZE_SUITE
  --history-audit HISTORY_AUDIT
```

Command:

```bash
python -m ruff check capmas/contracts/calibration.py scripts/create_p56_object6_manifests.py scripts/run_libero_p56_collect.py tests/test_libero_p56_collection.py
```

Output:

```text
All checks passed!
```

Command:

```bash
python -m compileall capmas/contracts/calibration.py scripts/create_p56_object6_manifests.py scripts/run_libero_p56_collect.py tests/test_libero_p56_collection.py
```

Output:

```text
Compiling 'scripts/run_libero_p56_collect.py'...
Compiling 'tests/test_libero_p56_collection.py'...
```

## Concerns

- No real LIBERO physical collection was executed, by instruction. The controller still needs to run the actual block after review.
- The worktree contains unrelated pre-existing untracked `outputs/phase5/...` artifacts; I did not modify or stage them.

## Review remediation

Independent task review found four collection-boundary defects: evaluator labels
could fall back to non-evaluator success, post-run failures could discard raw
evidence, `fail_fast` could stop on non-infrastructure validation failures, and
summary mode could miss duplicate failed cases. Commit `48cd8f1` adds focused
regressions and fixes those paths. The regressions first failed before the
implementation and then passed.

Controller verification after a lint-only cleanup:

```text
ruff check scripts/run_libero_p56_collect.py tests/test_libero_p56_collection.py
All checks passed!

/data/MLLM/wzh/agent/paper/infiAgent/workspace/cap-x/.venv-libero/bin/python -m pytest -q tests/test_libero_p56_collection.py tests/test_libero_p53_online.py tests/test_libero_p55_ood.py
....................................                                     [100%]
36 passed in 3.66s
```

## Persistence-boundary remediation

The follow-up review found that a raw-artifact retry could obscure its original
typed failure, case-manifest finalization could escape the case boundary, and
context or pool construction was classified as an online-runner failure. The
runner now records raw-artifact retry failures alongside the original failure,
converts finalization failures to non-infrastructure persistence failures, and
sets the infrastructure stage only immediately before the runner invocation.
New regressions cover raw-artifact persistence, finalization, context and pool
preparation, and normalization under `fail_fast=True`.

The requested distinct-case/shared-lineage scenario cannot occur in a valid
P5.6 collection manifest because the binding contract requires
`lineage_group_id == case_id`. The summary decoder rejects a deliberately
malformed persisted identity before aggregation; the valid failed-case duplicate
case regression remains covered.

Controller verification:

```text
ruff check scripts/run_libero_p56_collect.py tests/test_libero_p56_collection.py
All checks passed!

/data/MLLM/wzh/agent/paper/infiAgent/workspace/cap-x/.venv-libero/bin/python -m pytest -q tests/test_libero_p56_collection.py tests/test_libero_p53_online.py tests/test_libero_p55_ood.py
..........................................                               [100%]
42 passed in 4.23s

python scripts/create_p56_object6_manifests.py --project-root . --check
validated p56_object6_id_seeds_11_20.json sha256=7104dee4da0a59e3aa4f3dbb11f65924c2b3242423d895aff3c9fe355282726b
validated p56_object6_id_seeds_21_30.json sha256=97ba76e1918fb4f087ce522d79599be6da8b1333026e0292da7cb807b3f6ca1b
```

## Best-effort persistence remediation

The final task review found that raw evidence writes still stopped at the first
storage error and that a finalization error after an existing case failure was
not retained. Each raw or placeholder artifact is now attempted independently;
all write failures are collected, and any partial raw write produces a typed
non-infrastructure persistence failure. A finalization error augments an
existing failure record rather than being discarded. Regressions cover both a
single failed raw artifact while later evidence remains written and a manifest
finalization failure after an online-runner failure.

Controller verification:

```text
/data/MLLM/wzh/agent/paper/infiAgent/workspace/cap-x/.venv-libero/bin/python -m pytest -q tests/test_libero_p56_collection.py tests/test_libero_p53_online.py tests/test_libero_p55_ood.py
............................................                             [100%]
44 passed in 4.35s

ruff check scripts/run_libero_p56_collect.py tests/test_libero_p56_collection.py
All checks passed!

python scripts/create_p56_object6_manifests.py --project-root . --check
validated p56_object6_id_seeds_11_20.json sha256=7104dee4da0a59e3aa4f3dbb11f65924c2b3242423d895aff3c9fe355282726b
validated p56_object6_id_seeds_21_30.json sha256=97ba76e1918fb4f087ce522d79599be6da8b1333026e0292da7cb807b3f6ca1b
```
