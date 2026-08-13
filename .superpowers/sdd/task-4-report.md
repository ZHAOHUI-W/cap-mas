# Task 4 Report

## Outcome

- Added exact `CalibrationLineage` and `CalibrationDatasetManifest` contracts.
- Added deterministic lineage-group splitting, conservative three-tier outcome normalization, content-addressed dataset construction, and fail-closed auditing.
- Kept executor and Arbiter behavior unchanged; no model fitting, capability diagnosis, or collection was added.

## TDD Evidence

- Initial dataset test collection failed with missing `CalibrationDatasetManifest`.
- After adding contracts, collection failed with missing `capmas.evaluation.dataset`.
- Added a regression proving inconclusive physical execution leaked graph/verifier/failure fields into Tier C; it failed before the normalizer was corrected.
- Added real graph-validator rejection cases (`DUPLICATE_NODE`, `DANGLING_EDGE`); they failed before schema-status mapping was corrected.

## Verification

- Brief Step 5: `pytest -q tests/test_p56_dataset.py tests/test_p56_contracts.py`
- Direct snapshot regression: `pytest -q tests/test_p56_feature_snapshots.py`
- Changed-file lint: `ruff check capmas/contracts/calibration.py capmas/evaluation/dataset.py capmas/evaluation/__init__.py tests/test_p56_dataset.py`
- Compile: `python -m compileall -q capmas tests`
- Whitespace: `git diff --check`

## Concern

The brief requires `dataset_id == "sha256:<manifest digest>"` while saying only
`manifest_sha256` is omitted from the digest payload. Including the final
`dataset_id` in its own digest is self-referential. The implementation computes
the canonical ASCII JSON digest with `dataset_id=""`, excludes
`manifest_sha256`, then binds both final fields to that digest. Audit recomputes
the same canonical preimage and rejects any mismatch.

## Reviewer 4 Fix

### RED

- The added split-integrity regression fails against the pre-fix behavior because
  construction accepted caller-provided assignments without recomputing the
  salted lineage-group split, and audit did not detect a tampered default split.
- The exact feature-group regressions fail against the pre-fix behavior because
  schema v1 validation checked matching keys but did not require the literal
  `FEATURE_GROUPS_V1` mapping.
- The decoder rejection regressions fail against the pre-fix behavior because
  the concrete JSON/structured/request mismatch codes were not all mapped to
  `rejected_schema`.
- The fail-closed audit regressions fail against the pre-fix behavior because a
  forged `DatasetAudit(passed=True, findings=...)` could pass the eligibility
  gate, and `DatasetAudit` did not enforce flag/findings consistency.

The report did not preserve a numeric RED test count; no count is inferred here.

### GREEN

- `build_calibration_dataset` and `audit_calibration_dataset` recompute the
  salted default lineage split with the manifest split salt.
- v1 audit requires exact feature keys, statuses, and correlation-group mapping
  from `FEATURE_GROUPS_V1`.
- JSON, structured-payload, and request/mission mismatch decoder codes are
  mapped to `rejected_schema` alongside graph-validation codes.
- `DatasetAudit` and `assert_dataset_eligible` fail closed when `passed` does
  not equal the absence of findings.

### Reviewer 4 Follow-up: Remaining Decoder Rejection Codes

#### RED

- Added `EMPTY_RESPONSE`, `SUBGRAPH_ID_MISMATCH`, and
  `SUBGOAL_ID_MISMATCH` to the existing rejection-code parameterization with
  `rejected_schema` expectations.
- `pytest -q tests/test_p56_dataset.py::test_rejection_codes_map_to_unlabeled_statuses`
  failed as expected: 16 passed, 1 failed; `SUBGOAL_ID_MISMATCH` remained
  `not_selected` before the production change.

#### GREEN

- Added all three concrete decoder codes to
  `_DECODER_SCHEMA_REJECTION_CODES`; unknown codes remain `not_selected`.
- `pytest -q tests/test_p56_dataset.py`: 47 passed.
- `ruff check capmas/evaluation/dataset.py tests/test_p56_dataset.py`,
  `python -m compileall -q capmas/evaluation/dataset.py`, and `git diff --check`
  all passed.

### Task 4 Review-Fix: Graph Validator Rejection Status Coverage

#### RED

- Command:
  `pytest -q tests/test_p56_dataset.py::test_rejection_codes_map_to_unlabeled_statuses tests/test_p56_dataset.py::test_graph_validator_rejection_code_set_covers_validator_contract`
- Output:
  `8 failed, 17 passed in 0.44s`
- Expected failures showed `PORT_TYPE_MISMATCH`, `ACTION_WITHOUT_SKILL`,
  `UNBOUND_INPUT`, `PARALLEL_RESOURCE_CONFLICT`,
  `UNESTABLISHED_PRECONDITION`, and `UNBOUNDED_CYCLE` mapped to
  `not_selected`; a fake `DANGLING_REVIEWER_UNKNOWN` was misclassified as
  `rejected_schema`; and no centralized `_GRAPH_VALIDATION_REJECTION_CODES`
  existed for validator-contract coverage.

#### GREEN

- Command:
  `pytest -q tests/test_p56_dataset.py::test_rejection_codes_map_to_unlabeled_statuses tests/test_p56_dataset.py::test_graph_validator_rejection_code_set_covers_validator_contract`
- Output:
  `25 passed in 0.29s`
- Command: `pytest -q tests/test_p56_dataset.py`
- Output: `55 passed in 0.31s`
- Command: `ruff check capmas/evaluation/dataset.py tests/test_p56_dataset.py`
- Output: `All checks passed!`
- Command: `python -m compileall -q capmas/evaluation/dataset.py`
- Output: passed with exit code 0.
- Command: `git diff --check`
- Output: passed with exit code 0.

#### Files Changed

- `capmas/evaluation/dataset.py`
- `tests/test_p56_dataset.py`
- `.superpowers/sdd/task-4-report.md`

#### Commit

- Committed as `fix: fail closed P56 dataset review findings`; final hash
  recorded in handoff.

### Task 4 Second Review-Fix: Fail-Closed Tier Audit and Explicit Schema Rejections

#### RED

- Command:
  `pytest -q tests/test_p56_dataset.py::test_dataset_rejects_unknown_tier_even_when_shadow_split_is_consistent tests/test_p56_dataset.py::test_rejection_codes_map_to_unlabeled_statuses`
- Output:
  `2 failed, 25 passed in 0.42s`
- Expected failures showed an internally consistent manifest tampered to
  `tier="D"` and `dataset_split="shadow"` only produced
  `MANIFEST_DIGEST_MISMATCH`, without a stable tier finding code, and
  `FUTURE_SCHEMA_INVALID` was misclassified as `rejected_schema`.

#### GREEN

- Added explicit audit finding code `INVALID_TIER` for any outcome tier outside
  exactly `A`, `B`, or `C`.
- Added `GRAPH_SCHEMA_INVALID` to the explicit known decoder/schema rejection
  code set.
- Removed broad `_SCHEMA_INVALID` suffix classification so unknown future codes
  remain `not_selected`.

#### Verification

- Command:
  `pytest -q tests/test_p56_dataset.py::test_dataset_rejects_unknown_tier_even_when_shadow_split_is_consistent tests/test_p56_dataset.py::test_rejection_codes_map_to_unlabeled_statuses`
- Output:
  `26 passed in 0.34s`
- Command:
  `pytest -q tests/test_p56_dataset.py tests/test_p56_contracts.py`
- Output:
  `81 passed in 0.29s`
- Command:
  `ruff check capmas/evaluation/dataset.py tests/test_p56_dataset.py`
- Output:
  `All checks passed!`
- Command:
  `python -m compileall -q capmas/evaluation/dataset.py`
- Output:
  passed with exit code 0.
- Command:
  `git diff --check`
- Output:
  passed with exit code 0.

#### Files Changed

- `capmas/evaluation/dataset.py`
- `tests/test_p56_dataset.py`
- `.superpowers/sdd/task-4-report.md`

#### Commit

- Pending until commit is created.
