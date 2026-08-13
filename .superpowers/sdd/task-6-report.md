## Task 6 recovery evidence - 2026-08-13

Scope recovered in shared worktree:

- Worktree: `/data/MLLM/wzh/agent/paper/infiAgent/workspace/cap-mas/.worktrees/p56a-data-foundation`
- Branch: `feature/p56a-data-foundation`
- Starting HEAD: `56fa4d71f5e8c3d1e6f9ad05fd47bce3604fdd58`
- Partial implementation assessed rather than discarded:
  - `capmas/evaluation/history_audit.py`
  - `scripts/audit_p56_history.py`
  - `tests/test_p56_history_audit.py`
  - `capmas/evaluation/__init__.py`

Baseline partial implementation:

```text
$ python -m pytest -q tests/test_p56_history_audit.py
.......                                                                  [100%]
7 passed in 0.25s
```

TDD RED tests added for concrete missing behavior:

- `test_history_audit_rejects_evaluator_observation_before_execution_start`
- `test_history_audit_rejects_missing_evaluator_observation_for_native_snapshot`
- `test_history_audit_rejects_inconsistent_retained_evaluator_outcome`
- `test_history_audit_rejects_non_structural_graph_events`
- `test_history_audit_rejects_missing_case_level_evaluator_outcome`

Observed RED failure:

```text
$ python -m pytest -q tests/test_p56_history_audit.py
...FFFF....                                                              [100%]
FAILED tests/test_p56_history_audit.py::test_history_audit_rejects_evaluator_observation_before_execution_start
FAILED tests/test_p56_history_audit.py::test_history_audit_rejects_missing_evaluator_observation_for_native_snapshot
FAILED tests/test_p56_history_audit.py::test_history_audit_rejects_inconsistent_retained_evaluator_outcome
FAILED tests/test_p56_history_audit.py::test_history_audit_rejects_non_structural_graph_events
4 failed, 7 passed in 0.61s
```

Minimal GREEN implementation:

- Added native evaluator observation timestamp checks:
  - `MISSING_EVALUATOR_OBSERVATION_TIMESTAMP`
  - `EVALUATOR_OBSERVED_BEFORE_EXECUTION_START`
- Added retained summary/evidence/physical evaluator outcome consistency:
  - `EVALUATOR_OUTCOME_MISMATCH`
  - case-level and online evidence must both retain boolean evaluator outcomes
- Strengthened graph-events check from nonempty list to structural graph-event records:
  - `INVALID_GRAPH_EVENTS`

Focused Task 6 test result after implementation:

```text
$ python -m pytest -q tests/test_p56_history_audit.py
............                                                             [100%]
12 passed in 0.29s
```

Changed verification:

```text
$ python -m pytest -q tests/test_p56_history_audit.py
............                                                             [100%]
12 passed in 0.29s

$ python scripts/audit_p56_history.py --help
usage: audit_p56_history.py [-h] --suite-dir SUITE_DIR --family FAMILY
                            --output-root OUTPUT_ROOT

Audit retained P5.5 family rows for native P5.6 Tier-A history compatibility.

options:
  -h, --help            show this help message and exit
  --suite-dir SUITE_DIR
                        Retained P5.5 frozen replay suite
  --family FAMILY       Task family to audit
  --output-root OUTPUT_ROOT
                        Phase 5 output root

$ python -m ruff check capmas/evaluation/history_audit.py scripts/audit_p56_history.py tests/test_p56_history_audit.py capmas/evaluation/__init__.py
All checks passed!

$ python -m compileall -q capmas/evaluation/history_audit.py scripts/audit_p56_history.py tests/test_p56_history_audit.py capmas/evaluation/__init__.py

$ git diff --check -- capmas/evaluation/history_audit.py scripts/audit_p56_history.py tests/test_p56_history_audit.py capmas/evaluation/__init__.py
```

Real frozen-suite audit:

```text
$ python scripts/audit_p56_history.py --suite-dir outputs/phase5/P5.5_matched_provenance_10seed_retry2_20260807/P5.5_frozen_ood_replay/20260807_024842_suite_20674432 --family object-6 --output-root outputs/phase5
outputs/phase5/P5.6.2a_object6_history_audit/20260813_092723_history_1188452c
object-6: examined=20 admissible_tier_a=0 rejected=20
rejection_counts={'MISSING_GRAPH_EVENTS': 20, 'MISSING_HORIZON_LINEAGE': 20, 'MISSING_PREEXECUTION_FEATURE_SNAPSHOT': 20}
```

Real audit artifact verification:

```text
run_dir=outputs/phase5/P5.6.2a_object6_history_audit/20260813_092723_history_1188452c
manifest_entries=5
logs/runner.log	size=497	sha256=0bf0d19402cd92102e6d3928fbc127e22861a17ca27e8b6ace2b095658384e6a
results/admissible_rows.json	size=3	sha256=37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570
results/history_audit.json	size=13266	sha256=cbabf566d5f2007332e539acf3bdf2ca6b07462912a2a36a414672df2335b980
run_config.json	size=362	sha256=269df757f28147cd5e42a9e6b02baa20791e8fdcba24b68520a2a646b67c2fa3
summary.md	size=386	sha256=4448460e8d2b4b9eed1896f6ef5e765b7e6e782a26dc2699cbc54ae18ef7274d
errors=[]
```

Real audit computed counts:

```text
examined_count=20
admissible_tier_a_count=0
rejected_count=20
rejection_counts={'MISSING_GRAPH_EVENTS': 20, 'MISSING_HORIZON_LINEAGE': 20, 'MISSING_PREEXECUTION_FEATURE_SNAPSHOT': 20}
admissible_rows=0
all_rows_inadmissible=True
unique_reason_sets=[('MISSING_PREEXECUTION_FEATURE_SNAPSHOT', 'MISSING_GRAPH_EVENTS', 'MISSING_HORIZON_LINEAGE')]
```

Source tree digest before and after real audit:

```text
source_suite=outputs/phase5/P5.5_matched_provenance_10seed_retry2_20260807/P5.5_frozen_ood_replay/20260807_024842_suite_20674432
files=1028
before_source_tree_sha256=befc822a0fd03a28811bbfa730d092374a5535ccd103e004c1b4f1fd5452b4ea
after_source_tree_sha256=befc822a0fd03a28811bbfa730d092374a5535ccd103e004c1b4f1fd5452b4ea
source_byte_identical=True
```

## Task 6 review-fix evidence - 2026-08-13

Scope:

- Worktree: `/data/MLLM/wzh/agent/paper/infiAgent/workspace/cap-mas/.worktrees/p56a-data-foundation`
- Branch: `feature/p56a-data-foundation`
- Starting HEAD: `f4ed8317174fa4dc33b63c2d4ea6dd0951209622`
- Fixed reviewer findings in:
  - `capmas/evaluation/history_audit.py`
  - `tests/test_p56_history_audit.py`

TDD RED - retained primary scene identity:

```text
$ python -m pytest -q tests/test_p56_history_audit.py::test_history_audit_rejects_snapshot_from_different_source_scene
F                                                                        [100%]
FAILED tests/test_p56_history_audit.py::test_history_audit_rejects_snapshot_from_different_source_scene
AssertionError: assert True is False
```

GREEN - scene identity now requires native top-level `source_scene_version`
to exactly match `CandidateFeatureSnapshot.scene_version`. Missing or malformed
top-level scene version rejects admission with stable
`FEATURE_SNAPSHOT_IDENTITY_MISMATCH`.

```text
$ python -m pytest -q tests/test_p56_history_audit.py::test_history_audit_accepts_only_timestamp_ordered_native_record tests/test_p56_history_audit.py::test_history_audit_rejects_snapshot_from_different_source_scene
..                                                                       [100%]
2 passed in 0.32s
```

TDD RED - nested `physical_result` must not backfill required native fields:

```text
$ python -m pytest -q tests/test_p56_history_audit.py::test_history_audit_requires_native_top_level_lineage_timing_and_evaluator
F                                                                        [100%]
FAILED tests/test_p56_history_audit.py::test_history_audit_requires_native_top_level_lineage_timing_and_evaluator
AssertionError: assert {'INCONCLUSIVE_EVALUATOR', ...} <= {'INCONCLUSIVE_EVALUATOR'}
```

GREEN - removed nested `physical_result` fallbacks from horizon lineage,
graph-events, execution-start timing, evaluator-observation timing, and
evaluator consistency checks. Required native telemetry must now be retained
at top level.

```text
$ python -m pytest -q tests/test_p56_history_audit.py::test_history_audit_requires_native_top_level_lineage_timing_and_evaluator tests/test_p56_history_audit.py::test_history_audit_accepts_only_timestamp_ordered_native_record tests/test_p56_history_audit.py::test_history_audit_rejects_snapshot_from_different_source_scene
...                                                                      [100%]
3 passed in 0.37s
```

Focused Task 6 test suite:

```text
$ python -m pytest -q tests/test_p56_history_audit.py
.................                                                        [100%]
17 passed in 0.31s
```

Changed verification:

```text
$ python scripts/audit_p56_history.py --help
usage: audit_p56_history.py [-h] --suite-dir SUITE_DIR --family FAMILY
                            --output-root OUTPUT_ROOT

Audit retained P5.5 family rows for native P5.6 Tier-A history compatibility.

options:
  -h, --help            show this help message and exit
  --suite-dir SUITE_DIR
                        Retained P5.5 frozen replay suite
  --family FAMILY       Task family to audit
  --output-root OUTPUT_ROOT
                        Phase 5 output root

$ python -m ruff check capmas/evaluation/history_audit.py scripts/audit_p56_history.py tests/test_p56_history_audit.py capmas/evaluation/__init__.py
All checks passed!

$ python -m compileall -q capmas/evaluation/history_audit.py scripts/audit_p56_history.py tests/test_p56_history_audit.py capmas/evaluation/__init__.py

$ git diff --check -- capmas/evaluation/history_audit.py scripts/audit_p56_history.py tests/test_p56_history_audit.py capmas/evaluation/__init__.py .superpowers/sdd/task-6-report.md
```

Real frozen-suite audit in a new output root:

```text
$ python scripts/audit_p56_history.py --suite-dir outputs/phase5/P5.5_matched_provenance_10seed_retry2_20260807/P5.5_frozen_ood_replay/20260807_024842_suite_20674432 --family object-6 --output-root outputs/phase5/task6-review-20260813
outputs/phase5/task6-review-20260813/P5.6.2a_object6_history_audit/20260813_100401_history_2a994b7a
object-6: examined=20 admissible_tier_a=0 rejected=20
rejection_counts={'MISSING_GRAPH_EVENTS': 20, 'MISSING_HORIZON_LINEAGE': 20, 'MISSING_PREEXECUTION_FEATURE_SNAPSHOT': 20}
```

Real audit artifact verification:

```text
run_dir=outputs/phase5/task6-review-20260813/P5.6.2a_object6_history_audit/20260813_100401_history_2a994b7a
manifest_entries=5
errors=[]
examined_count=20
admissible_tier_a_count=0
rejected_count=20
rejection_counts={'MISSING_GRAPH_EVENTS': 20, 'MISSING_HORIZON_LINEAGE': 20, 'MISSING_PREEXECUTION_FEATURE_SNAPSHOT': 20}
admissible_rows=0
logs/runner.log	size=497	sha256=0bf0d19402cd92102e6d3928fbc127e22861a17ca27e8b6ace2b095658384e6a
results/admissible_rows.json	size=3	sha256=37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570
results/history_audit.json	size=13266	sha256=cbabf566d5f2007332e539acf3bdf2ca6b07462912a2a36a414672df2335b980
run_config.json	size=362	sha256=269df757f28147cd5e42a9e6b02baa20791e8fdcba24b68520a2a646b67c2fa3
summary.md	size=386	sha256=4448460e8d2b4b9eed1896f6ef5e765b7e6e782a26dc2699cbc54ae18ef7274d
```

Source tree digest before and after real audit:

```text
source_suite=outputs/phase5/P5.5_matched_provenance_10seed_retry2_20260807/P5.5_frozen_ood_replay/20260807_024842_suite_20674432
files=1028
before_source_tree_sha256=befc822a0fd03a28811bbfa730d092374a5535ccd103e004c1b4f1fd5452b4ea
after_source_tree_sha256=befc822a0fd03a28811bbfa730d092374a5535ccd103e004c1b4f1fd5452b4ea
source_byte_identical=True
```
