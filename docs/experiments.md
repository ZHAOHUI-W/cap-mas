# Experimental Protocol

## 1. Primary comparisons

| ID | System | Purpose |
| --- | --- | --- |
| B0 | CAP-X single-agent legacy | Direct baseline |
| B1 | CAP-X multi-turn with visual differencing | CAP-X strongest compatible baseline |
| B2 | CAP-MAS fixed contracts, one agent | Isolates contract benefit |
| B3 | CAP-MAS deterministic fixed graph | Regression baseline for graph/runtime safety |
| B3-LLM | CAP-MAS LLM Manager + local Policy fan-out | Tests typed multi-agent proposal and arbitration |
| B4 | CAP-MAS adaptive sparse graph | Tests topology adaptation |
| B5 | CAP-MAS plus asynchronous scene estimator | Tests real-time world model |
| B6 | CAP-MAS plus Memory Skills and experience memory | Tests memory contribution |
| B7 | B6 plus Memory Controller Stage 1/2/3 | Tests learned memory selection |
| B8 | B7 plus sequential Robot Skill evolution | Tests full self-evolution |

## 2. Required ablations

### Coordination

- Remove version checks.
- Remove precondition validation.
- Remove postcondition verification.
- Remove action lease.
- Replace structured artifacts with natural-language messages.
- Replace sparse graph with fully connected communication.

### Real-time perception

- Synchronous semantic perception.
- No object tracking between semantic updates.
- Full-scene rebuild instead of incremental mapping.
- No freshness-aware safety policy.

### Skill evolution

- No evolution.
- CAP-X occurrence-based skill library.
- Quarantine without promotion.
- Same-episode activation at safe boundaries.
- Later-episode activation only.
- Memory Skill evolution only with Robot Skills frozen.
- Robot Skill evolution only with Memory Skills and controller frozen.
- Joint evolution as a later ablation.

### Reward and RL

- CAP-X binary terminal reward.
- CAP-MAS binary terminal reward only.
- Verified potential-based subgoal shaping.
- Shaping without cost and intervention terms.
- Unverified intermediate detector reward.
- Memory Controller rules, Stage 1, Stage 2, and Stage 3.
- Hard-case buffer disabled.
- Automatic rollback disabled.

## 3. Task splits

- LIBERO spatial and object generalization for initial controlled comparisons.
- LIBERO goal/task perturbations for compositional and goal OOD tests.
- Long-horizon composition created by chaining compatible subtasks with known predicate interfaces.
- Hold out object layouts, language paraphrases, and subgoal combinations.

## 4. Metrics

### Task performance

- Full-task success rate.
- Success by sequential subgoal count.
- Success-rate decay slope versus horizon.
- First-attempt success.
- Recovery success after induced failure.

### Reliability and constraint quality

- Stale-action rejection rate.
- Precondition violation rate.
- Postcondition false-positive and false-negative rates.
- False completion rate.
- Collision/invariant violation rate.
- Episode invalidation handling rate.

### Real-time system metrics

- Control deadline miss rate.
- Scene snapshot freshness: median and P95.
- Map update rate.
- Semantic perception latency: median and P95.
- Planner-to-action latency.
- Queue drops and stale-frame ratio.

### Efficiency and intervention

- Total model calls and tokens.
- Per-call LLM latency: median and P95, separated by Manager and Policy role.
- Compile latency, physical execution latency, and total wall-clock latency.
- Schema mode, schema fallback count, retry count, and provider error count.
- Policy fan-out wall time at `max_workers=1` and `max_workers=2`.
- Wall-clock completion time.
- GPU/CPU utilization.
- Human interventions per episode.
- Intervention-free success rate.
- Communication messages and artifact bytes.

### Generalization and evolution

- ID versus OOD success gap.
- Cross-task skill reuse rate.
- Skill regression rate.
- Candidate-to-active promotion rate.
- Performance after distribution shift.
- Memory update precision and contradiction rate.
- Memory Skill selection regret or downstream utility.
- Promotion rollback rate and time-to-promotion.

## 5. Statistical reporting

Use fixed task seeds and report per-task paired outcomes where possible. Include confidence intervals or bootstrap intervals, not only aggregate means. The primary claim should be evaluated at matched budgets and tested with a horizon-stratified analysis rather than a single pooled success rate.

## 5.1 Staged experiment order

Run the staged experiments as separate artifacts with one changed variable at a
time:

1. **P3.1a observability** — verify that `run_config` and `llm_calls` are
   present, secret-free, and complete for strict and fallback modes. Preserve
   the terminal stream in a distinct per-run `.log`; repeated output paths must
   not overwrite prior logs.
2. **P3.1b fan-out A/B** — compare `max_workers=1` and `max_workers=2` with
   identical task seeds, model, prompts, retry budgets, and Policy count. The
   physical executor remains unchanged.
3. **P3.1c candidate quality** — compare homogeneous `balanced` Policies with
   explicit `safety`, `robust`, or `efficient` Policy strategies and attach
   evidence before arbitration; report candidate validity, disagreement,
   selected-agent distribution, rehearsal success, and recovery cost. If
   evidence is absent, report `confidence_fallback`; if evidence exists but
   scores tie, report `evidence_tie_break` rather than a unique quality
   ranking.
4. **P3.2 policy specialization** — compare Prompt-only strategy guidance with
   typed StrategyProfile payloads and SceneSnapshot-backed perception evidence.
   Report raw/normalized candidate fingerprints, rewrite convergence,
   perception-gate rejection, evidence score breakdown, and evidence-backed
   selected-agent distribution.
5. **P3.3 rolling replan** — compare full upfront compilation with fixed-topology
   ready-frontier compilation and post-verification replanning. Report Manager
   topology calls, frontier-scoped Policy calls, stale proposal rejection,
   horizon success decay, and recovery-free success. Adaptive topology edits are
   excluded and reserved for Phase 8.
6. **Phase 5 evidence evolution** — add isolated simulator workers, dynamic
   geometry/verifier evidence, OOD replay, and evidence calibration only after
   the preceding phases have locked their baseline artifacts. Any robot-action
   parallelism requires disjoint resources and an explicit graph join.

### P5.1 verifier evidence experiment

P5.1 records two evidence timings. Static VerifierEvidence is candidate and
source-scene bound and may be consumed by Arbiter before execution. Dynamic
evidence is converted from the post-execution VerificationResult; it must
record the candidate fingerprint, checked scene version, and execution
provenance, and cannot influence the candidate that has already run. A
pass_rate without evidence coverage is not treated as a successful verifier
measurement.

The empirical P5.1 gate is separate from the code gate. Run one CUDA
CUDA_VISIBLE_DEVICES=5 CAP-X/LIBERO smoke episode, then the matched seed
set. Every seed uses a new directory containing logs/, results/,
summary.json, summary.md, and manifest.json; logs must retain the full
run and redact provider secrets. The gate remains open until artifacts contain
both typed static verifier evidence and at least one post-execution dynamic
verifier result with valid scene/fingerprint provenance.

The 2026-07-31 condition-default follow-up uses the same artifact rule. Each
action candidate is enriched from registered typed skills before validation;
the selected candidate's static evidence should therefore include at least
one `scene_fresh(2000)` or uniquely resolved `track_exists:*` result. Static
coverage is reported separately from evaluator success and graph completion,
and an improved coverage measurement is not treated as a downstream success
rate improvement. The matched run must use a fresh directory for every seed
and retain the complete log, result, summary, manifest, and verifier evidence.

The fresh 2026-07-31 condition-default rerun used the non-privileged
`franka_libero_spatial_0.yaml` CAP-X backend, CUDA device 5, gpt-5.5, staged
`ready_wave`, two Policies (`balanced,safety`), fixed-graph execution, and
`geometry_mode=disabled`. The matched seeds 1--5 are stored under
`outputs/phase5/P5.1_condition_defaults_20260731/`, with one run-scoped
directory per invocation. The post-fix seed-1 smoke was used as the matched
seed-1 result; the earlier pre-refresh seed-1 failure remains separately
preserved as a diagnostic artifact.

| seed | evaluator success | graph completed | run outcome |
| --- | --- | --- | --- |
| 1 | true | true | normal success |
| 2 | false | false | normal execution, task failure |
| 3 | true | false | evaluator success, graph did not fully converge |
| 4 | unavailable | unavailable | both Policy requests hit the 60 s upstream LLM read timeout |
| 5 | false | false | normal execution, task failure |

The matched pilot therefore measured evaluator success `2/5`, graph
completion `1/5`, and four normal summary runs. Those four runs emitted eight
static candidate evidence records with positive coverage (`1.0` per selected
collection) and seven dynamic verifier records. Static `scene_fresh(...)`
pass rates remain below one because static evidence is captured before the
long LLM compilation interval; the pre-dispatch refresh fixes the execution
boundary but does not retroactively make the compile-time evidence fresh.
Seed 4 retained `failure.json`, `episode.failure.json`, `manifest.json`, and
the full log; it has no verifier artifact because no candidate reached
arbitration. This closes the empirical condition-default/evidence-publication
check, but does not establish a downstream success-rate improvement or close
the upstream LLM availability/latency risk.

The 2026-07-30 run used gpt-5.5 on `franka_libero_spatial_0` with two staged
Policy Agents. The seed-1 smoke and matched seeds 2--5 all produced complete
run directories and dynamic evidence with valid trace identity, scene version,
and effective-candidate fingerprint. Evaluator success was 2/5; graph
completion was 1/5. Static typed evidence was emitted in every run, but its
coverage was zero in all ten selected-candidate collections because the LLM
graphs did not expose a compile-time `track_exists:*` precondition. This is a
valid runtime/artifact integration result, but not closure of the meaningful
static-coverage gate.

### Phase 5 experiment artifact rule

Every Phase 5 experiment and every seed gets a new run-scoped directory. No
runner may reuse an output path or overwrite a prior log:

```text
outputs/phase5/<experiment_name>/<timestamp>_<run_id>/
  run_config.json  manifest.json  summary.json  summary.md
  logs/  results/  traces/  evidence/  artifacts/
```

The manifest records file size and SHA-256. Failed runs retain their failure
artifact and complete stdout/stderr log. API keys, Authorization headers, and
other provider secrets are redacted before publication. A run is incomplete
for comparison unless its configuration, logs, result, and manifest are all
present.

The Phase 5 baseline gate uses a five-seed pilot followed by 20 paired trials
per horizon bucket. The controlled horizon buckets are H2, H4, and H6 based
on verified subgoal count; native LIBERO tasks are a separate external
validity set. CAP-X, deterministic B3, and B3-LLM share the same initial
state seeds, model budget, retry/deadline budget, and action budget.

The P5.2 pilot is launched with:

```bash
CUDA_VISIBLE_DEVICES=5 python scripts/run_libero_p52_geometry.py \
  --config-path <cap-x-libero-yaml> \
  --server-url <endpoint> --model gpt-5.5 \
  --output-root outputs/phase5
```

It runs `geometry_disabled`, `geometry_shadow`, and
`geometry_online_bounded` for seeds 1 through 5. Every mode/seed pair gets an
independent artifact directory and `used_privileged_state=false` in its
published configuration.

The complete P3.2-to-Phase-5 handoff is in
[`phase5-evidence-evolution.md`](phase5-evidence-evolution.md). In particular,
the current `ProcessRehearsalPool` is an offline execution boundary, not proof
that online candidates already have rehearsal evidence.

The 2026-07-29 pre-transport P5.2 endpoint pilot completed all 15 isolated mode/seed runs.
`geometry_disabled`, `geometry_shadow`, and `geometry_online_bounded` each
obtained 2/5 CAP-X evaluator successes. The online provider emitted four
candidate geometry records per run with distinct normalized fingerprints,
`used_privileged_state=false`, and approximately 0.152--0.183 ms per-candidate
P95 latency. However, because that pilot predates live local-map transport, only
conservative reachability was measurable; clearance, collision risk, and grasp
quality stayed unknown. It is retained as a baseline/diagnostic artifact.

The subsequent single-run transport closure used
`--geometry-depth-subsample 16` and a fresh run-scoped directory. It produced
`map_version=4`, four candidate records, 23.83 ms maximum geometry latency,
candidate-specific clearance scores, and `evaluator_success=true`. This proves
the real geometry path is wired through CAP-X RGB-D without privileged state,
and the matched five-seed post-transport pilot is now complete at
`outputs/phase5/P5.2_geometry_evidence_posttransport_20260729/`:

| mode | evaluator successes | geometry records/run | map version | processed observations | observed geometry latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| `geometry_disabled` | 2/5 | 0 | 0 | 0 | n/a |
| `geometry_shadow` | 4/5 | 4 | 3--4 | 3--4 | 23.70--24.28 ms |
| `geometry_online_bounded` | 4/5 | 4 | 3--4 | 3--4 | 23.70--24.14 ms |

All 15 runs have complete logs, summaries, results, and manifests; all 75
manifest entries passed size/SHA-256 verification. The enabled modes produced
two distinct clearance scores per run (global range 0.3604--0.4895), but all
10 enabled action-subgoal arbitration decisions were still
`evidence_tie_break`. Consequently the 4/5 online result is an operational
and safety result, not yet evidence of a unique geometry-driven winner or a
causal downstream improvement. The P5.2 selection-quality and grasp-quality
gates remain open.

The P5.3 matched process gate was completed on 2026-07-30 at
`outputs/phase5/P5.3_process_rehearsal_matched_fix_20260730/`. Two candidates
were run against LIBERO Spatial-0 with the same five reset seeds and isolated
CAP-X workers. `policy-0:0` reached 0/5 evaluator successes and
`policy-1:safety:1` reached 2/5; the candidates differed on seeds 1 and 5,
and all remaining failures were classified as `postcondition_failure`. Each
seed retained its own log, result, summary, and SHA-256 manifest. This closes
the process rehearsal/candidate-discrimination gate, but does not claim an
online Arbiter improvement or a downstream success-rate gain.

The input artifact is a full MissionGraph and therefore uses a graph-scoped
fingerprint. The P5.3 identity closure now requires an explicit
`arbiter_subgraph_id` and carries the derived local subgraph fingerprint into
`RehearsalEvidence`; mismatched or unmapped evidence is rejected. The pure
`run_shadow_arbitration()` path reports whether mapped rehearsal evidence would
change the winner and records baseline/shadow selection bases, but it never
executes the hypothetical winner. Thus these artifacts can support a shadow
selection analysis without claiming an online Arbiter or downstream success
improvement.

The five-seed, one-task process result is still insufficient for the remaining
statistical gate: ten-plus seeds, multiple tasks, controlled physical online
selection, and the P5.4 evidence cache remain open.

P5.3.1 adds the controlled online selection driver at
`scripts/run_libero_p53_online.py`. It consumes the same graph-scoped
candidate artifact, runs one matched rehearsal batch, maps each result to the
local Arbiter identity, and executes only the selected live candidate. Every
run is stored under a new
`outputs/phase5/P5.3.1_online_rehearsal_arbiter/<timestamp>_<run_id>/`
directory with `run_config.json`, `results/rehearsal.json`,
`results/selection.json`, `logs/runner.log`, `summary.json`, `summary.md`,
and `manifest.json`. The selection artifact records baseline/evidence-aware/
live winners, selection bases, accepted evidence IDs, rejected evidence,
provider latency, fallback reason, and physical execution count.

The driver supports `disabled`, `shadow`, and `online_bounded`. The first two
are safe controls; `shadow` cannot promote its hypothetical winner. The
`online_bounded` mode is fail-closed and falls back to baseline when the
provider fails or the evidence-aware Arbiter has no winner. The focused code
gate and the matched one-task evaluation are now closed. The first five pairs
are at `outputs/phase5/P5.3.1_matched_spatial0_20260731/` and
`outputs/phase5/P5.3.1_matched_spatial0_seeds6_10_20260731/` retain seeds 1--10
with the same two graph candidates, config, object/target names, and reset
seeds for both modes on LIBERO spatial task 0. Both suites used
`max_workers=1`, `timeout_s=360`, and `CUDA_VISIBLE_DEVICES=5`:

| mode | evaluator successes | physical executions |
| --- | ---: | ---: |
| `disabled` baseline | 0/10 | 10/10 |
| `online_bounded` | 2/10 | 10/10 |

The matched delta is `+2/10` episodes. Both online successes (seeds 1 and 5)
used the evidence-aware `policy-1:safety:1` winner; the baseline selected
`policy-0:0` on all ten seeds. Evidence attached for both candidates in all
ten online runs, with zero identity/version rejections. The other eight online
decisions were `evidence_tie_break` and kept the baseline winner. This is a
controlled single-task evaluation showing that online evidence can alter the
physical choice and downstream outcome, not a statistically significant or
multi-task success-rate claim. Multiple tasks, larger seed sets, and
confidence intervals remain required.

P5.4 now has a process-local versioned cache implementation with exact
candidate/scene keys and observable hit, miss, stale-rejection, invalidation,
and eviction counters. The online seam is opt-in through
`select_with_rehearsal(..., evidence_cache=...)`, scheduler forwarding, and
`scripts/run_libero_p53_online.py --cache-mode enabled`; enabled runs persist
cache events and selection statistics. `--selection-repeats N` exercises
repeated same-scene arbitration while preserving one physical execution. The
focused runner test proves provider-call reduction and cache hits, but no real
CAP-X multi-seed artifact has yet established a multi-seed downstream or
latency claim. The single-episode smoke below covers the first repeated
selection comparison; a multi-seed extension remains open.

The first real runner-level smoke is retained under
`outputs/phase5/P5.4_online_cache_smoke_20260803_venv/` and its matched control
under `outputs/phase5/P5.4_online_cache_smoke_20260803_disabled_venv/`. On
Spatial-0 seed 1 with two repeated selections, enabled mode produced 2
rehearsal records, 1 provider call, and 2 cache hits, while disabled mode
produced 4 rehearsal records and 2 provider calls. Total selection latency was
approximately 341.4 s versus 683.8 s; both modes executed the physical
candidate once and both had evaluator success `false`. The provider call count
is repeated in `run_config.json`, `results/selection.json`, `summary.json`,
each `selection_history` entry, and `logs/runner.log`. This is a real
cache/latency smoke, not a downstream success claim.

The matched multi-seed extension is
`scripts/run_libero_p54_matched.py`. It creates
`outputs/phase5/P5.4_matched_online_cache/<suite>/pairs/<task>_seed<seed>/`
with independent `cache_disabled` and `cache_enabled` child runs. The two
lanes use the same candidate artifact hash, scene version, reset seed, and
`selection_repeats`, but never share a cache. The suite aggregate reports
provider-call reduction, cache hits, selection latency, physical execution
count, evaluator success, and pair status. It must be run on multiple seeds
and the locked task set before treating P5.4 as a multi-task empirical gate.

The first real matched run completed at
`outputs/phase5/P5.4_matched_online_cache_20260803/P5.4_matched_online_cache/20260803_054828_suite_d64ab784/`
using the CAP-X `.venv-libero` environment, CUDA device 5, LIBERO Spatial-0,
seeds 1--5, and two repeated selections per lane. All five pairs completed;
the control made 10 provider calls and the enabled lane made 5 with 10 cache
hits. Aggregate selection latency was 1792.70 s for the control and 899.77 s
with caching. Both lanes executed one physical candidate per seed and both
had evaluator success `2/5`. The selected candidate was identical within
every pair. This closes the single-task five-seed cache-efficiency check, but
does not establish a downstream success-rate improvement; multi-task and
larger-seed evaluation remain open.

Current evidence status: the P3.1b `max_workers=2` ready-wave path has a
successful endpoint-backed LIBERO run, but the matched `max_workers=1`
control is still required before claiming a parallel speedup. P3.1c has a
successful `balanced,safety` staged run with `evaluator_success=True`; its
artifact records the ordered strategy profiles and candidate selection basis.
The existing
`outputs/capmas_libero_b3_llm/staged_gpt55_p32_perception_retry2.json` is a
pre-closure historical artifact: it contains synthetic confidence and lacks
the new SceneSnapshot evidence provenance. It remains useful as a regression
record, but it does not satisfy the P3.2 closure evidence gate. A fresh
endpoint-backed multi-seed rerun is still required before making a P3.2
downstream quality claim; the code-level closure is covered by the local test
suite and compile checks.

An artifact is incomplete for latency or parallelism claims unless it contains
`run_config` and per-request `llm_calls`. A successful evaluator score alone is
not evidence of a multi-agent quality improvement.

### P5.5 frozen OOD replay implementation (2026-08-03)

The P5.5 runner is `scripts/run_libero_p55_ood.py`. It consumes the frozen
manifest `configs/phase5/p55_ood_smoke.json`, validates candidate SHA-256
digests and split/leakage constraints before any online call, and creates a
new suite and case directory for every run. The initial gate uses
`max_workers=1`, `max_restarts=0`, `cache_mode=disabled`, and at most one
physical execution per case:

```bash
CUDA_VISIBLE_DEVICES=5 \
/data/MLLM/wzh/agent/paper/infiAgent/workspace/cap-x/.venv-libero/bin/python \
scripts/run_libero_p55_ood.py \
  --manifest configs/phase5/p55_ood_smoke.json \
  --output-root outputs/phase5/P5.5_ood_replay_20260803 \
  --max-workers 1 --max-restarts 0 --gpu 5
```

This smoke is intentionally a structural gate. The layout-OOD entry is a
manual membership label and the fixture does not generate or apply a physical
layout perturbation. Results therefore validate replay isolation, explicit
pairing, provenance, shadow-only evidence, and retained logs/manifests; they
must not be reported as an OOD success-rate improvement. A real OOD pilot
requires physically distinct layout variants, matched seeds, multiple task or
layout families, Wilson intervals, paired deltas, and all infrastructure
unknowns reported separately.

The real smoke completed in the fresh directory
`outputs/phase5/P5.5_ood_replay_20260803_smoke1/`. It used the CAP-X
`.venv-libero` environment, CUDA device 5, one worker, zero restarts, and the
Spatial-0 seed-1 candidate artifact. Both cases completed the runner and
executed one physical candidate; ID and layout-OOD-labeled evaluator results
were both `0/1`, with paired tie `0/0`, `selection_basis=evidence_tie_break`,
and `shadow_only=true`.

This is deliberately not an OOD performance result: the fixture's OOD case
uses the same physical CAP-X config and candidate artifact as ID and carries a
manual placeholder layout label. It validates real replay execution,
pairing, provenance, leakage checks, and artifact retention only. A physical
The placeholder smoke was followed by the real layout-variant pilot below.

### P5.5 real layout-variant five-seed pilot (2026-08-04)

`scripts/create_p55_real_layout_manifest.py` froze three task families
(`spatial-0`, `goal-1`, and `object-6`), five matched seeds, and 15 ID/OOD
pairs. ID uses the native reset layout; OOD applies a deterministic free-joint
translation to the task objects and records the resulting state fingerprint.
The manifest is
`outputs/phase5/P5.5_real_layout_assets_20260803/p55_real_layout_3family_5seed.json`.

The formal run used CUDA device 5, the CAP-X `.venv-libero` environment,
`max_workers=2`, `max_restarts=0`, `max_steps=32`, and disabled evidence
cache. Its retained suite is
`outputs/phase5/P5.5_real_layout_pilot_20260803/P5.5_frozen_ood_replay/20260803_113118_suite_b4bbc31b/`.
All 30 cases completed without infrastructure failures and all case-level
logs and manifests passed digest/size verification.

| split | cases | evaluator success | graph completed | unknowns |
| --- | ---: | ---: | ---: | ---: |
| ID | 15 | 0/15 | 0/15 | 0 |
| real layout OOD | 15 | 0/15 | 0/15 | 0 |

The aggregate report gives ID/OOD Wilson intervals of `0.0` with upper
bound `0.2039`, paired delta `0`, 15 paired ties, and exact McNemar `p=1.0`.
All failures are classified as `task_failure`; no recovery or human
intervention was recorded. All 15 pairs have different layout fingerprints,
and all 30 evidence records are `shadow_only=true`. Selection used
`evidence_tie_break` in every case, so this pilot closes the real layout,
pairing, provenance, and measurement gate but does not demonstrate a
downstream success-rate improvement or causal Arbiter gain. The ten-seed
formal gate is recorded below; P5.6 calibration remains open.

### P5.5 real layout-variant ten-seed formal gate (2026-08-05)

The formal run expanded the frozen manifest to seeds 1--10 for the same
`spatial-0`, `goal-1`, and `object-6` families. It contains 60 cases and 30
matched ID/OOD pairs. The manifest is
`outputs/phase5/P5.5_real_layout_assets_20260803/p55_real_layout_3family_10seed.json`
with canonical manifest digest (SHA-256 excluding the self-digest field)
`5aeff85dae764c72fe9c0b1f3a0a07f4070e95247baea1b8d93f15311ea72141`.
The retained suite is
`outputs/phase5/P5.5_real_layout_formal_20260804/P5.5_frozen_ood_replay/20260804_014522_suite_dda9defe/`.

The run used the CAP-X `.venv-libero` interpreter, CUDA device 5,
`max_workers=2`, `max_restarts=0`, `max_steps=32`, `timeout_s=360`, one
selection repeat, and disabled evidence cache. All 60 cases completed with
zero infrastructure unknowns, and each case retained its own log, evidence,
manifest, and result boundary.

The zero-unknown aggregate above is a historical result from the
pre-diagnostic runner and must not be interpreted as a valid task-success
measurement. Its case-level rehearsal artifacts contain 120 candidate
attempts: 117 failed at depth initialization with
`LIBERO depth initialization failed`, while three entered the first skill
path and failed there. Because two spawned workers shared CUDA device 5,
renderer/model initialization was not isolated. The old physical result also
omitted an explicit failure class, so an incomplete graph was incorrectly
reported as `task_failure`.

| family | ID success | OOD success | pairs | mean latency |
| --- | ---: | ---: | ---: | ---: |
| `spatial-0` | 0/10 | 0/10 | 10 | 74.99 s |
| `goal-1` | 0/10 | 0/10 | 10 | 78.34 s |
| `object-6` | 0/10 | 0/10 | 10 | 82.86 s |
| **all** | **0/30** | **0/30** | **30** | **78.73 s** |

Both success estimates are `0.0` with Wilson 95% upper bound `0.1135`.
There were 30 paired ties, paired delta `0`, and exact McNemar `p=1.0`.
All 60 failures were `task_failure`; infrastructure unknowns, recoveries,
and human interventions were all zero. Reported latency had median
`72.68 s` and range `55.18--211.06 s`.

Horizon buckets are not reported for this run: the current P5.5 evidence
artifact does not record realized subgoal count, and `max_steps=32` is a
budget rather than an observed horizon. Since every case had
`graph_completed=false`, assigning H2/H4/H6 would be invalid. The runner
needs an explicit verified-horizon field before horizon-stability analysis.

All 30 pairs had distinct ID/OOD layout fingerprints and all case manifests
passed SHA-256/size verification. Every evidence record remained
`shadow_only=true`, and all 60 selections used `evidence_tie_break`. Thus
this closes the P5.5 ten-seed measurement/provenance gate only. It does not
support an OOD generalization, downstream success-rate, or causal Arbiter
claim; calibration and active weighting move to P5.6.

### P5.5 failure-diagnostics correction (2026-08-05)

The corrected runner enforces `max_workers=1` for a configured GPU. Rehearsal
exceptions matching CAP-X depth initialization are classified as
`reset_failure`; other isolated-process failures retain `worker_crash`, and
physical `GraphExecutionResult.failure` metadata is preserved in
`results/online.json`. The OOD adapter excludes `reset_failure`,
`worker_crash`, `timeout`, and `infrastructure_unknown` from success-rate
denominators. An incomplete physical graph without an explicit task failure is
also `infrastructure_unknown`, while explicit `EXECUTION_ERROR`,
`MOTION_TIMEOUT`, and `POSTCONDITION_FAILED` remain valid task-failure
evidence.

Each completed case now retains
`evidence/rehearsal_failure_summary.json`, including candidate IDs, failure
classes, reasons, and failure steps. The 2026-08-04 two-worker formal suite
is diagnostic-only and is not eligible for an ID/OOD success claim. A fresh
CUDA-5, single-worker matched smoke is required before expanding the run to
five or ten seeds.

The corrected single-worker matched smoke completed at
`outputs/phase5/P5.5_failure_diag_smoke_20260805/P5.5_frozen_ood_replay/20260805_032606_suite_d733ad12/`.
It used the three-family, one-seed real-layout manifest on CUDA device 5 with
`max_workers=1`, `max_restarts=0`, and disabled cache. All six cases completed
with zero runner failures and zero infrastructure unknowns. The 12 rehearsal
candidate attempts were classified as `skill_failure` rather than depth
initialization failure; all six physical executions retained explicit
`EXECUTION_ERROR` metadata.

| split | evaluator success | known failures | infrastructure unknowns |
| --- | ---: | ---: | ---: |
| ID | 0/3 | 3 | 0 |
| real layout OOD | 0/3 | 3 | 0 |

This confirms the single-GPU isolation and provenance fixes. It does not claim
success-rate improvement: the smoke has one seed, all physical candidates
failed during execution, and selection remained `evidence_tie_break`. A new
five-seed run is allowed by the infrastructure gate, but its task performance
must be reported independently from the invalidated two-worker formal suite.

### P5.5 execution-grounding smoke (2026-08-05)

After the failure-diagnostics smoke, the live executor and isolated rehearsal
worker were changed to call `ground_libero_mission_graph()` after the
post-reset `SceneSnapshot` is available. This rebases scene-dependent LIBERO
target poses without changing candidate fingerprints. The six-case smoke used
the real-layout three-family, one-seed manifest
`outputs/phase5/P5.5_real_layout_assets_20260803/p55_real_layout_3family_1seed.json`
on CUDA device 5, the CAP-X Python 3.10 environment, `max_workers=1`,
`max_restarts=0`, `max_steps=32`, and disabled cache. Its retained suite is
`outputs/phase5/P5.5_grounding_smoke_venv_20260805/P5.5_frozen_ood_replay/20260805_085352_suite_fd89ecef/`.

| split | cases | evaluator success | infrastructure unknowns |
| --- | ---: | ---: | ---: |
| ID | 3 | 1/3 | 0 |
| real layout OOD | 3 | 0/3 | 0 |

The ID success is the native `spatial-0` case. The five failures are explicit
`POSTCONDITION_FAILED` task failures, not reset, renderer, or worker failures.
The paired result is one ID-only success and two ties; the Wilson estimates are
`0.3333` (95% CI `[0.0615, 0.7923]`) for ID and `0.0` (95% CI `[0.0, 0.5615]`)
for OOD. Selection used `evidence_score` once and `evidence_tie_break` five
times. This is a grounding/regression smoke, not a multi-seed quality result.

The execution trace confirms that grounding reached the physical action path:
for spatial-0, placement `goto_pose` moved from native `x=0.72409` to OOD
`x=0.81099`, while the layout report recorded the same positive object
translation. Remaining failures are candidate/task-specific: OOD spatial
failed `object_at_target` after release, both goal cases failed
`object_in_gripper`, and both object cases exposed a `gripper_closed`
postcondition failure. These failures are the next grasp/coordinate and task
mapping investigation; they are not evidence that execution grounding was
ignored. Full regression verification remains `421 passed` plus compileall.

## 6. Failure taxonomy

Every failed episode should be assigned one primary cause and optional secondary causes: stale state, perception uncertainty, invalid contract, motion/planning failure, execution error, postcondition failure, recovery failure, budget timeout, or evaluator failure.

## 7. Artifact parity boundary

CAP-X writes trial directories whose name encodes `sandboxrc`, reward, and
`taskcompleted`, with a `summary.txt` payload. CAP-MAS writes structured JSON
episode artifacts. `capmas.evaluation.parity` normalizes both formats without
starting a backend and requires `task_id` and `seed` to be supplied by the
experiment driver. `scripts/compare_artifacts.py` produces one matched record;
batch execution and statistical aggregation remain outside this read-only
normalizer.

### P5.4 isolated evidence-cache evaluation (2026-07-31)

The run-scoped driver is `scripts/run_p54_evidence_cache.py`. It executes the
same deterministic eleven-operation trace in two independent lanes:
`cache_disabled` calls the local evidence provider for every logical query,
while `cache_enabled` uses `VersionedEvidenceCache` keyed by candidate
fingerprint and scene version. Each lane writes its own directory under
`outputs/phase5/P5.4_cache_evaluation/` with trace, summary, log, and manifest
artifacts. Failure runs retain partial trace and redacted error artifacts.

The isolated evaluation is designed to measure exact hits, scene-version
invalidation, stale rejection, and provider-call reduction. It does not start
CAP-X, call an LLM, execute LIBERO, or establish a downstream robot success
improvement. The seed-1 run completed at
`outputs/phase5/P5.4_cache_evaluation/20260731_112755_cache_disabled_seed1/`
and
`outputs/phase5/P5.4_cache_evaluation/20260731_112755_cache_enabled_seed1/`.
The disabled lane made 9 provider calls; the enabled lane made 5, a 44.44%
reduction, with 3 exact hits, 5 stores, 2 invalidations, 1 stale rejection,
and 0 stale attachments. The enabled cache ended at scene version 2 with
three current entries. Both manifests passed SHA-256 verification. This is an
isolated cache-contract result and does not establish downstream task success.

### P5.5 gripper-state semantic correction (2026-08-05)

The object-6 failure was traced to two different gripper signals being treated
as one. CAP-X's measured finger opening remained about `0.486` while a held
butter object prevented the fingers from closing further, but CAP-X's target
command was `0.0`. `CAPXObservationProvider` now carries the target as
`gripper_commanded_fraction`, the YAML factory reads it from the low-level
environment, and `PredicateBasedVerifier` uses it for open/closed predicates
with a legacy opening fallback.

The independent grounded probe was retained at
`outputs/phase5/P5.5_grasp_probe_object6_commanded_20260805/20260805_101645_c98dd434/`.
It physically lifted butter (`z=0.0087` to `z=0.1234`) and reported both
`object_in_gripper(butter)` and `gripper_closed()` as passed. It did not run
placement, so `task_completed=false` is expected. This is a pick-checkpoint
semantic regression result, not a full P5.5 success-rate measurement.

### P5.5 target-pose verified object-6 online closure (2026-08-06)

The placement fix was validated first in a disabled-mode physical run and then
through the full `online_bounded` rehearsal-Arbiter lane. The retained online
run is
`outputs/phase5/P5.5_target_pose_verified_object6_online_20260806/P5.3.1_online_rehearsal_arbiter/20260806_052654_seed1_d1b5f0d1/`.
It used the real CAP-X object-6 configuration, CUDA device 5,
`max_workers=1`, `max_restarts=0`, `max_steps=32`, one selection repeat, and
disabled evidence cache. Both candidates completed isolated rehearsal
successfully (`122301.288867 ms` and `91026.945827 ms`); the provider call
took `215962.848232 ms` and the single physical execution was reserved for
the selected winner.

The online Arbiter attached both candidates and selected
`sg_pick_butter:policy-0:0` with `selection_basis=evidence_tie_break`.
`baseline_selection_basis` remained `confidence_fallback`, the winner did
not change, and `would_change_selection=false`; this verifies the
evidence-aware online path and execution gating, but does not claim that the
Arbiter causally preferred one candidate over another.

The physical result closed end to end: `completed=true`,
`evaluator_success=true`, `success=true`, no failure artifact, and two
completed action traces. The placement trace contains pre-place approach,
target descent, gripper release, and retreat. The earlier online failure was
not a motion/evaluator failure: the basket was partially occluded, its
semantic body-center pose was used as the placement target, and the verifier
therefore measured `object_at_target` distance `0.1369 m` after the physical
evaluator had already succeeded. The current path uses a clipped point-cloud
placement center, top release clearance, robust XY with semantic Z, and emits
`placement_pose_wxyz_xyz` in scene diagnostics. This closes the object-6
target-pose regression and full online smoke, but not a multi-seed or
multi-family P5.5 success-rate gate. The original artifact predates placement
provenance in `_scene_debug_payload`; the field was absent rather than a
confirmed provider-side `null`.

The observability follow-up is now implemented in code. Every attempted target
placement estimate records `placement_pose_source` and
`placement_pose_reason`: successful point-cloud estimates use
`geometry_pointcloud`; unavailable or invalid geometry uses
`semantic_pose_fallback` with the exception or
`invalid_or_insufficient_pointcloud`. Physical fallback behavior is unchanged.
A fresh real capture is still required to identify the reason emitted by the
object-6 provider and to verify the new artifact fields outside unit tests.

That real capture completed at
`outputs/phase5/P5.5_placement_provenance_object6_20260806/P5.3.1_online_rehearsal_arbiter/20260806_064328_seed1_5168c2e2/`.
It used disabled rehearsal mode to isolate the physical CAP-X observation and
execution path. Before execution, basket placement provenance was
`geometry_pointcloud` with pose
`[0,1,0,0,0.60293,0.24965,0.12755]`; after execution it remained
`geometry_pointcloud` with pose
`[0,1,0,0,0.62065,0.25059,0.13144]`. Both reasons were `null`. The physical
boundary again reported `completed=true`, `evaluator_success=true`, and
`success=true`. This closes the real placement-provenance capture gate; the
fallback reason paths remain covered by deterministic tests.

### P5.5 matched provenance five-seed pilot (2026-08-06)

The post-fix matched suite is retained at
`outputs/phase5/P5.5_matched_provenance_5seed_20260806/P5.5_frozen_ood_replay/20260806_091429_suite_e169a480/`.
It used CUDA device 5, one rehearsal worker, three task families, five paired
seeds per family, and a frozen manifest with SHA-256
`cb106ca9ccc57785a38cd0c08f3d59f447bb879bc39a4b6d66db9efe9d1f320f`.
All 30 cases completed with no failure artifact or infrastructure-unknown
record, and all target tracks recorded
`placement_pose_source=geometry_pointcloud`.

| split | evaluator success | graph completion | verifier success |
| --- | ---: | ---: | ---: |
| ID | 3/15 | 2/15 | 2/15 |
| layout OOD | 5/15 | 4/15 | 4/15 |

ID evaluator success was 20.0% with Wilson 95% CI `[0.0705, 0.4519]`; OOD
success was 33.3% with CI `[0.1518, 0.5829]`. The paired table contains zero
ID-only successes, two OOD-only successes, and 13 ties. The estimated
`ID - OOD` gap is `-0.1333` with bootstrap CI `[-0.3333, 0]`, and exact
McNemar `p=0.5`. This pilot therefore does not establish an OOD improvement.
Per-family evaluator results were `0/5` versus `0/5` for spatial-0, `0/5`
versus `0/5` for goal-1, and `3/5` versus `5/5` for object-6.

A reporting audit found 24 raw `POSTCONDITION_FAILED` graph outcomes but only
22 physical task failures. The remaining two records,
`id-object-6-seed4` and `ood-object-6-seed2`, had
`evaluator_success=true` while `object_at_target(butter,basket)` rejected
distances of 0.0928 m and 0.0609 m. They are 2 verifier false negatives, not
downstream task failures. The offline correction, which did not rerun the
robot or modify the source suite, is retained at
`outputs/phase5/P5.5_matched_provenance_5seed_report_correction_20260807/P5.5_offline_reaggregation/20260807_013832_suite_e169a480/`.

Selection used `evidence_tie_break` 28 times and `evidence_score` twice, with
no `confidence_fallback`. This is a valid five-seed pilot, not the formal P5.5
gate: closure still requires at least ten paired seeds across all three
families using the corrected single-worker protocol.

### P5.5 corrected matched-provenance ten-seed formal gate (2026-08-07)

The corrected formal run is retained at
`outputs/phase5/P5.5_matched_provenance_10seed_retry2_20260807/P5.5_frozen_ood_replay/20260807_024842_suite_20674432/`.
It used CUDA device 5, one worker, no restarts, `max_steps=32`,
`timeout_s=360`, one selection repeat, disabled cache, and frozen manifest
SHA-256
`5aeff85dae764c72fe9c0b1f3a0a07f4070e95247baea1b8d93f15311ea72141`.
All 60 cases and 30 ID/OOD pairs completed with zero case-level failures and
zero infrastructure unknowns.

| family | ID evaluator | OOD evaluator | ID graph/verifier | OOD graph/verifier | mean latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| `spatial-0` | 0/10 | 0/10 | 0/10 | 0/10 | 190.98 s |
| `goal-1` | 0/10 | 0/10 | 0/10 | 0/10 | 165.10 s |
| `object-6` | 4/10 | 10/10 | 4/10 | 8/10 | 223.35 s |
| **all** | **4/30** | **10/30** | **4/30** | **8/30** | **193.15 s** |

The ID Wilson estimate was `0.1333` with 95% CI `[0.0531, 0.2968]`; OOD was
`0.3333` with CI `[0.1923, 0.5122]`. The paired `ID - OOD` gap was `-0.2`
with bootstrap CI `[-0.3667, -0.0667]`. There were zero ID-only successes,
six OOD-only successes, and 24 ties, giving exact McNemar `p=0.03125`.
Overall latency had median `175.14 s` and range `137.98--323.84 s`; ID and OOD
mean latency was `172.29 s` and `214.00 s`, respectively. Recovery and human
intervention counts were zero for every case. Each case made one provider call
and no case used a cache hit.

There were 48 graph-level `POSTCONDITION_FAILED` outcomes. Evaluator-based
reclassification gives 46 physical task failures and two verifier false
negatives: `ood-object-6-seed1` and `ood-object-6-seed2`. Both false negatives
passed the LIBERO evaluator but failed CAP-MAS `object_at_target` verification.
Selection used `evidence_tie_break` 57 times and `evidence_score` three times,
with no `confidence_fallback`.

All 30 pairs had distinct, non-null ID/OOD layout fingerprints and all 60
records remained shadow-only. Independent reaggregation exactly matched the
retained aggregate. The suite manifest covered 907 files, and its entries plus
all 60 case manifests passed size and SHA-256 checks; secret-pattern checks
found no API key or Authorization value.

This closes the corrected P5.5 measurement, pairing, provenance, and artifact
retention gate. It does not demonstrate general OOD robustness or causal
Arbiter improvement: the success difference is entirely from the `object-6`
layout, two families stayed at zero, P5.5 evidence was shadow-only, and 57/60
selections were ties. Realized horizon was not recorded, so no horizon bucket
or stability claim is reported; adding that field and calibrated active
evidence weighting is the P5.6 handoff.

### P5.6A data foundation fixed-block collection (2026-08-19)

P5.6A records `p56.feature.v1` decision-time feature snapshots, typed physical
outcomes, lineage, and planned/realized horizon before any calibration fit.
`max_steps=32 is not a horizon`; it is a fixed executor budget. The read-only
capability diagnosis at
`outputs/phase5/P5.6.0_capability_diagnosis/20260813_072308_capability_c9df3f4b/`
found 10/10 physical execution reach and zero infrastructure unknowns for each
family, but evaluator success was `0/10` for `spatial-0`, `0/10` for `goal-1`,
and `4/10` for `object-6`. `spatial-0` and `goal-1` are deferred to the
separate `P5.3.2 Task-Family Capability Repair` work package.

The history audit
`outputs/phase5/P5.6.2a_object6_history_audit/20260813_103434_history_a6bc49b1/`
accepted zero legacy rows. Consequently, immutable object-6 ID seed blocks
11-20 and 21-30 ran at
`outputs/phase5/P5.6.2a_object6_collection/20260818_090102_suite_63248cf1/`
and
`outputs/phase5/P5.6.2a_object6_collection/20260818_095350_suite_50dc9bd3/`.
Both completed 10/10 cases with no case-level infrastructure failure, each
yielding 5 positive and 5 negative Tier A labels. The combined result is
20 Tier A outcomes, 10 positive and 10 negative, satisfying the pre-registered
20/5/5 collection gate without outcome-adaptive seed selection.
The capability diagnosis, history audit, and both collection suites each retain
`results/manifest_verification.json`; their regenerated manifests report zero
missing, size, SHA-256, and untracked-file mismatches.

This is a data-coverage gate only. It does not fit a model, emit calibrated
probabilities, change Arbiter ranking, or demonstrate a downstream success-rate
gain. P5.6B/C calibration, shadow arbitration, and canary evaluation remain
open.

### P5.6C fit stability and real offline calibration (2026-08-19)

P5.6C adds train-only design diagnostics, constant-column freezing,
availability-pattern abstention, and projected-KKT convergence reporting to
the offline constrained-logistic primitive. The verified synthetic fixture has
six train rows and a rank-two design matrix after the constant columns are
identified. It is an implementation regression test, not a physical
calibration experiment.

The real-data offline run is
`outputs/phase5/P5.6.4_offline_calibration/20260819_012907_p56b-object6-offline/`.
Its manifest verifies cleanly. The locked split is 12/4/4; the V2 model
converges at iteration 3,730 with projected-KKT norm `9.9696e-9`. Its first
held-out report had Brier `0.1111` and ECE `0.1667`, so ECE missed the
predeclared `<= 0.10` target. The four-row calibration split also gives wide
PAVA Wilson widths.

The corrected baseline comparison is retained at
`outputs/phase5/P5.6.4_offline_calibration/20260819_014928_p56b-object6-offline-baseline-v2/`.
It pairs predictions to test rows in lineage order, avoiding repeated
`candidate_id` overwrite. The calibrated model scores Brier `0.02778` and
ECE `0.08333`; the frozen fixed-weight/PAVA baseline scores Brier `0.00826`
and ECE `0.04545`. Consequently calibrated Brier improvement is `-2.3611`
and `offline_qualification_passed=false`: the ECE gate passes, but the
baseline-relative Brier gate fails. Both reports remain offline-only; there
is no `CalibrationSnapshot`, active probability, Arbiter selection change, or
physical Executor effect. Shadow/canary gates remain blocked.

### P5.6D same-runtime evidence transport gate (2026-08-19)

The completed seed-11--30 collection is retained for audit, not reused as a
P5.6D feature source: its synthetic decision scene made perception, verifier,
and geometry unavailable. The new collection path owns one CAP-X/LIBERO
runtime from reset through a single selected execution. It commits a real
version-one scene before arbitration, captures candidate-bound evidence before
the decision boundary, merges only isolated rehearsal evidence, and records
`evidence_mode=same_runtime` in the immutable suite/case artifacts. Geometry
or World Model failure remains `unknown` rather than becoming a negative
score.

This is a transport and provenance correction, not a new calibration result.
The CLI defaults to `same_runtime`; `rehearsal_only` is historical/test
compatibility only. A fresh pre-registered seed manifest and a verified
GPU-5 smoke are required before any new 20/5/5 collection or offline fit.

The real seed-31 retry at
`outputs/phase5/P5.6D_same_runtime_collection/P5.6.2a_object6_collection/20260819_030619_suite_6d229e5c/`
verified its run manifest and completed the same-runtime path: a real
`decision_scene_version=1`, two decision-time snapshots, and exactly one
physical submission. The selected graph reached a failed placement freshness
checkpoint (`POSTCONDITION_FAILED`, evaluator false), so it demonstrates
transport/provenance only, not downstream task success. Perception, geometry,
and rehearsal values were present; static verifier values correctly remained
unknown where no compile-time predicate was available.

P5.6D smoke manifests now use signed `p56.collection.v2`
`collection_purpose="transport_smoke"`. The eligibility summarizer preserves
their physical diagnostics but excludes them from the admissible Tier-A count;
only a new signed `qualification` collection may be used for the 20/5/5 gate.

The qualification collection is pre-registered before execution in
`configs/phase5/p56d_object6_id_seeds_32_51.json`: 20 native object-6 ID
cases, seeds 32--51, `p56.collection.v2`, purpose `qualification`, and
SHA-256 `225e94ac02117104ce9fa6fa4db4433eed42fb5524c3771504d7d0ca8af67c06`.
The runner fails closed if that manifest is asked to use `rehearsal_only`.
The block will run serially on GPU 5 with one retained same-runtime session
per case; its aggregate 20/5/5 status is intentionally not inferred before
the fixed block completes.
