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
gate and one endpoint-backed smoke are now closed. The retained smoke under
`outputs/phase5/P5.3.1_real_smoke_20260731_cleanfix/` used two candidates on
LIBERO spatial task 0. Rehearsal evidence changed the winner from the
baseline `confidence_fallback` policy-0 to policy-1 with `evidence_score`; the
single physical execution completed successfully and passed the CAP-X
evaluator. Ten-plus seeds, multiple tasks, a matched baseline physical
control, and downstream causal comparison are still required.

P5.4 now has a process-local versioned cache implementation with exact
candidate/scene keys and observable hit, miss, stale-rejection, invalidation,
and eviction counters. No experiment artifact has yet enabled the cache or
reported a hit-rate/downstream comparison; such a result must use a new
run-scoped directory and must compare cache-disabled, cache-shadow, and any
later explicitly enabled online mode under matched seeds.

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
