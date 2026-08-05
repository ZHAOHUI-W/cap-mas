# P5.5 Frozen OOD Replay Design

## 1. Purpose

P5.5 adds a reproducible, evaluation-only out-of-distribution (OOD) replay
lane for CAP-MAS. It measures how a frozen CAP-X-compatible agent behaves on
held-out layouts, task/object families, and language variants without
training on those cases or allowing their outcomes to modify active runtime
state.

P5.5 is an evaluation protocol, not an OOD-trained model. The LLM, perception
models, CAP-X skill registry, verifier, Memory Bank, and Robot Skill Registry
remain frozen for a run. An `OODReplayEvidence` record describes performance
observed during frozen replay; it is not a calibrated Arbiter weight and must
not promote a different physical candidate in the P5.5 primary lane.

## 2. Scope and non-goals

### In scope

- Immutable ID/OOD case manifests with provenance and split audits.
- Reuse of the existing CAP-X/LIBERO online execution seam.
- Paired ID/OOD replay under matched model, action, interaction, and time
  budgets.
- Candidate-level replay results, failure taxonomy, latency, recovery, and
  intervention metrics.
- Confidence intervals and paired ID/OOD gap reports.
- Shadow-only OOD evidence for later P5.6 calibration.
- Independent run directories, logs, manifests, and failure artifacts.

### Non-goals

- Training or fine-tuning a separate OOD model.
- Modifying Policy prompts from OOD outcomes during evaluation.
- Updating Memory Skills, Robot Skills, or the active Arbiter online.
- Cross-case cache reuse or replay-generated cross-case memory retrieval.
- TSDF implementation, semantic adapter implementation, or adaptive topology.
- Claiming a downstream success-rate improvement from OOD evidence alone.

## 3. Terminology

- **ID case:** a case drawn from the locked in-distribution task/layout
  families.
- **OOD case:** a frozen case whose declared task, object, layout, or language
  factor is outside the corresponding ID split.
- **OOD model:** not a P5.5 component. P5.5 evaluates the same frozen model
  stack on a different case distribution.
- **Replay:** executing the frozen candidate/runtime protocol against one
  manifest case and recording all outputs.
- **Primary lane:** the lane whose physical winner is selected using the
  pre-P5.5 policy and evidence configuration.
- **Shadow OOD evidence:** OOD replay information recorded for analysis but
  prohibited from changing the primary physical winner.
- **OOD gap:** `ID success rate - OOD success rate`, reported with uncertainty.

## 4. Evaluation conditions

The final matched evaluation should contain these conditions:

1. CAP-X single-agent baseline.
2. CAP-MAS fixed-graph baseline without OOD evidence promotion.
3. CAP-MAS with OOD evidence computed in shadow mode.

The first implementation may run CAP-MAS only for a smoke test, but the
formal comparison must include CAP-X because the research charter requires
identical task splits and initial-state budgets for CAP-X and CAP-MAS.

All paired conditions must use the same:

- model family, model version, temperature, and prompt version;
- task/case manifest and reset seed;
- candidate artifact or task-specific candidate generation protocol;
- model-call and token budget;
- action, retry, recovery, and timeout budget;
- CAP-X environment/configuration version;
- physical execution ownership and maximum execution count.

For layout OOD, the candidate artifact and task goal remain fixed while the
initial layout changes. For task/object OOD, the task-specific candidate may
be generated with the same frozen model and prompt version, but generation
must be counted and cannot use any OOD outcome. Instruction OOD is reported
separately from geometric/task OOD.

## 5. Frozen case manifest

The manifest is the only source of evaluation membership. A case is immutable
after the suite is created.

```python
@dataclass(frozen=True)
class OODCase:
    case_id: str
    split: Literal["id", "ood"]
    ood_type: Literal["none", "layout", "task_object", "instruction"]
    task_id: str
    task_goal: str
    task_family: str
    layout_family: str
    object_name: str
    target_name: str
    seed: int
    pair_id: str
    config_path: str
    candidate_artifact: str
    candidate_artifact_sha256: str
    environment_version: str
    generator_version: str
    parent_case_id: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class OODSplitManifest:
    suite_id: str
    manifest_version: str
    cases: tuple[OODCase, ...]
    id_task_families: tuple[str, ...]
    ood_task_families: tuple[str, ...]
    id_layout_families: tuple[str, ...]
    ood_layout_families: tuple[str, ...]
    memory_snapshot_version: str
    robot_skill_snapshot_version: str
    prompt_version: str
    code_revision: str
    created_at_utc: str
    manifest_sha256: str
```

`manifest_sha256` is the SHA-256 digest of the canonical manifest with the
`manifest_sha256` field omitted. It is not a digest of a self-referential
serialized object.

Validation must reject:

- duplicate `case_id` values;
- an ID case with a non-`none` OOD type;
- an OOD case with `ood_type="none"`;
- duplicate `(task_id, seed, split)` entries;
- missing candidate artifact or digest mismatch;
- a `task_object` OOD case whose `task_family` is present in the ID task-family
  set;
- a `layout` OOD case whose `layout_family` is present in the ID layout-family
  set;
- a `parent_case_id` that is not present in the same manifest;
- an OOD case whose parent does not have the same declared `task_goal` when
  the OOD type is `layout`;
- a layout OOD case whose candidate digest differs from its parent case.

The manifest digest, candidate digest, environment version, code revision,
and prompt version must be copied into every case run configuration. A frozen
baseline Memory Bank snapshot may be shared by cases, but replay-generated
memory records may not be read by another case during the suite.

## 6. Leakage audit

Each suite runs a deterministic `LeakageAudit` before replay. It checks:

```python
@dataclass(frozen=True)
class LeakageAudit:
    passed: bool
    duplicate_case_ids: tuple[str, ...]
    id_ood_family_overlap: tuple[str, ...]
    candidate_digest_mismatches: tuple[str, ...]
    forbidden_memory_versions: tuple[str, ...]
    forbidden_skill_versions: tuple[str, ...]
    cross_case_cache_keys: tuple[str, ...]
    details: tuple[str, ...]
```

The audit fails closed. Replay cannot start when it finds a split overlap,
candidate digest mismatch, non-frozen Memory/Robot Skill version, or cache key
that was created by another case. OOD failure outcomes are written to the
case artifact only; they are never appended to active Memory Bank, hard-case
promotion input, prompt examples, or skill registries during the suite.

## 7. Replay data flow

```text
OODSplitManifest
        |
        v
LeakageAudit --fail--> failure artifact, no replay
        |
        v
case-scoped CAP-X/LIBERO runner
        |
        +--> frozen observation / candidate / execution trace
        +--> primary selection and physical result
        +--> shadow OODReplayEvidence
        |
        v
case result + manifest + statistics
        |
        v
paired ID/OOD report
```

The runner must reuse `run_online_experiment` and the CAP-X factory rather
than introduce a second physical execution path. Each case receives an
independent artifact directory and cache namespace. The primary lane uses
`max_workers=1` for the first correctness gate and executes at most one
physical candidate. Parallel fan-out is a later performance ablation.

The P5.5 primary lane uses the pre-P5.5 Arbiter policy. OOD replay results may
be attached to a shadow report, but they cannot change the primary winner,
retry policy, prompt, Memory Bank, or Robot Skill Registry during the same
case.

## 8. Result contracts

```python
@dataclass(frozen=True)
class OODReplayEvidence:
    case_id: str
    pair_id: str
    condition: Literal["capx", "capmas"]
    candidate_id: str
    split: Literal["id", "ood"]
    ood_type: Literal["none", "layout", "task_object", "instruction"]
    source_scene_version: int
    candidate_fingerprint: str
    evaluator_success: bool | None
    verifier_success: bool | None
    graph_completed: bool
    failure_class: str | None
    recovery_count: int
    human_intervention_count: int
    latency_ms: float
    provider_call_count: int
    cache_hit_count: int
    selection_basis: str | None = None
    shadow_only: bool = True
```

`shadow_only` must be true for every P5.5 evidence record. The record is
invalid if its candidate fingerprint or scene version does not match the
case request. Unknown verifier outcomes remain `None`; they are not converted
to failure or zero confidence.

## 9. Metrics and statistics

Every report must include:

- ID and OOD evaluator success rate;
- ID and OOD observable verifier success rate;
- OOD gap with a Wilson or Jeffreys interval;
- paired success delta and paired bootstrap or exact McNemar result;
- success by horizon bucket;
- candidate validity/rejection and selection-basis distribution;
- recovery and human intervention counts;
- LLM, perception, rehearsal, and total latency;
- OOD evidence accept/reject/timeout counts;
- cache hits, misses, and stale rejections;
- failure-class histogram.

The report must distinguish evaluator success, observable verifier success,
graph completion, and infrastructure completion. A task that times out before
arbitration is not silently counted as an ordinary policy failure.

Recommended experiment sizes are:

- smoke: one paired ID/OOD case;
- pilot: five paired seeds;
- formal gate: at least ten paired seeds across at least three task/layout
  families.

The sample-size recommendation is an evaluation gate, not a claim that ten
seeds provide strong significance for every effect. Confidence intervals and
paired outcomes must be shown even when the sample is small.

## 10. Artifact layout and retention

Every case and suite gets a new run-scoped directory:

```text
outputs/phase5/P5.5_ood_replay/<suite_id>/
  run_config.json
  manifest.json
  summary.json
  summary.md
  logs/
  results/
  traces/
  evidence/
  artifacts/
  cases/<case_id>/...
```

Failed cases retain `failure.json`, partial traces, logs, configuration, and
completed manifests. API keys and Authorization headers must never be written
to any artifact.

## 11. Acceptance gates

P5.5 code closure requires:

1. Manifest validation rejects all listed duplicate, overlap, and digest
   violations.
2. Leakage audit fails closed before replay.
3. Case replay produces independent, versioned, manifest-covered artifacts.
4. OOD evidence is shadow-only and cannot change the primary physical winner.
5. ID/OOD reports contain success, verifier, horizon, latency, recovery, and
   confidence-interval metrics.
6. Failed cases preserve their failure artifact and full log.
7. No cross-case Memory, Skill, or cache state is observed.
8. A real ID/OOD smoke completes with CAP-X-compatible LIBERO execution.
9. Focused tests, full tests, compile check, diff check, and artifact manifest
   validation pass.

P5.5 empirical closure additionally requires the formal frozen suite to meet
the declared case-count and task-family thresholds. P5.5 closure does not
claim that CAP-MAS improves OOD success. P5.6 calibration is required before
OOD evidence can become a learned or causal Arbiter signal.

## 12. Implementation order

```text
P5.5.0  contracts and frozen manifest validation
P5.5.1  leakage audit and case-scoped artifact boundary
P5.5.2  single-case replay driver
P5.5.3  aggregation and confidence intervals
P5.5.4  real ID/OOD smoke
P5.5.5  five-seed pilot
P5.5.6  ten-plus-seed, multi-family formal gate
P5.5.7  experiment documentation and P5.6 handoff
```
