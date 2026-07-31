# P5.1 Typed Skill Condition Defaults Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with review checkpoints.

**Goal:** Make incomplete typed-skill action graphs pass deterministic contract validation, produce meaningful P5.1 static verifier evidence, and preserve explicit strategy-specific constraints.

**Architecture:** Add a reusable `SkillConditionEnricher` that reads default postconditions from registered typed skills and derives only safe strategy-controlled preconditions from explicit scene-grounded IDs. Apply the same idempotent enrichment before staged decoder validation and at the scheduler candidate boundary; keep graph validation and runtime skill argument validation authoritative. Relax only the wire-list cardinality needed for an empty LLM list, never the final action contract requirement.

**Tech Stack:** Python 3, frozen dataclass contracts, pytest, existing CAP-MAS staged graph decoder/scheduler, CAP-X typed skill adapter, LIBERO observable verifier.

## Global Constraints

- Preserve all existing uncommitted source changes and all `outputs/phase5/` experiment directories.
- Do not change CAP-X API function names, environment ownership, or privileged evaluator access.
- Defaults are additive and idempotent; explicit LLM predicates are never replaced.
- Never infer a track ID from free-form prose; only explicit `MotionIntent` IDs or exact current-scene track/label matches are eligible.
- `object_in_gripper` and `object_at_target` remain task predicates and are never synthesized from a generic skill call.
- Every fresh empirical run must use a new run-scoped directory and retain redacted logs, results, summaries, and manifests.

---

### Task 1: Expose typed-skill predicate metadata

**Files:**
- Modify: `capmas/skills/protocol.py`
- Modify: `capmas/backends/capx.py:211-273`
- Modify: `capmas/backends/capx_libero_factory.py:25-40,340-360`
- Modify: `capmas/skills/registry.py:35-80`
- Test: `tests/test_capx_adapter.py`
- Test: `tests/test_skill_registry.py`

**Interfaces:**
- `CAPXTypedSkill(..., default_postconditions: tuple[str, ...] = ())` exposes `default_postconditions`.
- `SkillRegistry.default_postconditions(reference: SkillRef) -> tuple[str, ...]` returns declared metadata or an empty tuple for compatible custom skills.
- `build_capx_skills(..., default_postconditions: Mapping[str, Sequence[str]] | None = None)` attaches metadata by skill ID.

- [ ] **Step 1: Write the failing tests**

Add tests asserting that a CAP-X typed skill stores its declared defaults and
that a custom skill without the optional attribute returns `()` from the
registry. Add a factory-level assertion for the five LIBERO skill defaults.

- [ ] **Step 2: Run the focused tests to verify failure**

Run:

```bash
pytest -q tests/test_capx_adapter.py tests/test_skill_registry.py
```

Expected: FAIL because `CAPXTypedSkill` and `SkillRegistry` do not expose
condition metadata yet.

- [ ] **Step 3: Implement the minimal metadata surface**

Add the optional protocol attribute, constructor parameter, registry lookup,
and the `DEFAULT_LIBERO_POSTCONDITIONS` mapping. Pass the mapping from
`build_capx_runtime_from_yaml()` into `build_capx_skills()`; do not alter
function binding or execution behavior.

- [ ] **Step 4: Run the focused tests to verify green**

Run the same pytest command. Expected: PASS.

- [ ] **Step 5: Commit the isolated metadata slice**

```bash
git add capmas/skills/protocol.py capmas/backends/capx.py capmas/backends/capx_libero_factory.py capmas/skills/registry.py tests/test_capx_adapter.py tests/test_skill_registry.py
git commit -m "feat: expose typed skill condition defaults"
```

### Task 2: Implement strategy-aware condition enrichment

**Files:**
- Modify: `capmas/contracts/strategy.py:1-125`
- Create: `capmas/graph/condition_defaults.py`
- Test: `tests/test_condition_defaults.py`

**Interfaces:**
- `SkillConditionEnricher(registry: SkillRegistry).enrich(subgraph: SubgraphSpec, scene: SceneSnapshot, strategy: str) -> SubgraphSpec`.
- `StrategyProfile.require_grounding_preconditions: bool` is true for `safety` and `robust`, false for `balanced` and `efficient`.

- [ ] **Step 1: Write the failing tests**

Cover one action with empty predicates, one action with explicit predicates,
one checkpoint, and one ambiguous/non-grounded skill argument. Assert:

```python
balanced = enricher.enrich(graph, scene, "balanced")
safety = enricher.enrich(graph, scene, "safety")
assert "scene_fresh(2000)" in balanced.node("action").preconditions
assert "track_exists:bowl" not in balanced.node("action").preconditions
assert "track_exists:bowl" in safety.node("action").preconditions
assert enricher.enrich(safety, scene, "safety") == safety
```

Also assert `close_gripper` adds `gripper_closed()` and does not add
`object_in_gripper(...)`.

- [ ] **Step 2: Run the focused test to verify failure**

```bash
pytest -q tests/test_condition_defaults.py
```

Expected: FAIL because the enrichment module and profile flag do not exist.

- [ ] **Step 3: Implement the minimal enricher**

For each action node with calls, union `scene_fresh(2000)` into preconditions,
union each registered call's `default_postconditions` into postconditions,
and for grounding profiles add `track_exists:<id>` only for explicit IDs that
resolve exactly to one current `ObjectTrack.track_id` or label. Preserve order
and deduplicate. Leave checkpoint/router nodes unchanged.

- [ ] **Step 4: Run the focused tests to verify green**

```bash
pytest -q tests/test_condition_defaults.py
```

Expected: PASS.

- [ ] **Step 5: Commit the enrichment slice**

```bash
git add capmas/contracts/strategy.py capmas/graph/condition_defaults.py tests/test_condition_defaults.py
git commit -m "feat: derive strategy-aware action predicates"
```

### Task 3: Integrate enrichment before staged validation and at candidate ingress

**Files:**
- Modify: `capmas/llm/staged_decoder.py:80-135`
- Modify: `capmas/agents/policy.py:150-250`
- Modify: `capmas/graph/normalizer.py:40-85`
- Modify: `capmas/runtime/llm_scheduler.py:115-155,680-870`
- Modify: `scripts/run_libero_b3_llm.py:540-650`
- Test: `tests/test_staged_protocol.py`
- Test: `tests/test_llm_scheduler.py`
- Test: `tests/test_llm_graph_decoder.py`

**Interfaces:**
- `LocalSubgraphDecoder(..., condition_enricher: Callable[[SubgraphSpec, SceneSnapshot], SubgraphSpec] | None = None)` enriches before `validate_subgraph()`.
- `LLMStagedGraphPolicyAgent(..., policy_strategy: str = "balanced")` exposes `policy_strategy` and passes the configured decoder.
- `LLMGraphScheduler(..., condition_enricher: Callable[[SubgraphSpec, SceneSnapshot, str], SubgraphSpec] | None = None)` enriches proposals before scene grounding and remains idempotent at normalization.

- [ ] **Step 1: Write the failing decoder and scheduler tests**

Create a staged local artifact whose action has `skill_calls` but
`postconditions=[]`. Decode it with a registry-backed enricher and assert the
result is accepted with `scene_fresh(2000)`. Add a scheduler test using a
callable policy that returns the same empty list and assert arbitration sees a
postcondition instead of `ACTION_WITHOUT_POSTCONDITION`.

- [ ] **Step 2: Run the focused tests to verify failure**

```bash
pytest -q tests/test_staged_protocol.py tests/test_llm_scheduler.py tests/test_llm_graph_decoder.py
```

Expected: FAIL with the existing missing-postcondition diagnostic.

- [ ] **Step 3: Implement the integration**

Apply the decoder callback immediately after `local_subgraph_from_dict()` and
before terminal-edge normalization/GraphValidator. In the scheduler, enrich a
proposal before `candidate_scene_rewriter` so LIBERO grounding can see the
injected `object_at_target`-independent freshness predicate. Apply the same
callback in `CandidateNormalizer` as an idempotent fallback. Configure each
staged LLM agent in the runner with its strategy-specific decoder callback and
configure the scheduler with the registry-backed callback.

- [ ] **Step 4: Run the focused tests to verify green**

Run the same pytest command. Expected: PASS.

- [ ] **Step 5: Commit the integration slice**

```bash
git add capmas/llm/staged_decoder.py capmas/agents/policy.py capmas/graph/normalizer.py capmas/runtime/llm_scheduler.py scripts/run_libero_b3_llm.py tests/test_staged_protocol.py tests/test_llm_scheduler.py tests/test_llm_graph_decoder.py
git commit -m "feat: enrich staged policy contracts before validation"
```

### Task 4: Open the wire path and static verifier coverage

**Files:**
- Modify: `capmas/llm/prompts.py:90-150,743-790`
- Modify: `capmas/verification/libero.py:97-110`
- Modify: `tests/test_llm_prompts.py`
- Modify: `tests/test_libero_verifier.py`
- Modify: `tests/test_verifier_evidence_collection.py`

**Interfaces:**
- Local node `postconditions` remains a required array field but permits
  `[]`; final action validation still requires the enriched result to be
  non-empty.
- `compile_time_preconditions()` includes `scene_fresh(` alongside stable
  track/visibility facts.

- [ ] **Step 1: Write the failing schema/evidence tests**

Assert the local schema reports no positive `minItems` requirement, the staged
prompt explains runtime default injection, and
`compile_time_preconditions(("scene_fresh(2000)",))` returns that predicate.
Extend static evidence collection to assert a passing `scene_fresh` result is
included when it is the only safe precondition.

- [ ] **Step 2: Run the focused tests to verify failure**

```bash
pytest -q tests/test_llm_prompts.py tests/test_libero_verifier.py tests/test_verifier_evidence_collection.py
```

Expected: FAIL because the schema requires at least one item and freshness is
currently filtered out.

- [ ] **Step 3: Implement schema, prompt, and selector changes**

Remove the `minItems: 1` override from the shared node `postconditions` list,
state that the runtime adds typed-skill defaults, and include `scene_fresh(` in
the LIBERO stable-prefix selector. Do not allow the selector to include
object/gripper or downstream target predicates.

- [ ] **Step 4: Run the focused tests to verify green**

Run the same pytest command. Expected: PASS.

- [ ] **Step 5: Commit the wire/evidence slice**

```bash
git add capmas/llm/prompts.py capmas/verification/libero.py tests/test_llm_prompts.py tests/test_libero_verifier.py tests/test_verifier_evidence_collection.py
git commit -m "feat: publish freshness defaults as static verifier evidence"
```

### Task 5: Update Phase 5 documentation and run repository verification

**Files:**
- Modify: `docs/phase5-evidence-evolution.md`
- Modify: `docs/implementation-roadmap.md`
- Modify: `docs/experiments.md`
- Modify: `docs/superpowers/plans/2026-07-30-p5-1-verifier-evidence.md`
- Test: `tests/test_phase5_docs.py`

- [ ] **Step 1: Write the documentation assertions**

Add assertions for the A+D condition-default design, the decoder/scheduler
double boundary, the profile distinction, and the still-open empirical
success-rate gate.

- [ ] **Step 2: Run documentation tests to verify failure**

```bash
pytest -q tests/test_phase5_docs.py
```

Expected: FAIL because the new terms are not recorded.

- [ ] **Step 3: Update the phase documents**

Record P5.1 condition-default implementation status, explicitly state that
static coverage is now measurable but downstream success is not claimed, and
preserve the separate-run/log retention rule.

- [ ] **Step 4: Run all local verification**

```bash
pytest -q
python -m compileall -q capmas scripts
git diff --check
```

Expected: all tests pass, compileall exits 0, and diff check is clean.

- [ ] **Step 5: Commit documentation and verified implementation**

```bash
git add docs/phase5-evidence-evolution.md docs/implementation-roadmap.md docs/experiments.md docs/superpowers/plans/2026-07-30-p5-1-verifier-evidence.md tests/test_phase5_docs.py
git commit -m "docs: record P5.1 condition-default closure"
```

### Task 6: Run a fresh matched LIBERO P5.1 validation

**Files:**
- Create only: `outputs/phase5/P5.1_condition_defaults_<timestamp>/`

- [x] **Step 1: Run one smoke seed with the configured CAP-X environment**

Use `CUDA_VISIBLE_DEVICES=5`, the existing CAP-X LIBERO Python environment,
the configured endpoint/model, and a new output root. Do not place the API key
in a file or shell history artifact; pass it through the existing runtime
secret mechanism.

- [x] **Step 2: Verify artifact integrity**

Check that every run has a retained log, `summary.json`, `summary.md`,
`manifest.json`, and `evidence/verifier.json`; confirm at least one selected
candidate has a non-empty static result list and positive coverage.

- [x] **Step 3: Run the matched seed set**

Run the same five seeds used by the locked P5.1 comparison, writing each seed
to its own child directory. Report evaluator success, graph completion,
static coverage, dynamic evidence count, and any compile/recovery failure
separately.

- [x] **Step 4: Inspect the final diff without reverting history**

```bash
git status --short
git diff --stat
rg -n "sk-[A-Za-z0-9]" outputs/phase5/P5.1_condition_defaults_<timestamp> || true
```

Expected: no provider key appears in logs or artifacts; all previous output
directories remain present.

Observed matched result (2026-07-31): evaluator success was 2/5 and graph
completion was 1/5. Four normal runs produced positive static coverage and
seven dynamic records. Seed 4 was an upstream LLM read-timeout failure and
retained its failure artifact and log. The pre-refresh seed-1 diagnostic run
is retained separately and is not counted in the matched five-seed result.
