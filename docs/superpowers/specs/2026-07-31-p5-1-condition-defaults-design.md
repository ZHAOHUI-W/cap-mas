# P5.1 Typed Skill Condition Defaults Design

## Goal

Make incomplete LLM action nodes executable and observable by deriving safe
default predicates from registered typed skills, while preserving explicit LLM
predicates and keeping policy strategy differences auditable.

## Root Cause

The staged local decoder validates a decoded `SubgraphSpec` before the runtime
scheduler sees it. An action node with an empty `postconditions` list is
therefore rejected as `ACTION_WITHOUT_POSTCONDITION`. The current
`CAPXTypedSkill` protocol carries argument validation and execution behavior,
but no predicate metadata. P5.1 static evidence also filters out
`scene_fresh(...)`, so graphs that lack `track_exists:*` declarations produce
typed artifacts with zero useful coverage.

## Design

### 1. Typed skill defaults

Registered skills expose optional `default_postconditions`. CAP-X LIBERO
bindings declare these defaults:

| Skill | Default postcondition |
| --- | --- |
| `goto_pose` | `scene_fresh(2000)` |
| `sample_grasp_pose` | `scene_fresh(2000)` |
| `lift_after_grasp` | `scene_fresh(2000)` |
| `close_gripper` | `gripper_closed()` |
| `open_gripper` | `gripper_open()` |

Observation/query skills do not create physical state predicates. A skill with
no declaration remains compatible and contributes no default.

`SkillConditionEnricher` applies defaults only to action nodes that have typed
skill calls. It unions defaults with explicit node postconditions, removes
duplicates while preserving order, and never overwrites explicit predicates.
Checkpoint and router nodes are unchanged.

### 2. Strategy-controlled preconditions

`StrategyProfile` declares whether default grounding preconditions are enabled.
`balanced` and `efficient` inject only `scene_fresh(2000)`. `safety` and
`robust` also inject `track_exists:<track_id>` when a track ID is explicitly
available from `MotionIntent` or a typed `object_name` argument that matches a
current `SceneSnapshot` track. Natural-language text is never parsed into a
track ID. Existing explicit preconditions are preserved and deduplicated.

The enricher is parameterized by the current scene only for safe identity
resolution. It does not add `object_in_gripper` or `object_at_target`, because
those are task predicates that cannot be inferred from a skill call alone.

### 3. Injection boundaries

The same idempotent enricher is used at two boundaries:

1. `LocalSubgraphDecoder` applies it before `GraphValidator`, so an LLM
   response with `postconditions: []` does not fail before scheduling.
2. `CandidateNormalizer`/scheduler applies it again before scene grounding,
   geometry evidence, skill validation, and arbitration. This covers legacy,
   callable, and non-LLM policy producers.

The second application is a safety net, not a second source of semantics.
Both paths use the candidate strategy and the same registered `SkillRegistry`.

### 4. Response schema and static evidence

The local graph schema permits an empty `postconditions` list. The prompt says
that the runtime fills safe defaults and that the Policy must still declare
task-specific success predicates. `compile_time_preconditions` treats
`scene_fresh(...)` as safe to evaluate against the current scene in addition
to `track_exists:*` and `object_visible:*`. This produces positive P5.1 static
evidence without evaluating downstream object/gripper postconditions against
the initial scene.

## Compatibility and Safety

- Explicit LLM predicates remain authoritative; defaults are additive only.
- Unregistered skills and invalid arguments remain rejected by the existing
  `SkillRegistry`.
- Empty action postconditions are accepted only after deterministic enrichment;
  a custom skill with no defaults still fails closed in the final validator.
- Checkpoint-only subgoals continue to bypass Policy skill generation.
- The feature is CAP-MAS-local and does not alter CAP-X API functions.

## Testing and Acceptance

Unit tests must prove:

1. CAP-X skills expose the declared defaults.
2. Empty action postconditions are enriched, explicit predicates survive, and
   enrichment is idempotent.
3. `balanced` and `safety` produce distinct, deterministic preconditions.
4. The staged decoder accepts an empty-postcondition action after enrichment.
5. Compile-time evidence includes `scene_fresh(...)` and has positive coverage
   on an otherwise empty-precondition candidate.
6. Existing full-suite behavior remains green.

The empirical gate is a fresh, run-scoped LIBERO experiment. It must retain
all logs and artifacts, report non-zero static verifier coverage for at least
one selected candidate, and separately report evaluator success versus graph
completion. This change does not claim downstream success improvement by
itself.

