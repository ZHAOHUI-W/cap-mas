# P5.3.2 Object-6 Effective Motion and Candidate Identifiability Design

## 1. Decision

P5.3.2 begins with an `object-6`-only task-capability repair increment. It
will make geometry evidence describe the full candidate graph that is actually
submitted to CAP-X/LIBERO, rather than the first `MotionIntent` on the local
subgraph selected by the Arbiter.

The increment introduces a read-only, deterministic `EffectiveMotionProgram`
at the decision boundary. It binds a candidate graph and the immutable
`SceneSnapshot` to the executable grasp, approach, lift, transfer, placement,
and release path. The same bound program is materialized into the execution
graph, so preview and execution cannot silently use different pose or motion
parameters.

The P5.6D seed-32--51 qualification artifact remains immutable. It supplied
the diagnosis for this work but is not retrained, relabelled, filtered, or
used to activate calibration.

## 2. Root Cause and Scope

In the P5.6D qualification block, all 20 candidate pairs had identical
evidence vectors. The current `candidate_geometry_evidence()` takes the first
motion intent from `GraphCandidate.subgraph`; for `object-6` this is the grasp
approach. `ReferenceMotionPreview` then uses its fixed
`approach_distance_m=0.15` and does not consume the candidate's effective
`z_approach`, lift, transfer, placement approach, or release trajectory.

At the same time, the Arbiter selects a local `sg_pick_*` candidate whereas
the collection labels the full pick-and-place `MissionGraph`. Two submitted
episodes passed pick but failed place. A pick-only feature vector cannot be a
valid predictor of that terminal label.

### In scope

- A typed, serializable `EffectiveMotionProgram` for a selected local
  candidate plus its complete executable mission graph.
- Decision-time binding from normalized graph metadata and the current
  `SceneSnapshot`, with no robot command, action lease, simulator evaluator
  state, or post-decision observation.
- Execution materialization that replaces symbolic grasp-result references
  with the exact decision-time bound poses and parameters used by preview.
- Segment-level read-only preview for grasp approach, lift, transfer, place
  approach, and release; conservative full-path aggregation for the existing
  Arbiter geometry dimensions.
- Candidate semantic-equivalence, evidence-identical, and selection-
  identifiability diagnostics.
- A bounded diversity-rejection path that requests genuinely different typed
  motion alternatives before arbitration, or abstains when none is available.
- A separate, pre-registered object-6 ten-seed capability experiment after
  the implementation tests pass.

### Out of scope

- Editing, deleting, excluding, or relabelling P5.6D rows.
- Fitting or activating a calibration model, changing P5.6 weights, or
  claiming an Arbiter causal improvement.
- Relaxing a collision/clearance gate to increase data volume before its
  coordinate-frame, occupancy, map-freshness, and preview/execution alignment
  diagnosis is recorded.
- Repairing `spatial-0` or `goal-1`; their existing P5.3.2 handoffs remain
  separate task-family work packages.
- Adding learned grasp selection, a TSDF backend, semantic adapters, or a
  synchronous rehearsal call to the control-critical path.

## 3. Candidate and Label Boundary

The current `GraphCandidate` remains the local Arbiter artifact and keeps its
subgraph fingerprint. P5.3.2 adds a companion execution context instead of
changing that identity:

```python
@dataclass(frozen=True)
class CandidateExecutionContext:
    candidate: GraphCandidate
    mission_graph: MissionGraph
    selected_subgraph_id: str
    execution_graph_fingerprint: str
```

The P5.6 collection loader already retains each policy record's full
`MissionGraph`. It must construct this context before evidence collection and
must execute the `materialized_graph` derived from the same context. The
Arbiter still scores and selects its local candidate. Geometry evidence keeps
`candidate_fingerprint` for that Arbiter connection and additionally records
the execution-graph and effective-program fingerprints for terminal-label
provenance.

For this increment, the label strategy is graph-level: geometry aggregates
the critical successful path from the selected local subgraph through the
full pick-and-place graph. The terminal evaluator outcome is therefore scoped
to the same graph. Per-subgraph postconditions remain diagnostics only. A
later rolling policy experiment may use local labels, but it must not mix them
with graph-level labels in one calibration dataset.

## 4. Effective Motion Program

### 4.1 Contracts

`capmas.perception.effective_motion` will define:

```python
@dataclass(frozen=True)
class EffectiveMotionSegment:
    segment_id: str
    kind: Literal["grasp_approach", "lift", "transfer", "place_approach", "release"]
    source_subgraph_id: str
    source_node_id: str
    start_pose_wxyz_xyz: tuple[float, ...] | None
    end_pose_wxyz_xyz: tuple[float, ...] | None
    approach_vector_xyz: tuple[float, float, float] | None
    approach_distance_m: float | None
    payload_track_id: str | None

@dataclass(frozen=True)
class EffectiveMotionProgram:
    candidate_fingerprint: str
    execution_graph_fingerprint: str
    program_fingerprint: str
    decision_scene_version: int
    selected_subgraph_id: str
    segments: tuple[EffectiveMotionSegment, ...]
    semantic_signature: str
```

The builder consumes only the normalized candidate graph, the full candidate
execution graph, and the version-matched `SceneSnapshot`. It must reject a
graph whose selected local subgraph does not match the candidate, whose
decision scene is stale, whose symbolic pose cannot be resolved from typed
metadata, or whose path reaches an unsupported control-flow branch. These are
typed `unknown` evidence outcomes, not invented poses.

For object-6, `sample_grasp_pose` result references are bound from the adjacent
normalized grasp `MotionIntent.target_pose_wxyz_xyz`. Literal `goto_pose`,
`z_approach`, and `lift_after_grasp.z_lift` values are retained exactly. The
program builder does not call `sample_grasp_pose` or a simulator planner. This
makes binding reproducible from decision-time state.

### 4.2 Preview/execution equivalence

`materialize_execution_graph(program, graph)` returns a fresh graph in which
all physical pose references used by `goto_pose` and `lift_after_grasp` have
been replaced with the bound literal pose values. It retains the CAP-X skill
IDs and existing execution API; no new robot command API is introduced.

The materializer rejects a graph if a motion-bearing call cannot be proven to
match a segment in the program. Before the executor receives it, it records
the program fingerprint and the materialized execution-graph fingerprint.
The evidence collector, ActionLease path, and physical outcome artifact must
all carry these same values. A mismatch is fail-closed before physical
execution.

## 5. Segment Preview and Geometry Evidence

`ReferenceMotionPreview` will implement a new read-only
`preview_program(program, scene, local_map)` operation while retaining its
existing single-intent method for compatibility. It will derive each segment's
corridor start/end and sample count from each effective segment's explicit
poses rather than its global fixed approach distance. `grasp_approach` begins
at its approach-offset pose and ends at the bound grasp pose; `lift` begins at
that grasp pose; `transfer` starts at the lifted pose; and `place_approach`
ends at the bound release pose. This makes the previewed swept path
reconstructable from the decision artifact.

The preview result contains one status for every segment:

```python
@dataclass(frozen=True)
class SegmentMotionPreview:
    segment_id: str
    ik_valid: bool | None
    collision_free: bool | None
    clearance_m: float | None
    path_length_m: float | None
    reason: str

@dataclass(frozen=True)
class ProgramMotionPreview:
    segments: tuple[SegmentMotionPreview, ...]
    aggregate_status: Literal["feasible", "infeasible", "unknown"]
```

The existing scalar Arbiter interface remains conservative:

- `reachability` fails if any measurable critical segment has invalid IK;
- `clearance` is the minimum measurable critical-segment clearance;
- `collision_risk` is the maximum measurable critical-segment risk;
- `grasp_quality` is derived only when bound grasp pose/approach evidence is
  available, otherwise remains `unknown`.

If no critical segment is measurable, the corresponding dimension is
`unknown`, never zero. A program preview may include a compact artifact ref
with segment details; raw RGB-D and privileged evaluator state are excluded.

`GeometryEvidence` gains trailing, defaulted provenance fields so current
callers remain compatible:

```python
execution_graph_fingerprint: str | None = None
program_fingerprint: str | None = None
program_scope: Literal["subgraph", "mission_suffix"] = "subgraph"
segment_artifact_refs: tuple[ArtifactRef, ...] = ()
```

## 6. Candidate Diversity and Audit Semantics

P5.3.2 must distinguish a genuine evidence decision from a deterministic tie.
It writes a decision-bound artifact for every candidate wave:

```python
@dataclass(frozen=True)
class CandidateIdentifiability:
    candidate_id: str
    semantic_signature: str
    program_fingerprint: str | None
    candidate_semantic_equivalent: bool
    candidate_evidence_identical: bool
    selection_identifiable: bool
    abstention_reason: str | None
```

The semantic signature includes normalized object/target identities and all
bound grasp approach, lift, transfer, place approach, and release parameters.
It excludes `candidate_id`, policy name, response text, timestamps, and
fingerprints. Thus renaming a policy cannot create a false alternative.

Before scoring, a `CandidateDiversityValidator` compares candidates for the
same local decision. If all candidates are semantically equivalent, the
scheduler receives a typed diversity rejection containing the required
different fields (`grasp pose/offset`, `approach`, `lift`, `transfer`, or
`place/release`). It allows one bounded regeneration wave. If the regenerated
set remains equivalent, the Arbiter abstains with
`candidate_semantic_equivalence`; it must not select by
`evidence_tie_break`.

If the programs differ but their measurable geometry vectors are identical,
the artifact records `candidate_evidence_identical=true` and
`selection_identifiable=false`. This is an honest tie: it may execute only
under an explicitly pre-registered fixed-policy capability lane, never as
evidence-selected data for an Arbiter-improvement claim.

## 7. Collision and Capability Diagnosis

Before changing any gate, a rejected program emits segment-level diagnosis:
map/scene version, coordinate-frame provenance, occupancy hits, clearance
samples, corridor radius, inflation, and previewed path endpoints. The
materialized execution graph records matching endpoints. This distinguishes a
real obstructed path from a frame conversion, stale map, fixed-corridor, or
preview/execution mismatch.

The P5.3.2 physical gate uses a new independently pre-registered object-6
ten-seed manifest. It records one reset and one selected execution per seed,
uses GPU 5 without disturbing other workers, and creates a separate output
directory containing all logs and artifacts. It is not a replacement for the
fixed P5.6D block.

The capability gate requires:

1. zero infrastructure-unknown cases and typed provenance for every terminal
   result;
2. at least 80 percent physical-execution reach after safety/density checks;
3. at least one evaluator success;
4. no execution outcome whose bound-program fingerprint differs from its
   pre-decision preview;
5. an explicit count of semantic-equivalence, evidence-identical,
   evidence-selected, fixed-policy tie, and safety-abstained decisions.

Only a later, separately pre-registered P5.6 collection may consider rows
with `selection_identifiable=true` for an Arbiter causal claim. The P5.3.2
trial itself remains a capability and alignment test.

## 8. Implementation and Test Sequence

1. Add contract tests for deterministic program binding, scene/version
   rejection, full-path extraction, and materialized graph equivalence. The
   tests must fail before production code is changed.
2. Implement the pure program builder and materializer, then extend preview
   tests for candidate-specific approach/lift/place differences and
   conservative aggregation.
3. Extend live evidence-session and collection tests to require an execution
   context, matching program fingerprints, and graph-level provenance while
   retaining legacy subgraph-only callers.
4. Add diversity-validator and Arbiter tests: semantic duplicates cause a
   bounded reproposal/typed abstention; non-identical programs with identical
   evidence remain non-identifiable; differing evidence selects by evidence.
5. Add collision diagnostic artifacts and test their schema without running a
   physical environment.
6. Run focused tests, the full suite, `compileall`, touched-file Ruff, and
   `git diff --check`. Only then pre-register and run the separate ten-seed
   capability experiment.

## 9. Non-goals and Rollback

This design does not make rehearsal synchronous, loosen hard safety gates, or
turn a policy-name tie into evidence. The single-intent P5.2 provider remains
available to legacy callers. The new full-program provider is activated only
by an explicit object-6 P5.3.2 runner/configuration. Disabling that
configuration returns the system to the existing P5.6D code path without
changing historical artifacts or calibration state.
