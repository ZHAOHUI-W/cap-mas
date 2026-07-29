# Phase 5.2 Candidate-Conditioned Geometry and Grasp Evidence

## Status

Accepted design. Implementation is pending TDD execution.

## Goal

Make candidate arbitration action-conditioned: two Policy candidates evaluated
against the same immutable `SceneSnapshot` must be able to receive different
geometry evidence when their typed motion intent, target pose, approach, or
trajectory preview differs.

P5.2 is an evidence provider and Arbiter integration increment. It does not
change the CAP-X Robot Skill API, execute physical actions during preview, or
claim task success from geometry alone.

## 1. Problem and boundary

P3.2 perception evidence is mostly candidate-independent. It describes scene
freshness, visibility, identity, tracking, and pose reliability, but does not
describe whether a particular candidate's grasp or motion is geometrically
feasible. P5.2 adds candidate-conditioned evidence while preserving the P3.2
provenance and fallback semantics.

The provider must:

- consume the candidate's effective, normalized `SubgraphSpec`;
- consume one immutable `SceneSnapshot` and a frozen map view;
- use only agent-visible sensor-derived state in the primary mode;
- use a side-effect-free motion preview backend;
- finish within the proposal-wave deadline or return explicit `unknown` values;
- never acquire `ActionLease`, call physical `goto_pose`, or mutate CAP-X state.

P5.2 does not implement evidence calibration, OOD replay, process rehearsal,
or learned Arbiter weights. Those remain later Phase 5 packages.

## 2. Domain contracts

### 2.1 Three-state evidence

Every geometry dimension is represented as a typed result, not as a nullable
float interpreted by callers:

```python
@dataclass(frozen=True)
class EvidenceDimension:
    name: str
    status: Literal["pass", "fail", "unknown"]
    score: float | None = None       # [0, 1] only for pass/fail when measurable
    threshold: float | None = None
    reason: str = ""
```

Rules:

- `pass` with a score may enter soft scoring;
- `fail` may trigger a declared hard gate;
- `unknown` never becomes zero and never becomes pass;
- a missing backend, timeout, stale map, or unresolved target is `unknown`;
- hard safety failures remain failures even when other dimensions score well.

### 2.2 MotionIntent

`MotionIntent` is typed planning metadata on an action node. It is not a
Robot Skill and is never sent to the CAP-X executor as a skill argument.

```python
@dataclass(frozen=True)
class MotionIntent:
    kind: Literal["grasp", "place", "move"]
    object_track_id: str | None = None
    target_track_id: str | None = None
    approach_vector_xyz: tuple[float, float, float] | None = None
    standoff_m: float | None = None
    target_pose_wxyz_xyz: tuple[float, ...] | None = None
```

`SubgraphNodeSpec` gains an optional `motion_intent` field. The
`CandidateNormalizer` canonicalizes it and validates consistency with the
actual registered `skill_calls`. Free-form descriptions and unregistered
arguments such as `approach` or `standoff` inside `skill_calls.args` are not
accepted.

The first implementation may derive `MotionIntent` from existing typed calls,
including `sample_grasp_pose(object_name, use_multiview)` and
`goto_pose(position, quaternion_wxyz, z_approach)`. A later Policy schema may
emit the same typed structure directly. If the effective candidate does not
expose enough information for a preview, the provider returns `unknown`
rather than inventing a pose.

### 2.3 MotionPreview

Preview is a read-only planning result:

```python
@dataclass(frozen=True)
class MotionPreview:
    status: Literal["feasible", "infeasible", "unknown"]
    target_pose_wxyz_xyz: tuple[float, ...] | None = None
    trajectory_ref: ArtifactRef | None = None
    ik_valid: bool | None = None
    collision_free: bool | None = None
    path_length_m: float | None = None
    reason: str = ""
    backend: str = ""
    backend_version: str = ""
```

```python
class MotionPreviewBackend(Protocol):
    def preview(
        self,
        intent: MotionIntent,
        scene: SceneSnapshot,
        local_map: LocalMapBackend,
    ) -> MotionPreview: ...
```

The backend must be side-effect free. It cannot receive an environment handle,
an `ActionLease`, or evaluator completion state.

Implementations:

- `ReferenceMotionPreview`: workspace, pose validity, target freshness, map
  corridor occupancy, and conservative clearance checks;
- `CAPXPlannerPreviewAdapter`: optional read-only IK, trajectory, and collision
  queries when a CAP-X/CuRobo planner seam is available.

FK alone is insufficient for reachability. A backend must use IK or a declared
conservative approximation and label its provenance. If a planner is absent,
reachability and collision dimensions remain `unknown`.

### 2.4 GeometryEvidence

Geometry is separate from `PerceptionEvidence`:

```python
@dataclass(frozen=True)
class GeometryEvidence:
    grasp_quality: EvidenceDimension
    reachability: EvidenceDimension
    clearance: EvidenceDimension
    collision_risk: EvidenceDimension
    candidate_fingerprint: str
    scene_version: int
    map_version: int | None
    map_backend: str
    provider: str
    provider_version: str
    captured_at_ns: int
    latency_ms: float
    used_privileged_state: bool = False
    artifact_refs: tuple[ArtifactRef, ...] = ()
```

`CandidateEvidence` exposes `geometry: GeometryEvidence | None` and declares
`"geometry"` in `available_metrics` only when at least one valid geometry
dimension was produced. The candidate fingerprint must be the effective
normalized fingerprint used for execution. If grounding or repair changes the
motion intent or executable args, geometry evidence is recomputed.

## 3. Geometry dimensions

The reference provider computes the following dimensions when its inputs are
available:

| Dimension | Reference calculation | Hard gate |
| --- | --- | --- |
| `grasp_quality` | pose/approach alignment and available surface/contact evidence | no |
| `reachability` | target pose validity plus IK or conservative workspace check | yes when `fail` |
| `clearance` | continuous clearance from map queries along the approach corridor | no |
| `collision_risk` | end-effector/approach corridor occupancy; swept-volume query when available | yes when `fail` |

The current `MapQueryResult.clearance_m` is used as a continuous signal. A
binary `occupied -> 0` conversion is not sufficient. Sparse voxel surface
normals are optional; if unavailable, normal-dependent grasp quality is
`unknown`. TSDF is an interchangeable backend, not a reason to fabricate
surface evidence in the sparse reference backend.

Target lookup uses the immutable `SceneSnapshot.objects` at the requested
scene version. A mutable tracker state at a newer version must not be read
for an older candidate.

## 4. Arbiter integration

P5.2 uses transparent fixed weights until P5.6 calibration:

```text
geometry_score =
    0.30 * grasp_quality
  + 0.30 * reachability
  + 0.25 * clearance
  + 0.15 * (1 - collision_risk)
```

Only dimensions with a measurable `pass` or `fail` score participate, and the
weights are renormalized over available dimensions. `unknown` is excluded,
not converted to zero.

Hard gates are evaluated before soft scoring:

- stale candidate, stale map, or mismatched fingerprint: reject;
- `reachability.fail`: reject when the strategy profile declares the gate;
- `collision_risk.fail`: reject when the strategy profile declares the gate;
- `unknown`: do not reject solely for being unknown, but record the reason and
  allow the configured fallback policy.

The Arbiter records separate breakdown fields for `perception`, `geometry`,
`verifier`, `latency`, and `recovery`. Legacy `confidence` is not mixed into
geometry evidence.

## 5. Deadline and scheduling

Geometry evidence is computed after candidate normalization and before
ActionLease acquisition:

```text
Policy proposal wave
  -> candidate normalization
  -> read-only geometry preview in parallel
  -> 50 ms global wave deadline
  -> Arbiter
  -> ActionLease and physical execution
```

The global geometry deadline is 50 ms. Each candidate receives a smaller
sub-budget, for example 35-40 ms, so the wave has scheduling margin. Map
queries should be batched where possible. Heavy planner adapters may run in an
isolated asynchronous worker and return `unknown` when the current decision
deadline expires.

Every run records queue wait, compute latency, global deadline misses,
candidate-level timeouts, and evidence status. Geometry never blocks the
high-frequency World Model update.

## 6. Privilege and evaluation modes

Primary realistic mode uses sensor-derived `SceneSnapshot` objects, maps, and
robot state. It must set `used_privileged_state=false`.

Diagnostic privileged mode may use LIBERO ground-truth poses only to isolate
perception error from geometry error. Its artifacts must set
`evaluation_mode=diagnostic_privileged` and cannot be pooled with the primary
task-success result.

The P4.5 privileged object-measurement transport remains valid for World Model
infrastructure tests, but is not the primary P5.2 candidate evidence source.

## 7. Validation and experiment gates

P5.2 is accepted only after all three layers pass:

1. Contract tests for three-state evidence, fingerprints, stale versions,
   intent/call consistency, no executor side effects, and timeout behavior.
2. Arbiter integration tests showing that distinct intents yield distinct
   score breakdowns and that hard gates and unknown fallback behave correctly.
3. Realistic LIBERO pilot with five seeds comparing geometry-disabled,
   geometry-shadow, and bounded geometry-online modes.

Each experiment has its own immutable directory:

```text
outputs/phase5/P5.2_geometry_evidence/<timestamp>_<run_id>/
  run_config.json
  manifest.json
  summary.json
  summary.md
  logs/
  results/
  traces/
  evidence/
  artifacts/
```

No API key or Authorization header is stored. Failed runs retain their
failure artifact and complete log.

## 8. Non-goals

- No direct physical execution during preview;
- no LLM/VLM call in the high-frequency control loop;
- no learned geometry weights before P5.6 calibration;
- no use of privileged evaluator success as geometry evidence;
- no replacement of CAP-X registered Robot Skills;
- no claim that geometry evidence alone proves task success.
