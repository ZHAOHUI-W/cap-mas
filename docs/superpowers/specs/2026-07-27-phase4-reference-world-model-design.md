# Phase 4 Reference World Model Design

Status: Approved; reference implementation and live thread-mode CAP-X B5 gate
complete; live process-mode capture gate pending
Date: 2026-07-27

## 1. Purpose

Phase 4 adds a real-time, asynchronous World Model Plane to CAP-MAS. The
World Model is a geometric state-estimation pipeline, not a generative model
and not another LLM agent. It converts timestamped RGB-D and robot-state
observations into immutable, versioned `SceneSnapshot` values without placing
semantic perception or LLM inference on the control path.

The first implementation is a deterministic reference system that can be
tested without third-party dependencies. The existing CAP-X
`ObservationProvider` remains the observation source for action-boundary
execution; recording, replay, and streaming adapters feed the same
`ObservationBundle` contract. CAP-X streaming and process isolation are added
behind the same public interfaces. A TSDF implementation is an optional Phase
4 backend, not a Phase 5 prerequisite.

## 2. Goals

- Synchronize RGB-D frames, camera metadata, and robot state by timestamp.
- Provide fast camera/FK and depth-to-world geometry seams.
- Incrementally maintain a local sparse occupancy voxel map.
- Track known task objects with stable IDs and confidence-aware prediction.
- Publish immutable scene snapshots with explicit processing latency and
  runtime age semantics.
- Emit bounded asynchronous semantic-perception requests from deterministic
  geometry/tracking triggers; semantic inference remains an optional adapter.
- Offer thread reference and process benchmark runtimes through one interface.
- Preserve CAP-X API and typed-skill compatibility.
- Record deadline, freshness, queue, map, and worker health metrics.

## 3. Non-goals

- No LLM, VLM, SAM, or learned multi-object tracker in the reference backend.
- No semantic inference is required for the reference trigger detector;
  external semantic adapters may consume its requests.
- No physical robot action parallelism.
- No adaptive mission topology changes.
- No Memory Skill, rehearsal, OOD, or Arbiter evidence evolution; those remain
  Phase 5 work.
- No claim that the reference sparse map is equivalent to a production TSDF.

## 4. Confirmed architecture decisions

### 4.1 Runtime modes

The public `WorldModelRuntime` interface is shared by:

```text
WorldModelRuntime
|- ThreadWorldModelRuntime   # deterministic reference, replay, unit tests
`- ProcessWorldModelRuntime  # CAP-X B5 benchmark and isolation tests
```

The estimator, tracker, and map implementations are shared. Runtime-specific
code owns queues, lifecycle, IPC, restart behavior, and health reporting.

The thread runtime uses a bounded queue and is the default for replay and
contract tests. The process runtime is the acceptance path for B5 because the
control path must not share failure or latency with the semantic worker.

### 4.2 Time semantics

`SceneSnapshot` keeps both sensor and publish timestamps. New implementations
must expose:

```text
processing_latency_ms = publish_timestamp_ns - sensor_timestamp_ns
snapshot_age_ms(now)  = now - publish_timestamp_ns
```

`SceneSnapshot` gains a trailing `processing_latency_ms: float = 0.0` field so
existing positional constructors remain valid. `ObjectTrack` gains trailing
defaulted fields for `velocity_xyz`, `prediction_timestamp_ns`, and
`track_status`; existing tracks default to no velocity and `observed` status.

`scene_fresh(threshold_ms)` uses `snapshot_age_ms(now)` only. The existing
`freshness_ms` field remains temporarily for CAP-MAS compatibility and is
deprecated for new code. Arbiter evidence must not treat processing latency,
snapshot age, and prediction uncertainty as the same signal.

### 4.3 Map backend

The Phase 4 reference backend is `SparseVoxelMap`. Its immutable map snapshot
contains map version, voxel size, occupied voxels or block references, dirty
blocks, and the source timestamp. A later `TSDFMapBackend` can implement the
same `LocalMapBackend` interface. TSDF configuration is reserved now but is
disabled by default.

### 4.4 Tracking scope

The reference tracker handles known task objects. CAP-X object-pose results
are the primary LIBERO measurement source. Generic RGB-D measurements use
label plus position gating. Association is deterministic; missing objects use
a bounded constant-velocity prediction with confidence decay and explicit
`observed`, `predicted`, `stale`, or `lost` status.

### 4.5 Observation-source compatibility

`ObservationBundle` is the normalized observation value, not a replacement for
`ObservationProvider` or `CAPXObservationProvider`. The existing action-boundary
path remains:

```text
CAPXObservationProvider.capture()
    -> ObservationBundle
    -> CAPXRobotBackend._snapshot()
    -> SceneSnapshot
```

Phase 4 adds compatible sources:

```text
CAPXObservationProvider.capture()
    -> CAPXStreamingObservationSource
    -> SensorSynchronizer
    -> WorldModelRuntime
```

`RecordingObservationSource` wraps any `ObservationProvider` and persists
serializable bundles. `ReplayObservationSource` implements the same source
interface over recorded bundles. Existing P3.3 code does not migrate to the
streaming path.

The canonical bundle preserves the existing CAP-MAS perception contract and
adds only trailing defaulted fields:

```python
@dataclass(frozen=True)
class ObservationBundle:
    timestamp_ns: int
    frames: tuple[CameraFrame, ...]
    robot_state: Mapping[str, object]
    episode_id: str | None = None
    episode_epoch: int | None = None
    source: str = ""
    sequence: int = 0
```

RGB/depth artifact references and camera parameters are obtained from each
`CameraFrame`; they are not duplicated as top-level fields. Object pose
measurements remain an optional tracking input because the current CAP-X
provider exposes them through `capture_object_tracks()` rather than the raw
bundle. Existing positional constructors such as
`ObservationBundle(timestamp_ns, frames, robot_state)` remain valid.

`CameraModel` and `CameraFrame` are existing contracts in
`capmas/perception/protocol.py` and are reused unchanged by Phase 4:

```python
@dataclass(frozen=True)
class CameraModel:
    camera_id: str
    intrinsics: tuple[float, ...]
    pose_world: tuple[float, ...]

@dataclass(frozen=True)
class CameraFrame:
    camera_id: str
    timestamp_ns: int
    rgb: ArtifactRef | None
    depth: ArtifactRef | None
    camera: CameraModel
```

## 5. Components and contracts

### 5.1 Observation sources and sensor synchronization

`capmas/perception/sensor_sync.py` owns timestamp ordering and pairing. The
source/replay seam should support:

```python
class ObservationSource(ObservationProvider, Protocol):
    def capture(self) -> ObservationBundle: ...

class RecordingObservationSource(ObservationSource, Protocol):
    def iter_records(self) -> Iterator[ObservationBundle]: ...

class ReplayObservationSource(ObservationSource, Protocol):
    def exhausted(self) -> bool: ...
```

The synchronizer exposes metrics through a small immutable value:

```python
@dataclass(frozen=True)
class SynchronizerMetrics:
    accepted: int
    rejected_out_of_order: int
    rejected_episode: int
    dropped_oldest: int
    queued: int
```

`CAPXStreamingObservationSource` is an adapter over the existing CAP-X
provider. It may poll `capture()` at a configured rate or consume a backend
callback; it does not change the CAP-X API registry or typed-skill path.

The synchronizer seam should support:

```python
class SensorSynchronizer(Protocol):
    def push(self, observation: ObservationBundle) -> None: ...
    def pop_ready(self) -> ObservationBundle | None: ...
    def metrics(self) -> SynchronizerMetrics: ...
```

The implementation rejects cross-episode and out-of-order observations and
drops the oldest queued observation when capacity is exhausted. Drops and
rejects are counted rather than hidden.

`ObservationBundle` may gain a trailing sequence/source field with a default
value so existing CAP-X and test constructors remain valid. The source must
preserve episode identity, source timestamp, and sequence ordering. A replay
source must not use privileged evaluator state.

### 5.2 Fast geometry

`capmas/perception/geometry.py` owns a dependency-light geometry seam:

```python
class KinematicsBackend(Protocol):
    def camera_pose(self, robot_state: Mapping[str, object], camera_id: str): ...

@dataclass(frozen=True)
class GeometryUpdate:
    timestamp_ns: int
    camera_poses: Mapping[str, tuple[float, ...]]
    points_world: tuple[tuple[float, float, float], ...] = ()
    source_artifacts: tuple[ArtifactRef, ...] = ()

class GeometryEstimator(Protocol):
    def estimate(self, observation: ObservationBundle) -> GeometryUpdate: ...
```

When CAP-X supplies a camera pose, the reference estimator uses it directly.
An FK backend is used when the pose is absent. Geometry conversion must be
bounded and must not import the LLM or agent planes.

### 5.3 Incremental local map

`capmas/perception/local_map.py` owns:

```python
@dataclass(frozen=True)
class MapRegion:
    center_xyz: tuple[float, float, float]
    extents_xyz: tuple[float, float, float]

@dataclass(frozen=True)
class MapQueryResult:
    map_version: int
    occupied: bool
    clearance_m: float | None
    confidence: float
    snapshot_timestamp_ns: int

@dataclass(frozen=True)
class MapSnapshot:
    map_version: int
    voxel_size_m: float
    source_timestamp_ns: int
    occupied_voxels: tuple[tuple[int, int, int], ...]
    dirty_blocks: tuple[tuple[int, int, int], ...] = ()

@dataclass(frozen=True)
class MapUpdate:
    map_version: int
    changed_voxels: tuple[tuple[int, int, int], ...]
    source_timestamp_ns: int
```

```python
class LocalMapBackend(Protocol):
    def integrate(self, geometry: GeometryUpdate, timestamp_ns: int) -> MapUpdate: ...
    def query(self, region: MapRegion) -> MapQueryResult: ...
    def freeze_snapshot(self) -> MapSnapshot: ...
```

`SparseVoxelMap` stores only changed local voxels/blocks. It supports bounded
local regions, confidence, last-seen timestamps, dirty-block reporting, and an
immutable snapshot operation. The existing `SceneSnapshot.local_map` field
already stores an `ArtifactRef` in `capmas/contracts/scene.py`; Phase 4 keeps
that contract and never places a mutable map object or raw point array inside a
snapshot.

### 5.4 Known-object tracking

`capmas/perception/tracking.py` owns a `KnownObjectTracker` whose state is
private to the tracker instance and whose public operations are:

```python
@dataclass(frozen=True)
class ObjectMeasurement:
    track_id: str | None
    label: str
    pose_wxyz_xyz: tuple[float, ...]
    confidence: float
    timestamp_ns: int
    covariance: ArtifactRef | None = None
    evidence: tuple[ArtifactRef, ...] = ()

class KnownObjectTracker:
    def __init__(
        self,
        max_match_distance_m: float,
        prediction_timeout_ms: float,
        confidence_decay: float,
    ) -> None: ...

    def update(
        self,
        measurements: Sequence[ObjectMeasurement],
    ) -> tuple[ObjectTrack, ...]: ...

    def predict(self, timestamp_ns: int) -> tuple[ObjectTrack, ...]: ...
```

The tracker preserves known IDs when available, otherwise applies deterministic
label and distance gating. It records velocity, prediction timestamp, status,
confidence decay, and stale/lost transitions. It does not infer privileged
task completion.

### 5.5 World Model service and snapshot publication

`WorldModelService` is the pure processing seam. Runtime implementations own
queues and worker lifecycle:

```python
@dataclass(frozen=True)
class WorldModelHealth:
    status: str  # starting / healthy / degraded / stopped
    restart_count: int
    last_error: str | None = None
    last_snapshot_version: int | None = None

@dataclass(frozen=True)
class WorldModelRuntimeConfig:
    queue_capacity: int = 4
    max_restarts_per_episode: int = 3
    restart_backoff_ms: tuple[int, ...] = (50, 100, 200)
    restart_timeout_ms: int = 500

@dataclass(frozen=True)
class SynchronizerConfig:
    queue_capacity: int = 4
    max_age_ms: float = 150.0

class WorldModelService(Protocol):
    def process(
        self,
        observation: ObservationBundle,
        previous: SceneSnapshot | None,
    ) -> SceneSnapshot: ...

class WorldModelRuntime(Protocol):
    def start(self, initial_scene: SceneSnapshot) -> None: ...
    def submit(self, observation: ObservationBundle) -> bool: ...
    def latest_observation(self) -> SceneSnapshot: ...
    def health(self) -> WorldModelHealth: ...
    def stop(self, timeout_ms: int = 500) -> None: ...
```

The concrete runtime constructors are:

```python
class ThreadWorldModelRuntime:
    def __init__(
        self,
        service: WorldModelService,
        synchronizer: SensorSynchronizer,
        config: WorldModelRuntimeConfig,
        clock: Callable[[], int],
    ) -> None: ...

class ProcessWorldModelRuntime:
    def __init__(
        self,
        service_factory: Callable[[], WorldModelService],
        synchronizer_config: SynchronizerConfig,
        config: WorldModelRuntimeConfig,
        artifact_store: SharedArtifactStore,
    ) -> None: ...
```

The process `service_factory` must be picklable and must not capture a
simulator, open file handle, thread lock, lambda, or clock callable. The
process runtime creates the service inside the worker, where worker timestamps
come from `time.time_ns()` and scheduling/deadline measurements use
`time.monotonic_ns()`. The thread runtime retains an injected clock for
deterministic tests.

`WorldModelService` combines synchronized observations, geometry, map, and
tracking into a new `SceneSnapshot`. Every published snapshot is immutable,
belongs to one episode epoch, and has a monotonically increasing scene version.
The publisher never mutates a snapshot already used by an active contract.

The existing action-boundary CAP-X snapshot path remains valid. The streaming
path adds continuous publications without changing the agent-facing snapshot
contract.

### 5.6 Deterministic triggers and semantic adapter boundary

`capmas/perception/semantic_triggers.py` has two separate responsibilities:

1. `FastTriggerDetector` emits requests using only geometric/tracking state.
2. `SemanticRequestQueue` deduplicates and dispatches requests to an optional
   semantic adapter.

The queue has an explicit asynchronous seam:

```python
@dataclass(frozen=True)
class SemanticQueueMetrics:
    submitted: int
    deduplicated: int
    dropped: int
    completed: int
    timed_out: int
    queued: int

class SemanticRequestQueue(Protocol):
    def submit(self, request: PerceptionRequest) -> bool: ...
    def poll(self, max_items: int = 1) -> tuple[PerceptionRequest, ...]: ...
    def complete(self, request_id: str, result: PerceptionResult) -> None: ...
    def cancel_episode(self, episode_id: str, episode_epoch: int) -> int: ...
    def metrics(self) -> SemanticQueueMetrics: ...
```

`submit()` performs priority ordering, capacity checks, and deduplication.
`complete()` releases any pinned artifacts and publishes a version-bound
semantic result. Queue timeout is measured from request submission and never
blocks the fast World Model path.

The reference backend implements the detector and queue, not VLM/SAM or other
semantic inference. Deterministic trigger classes include:

- low scene or track confidence;
- ambiguous data association (small label/position gating margin);
- depth dropout, occlusion heuristic, or track loss;
- action-conditioned refresh requests at a checkpoint.

`geometric_semantic_disagreement` is accepted only when an external semantic
adapter has already attached semantic evidence to the current snapshot. It is
not a required reference-backend trigger.

Requests are deduplicated by episode, target track IDs, evidence type, and
scene version. A semantic timeout produces an uncertainty event and does not
stall geometry, map publication, or control.

### 5.7 Cross-process artifact bridge

`capmas/perception/artifact_bridge.py` owns the process-safe artifact seam:

```python
class SharedArtifactStore(Protocol):
    def put(self, value: bytes, media_type: str) -> ArtifactRef: ...
    def open(self, reference: ArtifactRef) -> BinaryIO: ...
    def exists(self, reference: ArtifactRef) -> bool: ...
    def pin(self, reference: ArtifactRef, ttl_ms: int) -> None: ...
    def release(self, reference: ArtifactRef) -> None: ...
```

`FileArtifactStore` uses a configured run directory or content-addressed
storage, atomic writes, checksums, and bounded retention. Process queues carry
strict JSON observation/snapshot envelopes containing metadata and artifact
URIs, never in-memory pointers. Large RGB-D and map payloads remain in shared
storage; only small thumbnails may use bounded base64 inline data.

The producer pins references until the consumer acknowledges them. Worker
failure and episode completion release references through TTL/GC. The
in-memory artifact store is supported only by the thread runtime.

The process envelope is metadata-only and has an explicit schema version:

```json
{
  "schema_version": "capmas.observation.v1",
  "episode_id": "episode-1",
  "episode_epoch": 1,
  "timestamp_ns": 123456789,
  "source": "capx",
  "sequence": 7,
  "robot_state": {
    "joint_position": {"uri": "artifact://sha256/...", "media_type": "array/joint-position"},
    "ee_pose_wxyz_xyz": [1.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3],
    "gripper_opening": 0.25
  },
  "frames": [
    {
      "camera_id": "agentview",
      "timestamp_ns": 123456789,
      "rgb": {"uri": "artifact://sha256/...", "media_type": "image/rgb"},
      "depth": {"uri": "artifact://sha256/...", "media_type": "image/depth"},
      "camera": {
        "camera_id": "agentview",
        "intrinsics": [1.0, 0.0, 2.0, 0.0, 1.0, 2.0, 0.0, 0.0, 1.0],
        "pose_world": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
      }
    }
  ]
}
```

This is the JSON serialization of `ObservationBundle`; it intentionally has
no `scene_version`, because the World Model publisher assigns scene versions
to output `SceneSnapshotEnvelope` values. The snapshot envelope is a separate
schema and contains `scene_version`, `publish_timestamp_ns`, map URI, tracks,
and visual evidence references. Both codecs reject unknown fields and never
serialize in-memory store objects, arbitrary Python values, or raw large frame
bytes.

## 6. Data flow

```text
RGB-D + robot state
    -> SensorSynchronizer
    -> GeometryEstimator / FK
    -> SparseVoxelMap.integrate
    -> KnownObjectTracker.update or predict
    -> SceneSnapshotPublisher
    -> StateStore / Verifier / Policy
    -> low-confidence trigger
    -> asynchronous SemanticPerception
    -> future snapshot correction
```

Semantic results are evidence-backed and version-bound. A correction is
accepted only if its episode identity is current and it is merged into a new
snapshot; it cannot rewrite a committed history.

## 7. Failure and backpressure behavior

| Condition | Required behavior |
| --- | --- |
| Out-of-order frame | Reject and count; do not rewind map/version |
| Cross-episode frame | Reject and count |
| Bounded queue full | Drop oldest frame and increment drop metric |
| Semantic timeout | Publish uncertainty; keep fast path running |
| Worker exception | Mark degraded, retain last valid snapshot, restart with bounded backoff |
| Map backend startup failure | Fail explicitly; never silently change backend |
| Stale action snapshot | Existing runtime freshness/stale rejection applies |
| Snapshot publication conflict | Fail closed and preserve the last valid snapshot |

The default worker policy is three restarts per episode with exponential
backoff of 50, 100, and 200 ms and a 500 ms per-start timeout. After restart
exhaustion, the runtime remains degraded and retains the last valid snapshot;
it does not silently mark the episode successful or failed. Once the snapshot
age exceeds the configured freshness limit, the existing skill-level safety
policy must hold, slow, or reject actions.

### 7.1 Observation versus committed state

Asynchronous World Model publications are observational snapshots. They are
not automatically action commits. The state store maintains separate views:

```text
committed_snapshot: v5
pending_observations: [v6, v7]
latest_observation: v7
```

The Phase 4 state-store seam is:

```python
class StateStore(Protocol):
    def publish_observation(self, snapshot: SceneSnapshot) -> None: ...
    def latest_observation(self) -> SceneSnapshot: ...
    def latest_committed(self) -> SceneSnapshot: ...
    def commit_after_action(
        self,
        parent_version: int,
        action_finished_at_ns: int,
        after: SceneSnapshot,
    ) -> bool: ...
```

`commit_after_action()` accepts a scene version greater than the parent, but
the after snapshot must have matching episode identity, must be published,
and must have `sensor_timestamp_ns >= action_finished_at_ns`. It must not
replace the after snapshot with an arbitrary latest observation captured before
the action finished.

The existing strict `compare_and_commit(parent, snapshot)` remains the legacy
contiguous-commit seam for Phase 2 and older callers. New Phase 4 code must
use the explicit observation and action-commit methods. Policy may read
`latest_observation()`; committed-state and evaluator logic use
`latest_committed()`.

## 8. Configuration

```yaml
perception:
  fast_geometry_hz: 20
  object_tracker_hz: 10
  semantic_trigger: event
  max_snapshot_age_ms: 150
  max_processing_latency_ms: 50
  runtime: thread
  queue_capacity: 4
  worker:
    max_restarts_per_episode: 3
    restart_backoff_ms: [50, 100, 200]
    restart_timeout_ms: 500
    exhaustion_policy: retain_last_snapshot
  artifact_store:
    backend: file  # memory is used by thread/replay tests
    root: "${run_dir}/artifacts"
    checksum: sha256
    max_bytes: 10737418240
    ttl_ms: 3600000
    pin_ttl_ms: 600000
    cleanup_on_episode_end: true
  map:
    backend: sparse_voxel
    voxel_size_m: 0.01
    local_radius_m: 1.0
    tsdf:
      enabled: false
      truncation_distance_m: 0.04

state_store:
  max_pending_observations: 8
  pending_drop_policy: drop_oldest
  require_contiguous_commit: false  # Phase 4; legacy runners override to true

legacy:
  require_contiguous_commit: true
```

The values are initial defaults, not quality claims. B5 must report observed
P50/P95 values and deadline misses. TSDF fields are reserved and validated but
must reject `enabled: true` until the TSDF backend is implemented.

`fast_geometry_hz` and `object_tracker_hz` are target update frequencies, not
hard maximums. A geometry period of 20 Hz is 50 ms and a tracker period of 10
Hz is 100 ms. If the input queue is behind, the runtime processes the newest
eligible observation and drops stale queued frames instead of blocking the
control path. It records target frequency, achieved frequency, queue delay,
and dropped-frame ratio. Semantic perception is event-triggered and has no
periodic frequency requirement.

## 9. Testing strategy

Tests target public seams rather than implementation details:

- timestamp pairing, out-of-order rejection, and bounded queue drops;
- camera pose/depth geometry using known numeric fixtures;
- sparse voxel incremental integration and immutable map snapshots;
- stable known-object IDs, prediction, confidence decay, and lost recovery;
- snapshot version/episode invariants and processing latency calculation;
- semantic request deduplication and timeout isolation;
- thread runtime start/stop and no semantic blocking;
- process runtime serialization, worker restart, and last-snapshot fallback;
- CAP-X action-boundary and streaming adapter parity.
- recording/replay source parity and strict JSON observation envelopes;
- shared artifact round-trip, atomic publication, pin/release, and TTL cleanup;
- pending observation versus committed state, including post-action timestamp
  validation and bounded pending-observation eviction;
- worker restart backoff, exhaustion degradation, and freshness safety fallback.

The B5 runner must preserve one JSON artifact and one log per run and record
control deadline, snapshot freshness, map update, semantic latency, queue
drops, worker restarts, and evaluator-only completion separately.

## 10. Acceptance criteria

Phase 4 first implementation is complete only when:

1. Reference replay produces monotonically versioned snapshots from timestamped
   observations.
2. Known-object tracks remain stable through observed, predicted, and stale
   intervals.
3. Sparse map updates are incremental and snapshots are immutable.
4. Semantic requests never block the fast World Model path.
5. Process mode can isolate a worker failure and retain a last valid snapshot.
6. Process observation/snapshot envelopes can resolve all required artifacts
   without sharing in-memory pointers.
7. CAP-X streaming remains compatible with the existing typed-skill runner.
8. Observation and committed-state versions remain distinct and post-action
   commits reject snapshots captured before action completion.
9. B5 artifacts contain the required real-time metrics.
10. The roadmap clearly distinguishes implemented reference behavior from the
   deferred TSDF backend and Phase 5 evidence features.

The current B5 runner accepts a JSONL replay recording and supports thread or
process reference runtimes. It records the required metrics and writes paired
JSON/log artifacts. A live CAP-X evaluator result remains a separate gate and
is not synthesized from replay snapshots.
