# Phase 4 Real-Time World Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task with a
> checkpoint after each task. Every code task follows the TDD red-green loop.

**Goal:** Implement the dependency-light Phase 4 World Model reference path,
then expose thread/process runtimes and a CAP-X B5 streaming benchmark without
blocking robot control on semantic perception.

**Architecture:** Preserve the existing `ObservationProvider` and
`ObservationBundle` contracts while adding trailing metadata fields. Build a
pure `WorldModelService` from geometry, sparse voxels, and deterministic known
object tracking. Wrap it with bounded thread and process runtimes; process
transport uses strict metadata-only JSON envelopes and a shared file artifact
store. Keep legacy action-boundary CAP-X execution and strict state commits
unchanged until the new observation/commit APIs are explicitly used.

**Tech Stack:** Python 3.10+, standard library only for the reference path,
dataclasses, `typing.Protocol`, `threading`, `queue`, `multiprocessing`, JSON,
filesystem atomic rename, and the existing pytest suite. No new mandatory
NumPy, Open3D, VLM, SAM, or LLM dependency.

## Global Constraints

- Preserve `ObservationBundle(timestamp_ns, frames, robot_state)` positional
  construction and existing CAP-X action-boundary behavior.
- `CameraModel`, `CameraFrame`, and `SceneSnapshot.local_map` remain the
  existing contracts in the current codebase.
- `SceneSnapshot` and `ObjectTrack` additions must be trailing defaulted fields.
- `scene_fresh()` uses runtime snapshot age, not processing latency.
- Thread runtime may inject a clock for deterministic tests; process workers
  call `time.time_ns()` and `time.monotonic_ns()` internally.
- Process queues carry strict JSON metadata and artifact URIs, never in-memory
  store objects or large raw frame bytes.
- Observation snapshots and committed action state are separate state-store
  views; legacy `compare_and_commit()` keeps strict contiguous semantics.
- Sparse voxel is the only active map backend in this phase; TSDF configuration
  is validated but `enabled: true` is rejected.
- Semantic triggers may emit requests, but the reference path does not run
  VLM, SAM, LLM, or a learned tracker.
- Every vertical slice must run its focused pytest command before continuing.
- The parent repository ignores `workspace/`; do not force-add or commit the
  project into the parent repository during implementation.

## File Map

Create:

- `capmas/perception/serialization.py` - strict Observation and Snapshot JSON
  envelopes.
- `capmas/perception/artifact_bridge.py` - shared file artifact backend and
  pin/release lifecycle.
- `capmas/perception/sensor_sync.py` - observation sources, recording/replay,
  bounded synchronization, and metrics.
- `capmas/perception/geometry.py` - camera/FK seam and reference geometry
  conversion.
- `capmas/perception/local_map.py` - sparse voxel types and backend.
- `capmas/perception/tracking.py` - object measurements and
  `KnownObjectTracker`.
- `capmas/perception/semantic_triggers.py` - deterministic triggers and
  bounded semantic request queue.
- `capmas/perception/world_model.py` - pure service, health/config types, and
  thread/process runtimes.
- `capmas/perception/metrics.py` - real-time counters and latency summaries.
- `tests/test_phase4_contracts.py`
- `tests/test_phase4_artifact_bridge.py`
- `tests/test_phase4_serialization.py`
- `tests/test_phase4_sensor_sync.py`
- `tests/test_phase4_geometry.py`
- `tests/test_phase4_local_map.py`
- `tests/test_phase4_tracking.py`
- `tests/test_phase4_semantic_triggers.py`
- `tests/test_phase4_world_model.py`
- `tests/test_phase4_state_store.py`
- `tests/test_phase4_runtime.py`

Modify:

- `capmas/perception/protocol.py` - trailing ObservationBundle metadata.
- `capmas/contracts/scene.py` - processing latency and prediction fields.
- `capmas/contracts/__init__.py` - public contract exports if needed.
- `capmas/perception/__init__.py` - public Phase 4 exports.
- `capmas/runtime/state_store.py` - observation/commit separation while
  preserving legacy methods.
- `capmas/backends/capx.py` - processing-latency calculation and streaming
  source adapter.
- `configs/default.yaml` - Phase 4 runtime, artifact, map, and state-store
  defaults.
- `docs/implementation-roadmap.md` - Phase 4 progress and remaining B5 gate.
- `docs/design/world-model-api.md` - executable reference interfaces.
- `docs/real-time-perception.md` - queue, frequency, and failure semantics.
- `docs/adr/0002-three-plane-timing.md` or a new Phase 4 ADR - process and
  freshness decisions.

## Task 1: Extend Contracts Without Breaking CAP-X

**Files:**

- Modify: `capmas/perception/protocol.py:26-31`
- Modify: `capmas/contracts/scene.py:45-71`
- Test: `tests/test_phase4_contracts.py`

**Interfaces:**

- Consumes: existing `CameraModel`, `CameraFrame`, `ObservationBundle`,
  `ObjectTrack`, and `SceneSnapshot`.
- Produces: trailing `ObservationBundle` fields
  `episode_id`, `episode_epoch`, `source`, `sequence`; trailing
  `SceneSnapshot.processing_latency_ms`; trailing `ObjectTrack.velocity_xyz`,
  `prediction_timestamp_ns`, and `track_status`.

- [x] **Step 1: Write the failing tests.**

```python
def test_observation_bundle_keeps_legacy_positional_constructor() -> None:
    bundle = ObservationBundle(100, (), {"gripper_opening": 1.0})
    assert bundle.timestamp_ns == 100
    assert bundle.episode_id is None
    assert bundle.sequence == 0


def test_scene_and_track_phase4_fields_have_compatibility_defaults() -> None:
    scene = SceneSnapshot("episode", 1, 0, 100, 110, {})
    track = ObjectTrack("obj", "cube", (1, 0, 0, 0, 0, 0, 0), 0.9, 100)
    assert scene.processing_latency_ms == 0.0
    assert track.velocity_xyz is None
    assert track.prediction_timestamp_ns is None
    assert track.track_status == "observed"
```

- [x] **Step 2: Run the focused tests and verify the expected failure.**

Run:

```bash
pytest -q tests/test_phase4_contracts.py
```

Expected: FAIL because the new attributes do not yet exist.

- [x] **Step 3: Implement the minimum trailing fields.**

Append fields with defaults after the existing fields. Do not reorder existing
dataclass parameters and do not convert `Sequence[ObjectTrack]` in this task.

- [x] **Step 4: Run the focused and compatibility tests.**

```bash
pytest -q tests/test_phase4_contracts.py tests/test_capx_adapter.py
```

Expected: PASS.

- [x] **Step 5: Refactor only after green.**

Export new public types if an import boundary requires it; rerun the same
tests after the refactor.

## Task 2: Shared Artifacts and Strict JSON Envelopes

**Files:**

- Create: `capmas/perception/artifact_bridge.py`
- Create: `capmas/perception/serialization.py`
- Test: `tests/test_phase4_artifact_bridge.py`
- Test: `tests/test_phase4_serialization.py`

**Interfaces:**

```python
class SharedArtifactStore(Protocol):
    def put(self, value: bytes, media_type: str) -> ArtifactRef: ...
    def open(self, reference: ArtifactRef) -> BinaryIO: ...
    def exists(self, reference: ArtifactRef) -> bool: ...
    def pin(self, reference: ArtifactRef, ttl_ms: int) -> None: ...
    def release(self, reference: ArtifactRef) -> None: ...
```

`FileArtifactStore` must write to a configured root using a temporary file,
flush and close it, then use `os.replace`. The URI must include a SHA-256
content key. `pin` extends a sidecar expiry; `release` removes the pin but does
not delete the content before TTL cleanup.

- [x] **Step 1: Write the failing artifact tests.**

```python
def test_file_artifact_round_trip_uses_content_addressed_uri(tmp_path) -> None:
    store = FileArtifactStore(tmp_path, checksum="sha256")
    ref = store.put(b"rgb-bytes", "image/rgb")
    assert ref.uri.startswith("artifact://sha256/")
    assert store.exists(ref)
    assert store.open(ref).read() == b"rgb-bytes"


def test_release_does_not_delete_pinned_content_before_ttl(tmp_path) -> None:
    store = FileArtifactStore(tmp_path, checksum="sha256")
    ref = store.put(b"map", "application/octet-stream")
    store.pin(ref, ttl_ms=1000)
    store.release(ref)
    assert store.exists(ref)
```

- [x] **Step 2: Run the tests and verify they fail because the bridge is absent.**

```bash
pytest -q tests/test_phase4_artifact_bridge.py
```

- [x] **Step 3: Implement atomic file storage and lifecycle.**

Reject path traversal, unknown artifact schemes, and writes larger than the
configured `max_bytes`. Use `ArtifactRef(uri, media_type)` and store media type
in a sidecar metadata file. Do not use pickle for cross-process payloads.

- [x] **Step 4: Write the failing serialization tests.**

```python
def test_observation_envelope_round_trip_preserves_camera_and_artifact_refs():
    bundle = make_observation_bundle()
    encoded = observation_to_json(bundle)
    restored = observation_from_json(encoded)
    assert restored == bundle


def test_observation_envelope_rejects_scene_version_and_unknown_fields():
    payload = make_observation_payload()
    payload["scene_version"] = 4
    with pytest.raises(ValueError):
        observation_from_json(json.dumps(payload))
```

- [x] **Step 5: Implement strict ObservationBundle serialization.**

Serialize `timestamp_ns`, optional episode fields, source, sequence,
`robot_state`, and every frame's camera model and artifact references. Reject
unknown fields and arbitrary object values. Add a separate snapshot codec with
`scene_version`; do not reuse the observation schema for snapshot output.

- [x] **Step 6: Run both focused suites.**

```bash
pytest -q tests/test_phase4_artifact_bridge.py tests/test_phase4_serialization.py
```

Expected: PASS.

## Task 3: Recording, Replay, and Bounded Synchronization

**Files:**

- Create: `capmas/perception/sensor_sync.py`
- Test: `tests/test_phase4_sensor_sync.py`

**Interfaces:**

```python
class RecordingObservationSource(Protocol):
    def iter_records(self) -> Iterator[ObservationBundle]: ...

class ReplayObservationSource(Protocol):
    def capture(self) -> ObservationBundle: ...
    def exhausted(self) -> bool: ...

class SensorSynchronizer(Protocol):
    def push(self, observation: ObservationBundle) -> None: ...
    def pop_ready(self) -> ObservationBundle | None: ...
    def metrics(self) -> SynchronizerMetrics: ...
```

- [x] **Step 1: Write failing source and queue tests.**

Cover these independent behaviors:

```python
def test_replay_source_streams_one_bundle_without_full_materialization(tmp_path):
    first = make_observation_bundle(sequence=1)
    second = make_observation_bundle(sequence=2)
    recorder = JsonlObservationRecorder(tmp_path / "observations.jsonl")
    recorder.append(first)
    recorder.append(second)

    replay = JsonlReplaySource(tmp_path / "observations.jsonl")
    assert replay.capture() == first
    assert not replay.exhausted()
    assert replay.capture() == second
    assert replay.exhausted()
    with pytest.raises(StopIteration):
        replay.capture()


def test_synchronizer_rejects_out_of_order_sequence():
    synchronizer = BoundedSensorSynchronizer(capacity=2, episode_id="ep", episode_epoch=1)
    synchronizer.push(make_observation_bundle(sequence=2))
    with pytest.raises(ValueError, match="monotonic"):
        synchronizer.push(make_observation_bundle(sequence=1))
    metrics = synchronizer.metrics()
    assert metrics.accepted == 1
    assert metrics.rejected_out_of_order == 1


def test_synchronizer_rejects_wrong_episode():
    synchronizer = BoundedSensorSynchronizer(capacity=2, episode_id="ep", episode_epoch=1)
    with pytest.raises(ValueError, match="episode"):
        synchronizer.push(make_observation_bundle(episode_id="other", sequence=1))
    assert synchronizer.metrics().rejected_episode == 1


def test_synchronizer_drops_oldest_when_capacity_is_full():
    synchronizer = BoundedSensorSynchronizer(capacity=2, episode_id="ep", episode_epoch=1)
    synchronizer.push(make_observation_bundle(sequence=1))
    synchronizer.push(make_observation_bundle(sequence=2))
    synchronizer.push(make_observation_bundle(sequence=3))
    assert synchronizer.pop_ready().sequence == 2
    assert synchronizer.pop_ready().sequence == 3
    assert synchronizer.metrics().dropped_oldest == 1
```

Each test must assert the public result and `SynchronizerMetrics`, not private
queue contents.

- [x] **Step 2: Run the focused tests and confirm the missing module failure.**

```bash
pytest -q tests/test_phase4_sensor_sync.py
```

- [x] **Step 3: Implement append-only recording and iterator replay.**

Use one JSON envelope per line for metadata-only replay records. The replay
source reads one line per `capture()` call and raises `StopIteration` only
after `exhausted()` becomes true.

- [x] **Step 4: Implement bounded synchronization.**

Track the configured episode, last sequence, and queue capacity. Reject older
timestamps or sequence values, reject mismatched episode identity when both
values are present, and drop the oldest queued bundle on overflow.

- [x] **Step 5: Run compatibility and focused tests.**

```bash
pytest -q tests/test_phase4_sensor_sync.py tests/test_capx_adapter.py
```

## Task 4: Fast Geometry and Camera Pose Conversion

**Files:**

- Create: `capmas/perception/geometry.py`
- Test: `tests/test_phase4_geometry.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class GeometryUpdate:
    timestamp_ns: int
    camera_poses: Mapping[str, tuple[float, ...]]
    points_world: tuple[tuple[float, float, float], ...] = ()
    source_artifacts: tuple[ArtifactRef, ...] = ()

class KinematicsBackend(Protocol):
    def camera_pose(
        self,
        robot_state: Mapping[str, object],
        camera_id: str,
    ) -> tuple[float, ...] | None: ...

class GeometryEstimator(Protocol):
    def estimate(self, observation: ObservationBundle) -> GeometryUpdate: ...
```

- [x] **Step 1: Write failing numeric tests.**

```python
def test_geometry_uses_camera_pose_from_frame_for_world_points(tmp_path):
    depth = FileArtifactStore(tmp_path).put(b"0.5", "text/plain")
    frame = make_frame(depth=depth, pose_world=(1, 0, 0, 0, 1, 0, 0))
    observation = make_observation_bundle(frames=(frame,))
    estimator = ReferenceGeometryEstimator(
        artifact_store=FileArtifactStore(tmp_path),
        depth_decoder=SinglePixelDepthDecoder((0.0, 0.0, 0.5)),
    )

    update = estimator.estimate(observation)

    assert update.points_world == ((1.0, 0.0, 0.5),)
    assert update.camera_poses[frame.camera_id] == frame.camera.pose_world


def test_geometry_uses_fk_backend_when_camera_pose_is_empty(tmp_path):
    frame = make_frame(depth=None, pose_world=())
    observation = make_observation_bundle(frames=(frame,))
    fk = FixedKinematicsBackend({frame.camera_id: (1, 0, 0, 0, 1, 0, 0)})
    estimator = ReferenceGeometryEstimator(
        artifact_store=FileArtifactStore(tmp_path),
        depth_decoder=SinglePixelDepthDecoder((0.0, 0.0, 0.5)),
        kinematics=fk,
    )

    update = estimator.estimate(observation)

    assert update.camera_poses[frame.camera_id] == fk.poses[frame.camera_id]


def test_geometry_does_not_emit_points_for_missing_depth_artifact(tmp_path):
    frame = make_frame(depth=None)
    estimator = ReferenceGeometryEstimator(
        artifact_store=FileArtifactStore(tmp_path),
        depth_decoder=SinglePixelDepthDecoder((0.0, 0.0, 0.5)),
    )

    update = estimator.estimate(make_observation_bundle(frames=(frame,)))

    assert update.points_world == ()
```

Use a one-point fixture with an identity camera pose and a known depth value;
expected world coordinates must be literal values independent of the helper
implementation.

- [x] **Step 2: Run the tests and verify the expected missing-module failure.**

```bash
pytest -q tests/test_phase4_geometry.py
```

- [x] **Step 3: Implement the reference estimator.**

Keep point conversion dependency-free. Accept an injected depth decoder rather
than importing NumPy. Propagate source artifact references and timestamp.
Reject malformed poses and non-finite points instead of producing NaNs.

- [x] **Step 4: Run focused tests and existing perception tests.**

```bash
pytest -q tests/test_phase4_geometry.py tests/test_capx_adapter.py
```

## Task 5: Sparse Voxel Map

**Files:**

- Create: `capmas/perception/local_map.py`
- Test: `tests/test_phase4_local_map.py`

**Interfaces:**

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

class SparseVoxelMap:
    def __init__(self, voxel_size_m: float, local_radius_m: float) -> None: ...
    def integrate(self, geometry: GeometryUpdate, timestamp_ns: int) -> MapUpdate: ...
    def query(self, region: MapRegion) -> MapQueryResult: ...
    def freeze_snapshot(self) -> MapSnapshot: ...
```

- [x] **Step 1: Write failing map behavior tests.**

```python
def test_integrate_changes_only_voxels_touched_by_new_geometry():
    voxel_map = SparseVoxelMap(voxel_size_m=0.1, local_radius_m=1.0)
    first = voxel_map.integrate(make_geometry(points=((0.04, 0.04, 0.04),)), 100)
    second = voxel_map.integrate(make_geometry(points=((0.24, 0.04, 0.04),)), 200)

    assert first.changed_voxels == ((0, 0, 0),)
    assert second.changed_voxels == ((2, 0, 0),)
    assert second.map_version == first.map_version + 1


def test_query_reports_occupied_voxel_and_map_version():
    voxel_map = SparseVoxelMap(voxel_size_m=0.1, local_radius_m=1.0)
    voxel_map.integrate(make_geometry(points=((0.04, 0.04, 0.04),)), 100)

    result = voxel_map.query(MapRegion(center_xyz=(0, 0, 0), extents_xyz=(0.2, 0.2, 0.2)))

    assert result.map_version == 1
    assert result.occupied is True
    assert result.confidence > 0.0
    assert result.snapshot_timestamp_ns == 100


def test_freeze_snapshot_is_immutable_and_does_not_change_after_next_update():
    voxel_map = SparseVoxelMap(voxel_size_m=0.1, local_radius_m=1.0)
    voxel_map.integrate(make_geometry(points=((0.04, 0.04, 0.04),)), 100)
    frozen = voxel_map.freeze_snapshot()
    voxel_map.integrate(make_geometry(points=((0.24, 0.04, 0.04),)), 200)

    assert frozen.map_version == 1
    assert frozen.voxel_size_m == 0.1
    assert frozen.source_timestamp_ns == 100
    assert frozen.occupied_voxels == ((0, 0, 0),)


def test_points_outside_local_radius_are_excluded():
    voxel_map = SparseVoxelMap(voxel_size_m=0.1, local_radius_m=1.0)
    update = voxel_map.integrate(make_geometry(points=((1.1, 0, 0),)), 100)

    assert update.changed_voxels == ()
    assert voxel_map.freeze_snapshot().occupied_voxels == ()
```

- [x] **Step 2: Verify red.**

```bash
pytest -q tests/test_phase4_local_map.py
```

- [x] **Step 3: Implement sparse voxel indexing.**

Use integer voxel keys from `floor(position / voxel_size_m)`, a bounded local
region around the configured origin, monotonically increasing `map_version`,
and immutable tuple snapshots. Store occupancy/confidence internally; expose
only `MapSnapshot` and `MapQueryResult` at the public seam.

- [x] **Step 4: Verify green and run contract tests.**

```bash
pytest -q tests/test_phase4_local_map.py tests/test_phase4_contracts.py
```

## Task 6: Deterministic Known-Object Tracking

**Files:**

- Create: `capmas/perception/tracking.py`
- Modify: `capmas/contracts/scene.py:45-55`
- Test: `tests/test_phase4_tracking.py`

**Interfaces:**

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
    def update(self, measurements: Sequence[ObjectMeasurement]) -> tuple[ObjectTrack, ...]: ...
    def predict(self, timestamp_ns: int) -> tuple[ObjectTrack, ...]: ...
```

- [x] **Step 1: Write failing tracker tests.**

```python
def test_explicit_track_id_is_preserved_across_updates():
    tracker = KnownObjectTracker(max_match_distance_m=0.2)
    first = tracker.update((measurement("cube", (0, 0, 0), track_id="cube-7", timestamp_ns=100),))
    second = tracker.update((measurement("cube", (0.1, 0, 0), track_id="cube-7", timestamp_ns=200),))

    assert first[0].track_id == "cube-7"
    assert second[0].track_id == "cube-7"
    assert second[0].last_seen_ns == 200


def test_label_and_distance_gate_associates_measurement_without_id():
    tracker = KnownObjectTracker(max_match_distance_m=0.2)
    tracker.update((measurement("cube", (0, 0, 0), timestamp_ns=100),))

    tracks = tracker.update((measurement("cube", (0.1, 0, 0), timestamp_ns=200),))

    assert len(tracks) == 1
    assert tracks[0].track_id == "cube-0"


def test_missing_measurement_uses_constant_velocity_prediction():
    tracker = KnownObjectTracker(max_match_distance_m=0.2, prediction_timeout_ms=500)
    tracker.update((measurement("cube", (0, 0, 0), timestamp_ns=100),))
    tracker.update((measurement("cube", (0.1, 0, 0), timestamp_ns=200),))

    predicted = tracker.predict(300)

    assert predicted[0].track_status == "predicted"
    assert predicted[0].pose_wxyz_xyz[4] == 0.2


def test_prediction_confidence_decays_then_enters_stale_and_lost_states():
    tracker = KnownObjectTracker(
        max_match_distance_m=0.2,
        prediction_timeout_ms=100,
        stale_timeout_ms=200,
        confidence_decay=0.5,
    )
    tracker.update((measurement("cube", (0, 0, 0), timestamp_ns=100_000_000),))

    assert tracker.predict(250_000_000)[0].track_status == "stale"
    assert tracker.predict(400_000_001)[0].track_status == "lost"
```

- [x] **Step 2: Run red.**

```bash
pytest -q tests/test_phase4_tracking.py
```

- [x] **Step 3: Implement `KnownObjectTracker`.**

Maintain state per known track. Use explicit IDs first, then same-label nearest
neighbor under `max_match_distance_m`. Estimate velocity from two observed
positions, predict only through `prediction_timeout_ms`, and apply configured
confidence decay. Preserve evidence and covariance references.

- [x] **Step 4: Verify green and verifier compatibility.**

```bash
pytest -q tests/test_phase4_tracking.py tests/test_libero_verifier.py
```

## Task 7: Semantic Trigger Queue and Pure World Model Service

**Files:**

- Create: `capmas/perception/semantic_triggers.py`
- Create: `capmas/perception/world_model.py`
- Test: `tests/test_phase4_semantic_triggers.py`
- Test: `tests/test_phase4_world_model.py`

**Interfaces:**

```python
class SemanticRequestQueue(Protocol):
    def submit(self, request: PerceptionRequest) -> bool: ...
    def poll(self, max_items: int = 1) -> tuple[PerceptionRequest, ...]: ...
    def complete(self, request_id: str, result: PerceptionResult) -> None: ...
    def cancel_episode(self, episode_id: str, episode_epoch: int) -> int: ...
    def metrics(self) -> SemanticQueueMetrics: ...

class WorldModelService(Protocol):
    def process(
        self,
        observation: ObservationBundle,
        previous: SceneSnapshot | None,
    ) -> SceneSnapshot: ...
```

- [x] **Step 1: Write failing trigger tests.**

```python
def test_low_confidence_track_emits_deduplicated_request():
    queue = DeterministicSemanticRequestQueue()
    trigger = DeterministicSemanticTrigger(queue, confidence_threshold=0.5)
    scene = make_scene(objects=(track("cube-1", confidence=0.2),))

    trigger.inspect(scene)
    trigger.inspect(scene)

    requests = queue.poll(max_items=10)
    assert len(requests) == 1
    assert requests[0].target_track_ids == ("cube-1",)
    assert queue.metrics().deduplicated == 1


def test_queue_returns_high_priority_request_first():
    queue = DeterministicSemanticRequestQueue()
    queue.submit(make_request("low", priority=1))
    queue.submit(make_request("high", priority=10))

    assert [request.request_id for request in queue.poll(max_items=2)] == ["high", "low"]


def test_queue_timeout_does_not_change_fast_snapshot_processing():
    queue = DeterministicSemanticRequestQueue(max_latency_ms=1)
    service = make_world_model(semantic_queue=queue)
    before = make_scene(scene_version=3)

    after = service.process(make_observation_bundle(timestamp_ns=1_000_000), before)

    assert after.scene_version == 4
    assert after.processing_latency_ms >= 0.0
    assert queue.metrics().timed_out == 0
```

- [x] **Step 2: Verify red.**

```bash
pytest -q tests/test_phase4_semantic_triggers.py
```

- [x] **Step 3: Implement deterministic triggers and queue.**

Trigger only on scene/track confidence, association margin, depth dropout,
track loss, and explicit checkpoint refresh. Gate geometric/semantic
disagreement on the presence of external semantic evidence. Deduplicate by
episode, target IDs, evidence types, and scene version. Do not call any model.

- [x] **Step 4: Write failing World Model service tests.**

```python
def test_world_model_process_publishes_new_snapshot_with_latency():
    service = make_world_model(clock=SequenceClock(200))
    previous = make_scene(scene_version=4, sensor_timestamp_ns=100)

    snapshot = service.process(make_observation_bundle(timestamp_ns=100), previous)

    assert snapshot.scene_version == 5
    assert snapshot.processing_latency_ms == 0.0001
    assert snapshot.sensor_timestamp_ns == 100


def test_world_model_process_does_not_mutate_previous_snapshot():
    service = make_world_model(clock=SequenceClock(200))
    previous = make_scene(scene_version=4, objects=(track("cube-1"),))

    snapshot = service.process(make_observation_bundle(timestamp_ns=100), previous)

    assert previous.scene_version == 4
    assert tuple(previous.objects) == (track("cube-1"),)
    assert snapshot is not previous


def test_world_model_rejects_cross_episode_observation():
    service = make_world_model()
    previous = make_scene(episode_id="episode-a", episode_epoch=1)
    observation = make_observation_bundle(episode_id="episode-b", episode_epoch=1)

    with pytest.raises(ValueError, match="episode"):
        service.process(observation, previous)
```

- [x] **Step 5: Implement the pure service.**

Compose geometry, map, tracker, artifact publisher, and snapshot version
allocation. Set `processing_latency_ms` to publish time minus observation
timestamp. Use an injected clock only in this pure/thread path.

- [x] **Step 6: Run focused world-model tests.**

```bash
pytest -q tests/test_phase4_semantic_triggers.py tests/test_phase4_world_model.py
```

## Task 8: Observation/Committed State Store Semantics

**Files:**

- Modify: `capmas/runtime/state_store.py`
- Test: `tests/test_phase4_state_store.py`

**Interfaces:**

```python
class StateStore:
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

- [x] **Step 1: Write failing state-store tests.**

```python
def test_pending_observations_do_not_replace_committed_snapshot():
    store = InMemoryStateStore(max_pending_observations=3)
    initial = make_scene(scene_version=1, sensor_timestamp_ns=100)
    store.start_episode(initial)
    store.publish_observation(make_scene(scene_version=2, sensor_timestamp_ns=200))

    assert store.latest_committed() == initial
    assert store.latest_observation().scene_version == 2


def test_latest_observation_returns_highest_published_version():
    store = InMemoryStateStore(max_pending_observations=3)
    store.start_episode(make_scene(scene_version=1))
    store.publish_observation(make_scene(scene_version=2))
    store.publish_observation(make_scene(scene_version=3))

    assert store.latest_observation().scene_version == 3


def test_commit_after_action_accepts_noncontiguous_post_action_version():
    store = InMemoryStateStore(max_pending_observations=3)
    store.start_episode(make_scene(scene_version=1, sensor_timestamp_ns=100))
    store.publish_observation(make_scene(scene_version=2, sensor_timestamp_ns=150))
    after = make_scene(scene_version=3, sensor_timestamp_ns=300)
    store.publish_observation(after)

    assert store.commit_after_action(1, action_finished_at_ns=200, after=after)
    assert store.latest_committed() == after


def test_commit_after_action_rejects_snapshot_captured_before_action_end():
    store = InMemoryStateStore(max_pending_observations=3)
    store.start_episode(make_scene(scene_version=1, sensor_timestamp_ns=100))
    before_action = make_scene(scene_version=2, sensor_timestamp_ns=150)
    store.publish_observation(before_action)

    assert not store.commit_after_action(1, action_finished_at_ns=200, after=before_action)
    assert store.latest_committed().scene_version == 1


def test_pending_observations_drop_oldest_at_configured_capacity():
    store = InMemoryStateStore(max_pending_observations=2)
    store.start_episode(make_scene(scene_version=1))
    store.publish_observation(make_scene(scene_version=2))
    store.publish_observation(make_scene(scene_version=3))
    store.publish_observation(make_scene(scene_version=4))

    assert store.pending_versions() == (3, 4)


def test_legacy_compare_and_commit_stays_contiguous():
    store = InMemoryStateStore()
    store.start_episode(make_scene(scene_version=1))

    with pytest.raises(ValueError, match="increment"):
        store.compare_and_commit(1, make_scene(scene_version=3))
```

- [x] **Step 2: Run red.**

```bash
pytest -q tests/test_phase4_state_store.py
```

- [x] **Step 3: Implement separate state views.**

Keep `start_episode`, `publish`, `latest`, `get`, and strict
`compare_and_commit` behavior for existing callers. Add a bounded ordered
pending-observation collection. `commit_after_action` requires matching
episode identity, `after.scene_version > parent_version`, publication status,
and `after.sensor_timestamp_ns >= action_finished_at_ns`.

- [x] **Step 4: Run all state/runtime regression tests.**

```bash
pytest -q tests/test_phase4_state_store.py tests/test_runtime_cycle.py tests/test_rolling_scheduler.py
```

## Task 9: Thread and Process World Model Runtimes

**Files:**

- Modify: `capmas/perception/world_model.py`
- Modify: `capmas/perception/serialization.py`
- Test: `tests/test_phase4_runtime.py`

**Interfaces:**

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

- [x] **Step 1: Write failing runtime tests.**

```python
def test_thread_runtime_processes_observation_without_blocking_submit():
    runtime = ThreadWorldModelRuntime(
        service=make_world_model(),
        synchronizer=BoundedSensorSynchronizer(capacity=2),
        config=WorldModelRuntimeConfig(queue_capacity=2),
        clock=time.time_ns,
    )
    runtime.start()
    assert runtime.submit(make_observation_bundle(sequence=1))
    assert runtime.wait_until_processed(timeout_s=1.0)
    runtime.stop()


def test_thread_runtime_exposes_latest_snapshot_and_health():
    runtime = make_thread_runtime()
    runtime.start()
    runtime.submit(make_observation_bundle(sequence=1))
    assert runtime.wait_until_processed(timeout_s=1.0)

    assert runtime.latest_snapshot().scene_version == 0
    assert runtime.health().status == "healthy"
    runtime.stop()


def test_process_runtime_uses_picklable_factory_and_metadata_envelope(tmp_path):
    runtime = ProcessWorldModelRuntime(
        service_factory=top_level_world_model_factory,
        synchronizer_config=SynchronizerConfig(queue_capacity=2),
        config=WorldModelRuntimeConfig(restart_timeout_ms=500),
        artifact_store=FileArtifactStore(tmp_path),
    )
    runtime.start()
    runtime.submit(make_observation_bundle(sequence=1))

    assert runtime.wait_until_processed(timeout_s=2.0)
    assert runtime.last_transport_message().format == "scene_snapshot_v1"
    runtime.stop()


def test_process_runtime_restarts_worker_and_retains_last_snapshot(tmp_path):
    runtime = make_process_runtime(tmp_path, service_factory=crash_after_first_snapshot_factory)
    runtime.start()
    runtime.submit(make_observation_bundle(sequence=1))
    assert runtime.wait_until_processed(timeout_s=2.0)
    previous = runtime.latest_snapshot()
    runtime.force_worker_exit()

    assert runtime.wait_until_healthy(timeout_s=2.0)
    assert runtime.latest_snapshot() == previous
    runtime.stop()


def test_process_runtime_marks_degraded_after_restart_budget_exhaustion(tmp_path):
    runtime = make_process_runtime(tmp_path, service_factory=always_crashing_factory)
    runtime.start()

    assert runtime.wait_until_degraded(timeout_s=5.0)
    assert runtime.health().status == "degraded"
    assert not runtime.episode_completed()
    runtime.stop()
```

- [x] **Step 2: Run red.**

```bash
pytest -q tests/test_phase4_runtime.py
```

- [x] **Step 3: Implement thread runtime.**

Use `queue.Queue(maxsize=queue_capacity)`, a daemon worker, latest-frame
backpressure, explicit `start/stop`, and `WorldModelHealth`. The worker must
not call semantic inference.

- [x] **Step 4: Implement process runtime.**

Use a top-level picklable worker entrypoint and `multiprocessing` queues. Do
not pass a clock callable or simulator object. The worker creates the service
and uses `time.time_ns()`/`time.monotonic_ns()`. Parent/worker messages use the
strict observation/snapshot envelope and shared artifact URIs.

- [x] **Step 5: Implement bounded restart and fallback.**

Retry up to three times with 50/100/200 ms backoff and a 500 ms start timeout.
Keep the last valid snapshot during restart. After exhaustion, set health to
`degraded`; do not mark episode success or failure. Let freshness policy reject
actions after the snapshot age threshold.

- [x] **Step 6: Run focused and full runtime tests.**

```bash
pytest -q tests/test_phase4_runtime.py tests/test_phase4_serialization.py tests/test_phase4_state_store.py
```

## Task 10: CAP-X Streaming, Metrics, Configuration, and B5 Gate

**Files:**

- Modify: `capmas/backends/capx.py`
- Create: `capmas/perception/metrics.py`
- Create: `scripts/run_libero_b5.py`
- Modify: `configs/default.yaml`
- Modify: `docs/implementation-roadmap.md`
- Modify: `docs/design/world-model-api.md`
- Modify: `docs/real-time-perception.md`
- Modify: `docs/adr/0002-three-plane-timing.md` or add a Phase 4 ADR
- Test: `tests/test_phase4_capx_streaming.py`

**Interfaces:**

```python
class CAPXStreamingObservationSource:
    def __init__(
        self,
        provider: CAPXObservationProvider,
        source: str = "capx",
        episode_id: str | None = None,
        episode_epoch: int | None = None,
    ) -> None: ...

    def capture(self) -> ObservationBundle: ...
```

- [x] **Step 1: Write failing CAP-X adapter and metric tests.**

```python
def test_capx_streaming_source_adds_source_sequence_and_episode_metadata():
    provider = FakeCAPXObservationProvider(make_observation_bundle(timestamp_ns=100))
    source = CAPXStreamingObservationSource(
        provider, source="capx-libero", episode_id="ep-1", episode_epoch=4
    )

    bundle = source.capture()

    assert bundle.source == "capx-libero"
    assert bundle.sequence == 1
    assert bundle.episode_id == "ep-1"
    assert bundle.episode_epoch == 4


def test_capx_snapshot_records_processing_latency():
    service = make_world_model(clock=SequenceClock(250))
    previous = make_scene(scene_version=1, sensor_timestamp_ns=100)
    snapshot = service.process(make_observation_bundle(timestamp_ns=100), previous)

    assert snapshot.processing_latency_ms == 0.00015


def test_metrics_record_target_and_achieved_rates_and_drops():
    metrics = RealTimeMetrics(target_hz=10.0)
    metrics.record_observation(timestamp_ns=0)
    metrics.record_observation(timestamp_ns=100_000_000)
    metrics.record_drop()

    summary = metrics.summary(now_ns=200_000_000)

    assert summary.target_hz == 10.0
    assert summary.achieved_hz == 10.0
    assert summary.dropped_frames == 1
```

- [x] **Step 2: Run red.**

```bash
pytest -q tests/test_phase4_capx_streaming.py
```

- [x] **Step 3: Implement the streaming adapter and metrics.**

Reuse `CAPXObservationProvider.capture()` and preserve `CAPXRobotBackend`
action-boundary behavior. Add source/sequence metadata without exposing the
environment object. Compute processing latency when publishing snapshots.

- [x] **Step 4: Add the Phase 4 configuration.**

Update `configs/default.yaml` with target rates, `max_processing_latency_ms`,
thread runtime default, artifact store settings, worker restart settings,
sparse voxel settings, and `state_store.require_contiguous_commit: false`.

- [x] **Step 5: Add B5 runner and artifact metrics.**

`scripts/run_libero_b5.py` must accept the existing CAP-X task/config inputs,
select thread or process runtime, preserve one JSON artifact and one `.log`,
and record target/achieved rates, queue delay, dropped frames, processing
latency, snapshot age P50/P95, worker restarts, and evaluator-only success.

- [x] **Step 6: Update docs and run the complete test suite.**

```bash
pytest -q
python -m compileall -q capmas scripts
```

The roadmap must mark the live thread gate complete only after a real B5 CAP-X
run meets its declared metrics. TSDF remains disabled and Phase 5 evidence
work remains unchecked. A live process-mode run remains a separate gate until
CAP-X capture artifacts are emitted into the shared file store.

Execution status: Tasks 1-10 reference implementation steps are complete. The
the live thread-mode CAP-X B5 gate is complete with real timestamped RGB-D,
CAP-X object-pose tracks, and declared deadline/freshness metrics. Replay mode
is intentionally not an evaluator-success claim; process-mode live capture is
still pending the shared artifact-store integration.

## Validation Gates

After Task 3, replay must generate ordered bundles without loading the whole
recording. After Task 7, a pure service call must produce a new immutable
snapshot. After Task 8, existing P2/P3 runtime tests must remain green. After
Task 9, a worker failure must not block `latest_observation()` or fabricate
episode completion. Only after Task 10 may the project claim a Phase 4 B5
integration attempt.

## Plan Self-Review

- Spec coverage: observation contract and replay are Tasks 1-3; geometry is
  Task 4; sparse map is Task 5; tracking is Task 6; semantic triggers and
  service are Task 7; state semantics are Task 8; runtimes and restart policy
  are Task 9; CAP-X, metrics, config, docs, and B5 are Task 10.
- No task changes the CAP-X API registry or adds privileged evaluator state to
  the agent observation boundary.
- Every public interface used by a later task is introduced in an earlier
  task or already exists in the current repository.
- Each task has a focused red test command and a green verification command.
- TSDF, learned perception, and Phase 5 evidence evolution are explicitly
  outside this plan's active implementation.
