# Real-Time Perception and 3D Scene Mapping

## 1. Requirement

The CAP-MAS controller must meet the robot control deadline even when semantic perception is slow. A 20 Hz control loop has approximately 50 ms per cycle. VLM, SAM, and LLM calls are not placed in this loop.

## 2. Multi-rate pipeline

~~~mermaid
flowchart LR
    A[RGB-D + proprioception] --> B[Time synchronization]
    B --> C[Fast geometry]
    C --> D[Tracking and motion prediction]
    D --> E[Incremental local 3D map]
    E --> F[SceneSnapshot]
    F --> G[Controller and Verifier]
    F --> H{Low confidence or event?}
    H -->|yes| I[Semantic Perception Agent]
    I --> J[Identity and map correction]
    J --> E
~~~

## 3. Update rates

| Layer | Target rate | Blocking policy |
| --- | ---: | --- |
| Joint state and FK | 50–200 Hz | Must never block |
| Local depth/voxel update | 20–50 Hz | Drop stale frames rather than block |
| Known-object tracker | 10–30 Hz | Predict between observations |
| Semantic segmentation | 1–5 Hz or event-triggered | Runs asynchronously |
| Perception Agent reasoning | Event-triggered | Never in servo path |
| Global planning | At subgoal boundaries | Can be slow but has timeout |

## 4. Incremental map design

1. Capture RGB-D with sensor timestamps.
2. Synchronize with joint state and estimate camera pose through FK.
3. Transform depth points into the world frame.
4. Fuse only changed local voxels or TSDF blocks.
5. Track known objects using ROI depth, point registration, or a learned tracker.
6. Publish an immutable snapshot with a monotonically increasing version.
7. Let the semantic agent attach corrections to a future snapshot; never mutate a snapshot already used by an active contract.

## 5. Avoiding perception-induced latency

- Use process isolation for control, map fusion, and semantic inference.
- Use bounded queues and drop-oldest policies for camera frames.
- Use double-buffered snapshots or a read-copy-update store.
- Cache camera calibration, FK transforms, masks, and object identities.
- Crop semantic inference to the active subgoal ROI.
- Batch compatible segmentation requests.
- Coalesce duplicate requests from Manager and Verifier.
- Use confidence-aware prediction between semantic updates.
- Prioritize end effector, grasped object, target object, and collision geometry.
- Emit freshness_ms and covariance so downstream modules can make explicit decisions.

### Policy-facing visual grounding

The low-latency path publishes a compact scene graph rather than copying full
RGB-D frames into the Policy Agent context. Each object track can point to
RGB/depth crops, masks, or pose-support artifacts, while spatial relations and
uncertainty make the grounding decision explicit. The Policy Agent requests
only the evidence type and track IDs needed to resolve an ambiguity. Requests
are bounded by a latency budget, deduplicated, ROI-cropped, and served by the
asynchronous Perception Agent. A timeout yields a perception-uncertain result;
it does not stall servo control.

## 6. Safety under stale maps

The controller uses a freshness policy per skill. A high-speed motion may require a fresh geometric map, while a low-speed hold or gripper action may tolerate an older snapshot. If freshness or covariance thresholds fail, the action is rejected or slowed. The runtime never silently treats an old snapshot as current.

## 7. CAP-X compatibility

The first adapter can consume CAP-X observations from get_observation() and expose a SceneSnapshot at action boundaries. The research implementation should then add continuous or incremental updates without changing the agent-facing contract. This isolates the benefit of asynchronous mapping from changes to the robot API.

## 8. Phase 4 implementation boundary

The current reference implementation follows this split:

```text
CAPXObservationProvider
  -> CAPXStreamingObservationSource
  -> BoundedSensorSynchronizer
  -> ThreadWorldModelRuntime | ProcessWorldModelRuntime
  -> WorldModelService
  -> Geometry / SparseVoxelMap / KnownObjectTracker
  -> immutable SceneSnapshot
```

Thread mode is the deterministic replay path. Process mode uses strict JSON
observation/snapshot envelopes and file-backed ArtifactRef URIs, with bounded
worker restart and last-snapshot retention. The asynchronous semantic queue
only emits requests from deterministic trigger conditions; it never calls a
VLM or blocks snapshot publication.

The B5 runner is `scripts/run_libero_b5.py`. Replay mode remains the
dependency-light regression path; live mode accepts a CAP-X YAML, resets a
real LIBERO environment, and feeds `CAPXStreamingObservationSource` into the
same World Model service. Live mode uses the thread runtime because the
current CAP-X provider stores NumPy RGB-D values in an in-process artifact
store. Process mode is retained for replay/IPC acceptance until capture-side
artifacts are written to the shared file store.

The live gate records target and achieved rates, processing latency, snapshot
age, drops, worker health, final object tracks, source artifact media types,
and the local-map URI in both JSON and per-run log artifacts. The 2026-07-27
LIBERO Spatial task 0 run used 20 observations at a 20 Hz target and produced:

| Metric | Result |
| --- | ---: |
| Snapshots | 20, versions 1..20 |
| Dropped frames | 0 |
| Processing latency P95 | 39.54 ms |
| Snapshot age P95 | 14.52 ms |
| Achieved rate | 17.47 Hz |
| Worker restarts | 0 |
| Final tracks | 7 |

`evaluator_success=false` in this gate is expected: B5 observes the reset
scene and does not execute a robot action. The optional endpoint probe for
`gpt-5.5` completed separately; the World Model itself remains LLM-free.
