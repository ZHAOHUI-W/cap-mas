# P4.5 Process-Mode CAP-X World Model Implementation Plan

Status: Implemented and verified on 2026-07-28. The live gate used the
privileged LIBERO Spatial task 0, CUDA device 5, a shared `/dev/shm` artifact
root, `depth_subsample=16`, and `fsync=false`.

## Goal

Close the live CAP-X to `ProcessWorldModelRuntime` artifact boundary while
preserving CAP-MAS contracts and the Phase 4 thread/replay baselines.

## Non-goals

- VLM/SAM or non-privileged semantic perception;
- TSDF;
- robot-action parallelism or live ActionLease changes;
- Memory Skill, rehearsal, OOD evidence, or task-success claims.

## Public seams

1. `EncodedArtifactStore.put(value, media_type) -> ArtifactRef`
2. `NumpyArtifactCodec.encode/decode`
3. CAP-X observation provider using a replaceable artifact sink
4. Per-observation process envelope and acknowledgement
5. `ProcessWorldModelRuntime` latest-wins queue behavior
6. Worker artifact failure and restart health messages
7. Live B5 process runner and JSON/log artifact

## Implementation slices

### Slice 1 — Codec and shared store

- Add a codec protocol and NumPy `.npy` codec.
- Add an encoded store wrapper over `FileArtifactStore`.
- Preserve `InMemoryArtifactStore` behavior for thread/replay tests.
- Verify exact dtype, shape, values, checksum, and media type.

### Slice 2 — CAP-X capture adapter

- Allow `CAPXObservationProvider` to use the encoded store without changing
  action-boundary API behavior.
- Encode RGB, depth, joint state, and EE pose at capture time.
- Keep the CAP-X environment and API object in the parent process only.
- Record artifact byte size and write latency.

### Slice 3 — Process transport correlation

- Add observation correlation `(episode_id, episode_epoch, sequence)`.
- Return processed, dropped, and failed messages for each sequence.
- Replace count-only waiting with sequence-aware waiting and metrics.
- Ensure dropped observations do not create scene versions.

### Slice 4 — Latest-wins and failure behavior

- Make queue overflow non-blocking and drop the oldest pending observation.
- Preserve the last published snapshot across worker restart.
- Discard the in-flight observation on crash.
- Fail closed on missing, corrupt, or undecodable artifacts.
- Enter degraded after the configured restart or consecutive artifact-failure
  budget is exhausted.

### Slice 5 — Live process B5

- Add a process-mode live path to `scripts/run_libero_b5.py`.
- Use the privileged LIBERO YAML and `--skip-api-servers`.
- Use a run-scoped artifact root and cleanup only after worker shutdown.
- Record `perception_mode`, codec, artifact I/O metrics, transport outcomes,
  freshness, latency, drops, restart count, and evaluator-only status.

The runner records parent capture artifact I/O metrics and source byte sizes;
worker map artifacts remain content-addressed and are visible through the
published `local_map` URI.

## Test matrix

| Test | Required result |
| --- | --- |
| `.npy` RGB/depth round-trip | Exact shape/dtype/value recovery |
| Atomic publication | Reader never observes partial artifact |
| Missing/checksum-invalid artifact | Observation failed, no new snapshot |
| Queue overflow | Oldest pending sequence dropped, newest retained |
| Sequence acknowledgement | Specific sequence resolves correctly |
| Worker crash | Last snapshot retained, bounded restart occurs |
| Restart exhaustion | Health becomes degraded, no fabricated snapshot |
| Live privileged LIBERO process B5 | Real RGB-D, process snapshot stream, JSON/log pair |
| Regression suite | Existing CAP-MAS tests remain green |

## Acceptance thresholds

Hard:

- artifact round-trip and checksum correctness;
- monotonic scene versions;
- sequence-correlated acknowledgements;
- explicit drop/failure/restart artifacts;
- last-snapshot retention and degraded fail-closed behavior;
- processing latency P95 at or below 50 ms;
- snapshot age P95 at or below 150 ms.

Comparative:

- process overhead versus the live thread baseline;
- drop rate at or below 5% for the declared workload;
- artifact write/read latency and total bytes per observation.

## Exit artifact

The process B5 JSON and log must state:

```json
{
  "runtime": "process",
  "perception_mode": "privileged_object_pose_plus_real_rgbd",
  "semantic_models_enabled": false,
  "evaluator_success": false
}
```

`evaluator_success=false` is expected because this benchmark is observation
only and does not execute a robot action.

## Verification record

The accepted process run produced 20 snapshots with versions `1..20`, zero
drops, zero worker restarts, seven final object tracks, processing latency
P95 `23.64 ms`, and snapshot age P95 `5.63 ms`. Capture metrics were 126 puts
and 120,441,216 bytes. The retained artifacts are:

- JSON: `outputs/capmas_libero_b5/p45_process_final.json`
- log: `outputs/capmas_libero_b5/p45_process_final.log`
