# ADR-0012: P4.5 Cross-Process CAP-X Artifacts and World Model Recovery

## Status

Accepted and implemented. The live process B5 gate passed on 2026-07-28
with the declared privileged LIBERO workload; see the recorded JSON/log
artifacts under `outputs/capmas_libero_b5/`.

## Context

Phase 4 has a working `ProcessWorldModelRuntime` for replay and IPC tests, and
the live CAP-X B5 gate has passed in thread mode. P4.5 closes the remaining
live capture boundary without introducing semantic models, TSDF, robot-action
parallelism, or Memory/Skill evolution.

## Decision

### 1. Scope

P4.5 implements only the real CAP-X observation to process World Model loop:

```text
CAP-X capture
  -> encoded shared artifacts
  -> JSON observation envelope
  -> ProcessWorldModelRuntime
  -> decoded geometry/tracking/map
  -> JSON snapshot envelope
```

VLM/SAM, TSDF, semantic correction, robot-action parallelism, and evaluator
task completion remain outside P4.5.

### 2. Artifact encoding

RGB, depth, and numeric robot-state arrays are encoded as self-describing
NumPy `.npy` bytes before entering the shared store. The initial media types are:

- `image/rgb+npy`
- `image/depth+npy`
- `array/joint-position+npy`
- `array/ee-pose+npy`

`FileArtifactStore` remains bytes-only. An `EncodedArtifactStore` delegates
byte persistence to it, while a `NumpyArtifactCodec` performs encode/decode.
The codec is replaceable by PNG/JPEG or compressed-depth codecs later.

Publication remains atomic through temporary-file plus `os.replace`. The
capture path may set `fsync=false` for a run-scoped local or tmpfs artifact
root after the process boundary has been accepted; checksum, byte-size, and
atomic publication semantics remain unchanged. The reference process B5
profile uses `fsync=false`, `depth_subsample=16`, and records capture put
count, bytes, and put latency in the JSON artifact. `depth_subsample=8`
remains available when map density is preferred over the latency budget.

### 3. Store ownership and lifecycle

The parent process creates a run-scoped artifact root. The CAP-X capture side
writes RGB-D artifacts; the World Model worker reads those artifacts and writes
map artifacts into the same root. The root is not cleaned until the worker has
stopped and all required transport messages have been accounted for.

The worker receives only a picklable artifact-root string and transport
configuration. It does not receive a CAP-X environment, API object, provider,
or `InMemoryArtifactStore` instance. Worker restarts reuse the same root.

### 4. Backpressure

The process input path is non-blocking and latest-wins. When the bounded queue
is full, the oldest pending observation is dropped and the newest observation
is retained. Dropped observations are never assigned a snapshot version and
are recorded with their sequence and reason.

### 5. Per-observation IPC acknowledgement

Transport envelopes are correlated by `(episode_id, episode_epoch, sequence)`.
The worker returns a result for the specific observation, including the source
sequence and generated scene version. It can also return explicit
`observation_dropped` or `observation_failed` messages. Runtime wait and metric
logic must use this correlation, not only global submitted/processed counters.

### 6. Worker recovery

On worker failure, the runtime retains the last published snapshot and restarts
from it with bounded backoff `50/100/200 ms`, up to three restarts per episode.
The in-flight observation is discarded rather than replayed. A successful next
observation continues from the last published scene version. Restart-budget
exhaustion enters `degraded`; no fresh-dependent action may be dispatched.

### 7. Artifact failure semantics

Missing, checksum-invalid, truncated, or undecodable artifacts fail closed:
the current observation is rejected, no new snapshot is published, and the
last valid snapshot is retained. The same bad artifact is not retried forever.
Consecutive artifact failures eventually mark the worker degraded. Artifact
failures and worker crashes are separate counters and failure classes.

### 8. P4.5 acceptance

Hard gates:

- exact artifact round-trip, checksum, dtype, and shape validation;
- monotonic scene versions and sequence-correlated acknowledgements;
- auditable drop/failure/restart records;
- last-snapshot retention and degraded behavior after worker failure;
- `processing_latency_p95 <= 50 ms`;
- `snapshot_age_p95 <= 150 ms`.

Performance is reported comparatively rather than treated as a single binary
20 Hz gate. Process overhead is compared with the live thread baseline, and
the target drop rate is at most 5% under the declared workload.

The first live process run uses the privileged LIBERO configuration with
external API servers disabled. This isolates IPC and artifact correctness from
non-privileged semantic perception. The artifact must state that it uses real
RGB-D plus privileged object-pose measurements and that no semantic model is
enabled. The accepted reference run used a shared `/dev/shm` artifact root,
`depth_subsample=16`, and `fsync=false`; it produced 20/20 snapshots,
processing latency P95 `23.64 ms`, and snapshot age P95 `5.63 ms`. The
the artifact root is part of the workload declaration because an NFS-backed
root can exceed the 50 ms processing tail even when IPC is healthy.

## Consequences

Positive:

- The process worker can resolve real CAP-X RGB-D without shared memory
  pointers.
- Capture remains non-blocking under worker lag.
- Worker restart and bad-artifact behavior are explicit and fail closed.
- Codec and storage choices remain replaceable.
- P4.5 can be evaluated independently of Phase 5 semantic/evidence work.

Negative:

- `.npy` artifacts increase disk traffic and may require later compression.
- `fsync=false` trades crash-durability of the last uncommitted file write for
  the declared atomic publication and checksum guarantees; use `fsync=true`
  when durability is more important than the live deadline.
- Latest-wins drops frames and therefore cannot support frame-by-frame replay
  semantics in the live path.
- The privileged first gate does not measure non-privileged perception quality.
- A live process run may expose artifact I/O overhead not visible in thread mode.

## Alternatives rejected

- Passing NumPy objects or `InMemoryArtifactStore` through spawn IPC.
- Blocking CAP-X capture until every RGB-D frame is processed.
- Retrying a corrupt artifact indefinitely.
- Treating submitted/processed counters as per-frame acknowledgement.
- Mixing semantic model integration into the P4.5 infrastructure gate.
