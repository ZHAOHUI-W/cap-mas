# ADR-0002: Separate Control, World Model, and Agent Timing

## Status

Accepted for the initial research prototype.

## Context

Robot control has a strict deadline. Semantic perception and LLM inference have variable latency and cannot be safely placed in the servo loop.

## Decision

Use a high-frequency control plane, an asynchronous world-model plane, and an event-triggered agent plane. Controllers consume the newest snapshot and use explicit freshness and uncertainty policies.

Phase 4 realizes the world-model boundary with a dependency-light geometry,
sparse-voxel, and deterministic tracking reference path. Thread mode is used
for replay and deterministic tests. Process mode serializes only strict JSON
metadata and shared `ArtifactRef` URIs, and restarts a failed worker at most
three times with 50/100/200 ms backoff while retaining the last valid snapshot.
Semantic requests are bounded and event-triggered; semantic inference is not
called from the fast path.

`processing_latency_ms` is publish time minus sensor time. Snapshot freshness
is runtime age since publish, and neither value is used as a substitute for
the other. Observation snapshots remain pending until an explicit
post-action commit validates the action-end sensor timestamp.

## Consequences

Positive: agent latency does not directly stall control; map freshness and deadline misses are measurable.

Negative: the system must handle asynchronous data, prediction error, dropped frames, and stale snapshots.

## Alternatives rejected

- Synchronous Perception Agent call before every control action.
- Full-scene semantic rebuild on every RGB-D frame.
