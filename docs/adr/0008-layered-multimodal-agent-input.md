# ADR-0008: Layered Multimodal Agent Input

## Status

Accepted.

## Context

Object labels and track IDs alone do not ground a Policy Agent's spatial
reasoning. Conversely, placing complete RGB-D observations on the LLM path
would increase message size, duplicate perception work, and create an
unbounded latency dependency in a long-horizon control system.

## Decision

CAP-MAS uses two multimodal boundaries:

1. The Perception Agent consumes `ObservationBundle` and raw RGB-D artifacts.
   It performs 2D segmentation, 3D lifting, identity association, and scene
   publication outside the LLM and servo paths.
2. The Policy Agent consumes a compact `SceneSnapshot` containing structured
   object tracks, spatial relations, uncertainty, freshness, and typed visual
   evidence references. When grounding is insufficient, it emits one bounded
   `PerceptionRequest` targeted by track IDs and evidence types. It cannot read
   raw frames directly.

`PolicyDecision` is exclusive: a decision is either an `ActionContract` or a
targeted perception request. A perception timeout becomes an explicit
uncertainty/failure signal and never blocks high-frequency control.

## Consequences

Positive: visual grounding is auditable, most policy calls remain compact,
perception and policy can be independently ablated, and control latency is
protected.

Negative: the system needs artifact storage, request scheduling, and evidence
cache management. A policy may require an extra perception round trip when
identity or occlusion is genuinely ambiguous.
