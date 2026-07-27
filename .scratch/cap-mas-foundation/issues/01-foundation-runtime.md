Status: ready-for-agent
Type: task

# Implement CAP-MAS Foundation Runtime and Interfaces

Implement the CAP-MAS foundation described in `spec.md`: immutable contracts,
typed Robot Skill registry/execution, unified 2D/3D perception boundaries,
versioned runtime state, memory proposals and idempotent storage, and a
dependency-free `RuntimeOrchestrator.run_cycle()` seam. Preserve CAP-X
compatibility through adapters without importing CAP-X in core modules.

## Acceptance Criteria

- A valid action contract commits a new scene version and returns an execution
  trace through the public orchestrator seam.
- A stale parent scene or wrong episode epoch is rejected before backend action.
- Exactly one active lease can exist for a robot and lease expiry is observable.
- CAP-X-compatible binary benchmark score remains separate from learning return.
- 2D and 3D perception backends share normalized request/result artifacts while
  retaining separate implementations.
- Memory updates require provenance, base-version match, and idempotency.
- Public interfaces are covered by tests and the package imports without CAP-X,
  GPU, LIBERO, or network dependencies.
- Policy-facing scene objects expose artifact-backed visual grounding,
  spatial relations, and uncertainty without embedding raw RGB-D frames.
- A grounded Policy Agent can return either a typed action contract or a
  targeted, bounded perception request; raw observations remain exclusive to
  the Perception Agent boundary.
