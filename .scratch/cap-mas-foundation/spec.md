# CAP-MAS Foundation Runtime and Interfaces

## Problem Statement

CAP-MAS currently has an architecture specification but no executable project
foundation. The design also leaves the boundary between CAP-X perception/control
APIs and CAP-MAS typed contracts implicit. Core protocol objects such as
`EpisodeHandle`, `ExecutionTrace`, `MemoryContext`, and `MemoryUpdate` are named
but not defined, which makes it impossible to implement or test the runtime
without reinterpreting the architecture in each module.

## Solution

Build a dependency-light CAP-MAS foundation that defines versioned contracts,
runtime state management, typed Robot Skill execution, unified 2D/3D perception
boundaries, experience-memory updates, and a one-cycle orchestration seam.
Provide CAP-X adapters through protocols rather than importing CAP-X into the
core package. The initial runtime uses deterministic implementations and mock
backends; real LIBERO, SAM3, GraspNet, PyRoKi, and LLM services remain external
adapters.

## User Stories

1. As a researcher, I want to start a uniquely identified episode, so that traces from different task seeds cannot be confused.
2. As a researcher, I want reset to return an episode handle and an initial scene snapshot, so that identity and observable state are explicit.
3. As a runtime, I want episode epochs to invalidate stale contracts after reset, so that old actions cannot execute in a new episode.
4. As an agent, I want to consume a compact scene snapshot, so that I do not need direct access to a simulator or environment object.
5. As a perception agent, I want one request format for 2D segmentation and 3D geometry, so that I can switch backend implementations without changing agent logic.
6. As a perception agent, I want 2D detections to retain image evidence and camera metadata, so that their 3D lifting is auditable.
7. As a perception agent, I want 3D estimates to include frame, timestamp, covariance, and confidence, so that stale or uncertain facts can be rejected.
8. As a world-model publisher, I want to merge observations into monotonic scene versions, so that agents can reason against a known state.
9. As a policy agent, I want to propose an action contract referencing typed skills and versions, so that generated behavior is bounded and reproducible.
10. As a validator, I want to reject unknown skills, malformed arguments, stale parent scenes, and expired contracts, so that invalid actions never reach the robot.
11. As a verifier, I want to evaluate preconditions, safety invariants, and postconditions separately, so that approval and task completion are not conflated.
12. As a runtime, I want exactly one action lease for a robot, so that multiple software agents cannot control the same actuator concurrently.
13. As an executor, I want to record every skill invocation, timing, error, and artifact reference, so that an action can be replayed and diagnosed.
14. As a recovery agent, I want typed failure classes and a new observation after failure, so that recovery replans from current state instead of rewriting history.
15. As an evaluator, I want CAP-X binary benchmark success preserved, so that CAP-MAS remains comparable with CAP-X.
16. As a learning subsystem, I want verifier-derived intermediate progress signals, so that long-horizon credit assignment can use dense evidence without changing the benchmark label.
17. As a memory controller, I want a bounded context containing retrieved memories, current failure, budget, and trace span, so that selection does not require unrestricted database access.
18. As a memory executor, I want to emit a structured update proposal rather than mutate storage directly, so that provenance and conflict checks happen before commit.
19. As a memory store, I want idempotent updates based on an explicit base version, so that retries do not duplicate experience.
20. As a critic, I want failed and recovered spans stored as hard cases, so that rare failures are not hidden by frequent successes.
21. As a Skill Designer, I want Memory Skill candidates separated from Robot Skill candidates, so that physical behavior and experience extraction can evolve independently.
22. As a CAP-X adapter author, I want to wrap existing `get_observation`, segmentation, point-cloud, grasp, IK, and gripper functions, so that CAP-MAS reuses tested CAP-X capability implementations.
23. As a CAP-X adapter author, I want the core package to avoid importing heavy CAP-X dependencies, so that contract tests run without LIBERO or GPU services.
24. As an experimenter, I want mock and replay backends at the same public seams as CAP-X, so that deterministic tests and ablations are possible.
25. As an experimenter, I want each module to be replaceable through protocols, so that verifier, memory, perception, and scheduler ablations do not require invasive branches.
26. As an operator, I want LLM calls and semantic perception to remain outside the control path, so that high-frequency control never waits for a remote service.
27. As a researcher, I want execution traces to retain skill registry and memory-bank versions, so that performance changes can be attributed to a specific snapshot.
28. As a researcher, I want the runtime to expose no privileged evaluator result to agents, so that success claims reflect observable evidence.
29. As a maintainer, I want serialization-safe contract objects, so that artifacts can be stored as JSON-compatible records.
30. As a maintainer, I want clear status transitions for episodes, contracts, leases, traces, and memory updates, so that illegal transitions fail deterministically.
31. As a Policy Agent, I want every tracked object to expose artifact-backed visual evidence, so that labels and track IDs are spatially grounded.
32. As a Policy Agent, I want spatial relations and uncertainty in the scene graph, so that I can distinguish a confident action from an identity or occlusion ambiguity.
33. As a Policy Agent, I want to request only selected evidence for selected tracks, so that multimodal grounding does not require copying complete RGB-D frames into the LLM context.
34. As a Perception Agent, I want direct access to timestamped RGB-D observations, so that segmentation and identity association remain outside the LLM path.
35. As a control process, I want semantic perception requests to be bounded and asynchronous, so that slow VLM or LLM inference cannot stall high-frequency control.

## Implementation Decisions

- The foundation uses Python standard-library dataclasses, enums, protocols, and
  immutable value objects. Pydantic and heavy robotics dependencies are optional
  adapter concerns rather than core requirements.
- `EpisodeHandle` identifies an episode and its epoch. `EpisodeStart` pairs it
  with the initial `SceneSnapshot`.
- `SceneSnapshot` is the unified agent-facing world-model object. Raw RGB-D,
  masks, point clouds, and maps are referenced by `ArtifactRef`; they are not
  embedded in agent messages.
- Scene grounding is explicit. `ObjectTrack` can carry typed `VisualEvidence`,
  `SceneSnapshot` carries `SpatialRelation`, `SceneUncertainty`, and scene-level
  visual evidence, and all visual payloads remain artifact references.
- The default Policy Agent input is the structured scene graph. A
  `GroundedPolicyAgent` returns an exclusive `PolicyDecision`: either an
  `ActionContract` or a bounded, targeted `PerceptionRequest`. It never reads
  an `ObservationBundle` directly.
- The `PerceptionAgent` boundary consumes raw `ObservationBundle` values and
  emits normalized `PerceptionResult` values asynchronously, without entering
  the LLM or servo path.
- 2D and 3D perception use separate backend protocols but a shared
  `PerceptionRequest`/`PerceptionResult` facade. The 2D backend returns
  `Detection2D`; the 3D backend returns `ObjectPoseEstimate`, `PointCloudRef`,
  and `BoundingBox3D`; a fusion backend converts them into tracked objects in a
  `SceneSnapshot`.
- `ObservationProvider` is the only backend boundary that captures raw CAP-X
  observations. `Vision2DBackend`, `Geometry3DBackend`, `GraspProposalBackend`,
  and `RobotControlBackend` are separate capability boundaries.
- CAP-X implementations are wrapped by `CAPXPerceptionAdapter` and
  `CAPXRobotBackend`. The core runtime never receives the CAP-X environment
  handle and never executes arbitrary generated Python.
- `ActionContract` references registered immutable Robot Skill versions. The
  validator checks episode epoch, parent scene version, skill existence, typed
  arguments, duration, and simulation-step budgets.
- `ExecutionTrace` represents one action contract. `SkillTrace` represents one
  typed skill invocation. `EpisodeTrace` aggregates action traces for one
  episode.
- `MemoryContext` is a read-only, bounded decision view. It contains current
  subgoal, trace span, failure/recovery summaries, retrieved memory references,
  candidate Memory Skills, active snapshot versions, and budget; it contains no
  environment handle or privileged completion result.
- `MemoryUpdate` is a proposal with provenance, base memory version, operation,
  items, invalidations, evidence, idempotency key, and validation status. The
  store commits it only if the base version is current and the proposal passes
  validation.
- The first orchestration seam is `RuntimeOrchestrator.run_cycle()`: it accepts
  an `ActionContract`, validates/verifies it, acquires a lease, executes through
  the backend, observes a new snapshot, verifies postconditions, and returns a
  `CycleResult`. Agent implementation is replaceable above this seam.
- The benchmark score remains CAP-X-compatible binary success. The foundation
  exposes a separate `LearningReturn` interface for verifier-derived shaping;
  it is disabled by default and never changes benchmark success.
- Memory updates occur only at episode boundaries or explicit checkpoints. They
  cannot change an active lease or Robot Skill semantics.
- The first tests use a deterministic in-memory backend and a fake typed skill.
  CAP-X, filesystem, network, model, and GPU services are external boundaries
  and are not mocked as internal collaborators.

## Testing Decisions

- Tests verify public behavior at the `RuntimeOrchestrator`, `StateStore`,
  `SkillRegistry`, `MemoryStore`, and perception facade seams. They do not test
  private helper calls or internal collaborator invocation counts.
- The first vertical slice verifies that a valid action commits a new scene
  version and trace, while stale contracts are rejected before backend
  execution.
- Contract tests cover illegal episode, contract, lease, trace, and memory
  update transitions using independent literal expectations.
- Perception tests use a fake observation provider and verify the normalized
  2D/3D result shape, provenance, confidence, and scene publication.
- Multimodal contract tests verify visual evidence references, spatial
  relations, uncertainty, the exclusive Policy decision shape, and targeted
  perception request fields without exposing raw frames to `AgentContext`.
- Memory tests verify idempotent commit, base-version conflict rejection,
  provenance requirements, and hard-case retention through the public store.
- Integration tests will later run the same backend seam against CAP-X LIBERO;
  they are not required for the dependency-free foundation slice.

## Out of Scope

- Direct implementation of SAM3, Molmo, GraspNet, PyRoKi, LIBERO, or simulator
  internals inside CAP-MAS core.
- Arbitrary Python code execution by Policy Agent.
- Online RL parameter updates, PPO/GRPO training, or model fine-tuning.
- Full asynchronous multi-process scheduling and high-frequency servo control.
- Full artifact-store implementation, visual crop generation, and semantic
  model execution; these remain adapter concerns for the CAP-X integration.
- Production database, vector database, distributed event bus, or web UI.
- Automatic promotion of Memory Skill or Robot Skill candidates.
- Real-robot safety claims.

## Further Notes

The foundation is intentionally smaller than the full research architecture.
Its purpose is to make authority boundaries executable first. Once this seam is
stable, the real CAP-X adapter, asynchronous world-model process, multi-agent
scheduler, Memory Controller, and sequential evolution loop can be added without
changing the core artifact contracts.
