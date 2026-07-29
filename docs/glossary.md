# Glossary

| Term | Definition |
| --- | --- |
| Action Contract | A versioned proposal for a bounded robot state transition with typed skills, preconditions, postconditions, invariants, timeout, and recovery policy. |
| Action Lease | Runtime-issued exclusive authority to execute a contract on a physical robot. |
| Active Skill | An immutable, validated skill version callable by the active executor. |
| Agent Plane | Event-triggered multi-agent reasoning and coordination plane. |
| Arbiter | Deterministic or bounded decision component that selects one typed candidate before Verifier approval. |
| CAP-X Adapter | Compatibility layer that exposes CAP-X environments and APIs to CAP-MAS. |
| Committed State | A state snapshot accepted after verification; it is not a prediction. |
| Contract-Driven Coordination | Coordination in which agent proposals advance the robot only after version, permission, and predicate checks. |
| Episode Epoch | Identifier that invalidates contracts after reset, timeout, or cancellation. |
| Observable Postcondition | A task-relevant predicate inferred from agent-visible sensors and state estimators. |
| Privileged Completion | Simulator/evaluator success information unavailable to the normal agent. |
| Quarantine Registry | Isolated store for candidate skills that cannot affect active execution. |
| Scene Snapshot | Immutable, versioned world-model artifact with timestamps, geometry, semantics, and uncertainty. |
| Visual Evidence | An artifact-backed RGB, depth, mask, crop, or pose-support reference that grounds a scene fact without embedding raw sensor data in agent context. |
| Spatial Relation | A confidence-bearing relation between two tracked entities, such as `left_of`, `inside`, or `occluded_by`. |
| Grounded Policy Decision | A Policy Agent result containing exactly one bounded action contract or one targeted visual-evidence request. |
| Mission Graph | Versioned typed graph owned by the Manager that describes task-level subgoals, dependencies, ports, budgets, and exits. |
| SubgraphSpec | Bounded typed local policy graph assigned to one subgoal and lowered to one or more ActionContracts. |
| GraphValidator | Static validator for graph reachability, typed ports, checkpoints, cycles, and resource conflicts. |
| Graph Rehearsal | Offline or asynchronous execution of a graph over sampled scenes for failure localization and evolution. |
| Experiment Run Configuration | The non-secret protocol, model, seed, budget, schema, retry, and worker settings used to produce one experiment artifact. |
| LLM Call Trace | A sanitized immutable record of one Manager or Policy provider request, including latency, token usage, schema mode, fallback, and outcome. |
| Evidence Provenance | The provider, source SceneSnapshot version, capture timestamp, and optional artifact references attached to candidate evidence. |
| Evidence Dimension | A typed candidate-evidence result with `pass`, `fail`, or `unknown` status; unknown is not converted to zero. |
| Geometry Evidence | Candidate-conditioned evidence about grasp quality, reachability, clearance, collision risk, and target-pose feasibility. It is separate from scene-level Perception Evidence. |
| Motion Intent | Typed action-node metadata describing a grasp, place, or move target and approach. It is planning metadata, not an executable Robot Skill argument. |
| Motion Preview | A side-effect-free feasibility result produced before ActionLease acquisition; it may contain IK, trajectory, clearance, and collision information. |
| Motion Preview Backend | A read-only interface for computing Motion Preview from MotionIntent, SceneSnapshot, and local map state. |
| Candidate Normalizer | The component that canonicalizes a raw Policy subgraph, derives or validates MotionIntent, and recomputes candidate evidence after executable changes. |
| Realistic Evidence Mode | Primary evaluation mode using sensor-derived SceneSnapshot state without evaluator or simulator ground truth. |
| Diagnostic Privileged Mode | Isolated debugging mode allowed to use privileged simulator state; its results cannot enter primary success statistics. |
| Evidence Wave Deadline | Global pre-lease time budget for parallel candidate evidence computation; P5.2 uses 50 ms and converts timeouts to unknown. |
| Evidence Tie-Break | A deterministic structural choice made when candidates have usable evidence but equal final evidence scores. |
| Confidence Fallback | A legacy arbitration mode used only when no candidate-specific evidence is available. |
| Proposal Wave | A bounded set of Policy candidate requests launched for dependency-compatible subgoals from one immutable Scene Snapshot. |
| Rolling Replan | The execution mode that replans only the ready suffix after each verified Scene Snapshot update. |
| Targeted Perception Request | A bounded request for evidence about selected tracks, cameras, or regions; it is served asynchronously by the Perception Agent. |
| Semantic Perception Agent | Event-triggered agent for object identity, ambiguity resolution, and semantic map correction. |
| Skill Candidate | Newly generated or repaired skill awaiting validation and promotion. |
| State Version | Monotonic identifier for a scene snapshot used to detect stale proposals. |
| Subgoal Checkpoint | Safe boundary at which contracts, memory, topology, or skill versions may be updated. |
| World Model Plane | Asynchronous scene-estimation and incremental mapping plane. |
| P4.5 Process World Model | The Phase 4.5 infrastructure gate that connects real CAP-X observations to a spawned World Model worker through shared artifacts and JSON IPC. |
| Encoded Artifact Store | A codec-aware wrapper that converts runtime values such as NumPy arrays into bytes before delegating persistence to `FileArtifactStore`. |
| Artifact Codec | Replaceable encoder/decoder that defines how an in-memory observation value becomes a shared artifact. P4.5 initially uses NumPy `.npy`. |
| Latest-Wins Queue | Non-blocking bounded queue policy that drops the oldest pending observation when a newer observation arrives. |
| Observation Correlation | The `(episode_id, episode_epoch, sequence)` identity used to acknowledge one process observation independently of global counters. |
| In-Flight Observation | An observation accepted by a worker but not yet published as a `SceneSnapshot`; it is discarded if the worker crashes. |
| Last Valid Snapshot | The most recent successfully published immutable snapshot retained across worker restart or artifact failure. |
| Degraded Runtime | A fail-closed health state reached after restart or artifact-failure budgets are exhausted; fresh-dependent actions are rejected. |
| Artifact Failure | Missing, checksum-invalid, truncated, or undecodable shared data that prevents one observation from producing a snapshot. |
