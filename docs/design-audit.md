# CAP-MAS Design Audit

This audit separates decisions required for the dependency-free foundation from
decisions that require a real CAP-X/LIBERO integration experiment.

## 1. Defined in v0

| Area | v0 decision |
| --- | --- |
| Episode identity | `EpisodeHandle` plus monotonically scoped `episode_epoch` |
| Initial observation | `EpisodeStart(handle, initial_scene)` |
| Agent-facing state | immutable `SceneSnapshot` with version and artifact references |
| Action authority | `ActionContract` → validation → verifier → `ActionLease` → typed execution |
| Failure vocabulary | stable `FailureClass` constants shared by trace, recovery, and memory |
| Skill boundary | immutable `SkillRef` and `TypedSkill`; no arbitrary Python in core |
| State commit | `compare_and_commit(parent_version, next_snapshot)` |
| Action trace | `SkillTrace` per call, `ExecutionTrace` per contract, `EpisodeTrace` per episode |
| Perception boundary | separate 2D and 3D protocols behind a fused facade |
| Memory decision input | bounded read-only `MemoryContext` |
| Memory write | provenance-bearing `MemoryUpdate` proposal with idempotency key |
| Benchmark reward | CAP-X-compatible binary evaluator score |
| Learning reward | separate verifier-derived return, disabled by default |
| Evolution timing | memory and skill changes only at checkpoint/episode boundaries |
| Capability separation | observation, 2D vision, 3D geometry, grasp proposal, and robot control are separate adapter protocols |
| Multimodal agent boundary | Policy reads grounded scene graph plus on-demand evidence refs; Perception reads raw RGB-D asynchronously |
| CAP-X construction boundary | Existing YAML low-level node and CAP-X API registry are reused; CAP-X code execution is excluded by default |

## 2. Intentionally deferred

### Scene semantics

The exact object identity and data-association algorithm is not fixed. The v0
contract only requires stable `track_id`, confidence, frame, timestamp, and
evidence. The real adapter must decide whether tracking uses RGB-D geometry,
SAM masks, a learned tracker, or a combination.

### 2D-to-3D fusion

The interface is fixed, but camera calibration, depth filtering, mask lifting,
multi-view association, and covariance estimation require LIBERO fixtures and
should not be guessed from the API signatures alone.

### Robot action cancellation

The lease API has expiry, but the CAP-X motion helpers are currently blocking.
The adapter must define whether cancellation means a backend stop request,
bounded motion return, or a safe hold after the current simulator step.

### Postcondition vocabulary

The system has predicate reports but not a complete predicate language. v0
should implement a small explicit set (`scene_version`, gripper state, object
track confidence, distance, and artifact-backed facts) before adding a general
expression engine.

### Agent scheduling

The first orchestrator is a single-cycle seam. Fixed-graph multi-agent event
ordering, retries, timeout policy, and backpressure are deferred until the
single-cycle contracts are exercised with a mock backend.

### Memory retrieval

The memory item schema is fixed, but storage indexing is not. Start with an
in-memory or SQLite-compatible repository and deterministic lexical filtering;
vector retrieval is an optional later adapter.

### Memory conflict resolution

The update store rejects base-version conflicts and records contradiction IDs.
The policy for merging semantically similar but non-identical memories needs
hard-case data and is deferred to the Memory Designer phase.

### RL algorithm and coefficients

The three-stage algorithm progression is a research plan, not a v0 dependency.
Reward weights, discounting, offline-to-online data policy, and PPO/GRPO
hyperparameters must be selected only after trace schemas and verifier labels
are stable.

### CAP-X adapter lifecycle

The adapter must define when expensive SAM3, Molmo, GraspNet, and PyRoKi clients
are initialized, shared, warmed up, and closed. The core protocol intentionally
does not decide whether that lifecycle is process-scoped, worker-scoped, or
episode-scoped.

### Current V1 integration status

The code-level V1 path is implemented: YAML parsing, direct low-level
construction, registered API reuse, typed-skill binding, skill-output chaining,
observable LIBERO predicates, and CAP-MAS episode JSON output. A real simulator
run remains environment-dependent and must be validated in CAP-X's dedicated
LIBERO environment before claiming end-to-end success.

### Visual grounding and evidence policy

The artifact URI layout, crop generation, and cache implementation remain
adapter decisions. The semantic boundary is fixed: scene facts must be
traceable to evidence references, while raw frames stay behind the Perception
Agent boundary. Policy requests are targeted by track and evidence type and
must carry a bounded latency budget.

## 3. Risks exposed by implementation

1. CAP-X's current `ApiBase` still combines an environment reference with
   perception and control methods. CAP-MAS must wrap these methods behind
   separate capability objects rather than treating `ApiBase` as the new
   unified interface.
2. CAP-X observations contain large NumPy arrays and mutable dictionaries.
   SceneSnapshot must store immutable summaries plus artifact references, or
   asynchronous agents will accidentally share mutable simulator state.
3. CAP-X motion methods are synchronous and may exceed a lease deadline. The
   backend must report bounded execution explicitly instead of silently claiming
   completion.
4. CAP-X's task completion method is evaluator-facing. It must remain outside
   `SceneSnapshot` and outside normal AgentContext.
5. `get_object_pose` may call multiple expensive perception services. A direct
   one-to-one wrapper is valid for parity, but it is not yet a real-time
   Perception Agent implementation.
6. A single `SceneSnapshot` version cannot represent all asynchronous sensor
   streams without a publisher policy. v0 uses action-boundary snapshots; the
   world-model phase must define merge ordering and freshness thresholds.

## 4. Implementation gate before real CAP-X

The foundation is ready for CAP-X integration only when:

- contract tests pass without CAP-X imports;
- a mock backend can run at least one complete action cycle;
- stale and wrong-epoch contracts are rejected before backend calls;
- memory updates are idempotent and provenance-checked;
- 2D/3D result objects can be serialized without raw mutable environment state;
- the adapter has explicit lifecycle and cancellation behavior documented.
