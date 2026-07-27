# Module Interfaces

These interfaces are deliberately small. Concrete implementations may use dataclasses, Pydantic models, protobuf, or another transport, but the semantic fields and authority boundaries should remain stable. The executable foundation lives under `capmas/` and keeps CAP-X, network, GPU, and filesystem services at adapter boundaries.

## 1. Backend

~~~python
class RobotBackend(Protocol):
    def reset(
        self,
        seed: int | None = None,
        options: Mapping[str, object] | None = None,
    ) -> EpisodeStart: ...
    def observe(self) -> SceneSnapshot: ...
    def execute_skill(
        self,
        skill: TypedSkill,
        args: dict[str, object],
        budget: ExecutionBudget,
    ) -> SkillExecutionResult: ...
    def stop(self, lease: ActionLease) -> None: ...
    def evaluator_success(self) -> bool: ...  # evaluator-only
~~~

`evaluator_success()` is never included in the normal agent observation. The CAP-X adapter exposes it only to the evaluation plane.

## 2. State store

~~~python
class StateStore(Protocol):
    def publish(self, snapshot: SceneSnapshot) -> None: ...
    def latest(self) -> SceneSnapshot: ...
    def get(self, version: int) -> SceneSnapshot: ...
    def compare_and_commit(
        self,
        parent_version: int,
        snapshot: SceneSnapshot,
    ) -> bool: ...
~~~

compare_and_commit must fail atomically when the parent version is stale.

## 3. Agent

~~~python
class Agent(Protocol):
    name: str
    def handle(self, artifact: Artifact, context: AgentContext) -> list[Artifact]: ...
~~~

An agent returns artifacts or requests. It does not mutate the environment directly.

## 4. Skill

~~~python
class TypedSkill(Protocol):
    skill_id: str
    version: str
    def validate_args(self, args: dict[str, object]) -> None: ...
    def execute(self, args: dict[str, object], budget: ExecutionBudget) -> SkillExecutionResult: ...
~~~

## 4.1 Typed graph planning

The GaP-inspired planning seam is implemented in `contracts/graph.py` and
`graph/validator.py`:

~~~python
@dataclass(frozen=True)
class MissionGraph:
    mission_id: str
    task: str
    subgraphs: tuple[SubgraphSpec, ...]
    edges: tuple[MissionEdge, ...]
    bindings: tuple[MissionBinding, ...]
    entry_subgraph: str
    success_subgraphs: tuple[str, ...]
    failure_subgraphs: tuple[str, ...]
~~~

~~~python
@dataclass(frozen=True)
class SubgraphSpec:
    subgraph_id: str
    subgoal_id: str
    nodes: tuple[SubgraphNodeSpec, ...]
    edges: tuple[GraphEdge, ...]
    checkpoints: tuple[CheckpointSpec, ...]

    def to_action_contract(
        self,
        node_id: str,
        context: AgentContext,
    ) -> ActionContract: ...
~~~

`GraphValidator.validate(graph)` returns structured diagnostics instead of
silently repairing a candidate. It checks graph reachability, typed bindings,
cycle absence, observable checkpoints, and conflicts between exclusive resources
on parallel mission branches. The validator is static; scene-dependent
preconditions and postconditions remain the responsibility of `Verifier`.

The current lowering method preserves the P2.5 runtime seam. The fixed graph
runtime now also implements `EventBus`, `CandidateArbiter`, and
`FixedGraphInterpreter`; these are deterministic seams, not yet LLM-driven or
process-distributed implementations.

Typed dataflow is explicit: an action's last `SkillTrace.output` feeds a node
output port, `PortBinding` feeds a later node's first `SkillCall.args`, and
`SubgraphOutputBinding` plus `MissionBinding` feed the next subgraph. Required
mission inputs must have one binding from a reachable predecessor. The current
schema intentionally does not expose intermediate outputs from multiple
`SkillCall`s inside one action node.

Runtime failures are published as a write-once `FailureArtifact` through the
`ArtifactStore` and `EventBus`. A `RecoverySelector` can only choose a target
declared by a matching recovery edge; it cannot execute a robot action or alter
the graph arbitrarily.

## 5. Perception

~~~python
class SceneEstimator(Protocol):
    def update(self, sensor_frame: SensorFrame) -> SceneSnapshot: ...
    def predict(self, timestamp_ns: int) -> SceneSnapshot: ...
    def freshness(self, snapshot: SceneSnapshot, now_ns: int) -> float: ...
~~~

~~~python
class SemanticPerception(Protocol):
    def request(self, request: SemanticRequest) -> RequestId: ...
    def poll(self, request_id: RequestId) -> SemanticResult | None: ...
~~~

request() is asynchronous and never appears on the servo thread.

The raw-observation implementation boundary is:

~~~python
class PerceptionAgent(Protocol):
    def perceive(
        self,
        request: PerceptionRequest,
        observation: ObservationBundle,
    ) -> PerceptionResult: ...
~~~

This is deliberately not an LLM interface. It is the owner of RGB-D access;
its artifact-backed result is later fused into a `SceneSnapshot`.

The agent-facing perception contract is intentionally different from the raw
sensor contract:

~~~python
class GroundedPolicyAgent(Protocol):
    def decide(
        self,
        subgoal: AgentArtifact,
        scene: SceneSnapshot,
        context: AgentContext,
    ) -> PolicyDecision: ...
~~~

`PolicyDecision` contains exactly one of an `ActionContract` or a targeted
`PerceptionRequest`. `SceneSnapshot` carries artifact references for visual
grounding, but `AgentContext` does not carry an `ObservationBundle`. The
Perception Agent alone consumes raw RGB-D observations.

## 6. Agent and LLM

```python
class Agent(Protocol):
    name: str
    def handle(self, artifact: AgentArtifact, context: AgentContext) -> list[AgentArtifact]: ...

class LLMClient(Protocol):
    def complete(self, request: LLMRequest) -> LLMResponse: ...
```

Agents propose artifacts and contracts; they do not receive a backend environment object. LLM calls are outside the control process.

## 7. Verifier and learning

~~~python
class Verifier(Protocol):
    def approve(
        self,
        contract: ActionContract,
        scene: SceneSnapshot,
    ) -> VerificationResult: ...

    def commit(
        self,
        contract: ActionContract,
        before: SceneSnapshot,
        after: SceneSnapshot,
        trace: ExecutionTrace,
    ) -> VerificationResult: ...

class RewardEngine(Protocol):
    def benchmark(self, episode: EpisodeTrace) -> float: ...  # CAP-X binary score
    def learning_return(self, transition: VerifiedTransition) -> LearningReturn: ...

    # Safety violations are emitted as hard constraint events, not compensable reward.

class MemoryController(Protocol):
    def select(
        self,
        context: MemoryContext,
    ) -> MemorySelection: ...

class MemoryExecutor(Protocol):
    def apply(
        self,
        selection: MemorySelection,
        trace_span: TraceSpan,
    ) -> MemoryUpdate: ...
~~~

## 8. Memory

```python
class MemoryStore(Protocol):
    def snapshot(self) -> MemorySnapshot: ...
    def commit(self, update: MemoryUpdate) -> MemorySnapshot: ...
```

`MemoryUpdate` is a proposal and requires provenance, an idempotency key, and the current memory-bank version before commit.

## 9. Perception

2D and 3D implementations remain separate, but both emit artifact-backed results that the fused facade can publish as a `SceneSnapshot`.

## 10. Ablation seams

Each protocol must have a reference implementation and a disabled or CAP-X implementation. This prevents ablations from being implemented as ad hoc branches inside unrelated modules.
