# LLM Backend and API Boundary

## 1. Purpose

CAP-MAS reserves a dedicated LLM backend interface for every deliberative Agent. The backend is separate from the robot backend:

~~~text
LLMBackend   -> produces structured Agent artifacts
RobotBackend -> observes and executes robot actions
~~~

An Agent can request model inference through LLMBackend, but it cannot call RobotBackend directly. The Runtime Scheduler decides whether the resulting artifact is accepted.

## 2. Required interface

~~~python
class LLMBackend(Protocol):
    def generate(self, request: LLMRequest) -> LLMResponse: ...
    def stream(self, request: LLMRequest) -> Iterator[LLMDelta]: ...
    def cancel(self, request_id: str) -> None: ...
    def health(self) -> BackendHealth: ...
~~~

~~~python
@dataclass
class LLMRequest:
    request_id: str
    agent_role: str
    system_prompt: str
    input_artifacts: list[str]
    output_schema: str
    model: str
    temperature: float
    max_output_tokens: int
    deadline_ms: int | None
    episode_id: str
    budget_group: str
~~~

~~~python
@dataclass
class LLMResponse:
    request_id: str
    raw_text: str
    parsed_artifact: dict[str, object] | None
    schema_valid: bool
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    finish_reason: str
    error: str | None
~~~

## 3. Backend implementations

| Backend | Purpose | Required use |
| --- | --- | --- |
| capx_compatible | Calls the same CAP-X model/API path | Primary fairness baseline |
| openai_compatible | Calls an OpenAI-compatible endpoint | Remote model experiments |
| local | Calls a local model server | Cost and latency experiments |
| mock | Deterministic fixture responses | Unit and contract tests |
| replay | Replays recorded CAP-X responses | Reproducible ablations |

The initial implementation should preserve the CAP-X LLM client behavior behind capx_compatible rather than duplicate prompt or retry logic.

### 3.1 Implemented P3.1 path

`capmas.llm.capx_compatible.CAPXCompatibleLLMClient` implements the
CAP-X/OpenAI-compatible chat-completions boundary with only the Python
standard library. It preserves the endpoint/model/key separation, emits the
same GPT-5 `max_completion_tokens` convention used by CAP-X, and normalizes
provider content, structured JSON, usage, latency, finish reason, and HTTP
status into `LLMResponse`.

`capmas.llm.prompts` provides stable Manager and Policy prompt builders. Both
builders include the current `SceneSnapshot.scene_version`, skill allowlist,
bounded graph rules, and the strict versioned MissionGraph response schema.
They pass compact typed scene data rather than simulator or environment
handles.

The endpoint can be used without adding an SDK dependency:

```python
from capmas.llm.capx_compatible import CAPXCompatibleLLMClient

client = CAPXCompatibleLLMClient(
    endpoint="https://provider.example/v1/chat/completions",
    model="gpt-5.4",
    api_key_env="CAPMAS_LLM_API_KEY",
)
```

Transport errors, deadline exhaustion, malformed provider responses, and
non-2xx responses are explicit `LLMTransportError` failures. They never become
an empty plan or an implicit robot action. Retries are bounded and disabled by
default so the Agent deadline remains meaningful.

For providers that intermittently reject large provider-side schemas, the
runner supports `--no-provider-structured-output`. This removes only the
transport-level `response_format` field; the prompt still requires JSON and the
local `MissionGraphDecoder` continues to enforce the complete strict schema.
The CAP-X-compatible client also performs this downgrade once automatically
when a provider returns a clear schema-compatibility 400 mentioning
`invalid schema`, `response_format`, or `additionalProperties`. It never
downgrades arbitrary HTTP errors, and local artifact decoding remains required.

`SkillCall.args` is a runtime dictionary, but strict providers require every
object schema to set `additionalProperties: false`. Request builders derive a
parameter-name union from the registered typed skills, declare those names in
the provider schema, and encode unused parameters as `null`. The graph
serialization boundary strips those placeholders before typed CAP-X skill
validation, preserving the existing execution API while satisfying strict
provider validation.

For LIBERO, candidate compilation also separates stable scene facts from
dynamic robot state. Track existence and visibility may be checked before
execution; freshness, gripper state, and object attachment are checked only
when the action is dispatched against the latest scene. Sampled grasp and
placement poses are grounded by the CAP-X observation/skill output boundary,
while the LLM still selects the subgoal sequence and registered skills.

## 4. Structured-output rule

Every Agent request declares an output schema. The backend must return both raw model text and a parsed artifact. Invalid output is an inference failure; it must not be silently converted into FINISH, an empty plan, or an unverified action.

The following artifacts require structured output:

- SubgoalGraph;
- ActionContract;
- VerificationRequest;
- RecoveryContract;
- SemanticPerceptionResult;
- SkillCandidate;
- TopologyEdit.

The implemented `MissionGraphDecoder` is the first enforcement point for graph
responses. It accepts a provider's `structured` object or strict JSON content,
then applies the versioned graph codec, current `SceneSnapshot` version check,
and `GraphValidator`. Request-id mismatch, invalid JSON/schema, stale scene,
unbounded topology, missing checkpoints, and binding errors are explicit
rejections. No invalid response is converted into an empty plan or default
action.

`LLMGraphScheduler` implements the fixed-graph P3.1 coordination path:

```text
Manager -> MissionGraph
       -> concurrent read-only Policy proposals
       -> GraphValidator + CandidateArbiter
       -> one FixedGraphInterpreter
       -> one physical Executor / ActionLease
```

Policy fan-out uses a bounded thread pool because model calls are I/O-bound.
The workers never receive a `RobotBackend` and cannot execute actions. A
subgoal with no valid required candidate aborts before physical execution.
Manager-plan fallback is an explicit opt-in for controlled ablations only.

### 4.2 Experiment observability

`CAPXCompatibleLLMClient` accepts a thread-safe `LLMTraceSink`. Each logical
request emits one sanitized `LLMCallTrace` with request/agent identity,
provider status, attempts, latency, token usage, schema hash, schema mode,
fallback status, and bounded error diagnostics. The endpoint-backed LIBERO
runner stores these records under `llm_calls` and stores non-secret run controls
under `run_config`. It never stores API keys, authorization headers, raw
prompts, or raw model responses.

An artifact without these fields can still be used for success/parity checks,
but must not be used as evidence for latency, token, schema, or parallelism
claims.

### 4.1 Staged graph protocol

The default B3 LLM runner supports a compact staged protocol for reducing
repeated nested graph context:

```text
Manager -> MissionTopology
       -> concurrent local Policy requests
       -> direct SubgraphArtifact candidates
       -> LocalSubgraphDecoder + GraphValidator + CandidateArbiter
       -> server-side MissionGraph assembly
       -> one FixedGraphInterpreter
```

`MissionTopologyDecoder` validates Manager output and scene version before any
local Policy call. `LocalSubgraphDecoder` requires the Manager-selected IDs and
validates each local graph in isolation. The assembled graph is validated again
by the existing full `GraphValidator`; compact prompts do not reduce static
safety checks. Use `--graph-protocol legacy` to reproduce the original
full-MissionGraph Manager/Policy protocol for ablation.

## 5. Prompt and context boundary

The LLM receives:

- task text;
- permitted role instructions;
- relevant artifact references;
- compact SceneSnapshot fields;
- available skill metadata;
- remaining budgets;
- previous failure summaries.

The LLM does not receive:

- direct simulator handles;
- privileged task completion;
- mutable Python globals;
- unrestricted filesystem or network access;
- the ability to issue an actuator call outside the contract path.

## 6. Budget and observability

Each request records:

- provider and model;
- prompt and artifact hashes;
- input/output tokens;
- latency and queue wait;
- retries and cancellation;
- schema-validation result;
- downstream artifact acceptance or rejection.

Budgets are grouped by episode and role. The evaluation runner must be able to enforce equal total tokens, calls, and wall-clock budgets between CAP-X and CAP-MAS.

## 7. Latency policy

LLM calls are never placed in the high-frequency control path. A deadline applies only to the requesting Agent. If a response misses its deadline:

1. cancel or mark the request expired;
2. preserve the latest valid plan;
3. use a deterministic fallback or recovery policy;
4. emit a timeout artifact for evaluation.

The Perception Agent follows the same rule; fast geometric estimation remains independent of LLM inference.

## 8. CAP-X comparison modes

~~~text
capx_legacy:
    CAP-X prompt -> CAP-X LLM client -> CAP-X Python execution

capx_compatible:
    CAP-X prompt and model settings -> LLMBackend(capx_compatible)
    -> CAP-MAS-normalized response trace

capmas:
    role prompt + typed artifacts -> LLMBackend
    -> structured Agent artifact -> Runtime validation
~~~

The model and API provider should remain fixed when testing architecture modules. A local-model comparison is a separate experiment, not an implicit change to the main baseline.

## 9. LLM-specific ablations

- CAP-X legacy LLM client versus capx_compatible adapter.
- Same model with one Agent versus multiple roles.
- Natural-language output versus schema-constrained output.
- No retry versus bounded retry.
- Full context versus artifact-selected context.
- Synchronous request waiting versus deadline-aware asynchronous requests.
- Manager-only graph versus typed local Policy fan-out.
- One Policy Agent versus multiple candidate-producing Policy Agents.
