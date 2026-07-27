# Agent Plane 接口说明

## 1. 问题陈述

CAP-MAS 的 Agent Plane 包含 8 个角色（Manager / Policy / Verifier / Recovery / Monitor / Perception / Critic / Evolver），但 v0 代码中只有 Protocol 定义和确定性 mock。要将 Agent 接入真实 LLM 推理，需要：
- 每个角色的 Prompt 模板设计
- Context → Messages 的序列化规则
- Response Schema 定义（JSON Schema 格式）
- Default/fallback 行为
- Agent 之间的通信拓扑和 artifact 流转
- Manager 生成 MissionGraph、Policy Agent 生成局部 SubgraphSpec、GraphValidator 静态检查

## 2. 解决方案

定义 6 层 Agent 接口：
1. Agent 基类 Protocol
2. 各角色 Prompt 组装接口
3. 角色间的通信拓扑
4. 每个角色的输出 Schema
5. 多 Agent 执行生命周期
6. 测试接缝

## 3. Agent 基类 Protocol

基于 `capmas/contracts/agent.py`（第 31-34 行）：

```python
class Agent(Protocol):
    """所有 Agent 的统一接口。
    
    Agent 返回 artifacts 或 requests，不直接改变环境。
    """
    name: str

    def handle(
        self,
        artifact: "AgentArtifact",
        context: "AgentContext",
    ) -> list["AgentArtifact"]: ...
```

**AgentContext**（`capmas/contracts/agent.py:12-19`）：

```python
@dataclass(frozen=True)
class AgentContext:
    task_id: str
    episode_id: str
    episode_epoch: int
    scene: "SceneSnapshot"                       # 当前版本化场景快照
    memories: "MemoryContext" | None = None       # 检索到的记忆上下文
    budget: Mapping[str, int]                     # 剩余 token/调用预算
```

## 4. 各角色接口定义

### 4.1 Mission Manager

**职责**：将自然语言任务分解为子目标图，分配全局预算和调度。

```python
class MissionManager(Protocol):
    """全局任务分解与调度。
    
    LLM 调用：是（task → SubgoalGraph）
    """
    def propose_subgoal(self, task: str, scene: "SceneSnapshot") -> "AgentArtifact": ...

class LLMMissionManager(MissionManager):
    """基于 LLM 的任务分解器。"""
    def __init__(
        self,
        llm: "LLMClient",
        prompt_builder: "PromptBuilder",
        schema_validator: "SchemaValidator",
    ) -> None: ...
```

**System Prompt 模板**（伪代码）：

```
你是任务分解器（Mission Manager）。你的职责是将自然语言操作任务
分解为有序的子目标图（DAG）。

输入：
- 任务描述：{task_text}
- 初始场景：{scene_summary}
- 可用技能：{skill_metadata}

输出要求：
- 每个子目标必须可以独立验证（有明确的前置/后置条件）
- 子目标之间的依赖关系必须形成 DAG
- 不能假设 success 信号作为子目标完成的证据
- 必须给每个子目标分配预算（max_duration_ms, max_sim_steps）

输出格式：SubgoalGraph JSON Schema
```

**输出 Schema**：

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "title": "SubgoalGraph",
  "properties": {
    "subgoals": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["subgoal_id", "description", "preconditions", "expected_postconditions"],
        "properties": {
          "subgoal_id": {"type": "string"},
          "description": {"type": "string"},
          "preconditions": {"type": "array", "items": {"type": "string"}},
          "expected_postconditions": {"type": "array", "items": {"type": "string"}},
          "max_duration_ms": {"type": "integer", "default": 10000},
          "max_sim_steps": {"type": "integer", "default": 500},
          "depends_on": {"type": "array", "items": {"type": "string"}}
        }
      }
    },
    "budget_allocations": {"type": "object"}
  }
}
```

**当前行为**：`LLMMissionManager` 通过 `MissionGraphDecoder` 解析完整 typed graph。
请求 ID、JSON/schema、scene version 或 GraphValidator 任一检查失败都会抛出带 rejection
code 的 `GraphProposalError`；不会把无效输出回退成默认子目标。超时和恢复策略由后续
Scheduler/Recovery Agent 显式处理。

---

### 4.2 Policy Agent

**职责**：根据当前子目标和场景提出局部 `SubgraphSpec`，并通过兼容层降低为一个或多个 `ActionContract`。

```python
class PolicyAgent(Protocol):
    """策略生成器：子目标 → SubgraphSpec/ActionContract。"""
    def propose_action(
        self,
        subgoal: "AgentArtifact",
        scene: "SceneSnapshot",
        context: "AgentContext",
    ) -> "ActionContract": ...

class GraphPolicyAgent(Protocol):
    """GaP-inspired local graph proposal boundary."""
    def propose_subgraph(
        self,
        subgoal: "AgentArtifact",
        scene: "SceneSnapshot",
        context: "AgentContext",
    ) -> "SubgraphSpec": ...

class LLMPolicyAgent(PolicyAgent):
    """基于 LLM 的策略生成器。
    
    适配器模式：构造函数接受 Callable，可以替换为 LLM 或确定性规划器。
    与 capmas/agents/policy.py:CallablePolicyAgent 一致。
    """
    def __init__(
        self,
        llm: "LLMClient",
        prompt_builder: "PromptBuilder",
        schema_validator: "SchemaValidator",
    ) -> None: ...

class CallablePolicyAgent(PolicyAgent):
    """现有实现：proposer 通过 Callable 注入。"""
    def __init__(self, proposer: Callable) -> None: ...

当前代码中的 `LLMGraphPolicyAgent` 复用 `MissionGraphDecoder`，从模型返回的完整
`MissionGraph` 中按 Manager 发来的 `subgraph_id`/`subgoal_id` 提取唯一局部
`SubgraphSpec`。两者都不持有 RobotBackend，也不能直接取得 ActionLease。
```

**System Prompt 模板**：

```
你是策略生成器（Policy Agent）。你的职责是根据当前子目标和场景
生成 ActionContract（有界的动作计划）。

输入：
- 当前子目标：{subgoal_text}
- 场景状态：{scene_summary}（含物体追踪和置信度）
- 可用 Typed Skills：{skill_metadata}
- 剩余预算：{remaining_budget}
- 历史失败摘要：{failure_summaries}

约束：
- 只能使用已注册的 TypedSkill（skill_id + version）
- 必须声明 preconditions 和 expected_postconditions
- 必须设置 max_duration_ms 和 max_sim_steps
- 必须指定 recovery_policy
- 不能依赖特权状态变量（如 evaluator_success）

输出格式：ActionContract JSON Schema
```

**输出 Schema**（`ActionContract` — `capmas/contracts/action.py:21-34`）：

```json
{
  "required": ["contract_id", "episode_id", "parent_scene_version", "subgoal_id",
    "skills", "expected_postconditions", "max_duration_ms", "max_sim_steps", "proposed_by"],
  "properties": {
    "skills": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["skill", "args"],
        "properties": {
          "skill": {"$ref": "#/definitions/SkillRef"},
          "args": {"type": "object"}
        }
      }
    },
    "preconditions": {"type": "array", "items": {"type": "string"}},
    "expected_postconditions": {"type": "array", "items": {"type": "string"}},
    "safety_invariants": {"type": "array", "items": {"type": "string"}},
    "recovery_policy": {"type": "string", "enum": ["replan", "reacquire_and_retry", "skip", "abort"]}
  }
}
```

**当前行为**：LLM 输出无效时显式拒绝并交给上层 Recovery/重规划逻辑；decoder 不会
自动重试、伪造 ActionContract 或把空响应解释为 FINISH。

---

### 4.3 Verifier Agent

**职责**：校验前置/后置条件、安全不变量。

**LLM 调用**：可选。确定性规则检查器优先，LLM 仅用于语义验证（对象身份消歧等）。

```python
class Verifier(Protocol):
    """基于 capmas/runtime/orchestrator.py:19-28。"""
    def approve(self, contract: "ActionContract", scene: "SceneSnapshot") -> "VerificationResult": ...
    def commit(self, contract, before, after, trace) -> "VerificationResult": ...

class PredicateBasedVerifier(Verifier):
    """基于 SceneSnapshot 的确定性可观测谓词验证器。"""
    def __init__(
        self,
        gripper_open_threshold: float = 0.8,
        gripper_closed_threshold: float = 0.2,
        object_gripper_distance_threshold_m: float = 0.16,
        object_target_distance_threshold_m: float = 0.06,
    ) -> None: ...

# Built-in predicates include:
# - object_in_gripper(obj_id)
# - object_at_target(obj_id, target_id)
# - gripper_open() / gripper_closed()
# - scene_fresh(threshold_ms)
# Object identifiers match either ObjectTrack.track_id or ObjectTrack.label.

class LLMAssistedVerifier(Verifier):
    """确定性谓词 + LLM 语义辅助的验证器（Phase 3+）。"""
    def __init__(self, predicates, llm: "LLMClient") -> None: ...
```

**System Prompt 模板**（仅在 `LLMAssistedVerifier` 中使用）：

```
你是验证器（Verifier Agent）。你的任务是基于当前场景证据判断
前置条件/后置条件/安全不变量是否满足。

关键约束：
- 只能基于 SceneSnapshot 中的可观测证据做判断
- 不能假设 evaluator_success
- 置信度不足时返回 PERCEPTION_UNCERTAIN

输出格式：VerificationResult（decision, predicate_results, failure_class）
```

---

### 4.4 Recovery Agent

**职责**：分析失败 traces、分类失败原因、提出恢复合约。

```python
class RecoveryAgent(Protocol):
    """基于 capmas/contracts/agent.py:50-56。"""
    def recover(
        self,
        trace: object,               # ExecutionTrace
        verification: "VerificationResult",
        context: "AgentContext",
    ) -> "ActionContract | None": ...  # None = 放弃

class LLMRecoveryAgent(RecoveryAgent):
    def __init__(self, llm: "LLMClient", prompt_builder: "PromptBuilder") -> None: ...
```

**System Prompt 模板**：

```
你是恢复规划器（Recovery Agent）。根据失败 trace 和验证结果，
提出恢复动作合约。

失败信息：
- 失败分类：{failure_class}
- 验证报告：{verification_result}
- 执行迹：{trace_summary}
- 当前场景：{scene_summary}

可用的恢复策略：
- replan: 从最新场景重新规划
- reacquire_and_retry: 重新感知目标并重试
- skip: 跳过当前子目标
- abort: 终止 episode

关键约束：不能改写已发生的物理历史。恢复必须从新的 observe 开始。
```

---

### 4.5 Perception Agent

**职责**：事件触发式语义感知（目标识别、身份消歧、语义地图修正）。

**LLM 调用**：是（通常使用 VLM 进行视觉推理）。感知推理不在高频控制路径。

```python
class SemanticPerception(Protocol):
    """基于 capmas/perception/protocol.py:73-75。
    
    异步调用：request() 绝不出现于伺服线程。
    """
    def request(self, request: "SemanticRequest") -> "RequestId": ...
    def poll(self, request_id: "RequestId") -> "SemanticResult | None": ...
```

**触发条件**（低置信度 / 事件驱动）：
- 目标 tracking confidence < 0.7
- 场景中物体数量与已知状态不一致
- Policy 或 Verifier 请求语义消歧
- 新的未识别物体出现

---

### 4.6 Execution Monitor

**职责**：监控执行、记录 trace、报告后置条件和进度。

**LLM 调用**：否（确定性组件）。

```python
class ExecutionMonitor:
    """监听执行事件并在检查点提交时验证进度。
    
    非 LLM 调用的确定性组件。
    不修改 robot state，只记录和报告。
    """
    ...
```

---

### 4.7 Critic & Skill Evolvers

**职责**：离线分析 trace、归因失败、生成 Memory/Robot Skill 候选。

**LLM 调用**：是（离线，不在 active episode 内）。

```python
class Critic(Protocol):
    def analyze_trace(self, trace: "ExecutionTrace") -> "FailureAttribution": ...

class MemorySkillEvolver(Protocol):
    def propose_candidate(self, attribution: "FailureAttribution") -> "MemorySkillCandidate": ...

class RobotSkillEvolver(Protocol):
    def propose_candidate(self, attribution: "FailureAttribution") -> "RobotSkillCandidate": ...
```

## 5. Agent 通信拓扑

标准固定拓扑（Phase 3 目标）：

```
SceneSnapshot → Mission Manager → SubgoalGraph
    → Policy Agents → SubgraphSpec candidates
    → GraphValidator → Candidate Arbiter
    → ActionContract → Verifier.approve() → approve/reject
    → ActionLeaseManager → lease
    → TypedSkill Executor → SkillTrace
    → Execution Monitor → postcondition check
    → Verifier.commit() → commit/recover
    → [失败] Recovery Agent → new ActionContract
    → [成功] 下一个子目标

离线：
ExecutionTrace → Critic → FailureAttribution
    → MemorySkillEvolver → MemorySkillCandidate
    → RobotSkillEvolver → RobotSkillCandidate
```

**通信规则**：
- Agent 间通过 Typed Artifact 通信，不允许无限制对话
- 自然语言可作为解释字段，不能是权威状态表示
- 拓扑编辑仅允许在 action chunk 之间
- Manager → Verifier → Executor 路径不能被断开

## 6. 多 Agent 执行生命周期

```
INITIALIZE
  → observe → publish SceneSnapshot(v)
  → Manager 创建 SubgoalContract
  ↓
PLAN_LOOP:
  → Policy 提出 ActionContract(parent=v)
  → static validator 校验 schema、skill 版本、限制
  → Verifier.approve() 检查前置条件和安全不变量
  → Executor acquire action lease
  → execute bounded action chunk
  → observe → publish SceneSnapshot(v+1)
  → Monitor 检查 postconditions
  ↓
DECIDE:
  PASS: → commit SceneSnapshot(v+1), advance subgoal
  FAIL: → classify failure, invalidate suffix, invoke Recovery
  ↓
RECOVERY_LOOP:
  → RecoveryAgent.recover() → 新的 ActionContract
  → 回到 PLAN_LOOP
  ↓
EPISODE_END:
  → archive trace
  → Memory Skill evolution (异步)
  ↓
SAFE_BOUNDARY:
  → Robot Skill evolution (仅在 Memory Skill 冻结后)
```

## 7. 测试接缝

### 7.1 Agent Protocol 测试

```python
# tests/contract/test_agent_protocols.py 核心测试用例

def test_agent_cannot_mutate_environment():
    """Agent.handle() 只返回 artifacts，不改变环境。"""
    ...

def test_mission_manager_output_is_valid_subgoal_graph():
    """MissionManager 输出的 JSON Schema 可验证。"""
    ...

def test_policy_agent_only_uses_registered_skills():
    """Policy Agent 输出的 ActionContract 只引用已注册的 SkillRef。"""
    ...

def test_recovery_agent_does_not_rewrite_history():
    """Recovery Agent 从新 observe 开始规划，不改变已有 trace。"""
    ...

def test_verifier_rejects_stale_scene():
    """Verifier 拒绝基于过期场景版本的 contract。"""
    ...
```

### 7.2 Prompt 组装测试

```python
def test_prompt_builder_includes_all_required_context():
    """每个角色的 Prompt 包含 task、scene、skill、budget、failures。"""
    ...

def test_prompt_builder_does_not_include_privileged_info():
    """Agent Prompt 不包含 evaluator_success 或 simulator 句柄。"""
    ...

def test_response_schema_different_per_role():
    """不同角色的 response_schema 不同（SubgoalGraph vs ActionContract vs RecoveryContract）。"""
    ...
```

### 7.3 通信拓扑测试

```python
def test_topology_preserves_manager_to_executor_path():
    """任何拓扑变更必须保留 Manager → Verifier → Executor 路径。"""
    ...

def test_topology_edits_only_between_action_chunks():
    """拓扑编辑不能发生在 action 执行过程中。"""
    ...
```

## 8. 范围外

- 自适应拓扑的运行时学习（Phase 8）
- Agent 间的自主学习角色分配
- 多机器人协调
- 生产级分布式 Agent 调度
