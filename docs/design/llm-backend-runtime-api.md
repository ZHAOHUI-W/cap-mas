# LLM Backend 与 Runtime Plane 接口说明

## 1. 问题陈述

CAP-MAS 的 Agent Plane 中每个 Agent 都需要调用 LLM 进行推理决策。现有 v0 代码已定义 `LLMClient` Protocol 和 `LLMRequest`/`LLMResponse` 数据结构（`capmas/llm/protocol.py`），但缺少：
- 五种后端实现族的接口规范
- 多模态消息格式的类型约束
- SceneSnapshot → LLM-consumable 消息的桥接层
- Runtime Scheduler / Action Lease / Verifier 的完整接口边界

本文档补全上述缺口，为 LLM Backend 和 Runtime Plane 提供完整的接口设计和测试接缝。

## 2. 解决方案

定义三层接口：

1. **LLM Backend 层**：统一 `LLMClient` Protocol + 五种后端实现族接口
2. **多模态桥接层**：`MultiModalContent` + `SceneSerializer`，将 `SceneSnapshot` 转化为 Agent 可消费的多模态消息
3. **Runtime 调度层**：`Scheduler` + `ActionLeaseManager` + `Verifier` Protocol 的完整接口边界

## 3. LLM Backend 接口族

### 3.1 核心 Protocol

```python
from typing import Mapping, Protocol, Sequence, Iterator

class LLMClient(Protocol):
    """所有 LLM 后端的统一接口。"""
    def complete(self, request: "LLMRequest") -> "LLMResponse": ...
    def stream(self, request: "LLMRequest") -> Iterator["LLMDelta"]: ...
    def cancel(self, request_id: str) -> None: ...
    def health(self) -> "BackendHealth": ...
```

### 3.2 Request / Response 数据结构

```python
from dataclasses import dataclass, field

@dataclass(frozen=True)
class MultiModalContent:
    """多模态消息内容块。type 决定取哪个字段。"""
    type: str  # "text" | "image_url" | "artifact_ref"
    text: str | None = None
    image_url: str | None = None      # data: URL 或 artifact:// ref
    artifact_ref: str | None = None   # 指向 InMemoryArtifactStore 的 URI

@dataclass(frozen=True)
class MultiModalMessage:
    """一条多模态消息（对应 OpenAI content 数组格式）。"""
    role: str  # "system" | "user" | "assistant"
    content: Sequence[MultiModalContent]

@dataclass(frozen=True)
class LLMRequest:
    """向 LLM 后端发送的推理请求。"""
    request_id: str
    agent_name: str                     # 标识调用方角色
    messages: Sequence[Mapping[str, object]]   # 标准化后的多模态消息
    response_schema: Mapping[str, object] | None = None  # JSON Schema
    deadline_ms: int = 30_000
    max_output_tokens: int = 4096

@dataclass(frozen=True)
class LLMResponse:
    request_id: str
    content: str                        # 原始文本
    structured: Mapping[str, object] | None = None  # 解析后的结构化 artifact
    model: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None
    finish_reason: str = "stop"
    error: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

@dataclass(frozen=True)
class LLMDelta:
    """流式响应的增量块。"""
    request_id: str
    content: str
    finish_reason: str | None = None

@dataclass(frozen=True)
class BackendHealth:
    available: bool
    model: str = ""
    latency_ms: float | None = None
    error: str | None = None
```

### 3.3 五种后端实现

| Backend | 接口继承 | 用途 | 测试接缝 |
|---------|---------|------|---------|
| `CapxCompatibleBackend` | `LLMClient` | 调用 CAP-X 同款模型/API 路径 | `tests/contract/test_llm_backend.py` 中使用 mock 端点 |
| `OpenAICompatibleBackend` | `LLMClient` | 调用 OpenAI 兼容端点 | 替换 `remote_endpoint` 为 mock |
| `LocalModelBackend` | `LLMClient` | 调用本地模型服务器 | 替换 `local_endpoint` 为 mock |
| `MockBackend` | `LLMClient` | 确定性 fixture 响应 | 固定输入→固定输出 |
| `ReplayBackend` | `LLMClient` | 回放记录的 CAP-X 响应 | 从 trace 文件加载 |

```python
class CapxCompatibleBackend(LLMClient):
    """复用 CAP-X 的模型调用代码，包裹为 LLMClient 协议。
    
    关键设计约束：
    - 保持与 CAP-X 相同的 prompt 格式、温度和重试策略
    - 仅增加结构化输出的 schema 校验层
    """
    def __init__(
        self,
        capx_llm_client: object,       # CAP-X 的 LLM 客户端实例
        model: str,
        temperature: float = 0.0,
    ) -> None: ...

class OpenAICompatibleBackend(LLMClient):
    def __init__(
        self,
        remote_endpoint: str,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.0,
    ) -> None: ...

class LocalModelBackend(LLMClient):
    def __init__(
        self,
        local_endpoint: str,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> None: ...

class MockBackend(LLMClient):
    """测试用：返回预定义的 fixture 响应。"""
    def __init__(self, responses: dict[str, LLMResponse]) -> None: ...
    def add_fixture(self, request_hash: str, response: LLMResponse) -> None: ...

class ReplayBackend(LLMClient):
    """离线回放：从存储的 trace 文件中加载历史响应。"""
    def __init__(self, trace_file: str) -> None: ...
```

### 3.4 SceneSnapshot → LLM 消息桥接

```python
class SceneSerializer:
    """将 SceneSnapshot 转化为 Agent 可消费的多模态消息列表。
    
    设计原则：
    - 原始 RGB-D 图像不嵌入消息，仅传递 ArtifactRef → backend 层按需加载
    - 结构化摘要（track_id + label + pose + confidence）以文本形式传递
    - 最多包含 max_images 张场景图像以避免 token 爆炸
    """

    def to_messages(
        self,
        scene: "SceneSnapshot",
        agent_name: str,
        max_images: int = 3,
        include_depth: bool = False,
    ) -> Sequence[MultiModalMessage]:
        """将场景快照转化为角色专属的多模态消息。"""
        ...
```

### 3.5 Prompt 构建器

```python
class PromptBuilder:
    """从结构化上下文和角色模板构建 LLMRequest。
    
    每个 Agent 角色有一个对应的模板 + 输入/输出 Schema。
    """

    def __init__(self, templates: Mapping[str, str]) -> None:
        """templates: agent_name → system_prompt_template"""
        ...

    def build(
        self,
        agent_name: str,
        task_text: str,
        scene: "SceneSnapshot",
        context: "AgentContext",
        skill_metadata: Sequence[Mapping[str, object]],
        previous_failures: Sequence[Mapping[str, object]] = (),
    ) -> LLMRequest:
        """构建该角色的 LLM 请求。

        输入被序列化为 messages 数组：
        [0] system: 角色专属 system prompt + 技能元数据 + 预算
        [1...n]: 场景图像的多模态 user/assistant 交替消息
        [n+1]: task文本 + 场景结构化摘要 + 失败历史
        """
        ...
```

### 3.6 结构化输出校验

```python
class SchemaValidator:
    """校验 LLM 返回的 structured 字段是否匹配 response_schema。
    
    不合法输出是推理失败，不能静默转换为默认动作。
    """

    @staticmethod
    def validate(
        structured: Mapping[str, object] | None,
        schema: Mapping[str, object],
    ) -> bool:
        """验证 structured 对象是否符合 JSON Schema。"""
        ...

    @staticmethod
    def parse_or_reject(
        content: str,
        schema: Mapping[str, object],
    ) -> tuple[Mapping[str, object] | None, str | None]:
        """解析 LLM 文本输出为结构化 artifact，失败返回 error_message。"""
        ...
```

## 4. Runtime 调度层接口

### 4.1 RuntimeOrchestrator

基于 `capmas/runtime/orchestrator.py`（第 40-179 行）的形式化说明：

```
RuntimeOrchestrator.run_cycle(ActionContract) → CycleResult

执行流程：
1. 校验 contract 的 episode_id + episode_epoch
2. 校验 parent_scene_version == 当前最新版本（拒绝过期）
3. SkillRegistry.validate_contract(contract) （校验技能存在性和参数）
4. Verifier.approve(contract, scene) （前置条件验证）
5. ActionLeaseManager.acquire() （获取独占租约）
6. 逐 skill 执行：backend.execute_skill(skill, args, budget)
   - 成功：记录 SkillTrace
   - 失败：立即 observe 最新场景，构造 ExecutionTrace(status="failed")，返回
7. backend.observe() 获取执行后场景
8. StateStore.compare_and_commit(parent, after) （原子提交）
9. Verifier.commit(contract, before, after, trace) （后置条件验证）
10. finally: lease_manager.release()
```

**测试接缝**: `tests/test_runtime_cycle.py` — 已验证正常执行和过期拒绝 ✅

### 4.2 Scheduler Protocol

```python
class Scheduler(Protocol):
    """多 Agent 调度器的统一接口。"""
    def dispatch(self, contract: "ActionContract", scene: "SceneSnapshot") -> "CycleResult": ...

class FixedGraphScheduler:
    """固定拓扑调度器：Manager → Policy → Verifier → Executor → Monitor。
    
    与 FixedGraphScheduler 的当前实现（capmas/runtime/scheduler.py:15-22）完全一致。
    """
    orchestrator: RuntimeOrchestrator

    def dispatch(self, contract, scene) -> CycleResult:
        if scene.scene_version != self.orchestrator.state_store.latest().scene_version:
            raise ValueError("scheduler received a stale scene")
        return self.orchestrator.run_cycle(contract)
```

### 4.3 Action Lease Manager

基于 `capmas/runtime/action_lease.py`（第 20-57 行）：

```
ActionLeaseManager
├── acquire(holder, contract_id, duration_ms) → ActionLease
│   唯一性约束：同一时刻最多一个活跃租约
├── release(lease_id) → None
│   只能释放当前持有者的租约
├── expire_if_needed(now_ns) → bool
│   租约过期自动释放
└── active() → ActionLease | None
```

### 4.4 Verifier Protocol

基于 `capmas/runtime/orchestrator.py`（第 19-28 行）：

```python
class Verifier(Protocol):
    def approve(
        self,
        contract: "ActionContract",
        scene: "SceneSnapshot",
    ) -> "VerificationResult":
        """执行前验证：前置条件 + 安全不变量。
        
        决策：
        - "approve": 允许执行
        - "reject": 拒绝（场景过期/前置条件失败/碰撞风险）
        """
        ...

    def commit(
        self,
        contract: "ActionContract",
        before: "SceneSnapshot",
        after: "SceneSnapshot",
        trace: "ExecutionTrace",
    ) -> "VerificationResult":
        """执行后验证：后置条件检查。
        
        决策：
        - "commit": 接受场景变更
        - "recover": 后置条件失败，触发恢复
        """
        ...
```

### 4.5 Checkpoint Manager（待实现接口）

```python
class CheckpointManager(Protocol):
    """子目标检查点管理器。
    
    在检查点处可以：
    - 激活经过影子验证的 Memory Skill 候选
    - 更新拓扑（在固定图基础上增加/删除边）
    - 快照当前的 skill registry 和 memory bank 版本
    """

    def request_checkpoint(
        self,
        subgoal_id: str,
        reason: str,
    ) -> "CheckpointHandle": ...

    def commit_checkpoint(
        self,
        handle: "CheckpointHandle",
        updates: Mapping[str, object],
    ) -> None: ...

    def rollback_to(
        self,
        handle: "CheckpointHandle",
    ) -> None: ...
```

## 5. 测试接缝

### 5.1 LLM Backend 测试

```python
# tests/contract/test_llm_backend.py 的核心测试用例

def test_mock_backend_returns_fixture():
    """MockBackend 对预定义输入返回确定性响应。"""
    ...

def test_capx_compatible_backend_preserves_format():
    """CapxCompatibleBackend 保持与 CAP-X 相同的 prompt 格式。"""
    ...

def test_schema_validator_rejects_invalid_response():
    """Schema 不匹配的响应被拒绝，不静默转换为默认动作。"""
    ...

def test_scene_serializer_limits_image_count():
    """SceneSerializer.to_messages() 不超过 max_images 张图像。"""
    ...

def test_scene_serializer_does_not_embed_raw_arrays():
    """场景图像通过 ArtifactRef 引用，不嵌入消息。"""
    ...

def test_prompt_builder_output_contains_required_fields():
    """PromptBuilder 产出的 messages 包含 task、scene、skill_metadata、budget。"""
    ...
```

### 5.2 预算可观测性测试

```python
def test_llm_request_records_budget_group():
    """每个 LLMRequest 携带 agent_name 用于按角色统计消耗。"""
    ...

def test_deadline_exceeded_response_contains_timeout_error():
    """超过 deadline 的请求返回 timeout error，不阻塞控制路径。"""
    ...
```

## 6. 多模态适配边界

| 接口 | 多模态支持 | 状态 |
|------|----------|------|
| `LLMRequest.messages` | `Sequence[Mapping[str, object]]` 兼容 OpenAI 多模态格式 | ✅ 已定义 |
| `MultiModalContent` + `MultiModalMessage` | 强类型多模态消息格式 | 🆕 本文档新增 |
| `SceneSerializer.to_messages()` | SceneSnapshot → 多模态消息桥接 | 🆕 本文档新增 |
| `ObservationProvider.capture()` | RGB-D 采集 + ArtifactRef 存储 | ✅ `perception/protocol.py:96-97` |
| `CAPXObservationProvider` | CAP-X 观测标准化 | ✅ `backends/capx.py:32-84` |

## 7. 范围外

- LLM 模型的训练、微调或 RL 更新（在 reward-and-rl.md 中定义）
- 生产级数据库或消息队列（v0 使用内存实现）
- 分布式调度或多进程架构（Phase 4-8 规划中）
- Web UI 或 API 服务器