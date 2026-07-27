# CAP-MAS 公共接口总览

## 1. 总体架构

CAP-MAS 是一个七平面运行时系统：

| Plane | 频率 | 职责 | 模块 |
|-------|------|------|------|
| Interface Layer | 请求/响应 | Task API, Artifact API, Evaluation API | `capmas/contracts/agent.py` |
| Agent Plane | 事件触发 | Manager, Policy, Verifier, Recovery, Monitor, Perception, Critic | `capmas/agents/` |
| Contract & Runtime Plane | 同步 | Message Envelope, SceneSnapshot, ActionLease, State Store, Event Bus | `capmas/contracts/`, `capmas/runtime/` |
| World Model Plane | 5-30 Hz（异步） | Sensor Sync, FK, Geometry, Tracking, 3D Map, Semantic Trigger | `capmas/perception/` |
| Skill & Execution Plane | 20-100 Hz | TypedSkill, Registry, Validator, Executor, Safety Monitor | `capmas/skills/`, `capmas/execution/` |
| Backend Plane | 同步 | CAP-X Legacy/Typed, LIBERO, Robosuite, BEHAVIOR adapters | `capmas/backends/` |
| Evaluation Plane | 离线 | Baseline Runner, Budget Matcher, Ablation Controller, Metrics | `capmas/evaluation/` |

## 2. 核心数据流

```
Task → Mission Manager → MissionGraph
  → local Subgraph Policy Agents → GraphValidator → Arbiter
  → ActionContract(parent_scene_version)
  → Schema Validator → Verifier.approve()
  → ActionLease → TypedSkill Executor → [Robot]
  → observe → SceneSnapshot(version+1)
  → Verifier.commit() → commit/recover
  → [失败] Recovery Agent → new ActionContract
  → [成功] next subgoal
  → EpisodeTrace → Memory Updater → Evaluation
```

## 3. 关键接口清单

### 3.1 合约层

| 接口 | 文件 | 行号 |
|------|------|------|
| `ArtifactRef` | `contracts/core.py` | 7-12 |
| `SkillRef` | `contracts/core.py` | 15-18 |
| `EpisodeHandle` | `contracts/core.py` | 21-31 |
| `SceneSnapshot` | `contracts/scene.py` | 28-38 |
| `EpisodeStart` | `contracts/scene.py` | 41-43 |
| `ActionContract` | `contracts/action.py` | 20-34 |
| `VerificationResult` | `contracts/verification.py` | 16-29 |
| `FailureClass` | `contracts/failures.py` | 1-9 |
| `ExecutionTrace` | `contracts/trace.py` | 23-42 |
| `MemoryItem` | `contracts/memory.py` | 40-55 |
| `MemoryUpdate` | `contracts/memory.py` | 104-123 |
| `MemoryContext` | `contracts/memory.py` | 76-94 |
| `AgentContext` | `contracts/agent.py` | 12-19 |
| `SubgraphSpec` | `contracts/graph.py` | local typed policy graph |
| `MissionGraph` / `LoopSpec` | `contracts/graph.py` | graph contracts |
| `GraphCandidate` / `FailureArtifact` | `contracts/candidates.py`, `contracts/failures.py` | typed proposal/failure artifacts |
| `GraphValidator` | `graph/validator.py` | structural, typed-dataflow, mission-binding, and bounded-loop validation |
| `mission_graph_to_dict/from_dict` | `graph/serialization.py` | strict schema-versioned codec |

### 3.2 运行时层

| 接口 | 文件 | 行号 |
|------|------|------|
| `RobotBackend` Protocol | `backends/protocol.py` | 18-36 |
| `SkillExecutionResult` | `backends/protocol.py` | 10-15 |
| `RuntimeOrchestrator.run_cycle()` | `runtime/orchestrator.py` | 62-179 |
| `InMemoryStateStore` | `runtime/state_store.py` | 6-48 |
| `ActionLeaseManager` | `runtime/action_lease.py` | 20-57 |
| `FixedGraphScheduler` | `runtime/scheduler.py` | contract dispatch seam |
| `FixedGraphInterpreter` | `runtime/graph_interpreter.py` | fixed MissionGraph execution |
| `ArtifactStore` / `EventBus` | `runtime/artifact_bus.py` | write-once artifacts and typed events |
| `CandidateArbiter` | `agents/arbiter.py` | deterministic evidence-aware candidate selection |
| `TypedSkill` Protocol | `skills/protocol.py` | 10-20 |
| `SkillRegistry` | `skills/registry.py` | 10-37 |
| `LLMClient` Protocol | `llm/protocol.py` | 29-30 |

### 3.3 感知层

| 接口 | 文件 | 行号 |
|------|------|------|
| `ObservationProvider` | `perception/protocol.py` | 96-97 |
| `Vision2DBackend` | `perception/protocol.py` | 100-107 |
| `Geometry3DBackend` | `perception/protocol.py` | 110-115 |
| `GraspProposalBackend` | `perception/protocol.py` | 118-123 |
| `RobotControlBackend` | `perception/protocol.py` | 126-136 |
| `FusedPerceptionBackend` | `perception/protocol.py` | 139-151 |
| `CAPXObservationProvider` | `backends/capx.py` | 32-84 |
| `CAPXTypedSkill` | `backends/capx.py` | 87-115 |

### 3.4 记忆与评估层

| 接口 | 文件 | 行号 |
|------|------|------|
| `MemoryController` Protocol | `memory/protocol.py` | 13-14 |
| `MemoryExecutor` Protocol | `memory/protocol.py` | 17-18 |
| `RuleBasedMemoryController` | `memory/controller.py` | 8-18 |
| `InMemoryMemoryStore` | `memory/store.py` | 14-46 |
| `Evaluator` Protocol | `evaluation/interfaces.py` | 8-9 |
| `CAPXBinaryReward` | `evaluation/reward.py` | 29-39 |

### 3.5 Agent 层

| 接口 | 文件 | 行号 |
|------|------|------|
| `Agent` Protocol | `contracts/agent.py` | 31-34 |
| `MissionManager` Protocol | `contracts/agent.py` | 37-38 |
| `PolicyAgent` Protocol | `contracts/agent.py` | 41-47 |
| `RecoveryAgent` Protocol | `contracts/agent.py` | 50-56 |
| `SimpleMissionManager` | `agents/manager.py` | 10-19 |
| `CallablePolicyAgent` | `agents/policy.py` | 10-22 |
| `LLMMissionManager` / `LLMGraphPolicyAgent` | `agents/manager.py`, `agents/policy.py` | strict typed graph proposal seams |

## 4. 设计文档索引

| 文档 | 内容 |
|------|------|
| [code-framework.md](code-framework.md) | 项目目录结构、配置系统、集成脚本、Phase 映射 |
| [contracts-api.md](contracts-api.md) | 合约系统：状态机、引用关系、序列化规范、测试接缝 |
| [agent-plane-api.md](agent-plane-api.md) | Agent Plane：Prompt 模板设计、Schema 定义、通信拓扑、执行生命周期 |
| [llm-backend-runtime-api.md](llm-backend-runtime-api.md) | LLM Backend 五种实现、多模态消息格式、Runtime 调度层 |
| [world-model-api.md](world-model-api.md) | 几何感知管道：多速率融合、后端分离、场景发布管道 |
| [skill-execution-api.md](skill-execution-api.md) | Robot/Memory Skill 双注册表、进化循环、Quarantine & Promotion |
| [memory-eval-api.md](memory-eval-api.md) | 四层记忆、MemoryItem/MemoryUpdate Schema、双通道奖励、公平对比规则 |
