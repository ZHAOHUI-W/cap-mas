# CAP-MAS 代码结构分析

## 1. 项目概述

**CAP-MAS** (Code-as-Policy Multi-Agent System) 是一个**模块化多智能体 Code-as-Policy 运行时**，专为长时序机器人操控任务设计。它构建在 CAP-X 执行和 API 生态系统之上。

核心研究贡献：

1. **合约驱动的多智能体协调** — 所有模块间通信通过 frozen dataclass 合约
2. **可审计的自进化系统** — Memory Skill 和 Robot Skill 分离进化
3. **图即策略 (GaP)** — 类型化 MissionGraph/SubgraphSpec 提供结构化规划
4. **分阶段图协议 (Staged Protocol)** — Manager 只输出拓扑，Policy Agent 输出局部可执行图

- **首个环境**: LIBERO-PRO
- **首个实体**: Franka 机器人 + 多个软件智能体
- **主要目标**: 随任务时序增长提升成功率稳定性
- **基线对比**: CAP-X 单智能体循环

---

## 2. 顶层目录结构

```
cap-mas/
├── capmas/                    # 核心代码包
│   ├── agents/                # 智能体实现
│   ├── backends/              # 后端适配层
│   ├── contracts/             # 合约定义（核心数据模型）
│   ├── evaluation/            # 评估层
│   ├── execution/             # 执行层
│   ├── graph/                 # 图策略层
│   ├── llm/                   # LLM 后端
│   ├── memory/                # 记忆系统
│   ├── perception/            # 感知层
│   ├── runtime/               # 运行时核心
│   ├── skills/                # 技能系统
│   └── verification/          # 验证层
├── configs/
│   └── default.yaml           # 默认配置
├── scripts/                   # 运行脚本
├── tests/                     # 测试套件（21个测试文件）
├── docs/                      # 设计文档
├── outputs/                   # 输出目录
└── pyproject.toml             # 项目配置
```

---

## 3. 核心代码包详细结构

### 3.1 智能体层 (`capmas/agents/`)

```
agents/
├── __init__.py
├── base.py            # 智能体协议导出
├── manager.py         # Mission Manager（确定性 + LLM + 拓扑）
├── policy.py          # Policy Agent（Callable + LLM图 + LLM分阶段）
├── arbiter.py         # Candidate Arbiter（验证+评分+选择）
├── recovery.py        # Recovery Agent（Callable适配器）
└── libero.py          # LIBERO 确定性策略 + 图构建器
```

| 文件 | 类/函数 | 职责 |
|------|---------|------|
| `base.py` | — | 导出所有智能体协议 |
| `manager.py` | `SimpleMissionManager` | 确定性子目标生成 |
| `manager.py` | `LLMMissionManager` | LLM 驱动完整 MissionGraph 生成 |
| `manager.py` | `LLMTopologyManager` | 阶段一：LLM 驱动紧凑拓扑生成（支持重试+修复） |
| `policy.py` | `CallablePolicyAgent` | 函数适配器 → ActionContract |
| `policy.py` | `CallableGraphPolicyAgent` | 函数适配器 → SubgraphSpec |
| `policy.py` | `LLMGraphPolicyAgent` | LLM 驱动局部子图（从完整图中提取） |
| `policy.py` | `LLMStagedGraphPolicyAgent` | 阶段二：LLM 驱动直接局部图（支持重试+修复） |
| `arbiter.py` | `CandidateArbiter` | 验证+多因子评分+排序+选择最优候选 |
| `recovery.py` | `CallableRecoveryAgent` | 函数适配器 → 恢复合约 |
| `libero.py` | `LiberoSpatialTask0Policy` | V1 确定性单步策略 |
| `libero.py` | `LiberoSpatialTask0MultiStepPolicy` | P2.5 确定性分阶段策略 |
| `libero.py` | `build_libero_spatial_task0_mission_graph()` | 确定性图构建器 |

### 3.2 合约层 (`capmas/contracts/`)

```
contracts/
├── __init__.py
├── core.py            # 基础引用类型
├── action.py          # 动作合约
├── agent.py           # 智能体上下文与协议
├── graph.py           # 图合约
├── staged.py          # 分阶段协议合约
├── verification.py    # 验证结果
├── scene.py           # 场景快照
├── memory.py          # 记忆上下文
├── trace.py           # 执行轨迹
├── candidates.py      # 候选与仲裁
└── failures.py        # 失败分类
```

| 文件 | 核心类型 | 作用 |
|------|----------|------|
| `core.py` | `ArtifactRef`, `SkillRef`, `EpisodeHandle` | 基础引用和句柄 |
| `action.py` | `ActionContract`, `SkillCall`, `ExecutionBudget`, `SkillOutputRef` | 有界动作提案 |
| `agent.py` | `AgentContext`, `AgentArtifact`, `PolicyDecision`, `CycleHistory` + 协议族 | 智能体输入输出 |
| `agent.py` | `MissionManager`, `MissionGraphManager`, `MissionTopologyManager` | Manager 协议 |
| `agent.py` | `PolicyAgent`, `GraphPolicyAgent`, `GroundedPolicyAgent`, `RecoveryAgent` | 策略协议 |
| `graph.py` | `MissionGraph`, `SubgraphSpec`, `SubgraphNodeSpec`, `LoopSpec` | 图结构合约 |
| `graph.py` | `PortSpec`, `PortBinding`, `SubgraphOutputBinding`, `CheckpointSpec` | 端口与绑定 |
| `graph.py` | `MissionEdge`, `MissionBinding`, `ResourceRequirement` | 任务级边与资源 |
| `staged.py` | `MissionTopology`, `TopologySubgoal` | 分阶段协议：紧凑拓扑 |
| `verification.py` | `VerificationResult`, `PredicateReport` | 验证决策与谓词报告 |
| `scene.py` | `SceneSnapshot`, `ObjectTrack`, `VisualEvidence`, `SpatialRelation` | 版本化世界状态 |
| `memory.py` | `MemoryContext`, `MemoryItem`, `MemorySkillRef`, `MemoryBudget` | 记忆上下文 |
| `trace.py` | `ExecutionTrace`, `SkillTrace`, `EpisodeTrace` | 执行轨迹 |
| `candidates.py` | `GraphCandidate`, `CandidateEvidence`, `ArbitrationResult` | 候选与仲裁 |
| `failures.py` | `FailureClass`, `FailureArtifact` | 失败分类与结构化失败 |

### 3.3 运行时层 (`capmas/runtime/`)

```
runtime/
├── __init__.py
├── orchestrator.py        # 单周期编排
├── episode_runner.py      # Episode 运行器
├── graph_interpreter.py   # 图解释器
├── llm_scheduler.py       # LLM 多智能体调度器
├── scheduler.py           # 调度器协议
├── recovery.py            # 恢复选择器
├── action_lease.py        # 动作租约管理
├── state_store.py         # 版本化状态存储
└── artifact_bus.py        # 事件总线 + Artifact 存储
```

| 文件 | 类/函数 | 职责 |
|------|---------|------|
| `orchestrator.py` | `RuntimeOrchestrator` | 单周期编排：验证→租约→技能执行→观察→提交/恢复 |
| `orchestrator.py` | `CycleResult` | 单周期结果 |
| `episode_runner.py` | `EpisodeRunner` | 单合约单周期运行器 |
| `episode_runner.py` | `MultiCycleEpisodeRunner` | 多周期循环：重规划+有界恢复+目标检查 |
| `episode_runner.py` | `EpisodeRunResult` | Episode 运行结果 |
| `graph_interpreter.py` | `FixedGraphInterpreter` | 图执行：按控制流遍历 MissionGraph |
| `graph_interpreter.py` | `GraphExecutionResult` | 图执行结果 |
| `llm_scheduler.py` | `LLMGraphScheduler` | 多智能体编译+执行：Manager→Policy→Arbiter→Interpreter |
| `llm_scheduler.py` | `LLMGraphCompileResult`, `LLMGraphRunResult` | 编译/运行结果 |
| `scheduler.py` | `Scheduler` 协议, `FixedGraphScheduler` | 合约分发协议 |
| `recovery.py` | `RecoverySelector` 协议, `MappingRecoverySelector` | 失败类→子图恢复路由 |
| `action_lease.py` | `ActionLease`, `ActionLeaseManager` | 互斥执行器租约 |
| `state_store.py` | `InMemoryStateStore` | 版本化场景存储（CAS提交） |
| `artifact_bus.py` | `EventBus`, `ArtifactStore`, `ArtifactEnvelope` | 智能体间通信 |

### 3.4 LLM 层 (`capmas/llm/`)

```
llm/
├── __init__.py
├── protocol.py            # LLM 协议定义
├── capx_compatible.py     # CAP-X/OpenAI 兼容客户端
├── graph_decoder.py       # MissionGraph 严格解码器
├── staged_decoder.py      # 分阶段拓扑/子图解码器
└── prompts.py             # Prompt 构建器 + JSON Schema 生成
```

| 文件 | 类/函数 | 职责 |
|------|---------|------|
| `protocol.py` | `LLMClient` 协议, `LLMRequest`, `LLMResponse` | LLM 调用抽象 |
| `capx_compatible.py` | `CAPXCompatibleLLMClient` | HTTP 客户端：重试+结构化输出回退+超时 |
| `capx_compatible.py` | `CAPXCompatibleConfig`, `LLMTransportError` | 配置与传输错误 |
| `graph_decoder.py` | `MissionGraphDecoder` | LLM响应→MissionGraph：JSON解析→Schema→场景→图验证 |
| `graph_decoder.py` | `GraphDecodeResult`, `GraphDecodeRejection` | 解码结果与拒绝原因 |
| `staged_decoder.py` | `MissionTopologyDecoder` | LLM响应→MissionTopology |
| `staged_decoder.py` | `LocalSubgraphDecoder` | LLM响应→SubgraphSpec |
| `prompts.py` | `mission_graph_response_schema()` | MissionGraph JSON Schema |
| `prompts.py` | `mission_topology_response_schema()` | 拓扑 JSON Schema |
| `prompts.py` | `subgraph_response_schema()` | 子图 JSON Schema |
| `prompts.py` | `build_manager_request()`, `build_policy_request()` | Prompt 构建 |
| `prompts.py` | `build_topology_request()`, `build_staged_policy_request()` | 分阶段 Prompt 构建 |

### 3.5 图策略层 (`capmas/graph/`)

```
graph/
├── __init__.py
├── validator.py           # 图结构验证器
├── serialization.py       # MissionGraph 版本化序列化
└── staged.py              # 拓扑验证+序列化+组装
```

| 文件 | 类/函数 | 职责 |
|------|---------|------|
| `validator.py` | `GraphValidator` | 静态图验证（唯一性/入口/边/环路/可达性/终端/绑定/并行资源） |
| `validator.py` | `GraphValidationResult`, `GraphDiagnostic` | 验证结果与诊断 |
| `serialization.py` | `mission_graph_to_dict()`, `mission_graph_from_dict()` | 严格版本化序列化/反序列化 |
| `staged.py` | `TopologyValidator` | 拓扑验证（子目标唯一性/依赖完整性/边-依赖一致性/无环） |
| `staged.py` | `topology_to_dict()`, `topology_from_dict()` | 拓扑序列化/反序列化 |

### 3.6 后端适配层 (`capmas/backends/`)

```
backends/
├── __init__.py
├── protocol.py            # RobotBackend 协议
├── capx.py                # CAP-X 后端适配器
└── capx_libero_factory.py # CAP-X YAML → CAP-MAS 运行时工厂
```

| 文件 | 类/函数 | 职责 |
|------|---------|------|
| `protocol.py` | `RobotBackend` 协议, `SkillExecutionResult` | 机器人后端抽象 |
| `capx.py` | `CAPXRobotBackend` | CAP-X API → CAP-MAS RobotBackend |
| `capx.py` | `CAPXObservationProvider` | CAP-X 原始观察 → ObservationBundle |
| `capx.py` | `CAPXTypedSkill` | CAP-X API 函数 → TypedSkill |
| `capx_libero_factory.py` | `build_capx_runtime_from_yaml()` | YAML → 环境+API+后端+技能注册表 |
| `capx_libero_factory.py` | `CAPXRuntimeBundle` | 构建的运行时资源包 |

### 3.7 其他模块

| 模块 | 文件 | 核心类型 | 职责 |
|------|------|----------|------|
| **verification/** | `predicates.py` | `PredicateBasedVerifier`, `PredicateRegistry` | 基于谓词的确定性验证器 |
| | `libero.py` | `LiberoObservableVerifier` | LIBERO 专用验证器 |
| | `freshness.py` | `is_fresh()` | 场景新鲜度检查 |
| **memory/** | `protocol.py` | `MemoryController`, `MemoryExecutor` | 记忆控制/执行协议 |
| | `controller.py` | `RuleBasedMemoryController` | 规则 Top-K 选择 |
| | `store.py` | `InMemoryMemoryStore` | 版本化记忆存储 |
| **perception/** | `protocol.py` | `ObservationBundle`, `PerceptionRequest`, `PerceptionResult` | 多模态观察协议 |
| | `artifacts.py` | `InMemoryArtifactStore` | CAS 原型 Artifact 存储 |
| | `fusion.py` | `PerceptionFacade` | 感知融合门面 |
| **skills/** | `protocol.py` | `TypedSkill` 协议 | 类型化技能协议 |
| | `registry.py` | `SkillRegistry` | 版本化技能注册/验证/查询 |
| **evaluation/** | `interfaces.py` | `Evaluator`, `TraceSink`, `MetricsSink` | 评估器协议 |
| | `reward.py` | `CAPXBinaryReward`, `VerifiedTransition`, `LearningReturn` | 二值奖励+学习回报 |
| | `parity.py` | `NormalizedEpisode`, `ParityComparison` | CAP-X/CAP-MAS 归一化对比 |
| **execution/** | `typed_executor.py` | `TypedExecutor`, `BackendTypedExecutor` | 类型化执行器 |

---

## 4. 运行脚本

| 脚本 | 阶段 | 协议 | Manager | Policy | 特点 |
|------|------|------|---------|--------|------|
| `run_libero_b0.py` | B0 (V1) | 单合约 | 确定性 | 确定性单步 | 基线 |
| `run_libero_b1.py` | B1 (P2.5) | 多周期 | 确定性 | 确定性分阶段 | 重规划+恢复 |
| `run_libero_b3.py` | B3 | 固定图 | 确定性图 | 图解释器 | 确定性图基线 |
| `run_libero_b3_llm.py` | B3-LLM (P3.1) | legacy/staged | LLM | LLM多候选 | 完整多智能体LLM管线 |
| `compare_artifacts.py` | — | — | — | — | CAP-X vs CAP-MAS 对比工具 |

---

## 5. 测试文件

| 测试文件 | 覆盖模块 |
|----------|----------|
| `test_capx_adapter.py` | CAP-X 后端适配 |
| `test_capx_libero_factory.py` | YAML 工厂 |
| `test_episode_runner.py` | Episode 运行器 |
| `test_graph_contracts.py` | 图合约 |
| `test_graph_runtime.py` | 图运行时 |
| `test_libero_multistep_policy.py` | LIBERO 多步策略 |
| `test_libero_verifier.py` | LIBERO 验证器 |
| `test_llm_backend.py` | LLM 后端 |
| `test_llm_graph_decoder.py` | LLM 图解码器 |
| `test_llm_prompts.py` | LLM Prompt |
| `test_llm_scheduler.py` | LLM 调度器 |
| `test_memory_store.py` | 记忆存储 |
| `test_multicycle_runner.py` | 多周期运行器 |
| `test_multimodal_agent_context.py` | 多模态智能体上下文 |
| `test_parity.py` | CAP-X/CAP-MAS 对比 |
| `test_postcondition_and_episode.py` | 后置条件与 Episode |
| `test_reward_boundary.py` | 奖励边界 |
| `test_runtime_cycle.py` | 运行时周期 |
| `test_runtime_failures.py` | 运行时失败 |
| `test_staged_protocol.py` | 分阶段协议 |
| `contract/` | 合约子测试 |

---

## 6. 新增模块关系总结

| 新增模块 | 依赖 | 被依赖 | 核心作用 |
|----------|------|--------|----------|
| `contracts/staged.py` | `contracts/graph.py` | `graph/staged.py`, `llm/staged_decoder.py`, `agents/manager.py` | 分阶段协议的合约定义 |
| `graph/staged.py` | `contracts/staged.py` | `llm/staged_decoder.py` | 拓扑验证+序列化+组装 |
| `llm/capx_compatible.py` | `llm/protocol.py` | `scripts/run_libero_b3_llm.py` | 实际 LLM HTTP 调用 |
| `llm/graph_decoder.py` | `graph/serialization.py`, `graph/validator.py` | `agents/manager.py`, `agents/policy.py` | LLM→MissionGraph 严格解码 |
| `llm/staged_decoder.py` | `graph/staged.py`, `graph/serialization.py` | `agents/manager.py`, `agents/policy.py` | LLM→Topology/Subgraph 严格解码 |
| `llm/prompts.py` | `contracts/scene.py` | `scripts/run_libero_b3_llm.py` | Prompt+Schema 构建 |
| `runtime/llm_scheduler.py` | `agents/`, `contracts/`, `graph/` | `scripts/run_libero_b3_llm.py` | 多智能体编译+执行协调 |
| `runtime/recovery.py` | `contracts/failures.py` | `runtime/graph_interpreter.py` | 失败类→子图恢复路由 |
| `evaluation/parity.py` | — | `scripts/compare_artifacts.py` | CAP-X vs CAP-MAS 对比 |
