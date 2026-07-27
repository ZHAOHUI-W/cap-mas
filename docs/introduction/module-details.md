# CAP-MAS 模块功能详解

> 生成日期：2026-07-22

## 1. 设计原则

| 原则 | 实现方式 |
|------|----------|
| **合约驱动** | 所有模块间通信通过 frozen dataclass，不可变、可审计 |
| **三平面分离** | 控制平面(20-100Hz) / 世界模型平面(5-30Hz) / 智能体平面(事件触发) |
| **版本化状态** | SceneSnapshot 单调递增版本号，CAS 提交保证一致性 |
| **互斥租约** | ActionLeaseManager 确保同一时刻只有一个合约控制机器人 |
| **严格类型边界** | LLM 输出必须通过 Decoder（JSON→Schema→场景→验证）才能进入运行时 |
| **分阶段协议** | Manager 只输出拓扑（无技能细节），Policy Agent 输出局部可执行图 |
| **候选仲裁** | 多个 Policy Agent 并行提案，Arbiter 多因子评分选择最优 |
| **可替换可消融** | 每个模块都有 CAP-X 替代方案和消融配置标志 |
| **修复重试** | LLM 提案失败时支持 repair feedback + 重试 |

---

## 2. 合约层 (`contracts/`)

合约层是整个系统的"语言"，所有模块通过这些 frozen dataclass 通信。

### 2.1 ActionContract — 有界动作提案

```python
@dataclass(frozen=True)
class ActionContract:
    contract_id: str                    # 唯一标识
    episode_id: str                     # 所属 episode
    episode_epoch: int                  # episode 轮次
    parent_scene_version: int           # 基于的场景版本
    subgoal_id: str                     # 所属子目标
    skills: tuple[SkillCall, ...]       # 技能调用序列
    expected_postconditions: tuple[str, ...]  # 期望后置条件
    max_duration_ms: int                # 最大执行时间
    max_sim_steps: int                  # 最大仿真步数
    proposed_by: str                    # 提案者
    preconditions: tuple[str, ...]      # 前置条件
    safety_invariants: tuple[str, ...]  # 安全不变量
    recovery_policy: str                # 恢复策略 (默认 "replan")
```

**关键约束**：
- 技能调用序列中的 `SkillOutputRef` 引用前序技能输出，形成 DAG
- `parent_scene_version` 必须匹配当前场景版本，否则拒绝
- 执行预算 (`max_duration_ms`, `max_sim_steps`) 必须为正

### 2.2 SceneSnapshot — 版本化世界状态

```python
@dataclass(frozen=True)
class SceneSnapshot:
    episode_id: str
    episode_epoch: int
    scene_version: int                  # 单调递增版本号
    sensor_timestamp_ns: int
    publish_timestamp_ns: int
    robot: Mapping[str, object]         # 机器人状态
    objects: Sequence[ObjectTrack]       # 物体轨迹
    local_map: ArtifactRef | None       # 增量3D地图
    freshness_ms: float                 # 新鲜度
    visual_evidence: tuple[VisualEvidence, ...]  # 视觉证据
    spatial_relations: tuple[SpatialRelation, ...]  # 空间关系
    uncertainty: SceneUncertainty        # 不确定性
```

**关键约束**：
- `scene_version` 单调递增，CAS 提交保证一致性
- `freshness_ms` 用于控制平面判断是否使用安全响应
- `uncertainty` 包含模糊轨迹和过期轨迹，供策略智能体决策

### 2.3 MissionGraph — 全局任务图

```python
@dataclass(frozen=True)
class MissionGraph:
    mission_id: str
    task: str
    subgraphs: tuple[SubgraphSpec, ...]  # 子图集合
    edges: tuple[MissionEdge, ...]       # 任务级边
    bindings: tuple[MissionBinding, ...] # 子图间数据绑定
    entry_subgraph: str                  # 入口子图
    success_subgraphs: tuple[str, ...]   # 成功终端
    failure_subgraphs: tuple[str, ...]   # 失败终端
    parent_scene_version: int | None     # 基于场景版本
    graph_version: int                   # 图版本
    loops: tuple[LoopSpec, ...]          # 有界环路
```

**关键约束**：
- 所有环路必须声明 `max_visits`，解释器强制执行
- `SubgraphSpec.to_action_contract()` 将 action 节点降级为运行时合约
- 通过 `mission_graph_to_dict()` / `mission_graph_from_dict()` 严格序列化

### 2.4 MissionTopology — 紧凑拓扑（分阶段协议）

```python
@dataclass(frozen=True)
class MissionTopology:
    mission_id: str
    task: str
    subgoals: tuple[TopologySubgoal, ...]  # 子目标（无技能细节）
    edges: tuple[MissionEdge, ...]
    entry_subgraph: str
    success_subgraphs: tuple[str, ...]
    failure_subgraphs: tuple[str, ...]
    bindings: tuple[MissionBinding, ...]
    parent_scene_version: int | None
    graph_version: int
    loops: tuple[LoopSpec, ...]
```

**关键方法**：
- `assemble(subgraphs)` — 将拓扑 + 局部子图组装为完整 MissionGraph
  - 验证子图集合与拓扑子目标一一匹配
  - 推断缺失的数据绑定（精确名称匹配或唯一类型匹配）
  - 归一化语义边标签为 success/failure

### 2.5 CandidateEvidence — 候选证据

```python
@dataclass(frozen=True)
class CandidateEvidence:
    verifier_pass_rate: float       # 验证通过率 [0,1]
    rehearsal_success_rate: float   # 排练成功率 [0,1]
    ood_success_rate: float         # OOD成功率 [0,1]
    expected_latency_ms: float      # 预期延迟 (ms)
    recovery_cost: float            # 恢复成本
    evidence_refs: Sequence[str]    # 证据引用
```

---

## 3. 智能体层 (`agents/`)

### 3.1 Manager 角色

| 实现 | 协议 | 输入 | 输出 | LLM | 重试 |
|------|------|------|------|-----|------|
| `SimpleMissionManager` | `MissionManager` | task + scene | 子目标 Artifact | ✗ | ✗ |
| `LLMMissionManager` | `MissionGraphManager` | task + scene | MissionGraph | ✓ | ✗ |
| `LLMTopologyManager` | `MissionTopologyManager` | task + scene | MissionTopology | ✓ | ✓ (repair) |

**LLMTopologyManager 重试机制**：
1. 首次请求 → 解码
2. 若失败且有 `repair_request_builder`，将错误信息作为 feedback 构建修复请求
3. 重新调用 LLM → 解码
4. 最多 `proposal_retries` 次重试

### 3.2 Policy Agent 角色

| 实现 | 协议 | 输入 | 输出 | LLM | 重试 |
|------|------|------|------|-----|------|
| `CallablePolicyAgent` | `PolicyAgent` | subgoal + scene + context | ActionContract | ✗ | ✗ |
| `CallableGraphPolicyAgent` | `GraphPolicyAgent` | subgoal + scene + context | SubgraphSpec | ✗ | ✗ |
| `LLMGraphPolicyAgent` | `GraphPolicyAgent` | subgoal + scene + context | SubgraphSpec | ✓ | ✗ |
| `LLMStagedGraphPolicyAgent` | `GraphPolicyAgent` | subgoal + scene + context | SubgraphSpec | ✓ | ✓ (repair) |

**LLMGraphPolicyAgent 提取逻辑**：
- LLM 返回完整 MissionGraph
- 从中提取与 subgoal artifact 中 `subgraph_id` 或 `subgoal_id` 匹配的唯一子图
- 若匹配数量 ≠ 1，抛出 `GraphProposalError`

**LLMStagedGraphPolicyAgent 提取逻辑**：
- LLM 直接返回局部子图信封
- `LocalSubgraphDecoder` 解码并验证
- 检查 `subgraph_id` 和 `subgoal_id` 与拓扑子目标匹配

### 3.3 CandidateArbiter — 候选仲裁

**仲裁流程**：
1. **去重检查** — 拒绝重复 candidate_id
2. **场景版本检查** — 拒绝过期场景的候选
3. **图结构验证** — 通过 GraphValidator 验证子图
4. **子目标一致性** — 所有候选必须面向同一子目标
5. **多因子评分** — 选择最高分候选

**评分语义**：

Arbiter 只累加 `CandidateEvidence.available_metrics` 中已声明的维度：

```
verifier    = profile.verifier_weight   × verifier_pass_rate
rehearsal   = profile.rehearsal_weight  × rehearsal_success_rate
ood         = profile.ood_weight        × ood_success_rate
perception  = profile.perception_weight × mean(available perception fields)
latency     = -profile.latency_penalty  × min(latency / budget, 2.0)
recovery    = -profile.recovery_penalty × min(recovery_cost, 2.0)
```

未声明的维度不会被隐式记为零。当前 LIBERO P3.2 provider 只声明
`perception`；没有 evidence 时才使用显式 legacy confidence，并将结果标记为
`confidence_fallback`。`available_metrics` 为空仅用于兼容旧的离线
`CandidateEvidence` 调用，不是新的在线 provider 推荐格式。

Scheduler confidence is an optional legacy fallback. It is excluded when
candidate evidence is available; evidence-free selection is labeled
`confidence_fallback`.

---

## 4. 运行时层 (`runtime/`)

### 4.1 RuntimeOrchestrator — 单周期编排

**执行阶段**：

| 阶段 | 操作 | 失败处理 |
|------|------|----------|
| 1. 场景版本检查 | `contract.parent_scene_version == current.scene_version` | 抛出 ValueError |
| 2. 技能验证 | `SkillRegistry.validate_contract()` | 抛出 ValueError |
| 3. 前置验证 | `Verifier.approve()` | 抛出 ValueError |
| 4. 获取租约 | `ActionLeaseManager.acquire()` | 抛出 RuntimeError |
| 5. 技能执行 | 逐个执行技能调用，解析 SkillOutputRef | 技能失败 → 返回 CycleResult(committed=False) |
| 6. 观察新场景 | `Backend.observe()` | — |
| 7. 提交场景 | `StateStore.compare_and_commit()` | 场景变更 → 抛出 ValueError |
| 8. 后置验证 | `Verifier.commit()` | 失败 → 返回 CycleResult(committed=False) |
| 9. 释放租约 | `ActionLeaseManager.release()` | — (finally 块) |

### 4.2 MultiCycleEpisodeRunner — 多周期循环

**循环逻辑**：
1. 获取最新场景 → 构建 AgentContext（含 CycleHistory）
2. 调用 policy_step 获取 ActionContract
3. 运行 run_cycle
4. 若 committed：检查目标是否达成
5. 若未 committed：尝试恢复（最多 max_recoveries 次）
6. 恢复时构建新的 AgentContext（含恢复计数）

**停止原因**：
- `task_goal_reached` — 目标谓词满足
- `evaluator_success` — 评估器判定成功
- `policy_finished` — 策略返回 None
- `cycle_failed` — 周期失败且无恢复
- `recovery_exhausted` — 恢复次数用尽
- `recovery_declined` — 恢复策略返回 None
- `max_cycles` — 达到最大周期数

### 4.3 FixedGraphInterpreter — 图解释器

**执行逻辑**：
1. 验证 MissionGraph
2. 从 entry_subgraph 开始
3. 对每个子图：
   a. 从 entry_node 开始遍历
   b. action 节点 → 降级为 ActionContract → Scheduler.dispatch
   c. router 节点 → control_evaluator 决定下一节点
   d. checkpoint 节点 → 验证检查点谓词
4. 环路检查：每个子图访问次数不超过 LoopSpec.max_visits
5. 到达 success_node → 子图成功
6. 到达 failure_node 或超限 → 子图失败

### 4.4 LLMGraphScheduler — 多智能体调度器

**两种协议**：

| 协议 | Manager 输出 | Policy 输入 | 组装方式 |
|------|-------------|-------------|----------|
| `legacy` | MissionGraph | 子目标 artifact (含完整图信息) | 替换子图 |
| `staged` | MissionTopology | 拓扑子目标 (无技能细节) | assemble() |

**编译流程 (legacy)**：
1. Manager 产出 MissionGraph
2. 对每个子图，查找注册的 Policy Agent
3. 并行调用 Policy Agent 提案（线程池）
4. Arbiter 仲裁选择最优候选
5. 替换原图子图为选中候选
6. 重新验证编译后图

**编译流程 (staged)**：
1. Manager 产出 MissionTopology
2. 对每个拓扑子目标，查找注册的 Policy Agent
3. 并行调用 Policy Agent 提案
4. Arbiter 仲裁选择最优候选
5. 调用 `topology.assemble(selected_subgraphs)` 组装完整图
6. 验证编译后图

### 4.5 ActionLeaseManager — 互斥租约

**保证**：
- 同一时刻只有一个 ActionLease 活跃
- 租约有过期时间（基于 `duration_ms`）
- 过期租约自动失效
- 释放时验证 lease_id 匹配

### 4.6 InMemoryStateStore — 版本化状态存储

**保证**：
- 场景版本单调递增
- 同一版本只能发布一次
- `compare_and_commit()` 实现 CAS：只有最新版本匹配时才能提交
- 提交必须版本号 +1

---

## 5. LLM 层 (`llm/`)

### 5.1 CAPXCompatibleLLMClient — HTTP 客户端

**特性**：
- 仅依赖标准库（urllib），可选使用 requests
- 支持结构化输出（response_format: json_schema）
- 结构化输出不兼容时自动回退到纯 JSON prompting
- 可配置重试（指数退避），仅对可重试状态码重试（408, 429, 500, 502, 503, 504）
- 请求超时保护（deadline_ms）
- 可注入 transport 用于测试

**请求构建**：
- GPT-5/o 系列使用 `max_completion_tokens`
- 其他模型使用 `max_tokens`
- 结构化输出时附加 `response_format: { type: "json_schema", json_schema: {...} }`

### 5.2 MissionGraphDecoder — 严格解码器

**解码管线**：

| 阶段 | 检查 | 拒绝码 |
|------|------|--------|
| 1. 请求ID匹配 | `response.request_id == request.request_id` | `REQUEST_ID_MISMATCH` |
| 2. 载荷提取 | structured 或 content → JSON 对象 | `EMPTY_RESPONSE` / `JSON_INVALID` / `JSON_NOT_OBJECT` |
| 3. Schema 解析 | `mission_graph_from_dict(raw)` | `GRAPH_SCHEMA_INVALID` |
| 4. Mission ID 匹配 | `graph.mission_id == expected` | `MISSION_ID_MISMATCH` |
| 5. 场景版本检查 | `graph.parent_scene_version == scene.scene_version` | `MISSING_PARENT_SCENE` / `STALE_SCENE` |
| 6. 图结构验证 | `GraphValidator.validate(graph)` | 各种图验证错误码 |

**关键设计**：无空计划或默认动作回退。解码失败即拒绝。

### 5.3 MissionTopologyDecoder / LocalSubgraphDecoder

与 MissionGraphDecoder 类似的严格解码管线，但针对分阶段协议：
- `MissionTopologyDecoder` → `MissionTopology`
- `LocalSubgraphDecoder` → `SubgraphSpec`（含子图ID/子目标ID匹配检查）

### 5.4 Prompts — Prompt 构建器

**Schema 生成**：
- `mission_graph_response_schema()` — 完整 MissionGraph 的 JSON Schema
- `mission_topology_response_schema()` — 紧凑拓扑的 JSON Schema
- `subgraph_response_schema()` — 单个局部子图的 JSON Schema

**Schema 特点**：
- `additionalProperties: False` — 严格封闭世界
- 技能参数使用 nullable 类型（允许 null 表示未使用参数）
- 所有字段显式声明 required

**Prompt 构建**：
- `build_manager_request()` — Manager 完整图请求
- `build_topology_request()` — Manager 拓扑请求
- `build_policy_request()` — Policy 局部图请求（legacy）
- `build_staged_policy_request()` — Policy 局部图请求（staged）
- 支持 `repair_feedback` 参数用于修复重试

---

## 6. 图策略层 (`graph/`)

### 6.1 GraphValidator — 图结构验证

**验证项**：

| 检查 | 错误码 | 说明 |
|------|--------|------|
| 子图ID唯一性 | `DUPLICATE_SUBGRAPH` | 子图ID不能重复 |
| 非空任务 | `EMPTY_MISSION` | 必须有至少一个子图 |
| 入口子图存在 | `UNKNOWN_ENTRY` | entry_subgraph 必须在子图集合中 |
| 边引用有效 | `DANGLING_MISSION_EDGE` | 边的 source/target 必须存在 |
| 无非法环路 | `MISSION_CYCLE` | 环路必须通过 LoopSpec 声明 |
| 终端子图存在 | `UNKNOWN_TERMINAL` | success/failure 子图必须存在 |
| 可达性 | `UNREACHABLE_SUBGRAPH` | 所有子图从入口可达 |
| 绑定端口匹配 | `BINDING_PORT_MISMATCH` | 绑定的端口类型必须匹配 |
| 并行资源冲突 | `PARALLEL_EXCLUSIVE_RESOURCE` | 并行分支不能共享独占资源 |

### 6.2 TopologyValidator — 拓扑验证

**额外验证项**：

| 检查 | 错误码 | 说明 |
|------|--------|------|
| 子目标ID唯一 | `DUPLICATE_SUBGOAL` | 子目标ID不能重复 |
| 依赖存在 | `UNKNOWN_DEPENDENCY` | 依赖的子图必须存在 |
| 依赖有对应边 | `DEPENDENCY_EDGE_MISSING` | 依赖必须有对应的 success 边 |
| 无环 | 通过 `_cycle_diagnostics()` | 依赖图不能有环 |

### 6.3 Serialization — 严格序列化

**设计原则**：
- 线格式与运行时合约分离
- LLM 产出的 JSON 必须通过此解析器
- 每个对象有显式字段白名单
- 拼写错误的字段不会静默消失

---

## 7. 验证层 (`verification/`)

### 7.1 PredicateBasedVerifier — 基于谓词的验证器

**验证阶段**：

| 阶段 | 方法 | 检查内容 |
|------|------|----------|
| 前置验证 | `approve()` | 前置条件 + 安全不变量 |
| 后置验证 | `commit()` | 期望后置条件 |

**内置谓词**：
- `gripper_open` — 夹爪开度 > 阈值
- `gripper_closed` — 夹爪开度 < 阈值
- `object_at_gripper(object)` — 物体在夹爪附近
- `object_at_target(object, target)` — 物体在目标位置附近

**决策逻辑**：
- `approve`: 前置条件 + 安全不变量全部通过
- `reject`: 安全不变量违反 → `COLLISION_RISK`；前置条件失败 → `PRECONDITION_FAILED`
- `commit`: 后置条件全部通过
- `recover`: 后置条件失败 → `POSTCONDITION_FAILED`

---

## 8. 后端适配层 (`backends/`)

### 8.1 RobotBackend 协议

```python
class RobotBackend(Protocol):
    def reset(seed, options) -> EpisodeStart     # 重置环境
    def observe() -> SceneSnapshot                # 观察当前状态
    def execute_skill(skill, args, budget) -> SkillExecutionResult  # 执行技能
    def stop(lease) -> None                       # 停止执行
    def evaluator_success() -> bool               # 评估器判定
```

### 8.2 CAPXRobotBackend — CAP-X 适配

**适配逻辑**：
- `reset()` → 调用 CAP-X environment reset，构建 EpisodeStart
- `observe()` → 调用 CAP-X observation function，构建 SceneSnapshot
- `execute_skill()` → 调用 CAP-X API 函数，返回 SkillExecutionResult
- 技能输出通过 `SkillOutputRef` 在合约内引用

### 8.3 build_capx_runtime_from_yaml() — 工厂函数

**构建流程**：
1. 加载 CAP-X YAML 配置
2. 实例化低层环境（不实例化代码执行环境）
3. 通过 CAP-X API 注册表获取 API 工厂
4. 构建 CAPXTypedSkill 集合
5. 注册到 SkillRegistry
6. 构建 CAPXObservationProvider
7. 构建 CAPXRobotBackend
8. 返回 CAPXRuntimeBundle

---

## 9. 评估层 (`evaluation/`)

### 9.1 CAPXBinaryReward — 二值奖励

- `benchmark()`: 评估器成功 → 1.0，否则 → 0.0
- `learning_return()`: 基于 progress_delta - cost，安全违反 → -1.0

### 9.2 ParityComparison — CAP-X vs CAP-MAS 对比

**归一化字段**：
- `system` (capx / capmas)
- `task_id`, `seed`
- `success`, `reward`
- `action_count`
- `failure_reason`

**对比指标**：
- `success_delta` = CAP-MAS 成功 - CAP-X 成功（1/0/-1）

---

## 10. 模块消融配置

每个模块都有 CAP-X 替代方案和消融标志：

| 模块 | CAP-MAS 实现 | CAP-X 替代 | 消融标志 |
|------|-------------|-----------|----------|
| 调度器 | 事件驱动 Manager + 稀疏图 | CAP-X 单智能体循环 | `scheduler=capx_single` |
| 图策略层 | 类型化 MissionGraph + GraphValidator | CAP-X 自由策略代码 | `graph=disabled` |
| 状态存储 | 版本化黑板 + 快照存储 | Prompt/历史状态 | `state_store=unversioned` |
| 合约验证器 | 类型化 Schema + 限制 + 前置条件 | Markdown 代码提取 | `validator=syntax_only` |
| 策略生成器 | 代码/技能图提案 | CAP-X 代码生成 | `policy=capx_generator` |
| LLM 后端 | CAP-X 兼容/本地/Mock/回放 | CAP-X LLM 客户端 | `llm_backend=capx_compatible/local/mock` |
| 验证器 | 前置/不变/后置条件检查 | 执行后奖励/任务信号 | `verifier=none` |
| 执行器仲裁 | 单动作租约 | 隐式顺序执行 | `lease=disabled` |
| 执行器 | 有界类型化技能执行 | CAP-X exec 路径 | `executor=capx_exec` |
| 场景估计器 | 快速几何增量地图 | CAP-X 观察回调 | `scene_estimator=capx_observation` |
| 语义感知 | 事件触发感知智能体 | CAP-X VLM 视觉差异 | `semantic_perception=capx_vdm` |
| 恢复 | 失败分类 + 补偿动作 | 重新生成未来代码 | `recovery=regenerate` |
| 技能注册表 | 版本化 active/quarantine 注册表 | CAP-X 进化技能库 | `skill_registry=capx_library` |
| 拓扑控制器 | 事件触发稀疏图编辑 | 固定单循环/固定 MAS 图 | `topology=fixed` |
| 记忆 | 版本化 episode/experience/semantic/procedural | CAP-X prompt/history/library | `memory=none/episodic` |
| 记忆控制器 | Top-K Memory Skill 选择 | CAP-X 固定 prompt/history 检索 | `memory_controller=rules` |
| 奖励引擎 | CAP-X 二值 + 验证塑造 | CAP-X 评估器奖励 | `reward=binary_only` |
