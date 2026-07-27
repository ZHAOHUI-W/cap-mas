# CAP-MAS Graph-as-Policy 基础层

## 1. 设计动机

GaP 将机器人策略表示为由 typed skill nodes 和 data/control edges 组成的
有向计算图。其多 Agent 贡献不是简单增加 Agent 数量，而是把全局任务分解、
局部子图生成、静态结构验证、节点级 checkpoint 和并行 rehearsal 组合起来。

CAP-MAS 吸收这一方向，但保留 CAP-MAS 自己的在线闭环约束：版本化
`SceneSnapshot`、`ActionLease`、可观测后置条件和 bounded recovery。GaP 的
图是离线编译和反复 rehearsal 的策略载体；CAP-MAS 的图还必须能在在线执行中
被重新观察、局部失效和恢复。

参考论文：

- Chen et al., *GaP: A Graph-as-Policy Multi-Agent Self-Learning Harness for
  Variational Automation Tasks*, arXiv:2607.05369v1, 2026。
- 本地副本：`cap-x/docs/papers/GaP_graph_as_policy_2607.05369v1.pdf`。

## 2. 两层图模型

```text
MissionGraph
    └── SubgraphSpec（一个局部子目标）
          └── SubgraphNodeSpec（感知/规划/执行/检查节点）
                └── TypedSkill + ActionContract
```

### MissionGraph

`MissionGraph` 是 Manager 和多个局部 Policy Agent 共享的工作空间，描述：

- 任务和 mission 版本；
- subgraph 节点以及依赖边；
- 入口、成功出口和失败出口；
- 跨 subgraph 的 typed port binding；
- 生成时使用的 `parent_scene_version`。

它负责全局结构，不直接执行机器人动作。

### SubgraphSpec

`SubgraphSpec` 是一个 bounded local policy graph，属于一个子目标，包含：

- typed inputs/outputs；
- 局部节点和控制边；
- 节点间 port binding；
- success/failure 节点；
- 至少一个 `validate=True` checkpoint；
- 节点预算、恢复策略和资源声明。

一个 action 节点可以通过 `SubgraphSpec.to_action_contract()` 降低成已有的
`ActionContract`。这条兼容边界允许 P2.5 运行时先执行图节点，而不需要等待完整
的 Phase 3 Scheduler。

## 3. Agent 分工

```text
Manager
  → MissionGraph
  → Subgoal Policy Agents（可并行产生候选 SubgraphSpec）
  → GraphValidator
  → Arbiter
  → Verifier
  → Executor
```

Manager 负责全局图；Policy Agent 只负责一个局部 subgraph。候选 Agent 可以使用
不同的 skill allowlist、视觉证据、失败记忆和风险目标，但输出必须遵循相同的
typed schema。它们不直接执行机器人。

候选之间的通信使用 typed artifacts，而不是无界对话。当前实现由
`GraphCandidate`、write-once `ArtifactStore` 和 `EventBus` 承载；候选必须携带
`parent_scene_version`、`producer_agent`、`candidate_id`、资源需求和失败假设。

### 3.1 Typed dataflow 运行约定

图端口是运行时的数据依赖。当前解释器采用以下确定性路径：

```text
SkillTrace.output
  → action node output port
  → PortBinding
  → 后继 action 的第一个 SkillCall.args
  → SubgraphOutputBinding
  → MissionBinding
  → 下一个 subgraph 的输入
```

- 节点输入端口按端口名注入该节点第一个 `SkillCall` 的参数；
- literal skill 参数与图绑定值冲突时，运行时抛出 `GraphExecutionError`；
- 节点输出当前取该 action cycle 最后一个 `SkillTrace.output`；
- subgraph 的必需输入必须有唯一 `MissionBinding`，且 producer 从入口可达并能沿
  mission edge 到达 consumer；
- required output 必须通过 `SubgraphOutputBinding` 暴露；
- 当前版本不支持一个 action 内部多个 `SkillCall` 的中间输出路径（例如
  `call[0].pose`）。需要跨节点传递的值应先由最后一个 skill 输出；selector
  语义留给后续 schema 版本。

因此，`GraphValidator` 会在执行前报告
`UNBOUND_MISSION_INPUT`、`MULTIPLE_MISSION_INPUT_BINDINGS`、
`UNREACHABLE_BINDING_SOURCE` 和 `MISSION_BINDING_SOURCE_NOT_PREDECESSOR`，避免
错误图在机器人执行阶段才暴露。

## 4. GraphValidator 与 Verifier 的边界

### GraphValidator：静态检查

`capmas/graph/validator.py` 在动作执行前检查：

- subgraph/node ID 唯一性；
- 入口、出口和边的合法性；
- 节点和 subgraph 可达性；
- 循环是否由 `LoopSpec` 显式声明并具有有限预算；
- input/output port 是否存在且类型一致；
- required inputs/outputs 是否完成绑定；
- action 节点是否声明 TypedSkill 和可观测 postcondition；
- 每个 subgraph 是否至少有一个 validating checkpoint；
- 并行分支是否争用同一个 exclusive resource。

### Verifier：动态检查

`Verifier` 不负责图的静态结构，而是结合当前 `SceneSnapshot` 检查：

- 当前场景版本是否匹配；
- 前置条件和安全不变量是否满足；
- 执行后 observable predicates 是否满足；
- 是否需要 Recovery。

因此，结构正确不等于物理可执行，验证通过也不等于任务完成。

## 4.1 FailureArtifact 到 Recovery 的闭环

运行时失败不会直接改变图拓扑：

```text
dispatch failure
  → FailureArtifact
  → ArtifactStore（write-once）
  → EventBus(topic="failure")
  → RecoverySelector
  → 显式声明且条件匹配的 recovery edge
```

Recovery Selector 只能提出目标，不能取得 `ActionLease`。解释器会检查目标 subgraph
是否存在、是否是当前 subgraph 的 outgoing edge target，以及 edge condition 是否匹配
failure class、`failure` 或 `recovery_policy`。因此恢复是受图约束的控制流，而不是
Agent 任意跳转。

## 5. 并行策略

第一阶段允许并行的是只读推理和候选生成：

```text
Policy A ─┐
Policy B ─┼→ GraphValidator → Arbiter → Verifier → Executor
Policy C ─┘
```

单机器人动作仍由一个 `Executor` 和一个 `ActionLease` 独占。共享同一
exclusive resource 的并行分支会被 `GraphValidator` 拒绝；后续如果支持多臂
并行，必须显式声明互不冲突的资源集合。

GaP 式并行 rehearsal 属于离线/异步学习平面：在多个采样场景上并行运行图，按
节点 checkpoint 聚合失败，再生成图或参数更新候选。它不进入高频控制循环。

## 6. 候选证据仲裁

`CandidateArbiter` 在结构、scene version 和 evidence provenance 检查之后，使用
`CandidateEvidence` 进行确定性排序。证据模式不再把 scheduler 的 legacy
confidence 当作质量分：

The score is the sum of the dimensions explicitly declared by
`CandidateEvidence.available_metrics`:

```text
verifier    = profile.verifier_weight    × verifier_pass_rate
rehearsal   = profile.rehearsal_weight   × rehearsal_success_rate
ood         = profile.ood_weight         × ood_success_rate
perception  = profile.perception_weight  × mean(available perception fields)
latency     = -profile.latency_penalty   × min(latency / budget, 2.0)
recovery    = -profile.recovery_penalty  × min(recovery_cost, 2.0)
```

Unavailable dimensions are omitted rather than treated as zero. The current
LIBERO provider declares only `perception`, so P3.2 does not claim verifier,
rehearsal, or OOD quality. The legacy compatibility form with an empty
`available_metrics` sequence retains the historical full scalar fields for
older offline callers; new providers must declare their dimensions.

相同得分时使用执行时长、独占资源数量和 candidate id 做稳定 tie-break，并标记为
`evidence_tie_break`。完全没有证据时才使用 `confidence_fallback`。证据引用和
source scene version 会保留在候选上，便于解释选择结果和后续离线回放；该分数是
deterministic foundation 的默认策略，不代表最终学习到的 Arbiter。

### 6.1 P3.2 typed Policy 与感知证据

P3.1 的 strategy name 现在映射到 `StrategyProfile`。Profile 同时进入 Policy
请求的结构化 `strategy_profile` payload 和 Arbiter 的门控/评分，因此策略差异不
再只存在于 system prompt 文本中。

每个 `GraphCandidate` 保留 `raw_subgraph`、归一化后的 `subgraph`、策略名以及
`CandidateRewriteReport`。LIBERO 的 grounding/repair 仍然只允许 normalized graph
进入执行，但 raw/normalized fingerprint 会写入 arbitration artifact，用于区分
Policy 收敛和后处理抹平。

`PerceptionEvidence` 从当前 `SceneSnapshot` 计算 scene freshness、scene confidence、
目标可见性、track confidence、identity confidence 和 pose reliability。证据 provider
不调用机器人后端；未运行的 rehearsal/OOD 维度通过 `available_metrics` 显式保持
unknown，而不是当作零分。安全 Profile 在可用证据低于阈值时拒绝候选，其他可行候选
再按 Profile 权重由 Arbiter 选择。

## 7. 失败与演化

在线路径：

```text
ActionContract
  → execute
  → SceneSnapshot(v+1)
  → checkpoint verification
  ├── pass: commit and advance graph
  └── fail: classify failure and enter recovery subgraph
```

离线路径：

```text
ExecutionTrace
  → FailureAttribution
  → Graph/Memory Skill candidate
  → ProcessRehearsalPool
  → parallel rehearsal
  → regression + OOD validation
  → quarantine
  → promotion or rollback
```

结合 CAP-MAS 已确定的演化顺序，先冻结 Robot Skill Registry，只更新和验证
Memory Skill；Memory Skill 稳定后再进入 Robot Skill 演化。

仿真 privileged state 可以用于离线诊断，但在线 Verifier 只能使用
`SceneSnapshot` 和传感器可观测证据。

### 7.1 State-flow precondition validation

`GraphValidator` now performs a conservative must-analysis on a complete
`MissionGraph` before execution. For every action node, each precondition must
come from one of two sources:

1. a predicate that is true in the current initial `SceneSnapshot`; or
2. a postcondition of every mandatory normal-path predecessor.

At a join, predecessor facts are intersected, so a fact produced by only one
conditional branch is not treated as guaranteed. Failure/recovery edges do not
produce facts. This catches an invalid placement node that requires
`object_in_gripper(bowl)` without a successful grasp predecessor before the
robot runtime is called. Local Policy candidates are still validated in
isolation for structure; cross-subgraph state-flow is checked after Manager
assembly, where all predecessors are visible.

State-flow provenance is separate from predicate truth. The graph validator
proves where a fact can be established; `PredicateBasedVerifier` still decides
whether the fact is currently true at dispatch time.

### 7.2 Structured precondition rejection

An expected verifier rejection is not an exception path. `RuntimeOrchestrator`
returns a `CycleResult` with `rejected=true`, the failed predicate reports, an
`ExecutionTrace(status="rejected")`, and an unchanged scene version. No lease
is acquired and no skill is executed. `FixedGraphInterpreter` preserves the
original failure class, normalizes failure-like transition labels, writes a
`FailureArtifact` with predicate diagnostics, and lets the declared recovery
edge/selector handle the next transition.

`object_in_gripper(obj)` remains a held-object predicate: it requires both a
closed gripper and the object/end-effector geometry to be within threshold. A
geometry-only predicate should be named separately (for example,
`object_near_gripper(obj)`); weakening the held-object predicate would make
false-positive placement plans appear valid.

## 8. 当前实现状态

已实现：

- `capmas/contracts/graph.py`：MissionGraph、SubgraphSpec、节点、边、端口、
  checkpoint 和资源合约；
- `capmas/graph/validator.py`：静态结构、类型、checkpoint、可达性和资源冲突
  验证，以及基于初始 SceneSnapshot 和 mandatory predecessor postconditions
  的 state-flow precondition 验证；
- `SubgraphSpec.to_action_contract()`：向 P2.5 RuntimeOrchestrator 的兼容降低；
- 图结构和验证器的公开接口测试。

- strict graph serialization (`schema_version=1`)；
- `LoopSpec` 与 bounded-cycle validation；
- `FixedGraphInterpreter`，通过 `Scheduler.dispatch()` 进入现有 runtime；
- `CandidateArbiter`、write-once `ArtifactStore` 和 typed `EventBus`；
- `FailureArtifact` 统一 recovery/memory 失败输入；失败写入 ArtifactStore、发布
  `failure` 事件，并只能沿显式 recovery edge 跳转；前置条件拒绝也会保留
  predicate reports 和失败原因；
- typed local/mission dataflow 解析及 MissionBinding 完整性检查；
- 基于 verifier、rehearsal、OOD、latency 和 recovery cost 的 evidence-aware Arbiter。

已补充实现：

- `CAPXCompatibleLLMClient`：CAP-X/OpenAI-compatible HTTP 调用、structured
  response、usage/latency 记录和 bounded deadline/retry；
- `capmas.llm.prompts`：Manager/Policy 的 typed prompt builder 和严格图 schema；
- `LLMGraphScheduler`：Manager 图生成、只读 Policy fan-out、CandidateArbiter
  和单一 FixedGraphInterpreter 提交；
- `scripts/run_libero_b3_llm.py`：endpoint-backed LIBERO P3.1 runner。

尚未实现：

- LLM Recovery/Monitor 以及 deadline-aware recovery fallback；
- 多进程机器人/仿真并行执行；
- 并行仿真 rehearsal 执行器；
- streaming SceneSnapshot 和自适应 topology。

`scripts/run_libero_b3.py` 已完成 deterministic LIBERO smoke：它复用 CAP-X
YAML/API factory、同一 observable verifier 和单一 physical scheduler，并输出图、
trace 和 failure artifact。当前 smoke 已验证 5 个固定子目标和 scene version
`0→1→2→3→4→5` 的成功执行。

所以当前代码已经完成 Phase 3 的 deterministic foundation、B3 smoke、typed
LLM proposal path 和只读候选 fan-out，但真实 endpoint-backed B3-LLM 对比仍需
外部模型服务和匹配实验预算，不能用 deterministic B3 结果替代。
