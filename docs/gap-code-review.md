# GaP 本地实现复核与 CAP-MAS 落地边界

## 1. 复核范围

本次复核直接检查了本地 GaP 实现，而不是只依据论文描述：

- `gap/agent/multi_agent.py`：Coordinator、Subgraph Agent、Checkpoint Agent、验证修复与全图再生成；
- `gap/runtime/workflow.py`：严格 workflow schema、`$ref`、条件边和 `on_error`；
- `gap/runtime/executor.py`：frontier/super-step、回边、streaming node 和 Send fan-out；
- `gap/skills/`：技能元数据、canonical script、能力检查和 registry precedence；
- `gap/agent/parallel.py`：CUDA/MuJoCo 安全的多进程 rehearsal、超时、重试和 worker respawn；
- `tests/`：图 schema、checkpoint、loop、streaming、Send 和 worker 生命周期测试。

GaP 的工程核心不是 Agent 数量，而是：

```text
全局拓扑生成
  → 局部子图生成
  → 独立 checkpoint 编写
  → 严格结构验证
  → 局部修复 / 全图再生成
  → 可重放执行与 rehearsal
```

## 2. CAP-MAS 直接吸收的机制

### 2.1 分阶段图生成

CAP-MAS 保持以下职责边界：

- Manager 只产生 `MissionGraph` 拓扑；
- Local Policy Agent 只产生一个 `SubgraphSpec`；
- Verifier/Checkpoint 阶段独立产生 observable postconditions；
- `GraphValidator` 在动作提交前拒绝非法图；
- Candidate Arbiter 只选择一个候选，不执行机器人动作；
- 失败反馈只回传给负责的候选 Agent，拓扑级错误才触发 Manager 重生成。

这使 Agent 间通信成为有版本的 typed artifact，而不是不可审计的共享对话。

### 2.2 严格、有版本的图 schema

CAP-MAS 使用 `schema_version=1` 的显式 codec：

```python
from capmas.graph import mission_graph_from_dict, mission_graph_to_dict

payload = mission_graph_to_dict(graph)
restored = mission_graph_from_dict(payload)
```

每一级对象都拒绝未知字段。`MissionGraph.to_dict()` 和
`MissionGraph.from_dict()` 只是这个 codec 的便捷入口。LLM 返回的 JSON 必须先通过
codec，再通过 `GraphValidator`，不能直接交给 runtime。

### 2.3 有界循环与失败出口

默认图仍然是 DAG。需要回边时必须声明：

```python
LoopSpec(
    entry_node="reacquire",
    max_visits=3,
    max_duration_ms=20_000,
    exit_conditions=("success", "failure"),
)
```

没有 `LoopSpec` 的循环会被标记为 `UNBOUNDED_CYCLE`。运行时超出预算会生成
`FailureArtifact`，而不是静默跳转到成功节点。

### 2.4 固定图解释器

`FixedGraphInterpreter` 是 Phase 3 的第一版 scheduler seam：

```text
MissionGraph
  → Subgraph transition
  → Subgraph node transition
  → SubgraphSpec.to_action_contract()
  → Scheduler.dispatch()
  → RuntimeOrchestrator / ActionLease / Verifier
```

解释器不直接调用 `TypedSkill`。每个动作都经过现有
`RuntimeOrchestrator`，因此 scene version、lease、前置条件和后置条件仍由原有
runtime 负责。

当前解释器支持：

- 顺序 action node；
- 基于 outcome 的条件边；
- bounded loop；
- checkpoint node；
- 失败时生成带 scene/node/subgraph 归属的 `FailureArtifact`。

router node 需要注入 `control_evaluator`，避免把任意代码执行权限放进在线 runtime。

### 2.5 候选仲裁和 typed communication

`GraphCandidate` 必须携带：

- `candidate_id`；
- `parent_scene_version`；
- `producer_agent`；
- `confidence`；
- `SubgraphSpec`。

`CandidateArbiter` 会拒绝 stale scene、重复 candidate、错误 subgoal 和结构非法图，
然后按 confidence、总预算、资源数量和 candidate id 做确定性排序。Arbiter 没有
robot backend，也不能取得 `ActionLease`。

`ArtifactStore` 使用 write-once 语义；`EventBus` 只做同步 typed fan-out，订阅者异常
彼此隔离。后续可以把二者替换为持久化/跨进程实现而不改 Agent 接口。

## 3. 只改造、不直接复制的机制

### 3.1 GaP 的 super-step 并行

GaP 会并发运行同一 frontier 的 ready nodes。CAP-MAS 只允许以下对象并行：

- 多视角感知读取；
- memory retrieval；
- candidate generation；
- offline simulation rehearsal。

单机器人 physical action、lease acquisition、state commit 和 postcondition commit
必须串行。GaP 的 `ThreadPoolExecutor` 不能替代 CAP-MAS 的资源互斥和 lease authority。

### 3.2 GaP 的 streaming node

GaP 的 detached stream、latest snapshot 和 cooperative cancellation 是实时 3D scene
map 的合适参考，但不能把 VLM/LLM 放进 servo loop。CAP-MAS 的目标仍是：

```text
RGB-D / proprioception
  → fast geometric update
  → versioned SceneSnapshot
  → asynchronous semantic trigger
  → Agent Plane consumes latest snapshot
```

### 3.3 GaP 的 script 和 privileged World

GaP 允许 workflow script 调用工具，并在 simulator 中使用 privileged `World` 做
checkpoint。CAP-MAS 在线路径不复制这两个边界：

- Agent 只能看到 typed observable artifacts；
- 机器人动作只能通过 TypedSkill/ActionContract；
- privileged evaluator 只用于 offline evaluation、rehearsal 和 failure diagnosis。

## 4. 当前代码状态

已落地：

- `LoopSpec` 与 explicit bounded-cycle validation；
- strict graph serialization (`schema_version=1`)；
- `FixedGraphInterpreter`；
- `GraphCandidate` 与 deterministic `CandidateArbiter`；
- write-once `ArtifactStore` 与 typed `EventBus`；
- `FailureArtifact` 作为 recovery/memory 的统一失败输入；
- typed local/mission dataflow 解析，以及 MissionBinding 的完整性、可达性和前驱
  顺序检查；
- `CandidateEvidence` 驱动的 verifier/rehearsal/OOD/latency/recovery-cost 排序；
- 73 个测试通过，包含 graph runtime、failure routing、evidence ranking、parity 和
  strict LLM graph proposal 测试。

当前 dataflow 约定是“最后一个 SkillCall 的 output 作为 action node output”。一个
action 内多个 SkillCall 的中间输出尚未成为公开 graph port；需要该能力时应新增
显式 output selector schema，而不是依赖隐式数组下标。

仍未宣称完成：

- 真实 CAP-X-compatible LLM backend、PromptBuilder 和多模态请求适配；
- LLM Recovery/Monitor 以及 deadline-aware fallback；
- 真正的异步多进程在线 Agent scheduler；
- 多进程 LIBERO rehearsal worker pool；
- streaming Scene/World Model 的生产实现；
- adaptive topology 和 Memory/Robot Skill promotion pipeline。

## 5. 下一阶段顺序

1. 运行 `scripts/run_libero_b3.py` 完成 deterministic B3 smoke，并保存与 B1
   相同任务/seed/API/backend 的 artifact；
2. 接入 CAP-X-compatible LLM backend，只让 LLM 生成 typed artifacts；
3. 已完成 bounded read-only candidate fan-out，保持单一 Arbiter/Executor 提交；
4. 增加 GaP 式 process-level rehearsal、worker respawn 和 streaming snapshot；
4. 实现异步 SceneSnapshot stream 和 freshness/deadline；
5. 最后加入进程级 rehearsal、Memory Skill evolution 和 OOD regression。
