# CAP-MAS 项目代码框架设计

## 1. 问题陈述

CAP-MAS 当前 `capmas/` 目录已含运行时、合约、适配器和图基础设施；端到端
LIBERO B0/B1/B3 入口已经存在。仍缺少：
- 完整的 CAP-X vs CAP-MAS 批量公平对比脚本
- 配置系统说明（`configs/default.yaml` 的使用方式）
- 各 Phase 的代码演进路线（哪些模块已有、哪些待实现）

本文档补全上述缺口，提供项目整体代码框架的完整视图。

## 2. 解决方案

1. 基于 `docs/project-structure.md` 的 10 模块布局，标注每个模块的实现状态
2. 定义配置系统 YAML Schema
3. 设计两个关键集成脚本的接口
4. 映射各 Phase 到代码模块

## 3. 项目目录结构

基于 `docs/project-structure.md`（第 78-90 行）和实际代码结构：

```
cap-mas/
├── capmas/                         # 主包
│   ├── __init__.py
│   │
│   ├── contracts/                  # ✅ Phase 1 完成
│   │   ├── __init__.py
│   │   ├── core.py                 # ArtifactRef, SkillRef, EpisodeHandle
│   │   ├── scene.py                # SceneSnapshot, ObjectTrack, EpisodeStart, EpisodeStatus
│   │   ├── action.py               # ActionContract, SkillCall, ExecutionBudget
│   │   ├── verification.py         # VerificationResult, PredicateReport
│   │   ├── failures.py             # FailureClass, FailureArtifact
│   │   ├── candidates.py           # GraphCandidate, ArbitrationResult
│   │   ├── trace.py                # SkillTrace, ExecutionTrace, EpisodeTrace
│   │   ├── memory.py               # MemoryItem, MemoryUpdate, MemoryContext, MemorySelection, TraceSpan
│   │   └── agent.py                # AgentContext, AgentArtifact, Agent/MissionManager/PolicyAgent/RecoveryAgent Protocol
│   │
│   ├── backends/                   # ✅ Phase 0 完成（协议层）+ 适配器层
│   │   ├── __init__.py
│   │   ├── protocol.py             # RobotBackend Protocol, SkillExecutionResult
│   │   └── capx.py                 # CAPXObservationProvider, CAPXTypedSkill, CAPXRobotBackend, build_capx_skills
│   │
│   ├── runtime/                    # ✅ Phase 1-3 deterministic foundation
│   │   ├── __init__.py
│   │   ├── orchestrator.py         # RuntimeOrchestrator.run_cycle(), Verifier Protocol, CycleResult
│   │   ├── state_store.py          # InMemoryStateStore
│   │   ├── action_lease.py         # ActionLease, ActionLeaseManager
│   │   ├── scheduler.py             # Scheduler Protocol, FixedGraphScheduler
│   │   ├── graph_interpreter.py    # FixedGraphInterpreter, FailureArtifact routing
│   │   └── artifact_bus.py         # ArtifactStore, EventBus
│   │
│   ├── skills/                     # ✅ Phase 2 完成
│   │   ├── __init__.py
│   │   ├── protocol.py             # TypedSkill Protocol, skill_ref()
│   │   └── registry.py             # SkillRegistry
│   │
│   ├── perception/                 # ✅ Phase 0（协议层）完成
│   │   ├── __init__.py
│   │   ├── protocol.py             # ObservationProvider, Vision2DBackend, Geometry3DBackend,
│   │   │                           # GraspProposalBackend, RobotControlBackend, FusedPerceptionBackend
│   │   ├── artifacts.py            # InMemoryArtifactStore
│   │   └── fusion.py               # PerceptionFacade, tracks_from_result()
│   │
│   ├── llm/                        # ✅ protocol + strict graph decoder; backends待实现
│   │   ├── __init__.py
│   │   ├── protocol.py             # ✅ LLMClient Protocol, LLMRequest, LLMResponse
│   │   ├── graph_decoder.py        # ✅ response → schema/scene/graph validation
│   │   ├── capx_compatible.py      # ❌ 待实现：CAP-X 模型客户端适配
│   │   ├── openai_compatible.py    # ❌ 待实现：OpenAI 兼容端点
│   │   ├── local.py                # ❌ 待实现：本地模型服务器
│   │   ├── mock.py                 # ❌ 待实现：MockBackend（测试 fixture）
│   │   └── replay.py              # ❌ 待实现：ReplayBackend（trace 回放）
│   │
│   ├── graph/                      # ✅ Phase 3 deterministic graph foundation
│   │   ├── __init__.py
│   │   ├── validator.py             # GraphValidator and bounded-loop diagnostics
│   │   └── serialization.py         # strict schema_version=1 codec
│   │
│   ├── agents/                     # ✅ deterministic + typed LLM proposal seams
│   │   ├── __init__.py
│   │   ├── base.py                 # ✅ 重导出 Agent, AgentArtifact, AgentContext 等
│   │   ├── manager.py              # ✅ Simple/LLM graph manager（provider-independent）
│   │   ├── policy.py               # ✅ Callable/LLM graph policy seams
│   │   ├── arbiter.py              # ✅ deterministic CandidateArbiter
│   │   └── recovery.py             # ✅ CallableRecoveryAgent（适配器），❌ LLMRecoveryAgent 待实现
│   │
│   ├── execution/                  # ⚠️ Phase 2-4 部分待实现
│   │   ├── __init__.py             # ❌ 待创建
│   │   ├── typed_executor.py       # ❌ 待实现（当前 skill 执行在 orchestrator 内联）
│   │   ├── sandbox.py              # ❌ 待实现
│   │   ├── safety_monitor.py       # ❌ 待实现
│   │   └── trace.py                # ❌ 待实现（recording 层）
│   │
│   ├── verification/               # ⚠️ Phase 2-3 部分待实现
│   │   ├── __init__.py
│   │   ├── freshness.py            # ✅ 已存在（freshness check 空模块）
│   │   └── predicates.py           # ✅ 已存在（predicate 评估空模块）
│   │
│   ├── memory/                     # ✅ Phase 5 基础完成
│   │   ├── __init__.py
│   │   ├── protocol.py             # ✅ MemoryController/Executor Protocol
│   │   ├── controller.py           # ✅ RuleBasedMemoryController
│   │   └── store.py                # ✅ InMemoryMemoryStore
│   │
│   └── evaluation/                 # ⚠️ Phase 8 待完善
│       ├── __init__.py
│       ├── interfaces.py           # ✅ Evaluator/TraceSink/MetricsSink Protocol
│       └── reward.py               # ✅ CAPXBinaryReward, VerifiedTransition, LearningReturn
│
├── configs/
│   └── default.yaml                # ✅ 配置文件（需完善 Schema）
│
├── tests/
│   ├── test_runtime_cycle.py       # ✅ 正常执行 + 过期拒绝
│   ├── test_runtime_failures.py    # ✅ 技能失败→恢复
│   ├── test_memory_store.py        # ✅ 幂等更新 + 版本冲突
│   ├── test_capx_adapter.py        # ✅ CAP-X 观测 + 技能适配
│   ├── test_postcondition_and_episode.py  # ✅ 后置条件 + epoch 隔离
│   ├── test_reward_boundary.py     # ✅ 奖励边界
│   ├── test_graph_contracts.py     # ✅ 图 schema 和静态校验
│   └── test_graph_runtime.py       # ✅ 图执行、恢复、仲裁和 dataflow
│
├── scripts/                        # ✅ LIBERO smoke runners
│   ├── run_libero_b0.py            # 端到端单步循环
│   ├── run_libero_b1.py            # P2.5 多周期循环
│   ├── run_libero_b3.py            # deterministic fixed MissionGraph
│   ├── compare_artifacts.py        # ✅ single matched artifact comparison
│   └── compare_baselines.py        # ❌ CAP-X vs CAP-MAS 批量对比待实现
│
├── docs/
│   ├── design/                     # 🆕 本文档所在目录
│   └── adr/                        # ✅ 7 个 ADR
│
├── pyproject.toml                  # ✅ 项目配置
└── README.md                       # ✅ 项目文档
```

## 4. 配置系统

### 4.1 YAML Schema

```yaml
# configs/default.yaml 的完整结构
episode:
  suite: "libero_spatial"           # 测试套件名
  task_id: "libero_spatial_0"       # 任务 ID
  backend: "capx"                   # 后端：capx | mock
  max_steps: 500                    # 单集最大步数
  seed: 42                          # 随机种子（用于可复现对比）

llm:
  backend: "mock"                   # capx_compatible | openai_compatible | local | mock | replay
  model: "gpt-4"                    # 模型名
  temperature: 0.0                  # 温度（公平对比中使用 0）
  max_output_tokens: 4096
  deadline_ms: 30_000
  endpoint: ""                      # 远程端点 URL（openai_compatible/local 时）
  api_key: ""                       # API Key（openai_compatible 时）

runtime:
  verifier: "predicate_based"       # predicate_based | allow_all
  scheduler: "fixed_graph"          # fixed_graph | adaptive_sparse
  max_action_duration_ms: 5000      # 单次动作最大时长
  max_sim_steps_per_action: 120     # 单次动作最大仿真步数
  freshness_threshold_ms: 100       # 场景快照过期阈值

skills:
  allowed:                          # CAP-X API → CAP-MAS Skill 映射
    goto_pose: "goto_pose"
    open_gripper: "open_gripper"
    close_gripper: "close_gripper"
    get_object_pose: "get_object_pose"
    sample_grasp_pose: "sample_grasp_pose"

memory:
  enabled: false                    # Phase 5 feature flag
  active_bank_version: "0"
  max_retrieved_items: 3
  hard_case_buffer_size: 100

evaluation:
  baseline: "B0"                    # B0-B8 对照条件
  num_trials: 100                   # 可复现对比的 trial 数
  matched_budgets: true             # 是否匹配 token/调用/时间预算
  output_dir: "outputs/"
  seed_set: [42, 43, 44]           # 预定义的种子集合
```

### 4.2 配置加载接口

```python
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass(frozen=True)
class EpisodeConfig:
    suite: str
    task_id: str
    backend: str
    max_steps: int
    seed: int

@dataclass(frozen=True)
class LLMConfig:
    backend: str
    model: str
    temperature: float
    max_output_tokens: int
    deadline_ms: int
    endpoint: str | None
    api_key: str | None

@dataclass(frozen=True)
class RuntimeConfig:
    verifier: str
    scheduler: str
    max_action_duration_ms: int
    max_sim_steps_per_action: int
    freshness_threshold_ms: int

@dataclass(frozen=True)
class SkillsConfig:
    allowed: Mapping[str, str]

@dataclass(frozen=True)
class MemoryConfig:
    enabled: bool
    active_bank_version: str
    max_retrieved_items: int
    hard_case_buffer_size: int

@dataclass(frozen=True)
class EvaluationConfig:
    baseline: str
    num_trials: int
    matched_budgets: bool
    output_dir: str
    seed_set: tuple[int, ...]

class ConfigLoader:
    @staticmethod
    def load(path: Path) -> "RuntimeConfig":
        """从 YAML 文件加载配置并校验必填字段。"""
        ...
```

## 5. 集成脚本

### 5.1 LIBERO 端到端运行脚本

```python
# scripts/run_libero_b0.py

"""
运行 CAP-MAS 在 LIBERO 任务上的端到端单步循环。

用法：
    python scripts/run_libero_b0.py --config configs/libero_spatial_0.yaml

流程：
1. 加载 LIBERO 环境和 CAP-X API
2. 构建 CAPXRobotBackend + CAPXObservationProvider
3. 构建 SkillRegistry（从 CAP-X API 允许列表构建 TypedSkill）
4. 构建 Policy Agent（LLM 后端 + PromptBuilder）
5. 构建 RuntimeOrchestrator
6. 执行单步循环：
   - reset → observe → plan (Policy) → run_cycle → verify → next plan
7. 记录 EpisodeTrace 到 outputs/
8. 输出 benchmark score（evaluator_success）
"""

def build_runtime(config):
    # 1. LIBERO env
    env = create_libero_env(config.episode.task_id)
    # 2. CAP-X API
    api = create_capx_api(env)
    # 3. Observation provider
    artifacts = InMemoryArtifactStore()
    observation = CAPXObservationProvider(api.observe, artifacts)
    # 4. Robot backend
    backend = CAPXRobotBackend(env, observation, config.episode.task_id, config.episode.suite)
    # 5. Skill registry
    registry = SkillRegistry()
    for skill_id, fn_name in config.skills.allowed.items():
        skill = CAPXTypedSkill(SkillRef(skill_id, "capx-compat-1"), api.functions()[fn_name])
        registry.register(SkillRef(skill_id, "capx-compat-1"), skill)
    # 6. LLM Backend + Policy Agent
    llm = create_llm_backend(config.llm)
    prompt_builder = PromptBuilder(config.prompts)
    policy = LLMPolicyAgent(llm, prompt_builder, SchemaValidator())
    # 7. Runtime
    state_store = InMemoryStateStore()
    lease_manager = ActionLeaseManager()
    verifier = PredicateBasedVerifier(load_predicates(config.verification))
    orchestrator = RuntimeOrchestrator(backend, state_store, registry, lease_manager, verifier)
    return orchestrator, policy, config

def run_single_step_loop(runtime, policy, config):
    traces = []
    start = runtime.backend.reset(seed=config.episode.seed)
    runtime.start_episode(start)

    scene = start.initial_scene
    subgoal = SimpleMissionManager().propose_subgoal(
        config.episode.task_id, scene
    )
    context = AgentContext(
        task_id=config.episode.task_id,
        episode_id=start.handle.episode_id,
        episode_epoch=start.handle.episode_epoch,
        scene=scene,
        budget={"max_steps": config.episode.max_steps},
    )

    for _ in range(config.episode.max_steps):
        contract = policy.propose_action(subgoal, scene, context)
        result = runtime.run_cycle(contract)
        traces.append(result.trace)
        if result.committed:
            scene = result.after_scene
            # 检查任务是否完成（可观测后置条件）
            if check_all_postconditions(scene, subgoal):
                break
        else:
            # 失败 → Recovery
            recovery_agent = CallableRecoveryAgent(replan_fn)
            new_contract = recovery_agent.recover(
                result.trace, result.verification, context
            )
            if new_contract is None:
                break

    # 记录 evaluator 评分
    success = runtime.backend.evaluator_success()
    return EpisodeTrace(start.handle.episode_id, start.handle.episode_epoch, tuple(traces)), success
```

### 5.2 CAP-X vs CAP-MAS 对比脚本

```python
# scripts/compare_baselines.py

"""
CAP-X 与 CAP-MAS 的公平对比运行器。

强制执行公平对比规则：
- 相同 task config、task ID、初始 state 种子集
- 相同模型、温度、API 端点
- 相同 token 预算、调用预算、wall-clock 预算
- 分别记录 R_task（二元成功）和 trace 数据

输出：
- outputs/baseline_comparison_{timestamp}.json
- outputs/baseline_comparison_{timestamp}.csv
"""

@dataclass(frozen=True)
class TrialResult:
    trial_id: str
    system: str                       # "capx" | "capmas"
    seed: int
    success: bool                     # R_task
    num_actions: int
    num_model_calls: int
    total_tokens: int
    wall_clock_ms: float
    trace: EpisodeTrace

class FairComparisonRunner:
    def __init__(self, config: EvaluationConfig, capx_runner, capmas_runner):
        ...

    def run_comparison(self) -> Sequence[ComparisonReport]:
        """对每个种子运行 CAP-X 和 CAP-MAS，输出对比报告。"""
        results: list[TrialResult] = []
        for seed in self.config.seed_set:
            # 1. Run CAP-X
            capx_result = self.capx_runner.run(seed=seed)
            results.append(capx_result)

            # 2. Run CAP-MAS with matched budgets
            capmas_budget = self._derive_matched_budget(capx_result)
            capmas_result = self.capmas_runner.run(seed=seed, budget=capmas_budget)
            results.append(capmas_result)

        return self._generate_report(results)

    def _derive_matched_budget(self, capx_result):
        """从 CAP-X 结果中提取预算参数，匹配到 CAP-MAS。"""
        return {
            "max_model_calls": capx_result.num_model_calls,
            "max_tokens": capx_result.total_tokens,
            "max_wall_clock_ms": capx_result.wall_clock_ms,
        }
```

## 6. Phase 到代码模块的映射

| Phase | 涉及模块 | 当前状态 | 需实现 |
|-------|---------|---------|--------|
| Phase 0 (Foundation) | `contracts/`, `backends/protocol.py`, `backends/capx.py`, `perception/artifacts.py`, `memory/store.py`, `evaluation/reward.py` | ✅ 完成 | mock 后端集成测试 |
| Phase 1 (Contracts) | `contracts/`, `runtime/state_store.py` | ✅ 完成 | — |
| Phase 2 (Typed Executor) | `runtime/orchestrator.py`, `skills/`, `execution/` | ✅ 完成 | `execution/typed_executor.py` 独立化 |
| Phase 3 (Multi-Agent) | `contracts/graph.py`, `graph/`, `agents/`, `runtime/`, `scripts/run_libero_b3.py` | ✅ deterministic B3 foundation and LIBERO smoke complete | LLM Backend、Prompt 组装、LLM roles、process-level parallelism |
| Phase 4 (World Model) | `perception/`, `verification/freshness.py` | 协议就绪 | 增量地图、目标追踪、异步语义触发 |
| Phase 5 (Memory) | `memory/controller.py`, `memory/store.py` | ✅ 基础完成 | Memory Executor + Hard-Case Buffer |
| Phase 6 (Memory RL) | `memory/`, `evaluation/reward.py` | 协议就绪 | RL 控制器训练 |
| Phase 7 (Skill Evolution) | `skills/` 扩展 | 未开始 | Quarantine Registry, Shadow Executor |
| Phase 8 (Adaptive Topology) | `runtime/scheduler.py` 扩展 | 未开始 | 自适应拓扑切换 |

## 7. 测试接缝

```python
# tests/contract/test_project_structure.py 核心测试用例

def test_all_phases_have_entry_points():
    """每个 Phase 都有对应的 ___init___.py 入口。"""
    ...

def test_config_default_yaml_is_valid():
    """configs/default.yaml 包含所有必填字段。"""
    ...

def test_dependency_direction_is_correct():
    """contracts 不导入 agents, perception 不导入 agents 等。"""
    ...

def test_control_process_does_not_import_llm():
    """runtime/ 和 execution/ 不导入 llm/ 模块。"""
    ...

def test_mock_backend_can_run_complete_cycle():
    """Mock 后端可以运行完整的 action cycle。"""
    ...
```

## 8. 范围外

- CI/CD 管道配置
- 生产级容器化部署
- Web UI 或 API 服务器
- 分布式集群调度
