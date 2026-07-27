# Memory Plane 与 Evaluation Plane 接口说明

## 1. Memory Plane

### 1.1 四层存储架构

基于 `capmas/contracts/memory.py`（124 行）完整实现：

| Layer | 常量 | 生命周期 | 内容 |
|-------|------|---------|------|
| Episode Working | `EPISODE = "episode"` | 单集 | observations, contracts, traces, failures, recovery |
| Experience | `EXPERIENCE = "experience"` | 跨 episode | 成功/失败案例、技能统计、恢复结果、hard cases |
| Semantic/Procedural | `SEMANTIC = "semantic"` / `PROCEDURAL = "procedural"` | 长期 | 通用任务模式、物理先验、验证后的失败规则 |
| Memory Skill Bank | — (独立快照) | 版本化 | 提取、整合、验证、修订、遗忘记忆的程序 |

### 1.2 MemoryItem Schema

```python
@dataclass(frozen=True)
class MemoryItem:
    memory_id: str
    memory_version: str
    kind: str                        # "failure_rule" | "task_pattern" | "skill_boundary"
    content: Mapping[str, object]
    applicability: Mapping[str, object]  # {"task_family": "pick_place"}
    confidence: float
    evidence_count: int
    source_episode_ids: tuple[str, ...]
    source_trace_ids: tuple[str, ...]    # 来源追溯
    status: str = "candidate"            # candidate | active | stale | contradicted | retired
    contradiction_set: tuple[str, ...] = ()
    created_at_ns: int = 0
    last_validated_at_ns: int = 0
    ttl_seconds: int | None = None
```

### 1.3 MemoryContext — 有界决策视图

```python
@dataclass(frozen=True)
class MemoryContext:
    context_id: str
    episode_id: str
    task_id: str
    task_family: str
    scene_version: int
    current_subgoal: str
    trace_span: TraceSpan                       # 当前迹跨度
    retrieved_memories: tuple[MemoryItemRef, ...]
    hard_cases: tuple[MemoryItemRef, ...]
    memory_skill_candidates: tuple[MemorySkillRef, ...]
    active_memory_bank_version: str = "0"
    active_robot_registry_version: str = "0"
    budget: MemoryBudget                        # max_items, max_latency_ms, max_tokens
    novelty: float = 0.0
    uncertainty: float = 0.0
    current_failure: str | None = None          # FailureClass
    recent_recovery: str | None = None
```

**关键约束**：`MemoryContext` 不包含环境句柄或特权完成信号。

### 1.4 MemoryUpdate — 带来源的提案

```python
@dataclass(frozen=True)
class MemoryUpdate:
    update_id: str
    episode_id: str
    task_id: str
    base_memory_version: str           # 基于此版本
    target_layer: str                  # episode | experience | semantic | procedural
    operation: str                     # add | upsert | invalidate | retire | consolidate | noop
    items: tuple[MemoryItem, ...]
    invalidated_memory_ids: tuple[str, ...] = ()
    retired_memory_ids: tuple[str, ...] = ()
    source_trace_ids: tuple[str, ...] = ()     # 强制来源追溯
    evidence_refs: tuple[ArtifactRef, ...] = ()
    produced_by_skill: SkillRef | None = None
    controller_selection_id: str | None = None
    confidence: float = 0.0
    applicability: Mapping[str, object]
    ttl_seconds: int | None = None
    idempotency_key: str = ""
    status: str = "proposed"
```

**提交规则**（`capmas/memory/store.py:23-46`）：
- 必须有 `idempotency_key`（幂等去重）
- 必须有 `source_trace_ids`（来源追溯，noop 除外）
- `base_memory_version` 必须匹配当前版本（版本冲突检测）
- `add` 操作不能重复已有 `memory_id`

### 1.5 迹到记忆循环

```
ExecutionTrace
  → 检索相关记忆
  → MemoryController.select(context) → Top-K Memory Skills
  → MemoryExecutor.apply(selection, trace_span) → MemoryUpdate proposal
  → provenance/confidence/conflict/TTL checks
  → commit to episode/persistent memory
  → 下游结果评估
  → hard-case buffer 和 controller feedback
```

### 1.6 Hard-Case Buffer

有界滑动集合，按以下优先级采样：
1. 失败严重性
2. 新颖性
3. 复发频率
4. 不确定性
5. 覆盖不足的任务族

必须保留罕见的安全和恢复失败，即使它们不频繁。

## 2. Evaluation Plane

### 2.1 Evaluator Protocol

```python
class Evaluator(Protocol):
    def benchmark_success(self, episode: EpisodeTrace) -> bool: ...
```

### 2.2 双通道奖励

基于 `capmas/evaluation/reward.py`（39 行）和 `docs/reward-and-rl.md`：

```python
class CAPXBinaryReward:
    def benchmark(self, episode: EpisodeTrace, evaluator_success: bool) -> float:
        """CAP-X 二元基准分数。R_task = 1 if success else 0。"""
        return 1.0 if evaluator_success else 0.0

    def learning_return(self, transition: VerifiedTransition) -> LearningReturn:
        """Verifier 衍生的学习回报（分离通道，不改变 benchmark score）。"""
        progress_delta = transition.progress_after - transition.progress_before
        cost = 1.0 if transition.human_intervention else 0.0
        constrained = transition.safety_violation
        value = -1.0 if constrained else progress_delta - cost
        return LearningReturn(value, progress_delta, 0.0, cost, constrained)

@dataclass(frozen=True)
class LearningReturn:
    value: float
    progress_delta: float
    terminal: float
    cost: float
    constrained: bool
```

### 2.3 公平对比规则

```
CAP-X Legacy:
    Task prompt → single agent → Python exec → CAP-X API → environment
CAP-MAS:
    Task prompt → Manager → Policy → Verifier → typed skills
    → lease-controlled executor → environment
    → asynchronous snapshots → recovery and memory

匹配项：
- 相同模型、温度、trial 数、种子集合
- 相同总模型调用数、token 预算、action 预算、wall-clock 预算
- 分别报告 evaluator-only privileged success 和 agent-visible verification
```

### 2.4 指标接口

```python
class MetricsSink(Protocol):
    def record(self, name: str, value: float, tags: dict[str, str] | None = None) -> None: ...

class TraceSink(Protocol):
    def append(self, trace: EpisodeTrace) -> None: ...
```

**关键指标**：
- 全任务成功率、按子目标数分层的成功率
- 成功率随 horizon 增长而下降的斜率
- 过期动作拒绝率、前置条件违反率、后置条件假阳性/假阴性率
- 控制 deadline miss 率、场景快照新鲜度
- 总模型调用和 token 数、wall-clock 完成时间
- 跨任务技能复用率、技能回归率

## 3. 测试接缝

```python
# tests/contract/test_memory.py
def test_memory_update_is_idempotent():
    """相同 idempotency_key 的更新不重复生效。"""
    ...

def test_memory_update_rejects_stale_base_version():
    """base_memory_version 不匹配的更新被拒绝。"""
    ...

def test_memory_update_requires_trace_provenance():
    """没有 source_trace_ids 的非 noop 更新被拒绝。"""
    ...

def test_memory_context_does_not_contain_privileged_state():
    """MemoryContext 不包含 evaluator_success 或环境句柄。"""
    ...

# tests/contract/test_evaluation.py
def test_binary_reward_matches_capx():
    """CAPXBinaryReward 的 benchmark() 与 CAP-X 保持一致。"""
    ...

def test_learning_return_is_separate_channel():
    """LearningReturn 不修改 benchmark 二元分数。"""
    ...

def test_safety_violation_constrains_learning_return():
    """安全违规时 learning_return.constrained = True, value = -1。"""
    ...
```

## 4. 范围外

- RL 算法实现（PPO/GRPO）
- 向量数据库检索索引
- 记忆冲突的语义合并