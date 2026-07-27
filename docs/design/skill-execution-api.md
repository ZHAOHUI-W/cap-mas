# Skill & Execution Plane 接口说明

## 1. 问题陈述

CAP-MAS 区分 Robot Skills（物理状态转换）和 Memory Skills（迹→可复用信息），二者有独立注册表、隔离和进化管道。当前实现（`capmas/skills/` 和 `capmas/execution/`）覆盖了基础部分，但缺少完整的进化循环和隔离机制文档。

## 2. 技能类型对比

| 维度 | Robot Skill | Memory Skill |
|------|------------|-------------|
| 作用对象 | 物理世界（机器人） | 信息世界（记忆存储） |
| 注册表 | `SkillRegistry` | `InMemoryMemoryStore` |
| 隔离 | `DISCOVERED → QUARANTINED → SHADOW_VALIDATED → ACTIVE → RETIRED` | `candidate → active → stale → contradicted → retired` |
| 执行权限 | 需要 `ActionLease` | 无需租约，但只能发出提案 |
| 调用方式 | `TypedSkill.execute(args, budget)` | `MemoryExecutor.apply(selection, trace_span)` |
| 进化顺序 | 在 Memory Skill 冻结后 | 先于 Robot Skill 进化 |

## 3. Robot Skill 接口

### 3.1 TypedSkill Protocol

基于 `capmas/skills/protocol.py:10-20`：

```python
class TypedSkill(Protocol):
    skill_id: str                      # "goto_pose", "close_gripper"
    version: str                       # "1.0.0", "capx-compat-1"

    def validate_args(self, args: dict[str, object]) -> None:
        """类型校验：确保 args 匹配函数签名。失败抛 ValueError。"""
        ...

    def execute(self, args: dict[str, object], budget: ExecutionBudget) -> SkillExecutionResult:
        """有界执行。budget 提供 max_duration_ms 和 max_sim_steps。"""
        ...

def skill_ref(skill: TypedSkill) -> SkillRef:
    """从实现提取 SkillRef 标识。"""
    return SkillRef(skill_id=skill.skill_id, version=skill.version)
```

### 3.2 SkillRegistry

基于 `capmas/skills/registry.py:10-37`：

```python
class SkillRegistry:
    def __init__(self) -> None: ...

    def register(self, reference: SkillRef, skill: TypedSkill) -> None:
        """注册技能。reference 必须匹配 skill.skill_id + skill.version。"""
        ...

    def get(self, reference: SkillRef) -> TypedSkill:
        """获取技能。不存在时抛 ValueError。"""
        ...

    def validate_contract(self, contract: ActionContract) -> None:
        """校验 ActionContract 中的所有 SkillCall 都引用已注册技能且参数合法。"""
        ...

    def snapshot_version(self) -> str:
        """返回当前注册表快照标识。用于 trace 中的版本溯源。"""
        ...
```

### 3.3 Robot Skill 进化循环

```
ExecutionTrace (successful or failed)
  → Critic 归因失败
  → 生成候选（composition/parameter repair）
  → 静态验证 + sandbox test
  → 定向边界测试
  → 先前任务回归测试
  → OOD 验证
  → 在安全检查点激活或延迟到后续 episode 提升
```

### 3.4 Quarantine & Promotion 接口（Phase 7 新增）

```python
class RobotSkillQuarantine:
    """隔离的候选技能注册表。其中技能不能用于 active 执行。"""

    def add_candidate(self, skill: TypedSkill, source_trace_ids: tuple[str, ...]) -> None: ...
    def shadow_validate(self, skill_ref: SkillRef, traces: Sequence[ExecutionTrace]) -> ShadowValidationReport: ...
    def promote(self, skill_ref: SkillRef) -> None:  # → active registry
    def rollback(self, skill_ref: SkillRef) -> None:  # → retired
```

## 4. Memory Skill 接口

### 4.1 MemoryController

基于 `capmas/memory/protocol.py:13-14` 和 `capmas/memory/controller.py:8-18`：

```python
class MemoryController(Protocol):
    def select(self, context: MemoryContext) -> MemorySelection: ...

class RuleBasedMemoryController:
    """Phase 5 的基于规则的默认实现。
    
    从 context.memory_skill_candidates 中选取前 max_items 个候选。
    """
    def select(self, context: MemoryContext) -> MemorySelection:
        candidates = context.memory_skill_candidates[:context.budget.max_items]
        if not candidates:
            return MemorySelection(str(uuid4()), (), True, "no applicable Memory Skills")
        return MemorySelection(
            selection_id=str(uuid4()),
            selected_skills=tuple(candidates),
            skipped=False,
            rationale="selected active candidates within budget",
        )
```

### 4.2 MemoryExecutor

基于 `capmas/memory/protocol.py:17-18`：

```python
class MemoryExecutor(Protocol):
    def apply(self, selection: MemorySelection, trace_span: TraceSpan) -> MemoryUpdate: ...
```

### 4.3 Memory Skill Bank 的隔离设计

基于 `docs/memory-skill-evolution.md`：

```
(MemoryBank M_k, RobotRegistry R_k, Controller C_k)
        |
        | hold R_k fixed; evolve memory
        v
(M_(k+1), R_k, C_(k+1))
        |
        | hold M_(k+1) and C_(k+1) fixed; evolve robot skills
        v
(M_(k+1), R_(k+1), C_(k+1))
```

Active tuple 在安全检查点或 episode 之间原子切换。没有 mid-action 更新。

## 5. Execution 接口

### 5.1 有界执行

基于 `capmas/runtime/orchestrator.py` 中的内联 skill 执行循环（第 86-142 行）：

```
对 ActionContract.skills 中每个 SkillCall:
  1. 从 SkillRegistry 获取 TypedSkill
  2. 记录 SkillTrace 开始时间
  3. 调用 backend.execute_skill(skill, args, budget)
  4. 记录 SkillTrace 结束时间
  5. 如果失败 (result.ok == False):
     → 立即 observe 最新场景
     → 构造 ExecutionTrace(status="failed")
     → 返回 CycleResult(committed=False)
```

### 5.2 独立 Typed Executor（Phase 2 重构目标）

```python
class TypedExecutor:
    """独立的类型化执行器。当前逻辑内联于 orchestrator 中。

    职责：
    - 执行 ActionContract 中的所有技能调用
    - 逐技能记录 SkillTrace
    - 绑定 ActionLease 生命周期
    - 失败时早停并报告
    """
    def __init__(self, backend: RobotBackend, registry: SkillRegistry): ...

    def execute_contract(
        self,
        contract: ActionContract,
        lease: ActionLease,
    ) -> tuple[tuple[SkillTrace, ...], bool]:
        """执行合约，返回 (traces, all_ok)。"""
        ...
```

### 5.3 Safety Monitor

```python
class SafetyMonitor:
    """运行时光速安全监控。

    在每个仿真步长后检查：
    1. 关节限制
    2. 碰撞检测（如果可用）
    3. 租约过期
    4. 场景新鲜度
    """

    def check(self, state: Mapping[str, object], lease: ActionLease) -> list[str]:
        """返回违规的不变量列表。空列表 = 安全。"""
        ...
```

## 6. 注册表互斥规则

| 操作 | Robot Skill 注册表 | Memory Skill Bank |
|------|------------------|-------------------|
| 查看 active | 任意 Agent 可读 | 任意 Agent 可读 |
| 写入 candidate | 仅 Critic/Evolver | 仅 Memory Designer |
| 执行 | 仅通过 TypedExecutor + ActionLease | 仅通过 MemoryExecutor |
| 提升到 active | 仅 Promotion Manager（安全边界） | 仅 Promotion Manager（安全边界） |
| 修改 mid-action | ❌ 禁止 | ❌ 禁止 |

## 7. 测试接缝

```python
# tests/contract/test_skills.py
def test_registry_rejects_unregistered_skill():
    """未注册的 SkillRef 被拒绝。"""
    ...

def test_registry_validates_skill_args():
    """参数不匹配的 SkillCall 被拒绝。"""
    ...

def test_robot_skill_cannot_call_memory_skill():
    """Robot Skill 不能调用 Memory Skill 接口。"""
    ...

def test_memory_skill_cannot_access_actuator():
    """Memory Skill 不能访问机器人执行器。"""
    ...

def test_skill_registry_version_is_stable():
    """snapshot_version() 在注册表不变时返回相同标识。"""
    ...

# tests/contract/test_execution.py
def test_executor_early_stops_on_skill_failure():
    """Skill 执行失败时早停，后续技能不执行。"""
    ...

def test_executor_records_full_trace_on_success():
    """所有技能成功后，trace 包含全部 SkillTrace。"""
    ...

def test_safety_monitor_blocks_on_violation():
    """不变量违规时 SafetyMonitor 返回非空列表。"""
    ...
```

## 8. 范围外

- 跨任务自动技能发现（Phase 7）
- 技能蒸馏或模型压缩
- 在线学习技能参数
- Robot Skill 和 Memory Skill 的联合进化（已设计为后续 ablation）