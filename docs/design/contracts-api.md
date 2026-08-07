# 合约系统接口说明

## 1. 问题陈述

CAP-MAS 的合约系统（`capmas/contracts/`）已实现 `core`, `scene`, `action`, `verification`, `failures`, `trace`, `memory`, `agent` 八个模块，但缺少：
- 合约生命周期状态机
- 各合约之间的引用关系和版本依赖
- 测试接缝定义（已存在 6 个测试的覆盖度分析）
- 序列化/反序列化规范

本文档基于已实现的代码形式化合约系统的完整接口。

## 2. 合约模块总览

| 模块 | 文件 | 核心类型 | 行数 |
|------|------|---------|------|
| Core | `contracts/core.py` | `ArtifactRef`, `SkillRef`, `EpisodeHandle` | 31 |
| Scene | `contracts/scene.py` | `SceneSnapshot`, `ObjectTrack`, `EpisodeStart`, `EpisodeStatus` | 44 |
| Action | `contracts/action.py` | `ActionContract`, `SkillCall`, `ExecutionBudget` | 41 |
| Verification | `contracts/verification.py` | `VerificationResult`, `PredicateReport` | 29 |
| Failures | `contracts/failures.py` | `FailureClass` (8 个常量) | 9 |
| Trace | `contracts/trace.py` | `SkillTrace`, `ExecutionTrace`, `EpisodeTrace` | 49 |
| Memory | `contracts/memory.py` | `MemoryItem`, `MemoryUpdate`, `MemoryContext`, `MemorySelection`, `TraceSpan`, `MemoryBudget` | 124 |
| Agent | `contracts/agent.py` | `AgentContext`, `AgentArtifact`, `Agent/MissionManager/PolicyAgent/RecoveryAgent` Protocol | 56 |

## 3. 合约生命周期状态机

### 3.1 Episode 状态

```
                             reset()
                                │
                                ▼
┌──────────┐    start_episode    ┌──────────┐
│ CREATED  │ ──────────────────> │  ACTIVE  │
└──────────┘                    └────┬─────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
               task_completed   max_steps/   reset_or_timeout
                    │           abort/error         │
                    ▼                ▼                ▼
              ┌──────────┐   ┌──────────┐   ┌──────────────┐
              │COMPLETED │   │  FAILED  │   │   ABORTED    │
              └──────────┘   └──────────┘   └──────────────┘

每个 reset() 产生新的 episode_epoch，旧 epoch 的所有 contract 失效。
```

### 3.2 ActionContract 状态

```
                              Policy Agent
                                  │
                                  ▼
┌──────────┐   propose_action()  ┌──────────┐
│ DRAFT    │ ──────────────────> │ PROPOSED │
└──────────┘                    └────┬─────┘
                                     │
                          schema_validator
                                     │
                              ┌──────┴──────┐
                              │             │
                           valid         invalid
                              │             │
                              ▼             ▼
                     verifier.approve()   REJECTED
                              │
                    ┌─────────┼─────────┐
                    │         │         │
                approve     reject
                    │         │
                    ▼         ▼
              ┌─────────┐  REJECTED
              │ EXECUTING│
              └────┬─────┘
                   │
         ┌─────────┼─────────┐
         │         │         │
     commit       recover
         │         │
         ▼         ▼
    COMMITTED   RECOVERY
```

### 3.3 MemoryItem 状态

```
                               MemorySkillEvolver
                                      │
                                      ▼
┌───────────┐   propose_candidate()  ┌───────────┐
│ CANDIDATE │ <──────────────────── │ DISCOVERED │
└─────┬─────┘                        └───────────┘
      │
      │ validation gates (schema, replay, contradiction, OOD)
      │
      ├── ✓ ──> ┌───────┐
      │         │ ACTIVE │
      │         └───┬───┘
      │             │ aging / contradiction
      │             ▼
      │         ┌───────┐         ┌─────────────┐
      │         │ STALE │ ──────> │ CONTRADICTED │
      │         └───┬───┘         └─────────────┘
      │             │
      │             ▼
      │         ┌─────────┐
      └── ✗ ──> │ RETIRED │ (保留在存档中)
                └─────────┘
```

## 4. 核心合约接口

### 4.1 ArtifactRef — 内容寻址引用

基于 `capmas/contracts/core.py:7-12`：

```python
@dataclass(frozen=True)
class ArtifactRef:
    """内容寻址的二进制数据引用。
    
    大对象（RGB-D 图像、点云、地图 delta）通过此引用间接传递，
    不嵌入 Agent 消息中。
    """
    uri: str                       # artifact://array/... 或 artifact://voxel-delta/...
    media_type: str                # "image/rgb" | "image/depth" | "array/joint-position" | "array/ee-pose"
    sha256: str | None = None      # 内容哈希（可选的完整性校验）
    byte_size: int | None = None   # 字节大小（可选的估算）
```

**测试接缝**：
- `InMemoryArtifactStore` (`perception/artifacts.py:9-21`) — 可存储任意 Python 对象

### 4.2 EpisodeHandle — 集身份

基于 `capmas/contracts/core.py:21-31`：

```python
@dataclass(frozen=True)
class EpisodeHandle:
    episode_id: str                # UUID
    task_id: str                   # "libero_spatial_0"
    suite_name: str                # "libero_spatial"
    backend_id: str                # "capx" | "mock"
    seed: int | None               # 随机种子
    episode_epoch: int             # 单调递增（同一 episode 多次 reset 时递增）
    started_at_ns: int             # 启动时间戳
    status: str = "active"         # 见 EpisodeStatus 枚举
    metadata: Mapping[str, str]    # 附加元数据
```

**epoch 隔离规则**：
- 每个 `reset()` 产生 `episode_epoch + 1`
- `ActionContract.episode_epoch != handle.episode_epoch` → 拒绝执行
- 测试：`tests/test_postcondition_and_episode.py` 已验证 ✅

### 4.3 SceneSnapshot — 版本化场景快照

基于 `capmas/contracts/scene.py:28-38`：

```python
@dataclass(frozen=True)
class SceneSnapshot:
    episode_id: str
    episode_epoch: int
    scene_version: int             # 单调递增版本号
    sensor_timestamp_ns: int       # 传感器采集时间
    publish_timestamp_ns: int      # 发布时间
    robot: Mapping[str, object]    # measured gripper_opening plus optional gripper_commanded_fraction
    objects: Sequence[ObjectTrack] # 被追踪对象的列表
    local_map: ArtifactRef | None  # 增量局部 3D 地图（可选）
    freshness_ms: float            # 快照新鲜度
    source_artifacts: tuple[ArtifactRef, ...]  # 原始 RGB-D 帧引用
```

`ObjectTrack` 可选地携带 `placement_pose_wxyz_xyz`、
`placement_pose_source` 和 `placement_pose_reason`。点云估计成功时 source 为
`geometry_pointcloud`；感知异常或点云不足时 pose 保持为空，source 为
`semantic_pose_fallback`，reason 保留具体失败原因。这样 verifier 的语义位姿
fallback 不再表现为无法解释的 `null`。

**版本单调性**：
- `StateStore.publish()` 拒绝非单调版本号
- `compare_and_commit(parent_version, snapshot)` 要求 `snapshot.scene_version == parent_version + 1`
- 测试：`tests/test_runtime_cycle.py:143-160` 已验证过期拒绝 ✅

### 4.4 ActionContract — 动作合约

基于 `capmas/contracts/action.py:20-34`：

```python
@dataclass(frozen=True)
class ActionContract:
    contract_id: str               # UUID
    episode_id: str
    episode_epoch: int             # 与 EpisodeHandle.episode_epoch 匹配
    parent_scene_version: int      # 基于此场景版本的推理
    subgoal_id: str                # 关联的子目标 ID
    skills: tuple[SkillCall, ...]  # 有序的技能调用序列
    expected_postconditions: tuple[str, ...]  # "gripper.holds(obj-7)"
    max_duration_ms: int           # 执行时限
    max_sim_steps: int             # 仿真步数上限
    proposed_by: str               # "policy_agent"
    preconditions: tuple[str, ...] = ()       # "track(obj-7).confidence >= 0.85"
    safety_invariants: tuple[str, ...] = ()   # "distance_to_obstacle >= 0.02"
    recovery_policy: str = "replan"           # "replan" | "reacquire_and_retry" | "skip" | "abort"

    @property
    def budget(self) -> ExecutionBudget: ...  # max_duration_ms + max_sim_steps
```

### 4.5 ActionLease — 动作租约

基于 `capmas/runtime/action_lease.py:8-17`：

```python
@dataclass(frozen=True)
class ActionLease:
    lease_id: str                  # "lease-1"
    holder: str                    # 持有者（agent name）
    contract_id: str               # 关联的 ActionContract
    issued_at_ns: int
    expires_at_ns: int

    def is_expired(self, now_ns: int) -> bool: ...
```

**租约规则**：
- 同一时刻最多 1 个活跃租约（`acquire()` 拒绝重复获取）
- `release(lease_id)` 只能释放当前持有者的租约
- 租约过期 → 自动释放 → 可控安全 stop/hold

### 4.6 VerificationResult — 验证结果

基于 `capmas/contracts/verification.py:16-29`：

```python
@dataclass(frozen=True)
class VerificationResult:
    contract_id: str
    decision: str                  # "approve" | "reject" | "commit" | "recover"
    checked_scene_version: int     # 验证时的场景版本
    predicate_results: tuple[PredicateReport, ...] = ()
    violated_invariants: tuple[str, ...] = ()
    failure_class: str | None = None  # FailureClass 常量

    @property
    def passed(self) -> bool:
        return self.decision in {"approve", "commit"} and all(
            report.passed for report in self.predicate_results
        )
```

### 4.7 ExecutionTrace — 执行迹

基于 `capmas/contracts/trace.py:23-42`：

```python
@dataclass(frozen=True)
class ExecutionTrace:
    trace_id: str
    episode_id: str
    episode_epoch: int
    contract_id: str
    lease_id: str
    parent_scene_version: int
    start_scene_version: int       # 执行开始时的场景版本
    end_scene_version: int | None  # 执行结束时的场景版本
    started_at_ns: int
    finished_at_ns: int
    status: str                    # "completed" | "failed"
    skill_traces: tuple[SkillTrace, ...]
    precondition_result: VerificationResult | None
    postcondition_result: VerificationResult | None
    failure_class: str | None
    observation_before: ArtifactRef | None
    observation_after: ArtifactRef | None
    metadata: dict[str, object]
```

## 5. 合约序列化规范

所有合约对象支持 JSON 序列化（除 ArtifactRef 指向的二进制数据外）。

**序列化规则**：
1. 所有 `@dataclass(frozen=True)` 转换为 JSON 对象
2. `.` 前缀私有字段不序列化
3. `ArtifactRef` 序列化为 `{"uri": "...", "media_type": "...", "sha256": ..., "byte_size": ...}`
4. `Enum` 值序列化为其字符串值
5. `Sequence` 序列化为 JSON 数组
6. `Mapping` 序列化为 JSON 对象
7. `None` 序列化为 JSON `null`

## 6. 合约依赖图

```
EpisodeHandle ←── EpisodeStart ──→ SceneSnapshot
     │                                  │
     │                           ┌──────┼──────┐
     │                           │      │      │
     ▼                           ▼      │      │
ActionContract ──────────→ parent_scene_version
     │                           │      │      │
     │                    preconditions   │  postconditions
     │                           │      │      │
     ▼                           ▼      ▼      ▼
 ActionLease ←── Verifier.approve() → VerificationResult
     │                           │
     │                    Verifier.commit()
     │                           │
     ▼                           ▼
 SkillTrace ──→ ExecutionTrace ──→ EpisodeTrace
                    │
                    ▼
              MemoryItem ←── MemoryUpdate
```

## 7. 测试接缝

已存在的测试覆盖（`tests/` 目录）：

| 测试 | 合约覆盖 | 状态 |
|------|---------|------|
| `test_runtime_cycle.py` | `ActionContract`, `SceneSnapshot`, `VerificationResult`, `EpisodeHandle` | ✅ |
| `test_runtime_failures.py` | `ActionContract`, `ExecutionTrace`, `FailureClass.POSTCONDITION_FAILED`, `FailureClass.EXECUTION_ERROR` | ✅ |
| `test_memory_store.py` | `MemoryItem`, `MemoryUpdate`, `MemoryOperation.ADD` | ✅ |
| `test_postcondition_and_episode.py` | `SceneSnapshot`, `VerificationResult`, `EpisodeHandle.episode_epoch` | ✅ |
| `test_reward_boundary.py` | `EpisodeTrace`, `LearningReturn` | ✅ |
| `test_capx_adapter.py` | `ArtifactRef`, `SkillRef`, `ObservationBundle`, `CameraFrame` | ✅ |

待补充的测试（`tests/contract/` 目录）：

```python
# tests/contract/test_contracts.py 核心测试用例

def test_scene_snapshot_version_is_monotonic():
    """SceneSnapshot.scene_version 在 compare_and_commit 后单调递增。"""
    ...

def test_action_contract_rejects_wrong_epoch():
    """episode_epoch 不匹配的 contract 被拒绝。"""
    ...

def test_lease_cannot_be_acquired_twice():
    """同时只能有一个活跃租约。"""
    ...

def test_leased_contract_execution_is_atomic_per_lease():
    """一个租约内执行完整的技能序列。"""
    ...

def test_memory_update_requires_provenance():
    """没有 source_trace_ids 的 memory update 被拒绝。"""
    ...

def test_memory_update_requires_idempotency_key():
    """没有 idempotency_key 的 update 被拒绝。"""
    ...

def test_failure_class_enum_is_exhaustive():
    """8 个 FailureClass 常量完整。"""
    ...
```

## 8. 范围外

- 完整的谓词语言表达式引擎（Phase 2 使用显式小集合）
- Protobuf 或 gRPC 传输层
- 合同的数字签名或加密
- 合同 governance 策略
