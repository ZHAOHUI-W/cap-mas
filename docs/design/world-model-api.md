# World Model Plane 接口说明

## 1. 概述

CAP-MAS 中的 "World Model" 不是机器学习的生成式世界模型，而是**经典机器人学中的几何感知管道**——一个多速率异步传感器融合与增量 3D 地图构建器。运行频率 5-30 Hz，不使用任何学习式生成模型。

## 2. 接口族

基于 `capmas/perception/protocol.py`（151 行）的完整实现：

### 2.1 ObservationProvider

```python
class ObservationProvider(Protocol):
    """原始传感器采集接口。第 96-97 行。"""
    def capture(self) -> ObservationBundle: ...

@dataclass(frozen=True)
class ObservationBundle:
    timestamp_ns: int
    frames: tuple[CameraFrame, ...]      # 所有相机帧
    robot_state: Mapping[str, object]    # includes measured gripper_opening and optional gripper_commanded_fraction
```

**实现**：`CAPXObservationProvider`（`backends/capx.py:32-84`）— 将 CAP-X 观测标准化为 `ArtifactRef` 格式。

### 2.2 Vision2DBackend

```python
class Vision2DBackend(Protocol):
    """2D 检测/分割后端。第 100-107 行。"""
    def segment_text(self, frame: CameraFrame, text_query: str) -> Sequence[Detection2D]: ...
    def segment_point(self, frame: CameraFrame, point: tuple[float, float]) -> Sequence[Detection2D]: ...
```

### 2.3 Geometry3DBackend

```python
class Geometry3DBackend(Protocol):
    """3D 位姿估计后端。第 110-115 行。"""
    def estimate_pose(self, frame: CameraFrame, detection: Detection2D) -> ObjectPoseEstimate | None: ...
```

### 2.4 GraspProposalBackend

```python
class GraspProposalBackend(Protocol):
    """抓取姿态采样后端。第 118-123 行。"""
    def propose(self, observation: ObservationBundle, object_label: str) -> Sequence[GraspCandidate]: ...
```

### 2.5 RobotControlBackend

```python
class RobotControlBackend(Protocol):
    """运动执行后端。第 126-136 行。"""
    def goto_pose(self, position_xyz, quaternion_wxyz, z_approach=0.0) -> None: ...
    def open_gripper(self) -> None: ...
    def close_gripper(self) -> None: ...
```

### 2.6 FusedPerceptionBackend — 统一 facade

```python
class FusedPerceptionBackend(Protocol):
    """融合 2D+3D 的统一感知接口。第 139-151 行。"""
    def infer(self, request: PerceptionRequest, observation: ObservationBundle) -> PerceptionResult: ...
    def publish_scene(self, observation: ObservationBundle, result: PerceptionResult, previous: SceneSnapshot | None) -> SceneSnapshot: ...
```

**实现辅助**：`PerceptionFacade`（`perception/fusion.py:16-29`）和 `tracks_from_result()` （line 32-44）。

## 3. 多速率管道

| Layer | 接口 | 目标速率 | 阻塞策略 |
|-------|------|---------|---------|
| Joint state + FK | `RobotBackend.observe()` | 50-200 Hz | 绝不能阻塞 |
| Local depth/voxel update | `SceneEstimator.update()` | 20-50 Hz | 丢弃过期帧 |
| Object tracker | `SceneEstimator.predict()` | 10-30 Hz | 观测间预测 |
| Semantic segmentation | `Vision2DBackend.segment_text()` | 1-5 Hz 或事件触发 | 异步运行 |
| Perception Agent 推理 | `SemanticPerception.request()` | 事件触发 | 绝不在伺服路径 |

## 4. 场景发布管道

```
RGB-D + Proprioception
  → Sensor Synchronizer（时间戳对齐）
  → Fast Geometry（FK 计算相机位姿，深度→点云转换）
  → Object Tracker（ROI depth + 点注册）
  → Motion Predictor（置信度感知预测）
  → Incremental Local 3D Map（仅融合变化的 voxel/TSDF blocks）
  → SceneSnapshot Publisher（不可变快照 + 单调版本号）
  → [低置信度触发] Semantic Perception Agent（异步修正）
```

## 5. 测试接缝

```python
# tests/contract/test_perception.py
def test_observation_provider_capture_returns_artifactized_frames():
    """CAPXObservationProvider 将帧数据转化为 ArtifactRef。"""
    ...

def test_fused_backend_infer_returns_unified_result():
    """FusedPerceptionBackend.infer() 返回统一的 2D+3D 结果。"""
    ...

def test_tracks_from_result_preserves_confidence():
    """PerceptionResult → ObjectTrack 保留置信度信息。"""
    ...

def test_placement_pose_fallback_preserves_reason():
    """几何放置位姿不可用时保留 semantic fallback 的原因。"""
    ...

def test_scene_publish_does_not_mutate_active_snapshot():
    """场景发布不修改已被 active contract 使用的快照。"""
    ...

def test_stale_map_rejected_by_freshness_policy():
    """过期地图被 freshness 策略拒绝。"""
    ...
```

## 6. 范围外

- 实时多进程调度和伺服控制（Phase 4）
- 增量 TSDF 地图的具体实现（CAP-X LIBERO 集成后）
- 语义分割模型（SAM3/VLM）的直接实现（通过 adapter 解耦）

## 7. Phase 4 参考实现状态

已落地的公共实现位于 `capmas/perception/`：

- `protocol.py` 保留旧的 `ObservationBundle(timestamp_ns, frames, robot_state)`
  位置构造，并追加 episode/source/sequence 元数据。
- `sensor_sync.py` 提供 JSONL recording/replay 和有界同步队列。
- `geometry.py` 提供相机 pose/FK fallback 和注入式深度解码；不依赖 NumPy。
- `local_map.py` 提供 `SparseVoxelMap`，查询返回 occupancy、clearance、confidence
  和 source timestamp；TSDF 配置只校验且默认拒绝启用。
- `tracking.py` 提供显式 ID、标签+距离 gating 和常速度预测。
- `semantic_triggers.py` 只根据置信度/track 状态发出去重请求，不调用模型。
- `world_model.py` 提供 pure service、thread runtime、spawn process runtime、
  restart/degraded health 和 last-snapshot fallback。
- `serialization.py` 与 `artifact_bridge.py` 保证进程边界没有内存指针或原始
  大帧；文件 ArtifactStore 用 SHA-256 URI 和原子替换。

`scripts/run_libero_b5.py` 当前是 replay/reference benchmark。它会保留一份
JSON artifact 和一份 log，并明确将 `evaluator_success` 设为 `null`，直到真实
CAP-X backend 接入；几何快照版本不能代替 LIBERO evaluator。
