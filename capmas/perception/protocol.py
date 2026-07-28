from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence

from capmas.contracts.core import ArtifactRef
from capmas.contracts.scene import SceneSnapshot, VisualEvidence
from capmas.perception.tracking import ObjectMeasurement


@dataclass(frozen=True)
class CameraModel:
    camera_id: str
    intrinsics: tuple[float, ...]
    pose_world: tuple[float, ...]


@dataclass(frozen=True)
class CameraFrame:
    camera_id: str
    timestamp_ns: int
    rgb: ArtifactRef | None
    depth: ArtifactRef | None
    camera: CameraModel


@dataclass(frozen=True)
class ObservationBundle:
    timestamp_ns: int
    frames: tuple[CameraFrame, ...]
    robot_state: Mapping[str, object]
    episode_id: str | None = None
    episode_epoch: int | None = None
    source: str = ""
    sequence: int = 0
    object_measurements: tuple[ObjectMeasurement, ...] = ()


@dataclass(frozen=True)
class PerceptionRequest:
    request_id: str
    query: str | None = None
    point: tuple[float, float] | None = None
    camera_ids: tuple[str, ...] = ()
    require_3d: bool = True
    target_track_ids: tuple[str, ...] = ()
    evidence_types: tuple[str, ...] = ()
    purpose: str = "scene_understanding"
    max_latency_ms: int = 500
    priority: int = 0
    episode_id: str | None = None
    episode_epoch: int | None = None
    scene_version: int | None = None


@dataclass(frozen=True)
class Detection2D:
    camera_id: str
    label: str
    box_xyxy: tuple[float, float, float, float]
    mask: ArtifactRef | None
    confidence: float
    evidence: tuple[ArtifactRef, ...] = ()


@dataclass(frozen=True)
class ObjectPoseEstimate:
    label: str
    pose_wxyz_xyz: tuple[float, ...]
    confidence: float
    covariance: ArtifactRef | None
    frame: str
    evidence: tuple[ArtifactRef, ...] = ()
    track_id: str | None = None


@dataclass(frozen=True)
class PointCloudRef:
    artifact: ArtifactRef
    frame: str
    point_count: int


@dataclass(frozen=True)
class BoundingBox3D:
    center_wxyz_xyz: tuple[float, ...]
    extents_xyz: tuple[float, float, float]
    frame: str
    confidence: float
    evidence: tuple[ArtifactRef, ...] = ()


@dataclass(frozen=True)
class GraspCandidate:
    pose_wxyz_xyz: tuple[float, ...]
    width_m: float
    score: float
    evidence: tuple[ArtifactRef, ...] = ()


@dataclass(frozen=True)
class PerceptionResult:
    request_id: str
    timestamp_ns: int
    detections_2d: tuple[Detection2D, ...] = ()
    poses_3d: tuple[ObjectPoseEstimate, ...] = ()
    artifacts: tuple[ArtifactRef, ...] = ()
    warnings: tuple[str, ...] = ()
    visual_evidence: tuple[VisualEvidence, ...] = ()


class ObservationProvider(Protocol):
    def capture(self) -> ObservationBundle: ...


class Vision2DBackend(Protocol):
    def segment_text(self, frame: CameraFrame, text_query: str) -> Sequence[Detection2D]: ...

    def segment_point(
        self,
        frame: CameraFrame,
        point: tuple[float, float],
    ) -> Sequence[Detection2D]: ...


class Geometry3DBackend(Protocol):
    def estimate_pose(
        self,
        frame: CameraFrame,
        detection: Detection2D,
    ) -> ObjectPoseEstimate | None: ...


class GraspProposalBackend(Protocol):
    def propose(
        self,
        observation: ObservationBundle,
        object_label: str,
    ) -> Sequence[GraspCandidate]: ...


class RobotControlBackend(Protocol):
    def goto_pose(
        self,
        position_xyz: tuple[float, float, float],
        quaternion_wxyz: tuple[float, float, float, float],
        z_approach: float = 0.0,
    ) -> None: ...

    def open_gripper(self) -> None: ...

    def close_gripper(self) -> None: ...


class FusedPerceptionBackend(Protocol):
    def infer(
        self,
        request: PerceptionRequest,
        observation: ObservationBundle,
    ) -> PerceptionResult: ...

    def publish_scene(
        self,
        observation: ObservationBundle,
        result: PerceptionResult,
        previous: SceneSnapshot | None,
    ) -> SceneSnapshot: ...


class PerceptionAgent(Protocol):
    """Raw-observation boundary for asynchronous semantic perception."""

    def perceive(
        self,
        request: PerceptionRequest,
        observation: ObservationBundle,
    ) -> PerceptionResult: ...
