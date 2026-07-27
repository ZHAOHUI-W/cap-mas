from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Protocol, Sequence

from capmas.contracts.core import ArtifactRef
from capmas.perception.protocol import CameraFrame, ObservationBundle


@dataclass(frozen=True)
class GeometryUpdate:
    timestamp_ns: int
    camera_poses: Mapping[str, tuple[float, ...]]
    points_world: tuple[tuple[float, float, float], ...] = ()
    source_artifacts: tuple[ArtifactRef, ...] = ()


class KinematicsBackend(Protocol):
    def camera_pose(
        self,
        robot_state: Mapping[str, object],
        camera_id: str,
    ) -> tuple[float, ...] | None: ...


class DepthDecoder(Protocol):
    def decode(
        self,
        frame: CameraFrame,
        depth: ArtifactRef,
        artifact_store: object,
    ) -> Sequence[tuple[float, float, float]]: ...


class GeometryEstimator(Protocol):
    def estimate(self, observation: ObservationBundle) -> GeometryUpdate: ...


class ReferenceGeometryEstimator:
    def __init__(
        self,
        *,
        artifact_store: object,
        depth_decoder: DepthDecoder,
        kinematics: KinematicsBackend | None = None,
    ) -> None:
        self.artifact_store = artifact_store
        self.depth_decoder = depth_decoder
        self.kinematics = kinematics

    def estimate(self, observation: ObservationBundle) -> GeometryUpdate:
        camera_poses: dict[str, tuple[float, ...]] = {}
        points_world: list[tuple[float, float, float]] = []
        source_artifacts: list[ArtifactRef] = []
        for frame in observation.frames:
            for reference in (frame.rgb, frame.depth):
                if reference is not None and reference not in source_artifacts:
                    source_artifacts.append(reference)
            pose = self._resolve_pose(frame, observation.robot_state)
            if pose is None:
                continue
            camera_poses[frame.camera_id] = pose
            if frame.depth is None:
                continue
            for point in self.depth_decoder.decode(frame, frame.depth, self.artifact_store):
                _validate_point(point)
                points_world.append(_transform_point(pose, point))
        return GeometryUpdate(
            timestamp_ns=observation.timestamp_ns,
            camera_poses=camera_poses,
            points_world=tuple(points_world),
            source_artifacts=tuple(source_artifacts),
        )

    def _resolve_pose(
        self,
        frame: CameraFrame,
        robot_state: Mapping[str, object],
    ) -> tuple[float, ...] | None:
        pose = tuple(frame.camera.pose_world)
        if not pose and self.kinematics is not None:
            resolved = self.kinematics.camera_pose(robot_state, frame.camera_id)
            pose = tuple(resolved or ())
        if not pose:
            return None
        _validate_pose(pose)
        return pose


def _validate_pose(pose: tuple[float, ...]) -> None:
    if len(pose) not in (7, 16):
        raise ValueError("camera pose must be a quaternion pose or 4x4 matrix")
    if not all(math.isfinite(value) for value in pose):
        raise ValueError("camera pose contains a non-finite value")
    if len(pose) == 7:
        norm = math.sqrt(sum(value * value for value in pose[:4]))
        if norm == 0.0:
            raise ValueError("camera quaternion must be non-zero")


def _validate_point(point: Sequence[float]) -> None:
    if len(point) != 3 or not all(math.isfinite(float(value)) for value in point):
        raise ValueError("depth decoder emitted a malformed point")


def _transform_point(pose: tuple[float, ...], point: Sequence[float]) -> tuple[float, float, float]:
    x, y, z = (float(value) for value in point)
    if len(pose) == 16:
        world_x = pose[0] * x + pose[1] * y + pose[2] * z + pose[3]
        world_y = pose[4] * x + pose[5] * y + pose[6] * z + pose[7]
        world_z = pose[8] * x + pose[9] * y + pose[10] * z + pose[11]
        homogeneous = pose[12] * x + pose[13] * y + pose[14] * z + pose[15]
        if homogeneous == 0.0:
            raise ValueError("camera transform produced zero homogeneous coordinate")
        return (world_x / homogeneous, world_y / homogeneous, world_z / homogeneous)

    qw, qx, qy, qz, tx, ty, tz = pose
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = (value / norm for value in (qw, qx, qy, qz))
    return (
        (1 - 2 * (qy * qy + qz * qz)) * x
        + 2 * (qx * qy - qz * qw) * y
        + 2 * (qx * qz + qy * qw) * z
        + tx,
        2 * (qx * qy + qz * qw) * x
        + (1 - 2 * (qx * qx + qz * qz)) * y
        + 2 * (qy * qz - qx * qw) * z
        + ty,
        2 * (qx * qz - qy * qw) * x
        + 2 * (qy * qz + qx * qw) * y
        + (1 - 2 * (qx * qx + qy * qy)) * z
        + tz,
    )
