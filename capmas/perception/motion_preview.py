"""Side-effect-free candidate motion preview interfaces and reference backend."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol

from capmas.contracts.core import ArtifactRef
from capmas.contracts.graph import MotionIntent
from capmas.contracts.scene import ObjectTrack, SceneSnapshot
from capmas.perception.effective_motion import EffectiveMotionProgram, EffectiveMotionSegment
from capmas.perception.local_map import LocalMapBackend, MapRegion


@dataclass(frozen=True)
class MotionPreview:
    status: Literal["feasible", "infeasible", "unknown"]
    target_pose_wxyz_xyz: tuple[float, ...] | None = None
    trajectory_ref: ArtifactRef | None = None
    ik_valid: bool | None = None
    collision_free: bool | None = None
    clearance_m: float | None = None
    path_length_m: float | None = None
    reason: str = ""
    backend: str = ""
    backend_version: str = ""


@dataclass(frozen=True)
class MapQueryDiagnostic:
    """One sparse-map query retained for no-submit collision diagnosis."""

    point_xyz: tuple[float, float, float]
    map_version: int
    occupied: bool
    clearance_m: float | None
    confidence: float
    snapshot_timestamp_ns: int


@dataclass(frozen=True)
class SegmentMotionPreview:
    """Read-only map/IK result for one bound motion segment."""

    segment_id: str
    ik_valid: bool | None
    collision_free: bool | None
    clearance_m: float | None
    path_length_m: float | None
    reason: str
    start_pose_wxyz_xyz: tuple[float, ...] | None
    end_pose_wxyz_xyz: tuple[float, ...] | None
    sampled_points_xyz: tuple[tuple[float, float, float], ...] = ()
    occupied_points_xyz: tuple[tuple[float, float, float], ...] = ()
    map_queries: tuple[MapQueryDiagnostic, ...] = ()


@dataclass(frozen=True)
class ProgramMotionPreview:
    """Segment-level preview plus conservative program-level status."""

    segments: tuple[SegmentMotionPreview, ...]
    aggregate_status: Literal["feasible", "infeasible", "unknown"]
    scene_version: int
    map_version: int | None
    corridor_radius_m: float

    def by_segment(self, segment_id: str) -> SegmentMotionPreview:
        for segment in self.segments:
            if segment.segment_id == segment_id:
                return segment
        raise KeyError(f"unknown program preview segment: {segment_id}")


class MotionPreviewBackend(Protocol):
    def preview(
        self,
        intent: MotionIntent,
        scene: SceneSnapshot,
        local_map: LocalMapBackend | None,
    ) -> MotionPreview: ...


class ReferenceMotionPreview:
    """Conservative workspace and sparse-map preview without robot side effects."""

    backend = "reference_motion_preview"
    backend_version = "1"

    def __init__(
        self,
        *,
        workspace_bounds: tuple[tuple[float, float], ...] = (
            (-1.0, 1.0),
            (-1.0, 1.0),
            (0.0, 2.0),
        ),
        target_freshness_ms: float = 250.0,
        track_stale_ns: int = 500_000_000,
        corridor_radius_m: float = 0.025,
        approach_distance_m: float = 0.15,
        clearance_threshold: float = 0.5,
        collision_risk_threshold: float = 0.5,
        corridor_samples: int = 5,
    ) -> None:
        if len(workspace_bounds) != 3:
            raise ValueError("workspace bounds must contain three axes")
        if any(low >= high for low, high in workspace_bounds):
            raise ValueError("workspace bounds must have increasing limits")
        if target_freshness_ms < 0 or track_stale_ns < 0:
            raise ValueError("freshness limits must not be negative")
        if corridor_radius_m <= 0 or approach_distance_m <= 0:
            raise ValueError("preview distances must be positive")
        if corridor_samples <= 0:
            raise ValueError("corridor samples must be positive")
        self.workspace_bounds = workspace_bounds
        self.target_freshness_ms = target_freshness_ms
        self.track_stale_ns = track_stale_ns
        self.corridor_radius_m = corridor_radius_m
        self.approach_distance_m = approach_distance_m
        self.clearance_threshold = clearance_threshold
        self.collision_risk_threshold = collision_risk_threshold
        self.corridor_samples = corridor_samples

    def preview(
        self,
        intent: MotionIntent,
        scene: SceneSnapshot,
        local_map: LocalMapBackend | None,
    ) -> MotionPreview:
        if scene.freshness_ms > self.target_freshness_ms:
            return self._unknown("scene is stale")

        track = self._target_track(intent, scene)
        if intent.target_pose_wxyz_xyz is not None:
            target_pose = intent.target_pose_wxyz_xyz
        elif track is not None:
            target_pose = tuple(track.pose_wxyz_xyz)
        else:
            return self._unknown("target track is unresolved")

        if track is not None and scene.sensor_timestamp_ns - track.last_seen_ns > self.track_stale_ns:
            return self._unknown("target track is stale")
        if len(target_pose) != 7 or not all(math.isfinite(value) for value in target_pose):
            return self._unknown("target pose is unavailable or malformed")

        position = tuple(target_pose[4:7])
        if not self._in_workspace(position):
            return MotionPreview(
                status="infeasible",
                target_pose_wxyz_xyz=target_pose,
                ik_valid=False,
                collision_free=None,
                clearance_m=None,
                reason="target pose is outside conservative workspace",
                backend=self.backend,
                backend_version=self.backend_version,
            )

        approach = _normalize(intent.approach_vector_xyz)
        if approach is None:
            return MotionPreview(
                status="feasible",
                target_pose_wxyz_xyz=target_pose,
                ik_valid=True,
                collision_free=None,
                clearance_m=None,
                reason="approach vector is unavailable; map corridor not evaluated",
                backend=self.backend,
                backend_version=self.backend_version,
            )
        if local_map is None:
            return MotionPreview(
                status="feasible",
                target_pose_wxyz_xyz=target_pose,
                ik_valid=True,
                collision_free=None,
                path_length_m=self.approach_distance_m,
                reason="local map is unavailable",
                backend=self.backend,
                backend_version=self.backend_version,
            )

        collision_free = True
        min_clearance: float | None = None
        for index in range(1, self.corridor_samples + 1):
            distance = self.approach_distance_m * index / self.corridor_samples
            point = tuple(position[axis] - approach[axis] * distance for axis in range(3))
            result = local_map.query(
                MapRegion(
                    center_xyz=point,
                    extents_xyz=(self.corridor_radius_m,) * 3,
                )
            )
            if result.snapshot_timestamp_ns and result.snapshot_timestamp_ns < scene.sensor_timestamp_ns:
                return self._unknown("local map is stale")
            if result.occupied:
                collision_free = False
                min_clearance = 0.0
                break
            if result.clearance_m is not None:
                min_clearance = (
                    result.clearance_m
                    if min_clearance is None
                    else min(min_clearance, result.clearance_m)
                )
        if min_clearance is None:
            return self._unknown("local map has no clearance measurement")
        reason = "corridor is occupied" if not collision_free else "corridor is clear"
        return MotionPreview(
            status="feasible" if collision_free else "infeasible",
            target_pose_wxyz_xyz=target_pose,
            ik_valid=True,
            collision_free=collision_free,
            clearance_m=min_clearance,
            path_length_m=self.approach_distance_m,
            reason=reason,
            backend=self.backend,
            backend_version=self.backend_version,
        )

    def preview_program(
        self,
        program: EffectiveMotionProgram,
        scene: SceneSnapshot,
        local_map: LocalMapBackend | None,
    ) -> ProgramMotionPreview:
        """Preview every explicit program segment without robot side effects."""

        if scene.freshness_ms > self.target_freshness_ms:
            segments = tuple(
                self._unknown_segment(segment, "scene is stale")
                for segment in program.segments
            )
        else:
            segments = tuple(
                self._preview_segment(segment, scene, local_map)
                for segment in program.segments
            )
        if any(
            segment.ik_valid is False or segment.collision_free is False
            for segment in segments
        ):
            status: Literal["feasible", "infeasible", "unknown"] = "infeasible"
        elif segments and all(
            segment.ik_valid is True and segment.collision_free is True
            for segment in segments
        ):
            status = "feasible"
        else:
            status = "unknown"
        return ProgramMotionPreview(
            segments=segments,
            aggregate_status=status,
            scene_version=scene.scene_version,
            map_version=local_map.map_version() if local_map is not None else None,
            corridor_radius_m=self.corridor_radius_m,
        )

    def _target_track(self, intent: MotionIntent, scene: SceneSnapshot) -> ObjectTrack | None:
        track_id = intent.object_track_id if intent.kind == "grasp" else intent.target_track_id
        if track_id is None:
            track_id = intent.object_track_id
        if track_id is None:
            return None
        return next((track for track in scene.objects if track.track_id == track_id), None)

    def _preview_segment(
        self,
        segment: EffectiveMotionSegment,
        scene: SceneSnapshot,
        local_map: LocalMapBackend | None,
    ) -> SegmentMotionPreview:
        if segment.start_pose_wxyz_xyz is None or segment.end_pose_wxyz_xyz is None:
            return self._unknown_segment(segment, "segment endpoints are unavailable")
        start = tuple(segment.start_pose_wxyz_xyz[4:7])
        end = tuple(segment.end_pose_wxyz_xyz[4:7])
        if not self._in_workspace(start) or not self._in_workspace(end):
            return SegmentMotionPreview(
                segment.segment_id,
                False,
                None,
                None,
                _distance(start, end),
                "segment endpoint is outside conservative workspace",
                segment.start_pose_wxyz_xyz,
                segment.end_pose_wxyz_xyz,
            )
        samples = _line_samples(start, end, max(2, self.corridor_samples))
        path_length = _distance(start, end)
        if local_map is None:
            return SegmentMotionPreview(
                segment.segment_id,
                True,
                None,
                None,
                path_length,
                "local map is unavailable",
                segment.start_pose_wxyz_xyz,
                segment.end_pose_wxyz_xyz,
                samples,
            )

        minimum_clearance: float | None = None
        occupied: list[tuple[float, float, float]] = []
        queries: list[MapQueryDiagnostic] = []
        for point in samples:
            result = local_map.query(
                MapRegion(
                    center_xyz=point,
                    extents_xyz=(self.corridor_radius_m,) * 3,
                )
            )
            queries.append(
                MapQueryDiagnostic(
                    point_xyz=point,
                    map_version=result.map_version,
                    occupied=result.occupied,
                    clearance_m=0.0 if result.occupied else result.clearance_m,
                    confidence=result.confidence,
                    snapshot_timestamp_ns=result.snapshot_timestamp_ns,
                )
            )
            if result.snapshot_timestamp_ns and result.snapshot_timestamp_ns < scene.sensor_timestamp_ns:
                return self._unknown_segment(
                    segment,
                    "local map is stale",
                    path_length=path_length,
                    samples=samples,
                    map_queries=tuple(queries),
                )
            if result.occupied:
                occupied.append(point)
                minimum_clearance = 0.0
                continue
            if result.clearance_m is not None:
                minimum_clearance = (
                    result.clearance_m
                    if minimum_clearance is None
                    else min(minimum_clearance, result.clearance_m)
                )
        if minimum_clearance is None:
            return self._unknown_segment(
                segment,
                "local map has no clearance measurement",
                path_length=path_length,
                samples=samples,
            )
        collision_free = not occupied
        return SegmentMotionPreview(
            segment.segment_id,
            True,
            collision_free,
            minimum_clearance,
            path_length,
            "corridor is clear" if collision_free else "corridor is occupied",
            segment.start_pose_wxyz_xyz,
            segment.end_pose_wxyz_xyz,
            samples,
            tuple(occupied),
            tuple(queries),
        )

    @staticmethod
    def _unknown_segment(
        segment: EffectiveMotionSegment,
        reason: str,
        *,
        path_length: float | None = None,
        samples: tuple[tuple[float, float, float], ...] = (),
        map_queries: tuple[MapQueryDiagnostic, ...] = (),
    ) -> SegmentMotionPreview:
        return SegmentMotionPreview(
            segment.segment_id,
            None,
            None,
            None,
            path_length,
            reason,
            segment.start_pose_wxyz_xyz,
            segment.end_pose_wxyz_xyz,
            samples,
            (),
            map_queries,
        )

    def _in_workspace(self, position: tuple[float, float, float]) -> bool:
        return all(
            bounds[0] <= position[index] <= bounds[1]
            for index, bounds in enumerate(self.workspace_bounds)
        )

    def _unknown(self, reason: str) -> MotionPreview:
        return MotionPreview(
            status="unknown",
            reason=reason,
            backend=self.backend,
            backend_version=self.backend_version,
        )


def _normalize(vector: tuple[float, float, float] | None) -> tuple[float, float, float] | None:
    if vector is None:
        return None
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1e-9:
        return None
    return tuple(value / norm for value in vector)


def _distance(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> float:
    return math.sqrt(sum((first[index] - second[index]) ** 2 for index in range(3)))


def _line_samples(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    count: int,
) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        tuple(
            start[axis] + (end[axis] - start[axis]) * index / (count - 1)
            for axis in range(3)
        )
        for index in range(count)
    )
