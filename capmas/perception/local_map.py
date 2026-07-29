from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Protocol

from capmas.perception.geometry import GeometryUpdate


VoxelKey = tuple[int, int, int]


@dataclass(frozen=True)
class MapRegion:
    center_xyz: tuple[float, float, float]
    extents_xyz: tuple[float, float, float]


@dataclass(frozen=True)
class MapUpdate:
    map_version: int
    changed_voxels: tuple[VoxelKey, ...]
    source_timestamp_ns: int


@dataclass(frozen=True)
class MapQueryResult:
    map_version: int
    occupied: bool
    clearance_m: float | None
    confidence: float
    snapshot_timestamp_ns: int


@dataclass(frozen=True)
class MapSnapshot:
    map_version: int
    voxel_size_m: float
    source_timestamp_ns: int
    occupied_voxels: tuple[VoxelKey, ...]
    dirty_blocks: tuple[VoxelKey, ...] = ()


class LocalMapBackend(Protocol):
    """Read-only map seam used by candidate motion preview."""

    def query(self, region: MapRegion) -> MapQueryResult: ...

    def map_version(self) -> int: ...


class SparseVoxelMap:
    def __init__(
        self,
        voxel_size_m: float,
        local_radius_m: float,
        *,
        origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        if voxel_size_m <= 0.0 or not math.isfinite(voxel_size_m):
            raise ValueError("voxel_size_m must be finite and positive")
        if local_radius_m <= 0.0 or not math.isfinite(local_radius_m):
            raise ValueError("local_radius_m must be finite and positive")
        if len(origin_xyz) != 3 or not all(math.isfinite(value) for value in origin_xyz):
            raise ValueError("origin_xyz must contain three finite values")
        self.voxel_size_m = voxel_size_m
        self.local_radius_m = local_radius_m
        self.origin_xyz = origin_xyz
        self._occupied: dict[VoxelKey, float] = {}
        self._map_version = 0
        self._source_timestamp_ns = 0

    def integrate(self, geometry: GeometryUpdate, timestamp_ns: int) -> MapUpdate:
        changed: set[VoxelKey] = set()
        for point in geometry.points_world:
            if len(point) != 3 or not all(math.isfinite(value) for value in point):
                raise ValueError("geometry contains a malformed point")
            if _distance(point, self.origin_xyz) > self.local_radius_m:
                continue
            key = self._key(point)
            self._occupied[key] = min(1.0, self._occupied.get(key, 0.0) + 0.1)
            changed.add(key)
        self._map_version += 1
        self._source_timestamp_ns = timestamp_ns
        return MapUpdate(self._map_version, tuple(sorted(changed)), timestamp_ns)

    def query(self, region: MapRegion) -> MapQueryResult:
        if len(region.extents_xyz) != 3 or not all(
            math.isfinite(value) and value >= 0.0 for value in region.extents_xyz
        ):
            raise ValueError("region.extents_xyz must contain finite non-negative values")
        matches = tuple(
            sorted(
                key
                for key in self._occupied
                if all(
                    abs(self._voxel_center(key)[index] - region.center_xyz[index])
                    <= region.extents_xyz[index]
                    for index in range(3)
                )
            )
        )
        confidence = max((self._occupied[key] for key in matches), default=0.0)
        clearance = None if matches else min(
            (_distance(self._voxel_center(key), region.center_xyz) for key in self._occupied),
            default=None,
        )
        return MapQueryResult(
            map_version=self._map_version,
            occupied=bool(matches),
            clearance_m=clearance,
            confidence=confidence,
            snapshot_timestamp_ns=self._source_timestamp_ns,
        )

    def map_version(self) -> int:
        return self._map_version

    def freeze_snapshot(self) -> MapSnapshot:
        return MapSnapshot(
            map_version=self._map_version,
            voxel_size_m=self.voxel_size_m,
            source_timestamp_ns=self._source_timestamp_ns,
            occupied_voxels=tuple(sorted(self._occupied)),
            dirty_blocks=tuple(sorted(self._occupied)),
        )

    def _key(self, point: tuple[float, float, float]) -> VoxelKey:
        return tuple(
            math.floor((point[index] - self.origin_xyz[index]) / self.voxel_size_m)
            for index in range(3)
        )  # type: ignore[return-value]

    def _voxel_center(self, key: VoxelKey) -> tuple[float, float, float]:
        return tuple(
            self.origin_xyz[index] + (key[index] + 0.5) * self.voxel_size_m
            for index in range(3)
        )  # type: ignore[return-value]


def _distance(first: Iterable[float], second: Iterable[float]) -> float:
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(first, second)))


def validate_map_config(config: Mapping[str, object]) -> None:
    """Validate Phase 4 map settings without silently enabling TSDF."""
    if config.get("backend", "sparse_voxel") != "sparse_voxel":
        raise ValueError("only sparse_voxel is available in the reference backend")
    voxel_size = config.get("voxel_size_m", 0.01)
    local_radius = config.get("local_radius_m", 1.0)
    if not isinstance(voxel_size, (int, float)) or voxel_size <= 0:
        raise ValueError("voxel_size_m must be positive")
    if not isinstance(local_radius, (int, float)) or local_radius <= 0:
        raise ValueError("local_radius_m must be positive")
    tsdf = config.get("tsdf", {})
    if not isinstance(tsdf, Mapping):
        raise ValueError("tsdf configuration must be an object")
    if tsdf.get("enabled", False) is True:
        raise ValueError("TSDF is reserved and disabled in Phase 4")
    truncation = tsdf.get("truncation_distance_m", 0.04)
    if not isinstance(truncation, (int, float)) or truncation <= 0:
        raise ValueError("truncation_distance_m must be positive")
