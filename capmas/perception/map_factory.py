"""Explicit local-map backend configuration and construction boundary.

The factory is intentionally fail-closed.  SparseVoxelMap is the only active
backend in the reference path; selecting TSDF before its implementation exists
must raise a typed error instead of silently changing the requested backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

from capmas.perception.local_map import LocalMapBackend, SparseVoxelMap, validate_map_config


class UnsupportedMapBackendError(ValueError):
    """Raised when a requested map backend is not available."""


@dataclass(frozen=True)
class MapBackendConfig:
    backend: str = "sparse_voxel"
    voxel_size_m: float = 0.01
    local_radius_m: float = 1.0
    tsdf_enabled: bool = False

    @classmethod
    def from_mapping(cls, config: Mapping[str, object] | "MapBackendConfig") -> "MapBackendConfig":
        if isinstance(config, cls):
            return config
        backend = config.get("backend", "sparse_voxel")
        if not isinstance(backend, str) or not backend:
            raise ValueError("map backend must be a non-empty string")
        voxel_size = config.get("voxel_size_m", 0.01)
        local_radius = config.get("local_radius_m", 1.0)
        if not isinstance(voxel_size, (int, float)):
            raise ValueError("voxel_size_m must be numeric")
        if not isinstance(local_radius, (int, float)):
            raise ValueError("local_radius_m must be numeric")
        tsdf = config.get("tsdf", {})
        if not isinstance(tsdf, Mapping):
            raise ValueError("tsdf configuration must be an object")
        tsdf_enabled = tsdf.get("enabled", False)
        if not isinstance(tsdf_enabled, bool):
            raise ValueError("tsdf.enabled must be boolean")
        normalized = {
            "backend": backend,
            "voxel_size_m": float(voxel_size),
            "local_radius_m": float(local_radius),
            "tsdf": dict(tsdf),
        }
        try:
            validate_map_config(normalized)
        except ValueError as exc:
            message = str(exc)
            if backend != "sparse_voxel" or tsdf_enabled:
                raise UnsupportedMapBackendError(
                    f"requested map backend {backend!r} is unavailable: {message}"
                ) from exc
            raise
        return cls(
            backend=backend,
            voxel_size_m=float(voxel_size),
            local_radius_m=float(local_radius),
            tsdf_enabled=tsdf_enabled,
        )


def build_local_map_backend(
    config: Mapping[str, object] | MapBackendConfig,
) -> LocalMapBackend:
    """Build the configured backend without silently falling back."""

    parsed = MapBackendConfig.from_mapping(config)
    if parsed.backend != "sparse_voxel" or parsed.tsdf_enabled:
        raise UnsupportedMapBackendError(
            f"map backend {parsed.backend!r} is unavailable; TSDF remains disabled"
        )
    return SparseVoxelMap(
        voxel_size_m=parsed.voxel_size_m,
        local_radius_m=parsed.local_radius_m,
    )
