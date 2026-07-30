import pytest

from capmas.perception.local_map import SparseVoxelMap
from capmas.perception.map_factory import (
    MapBackendConfig,
    UnsupportedMapBackendError,
    build_local_map_backend,
)


def test_sparse_backend_factory_preserves_configured_geometry_limits():
    config = MapBackendConfig.from_mapping(
        {
            "backend": "sparse_voxel",
            "voxel_size_m": 0.02,
            "local_radius_m": 1.5,
        }
    )

    backend = build_local_map_backend(config)

    assert isinstance(backend, SparseVoxelMap)
    assert backend.voxel_size_m == 0.02
    assert backend.local_radius_m == 1.5


def test_factory_rejects_unknown_backend_without_silent_fallback():
    with pytest.raises(UnsupportedMapBackendError, match="tsdf"):
        build_local_map_backend({"backend": "tsdf"})


def test_factory_rejects_explicit_tsdf_enablement():
    with pytest.raises(UnsupportedMapBackendError, match="TSDF"):
        MapBackendConfig.from_mapping(
            {"backend": "sparse_voxel", "tsdf": {"enabled": True}}
        )
