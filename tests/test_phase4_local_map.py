from capmas.perception.geometry import GeometryUpdate
import pytest

from capmas.perception.local_map import MapRegion, SparseVoxelMap, validate_map_config


def make_geometry(*, points):
    return GeometryUpdate(timestamp_ns=100, camera_poses={}, points_world=tuple(points))


def test_integrate_changes_only_voxels_touched_by_new_geometry():
    voxel_map = SparseVoxelMap(voxel_size_m=0.1, local_radius_m=1.0)

    first = voxel_map.integrate(make_geometry(points=((0.04, 0.04, 0.04),)), 100)
    second = voxel_map.integrate(make_geometry(points=((0.24, 0.04, 0.04),)), 200)

    assert first.changed_voxels == ((0, 0, 0),)
    assert second.changed_voxels == ((2, 0, 0),)
    assert second.map_version == first.map_version + 1


def test_query_reports_occupied_voxel_and_map_version():
    voxel_map = SparseVoxelMap(voxel_size_m=0.1, local_radius_m=1.0)
    voxel_map.integrate(make_geometry(points=((0.04, 0.04, 0.04),)), 100)

    result = voxel_map.query(MapRegion(center_xyz=(0, 0, 0), extents_xyz=(0.2, 0.2, 0.2)))

    assert result.map_version == 1
    assert result.occupied is True
    assert result.confidence > 0.0
    assert result.snapshot_timestamp_ns == 100


def test_freeze_snapshot_is_immutable_and_does_not_change_after_next_update():
    voxel_map = SparseVoxelMap(voxel_size_m=0.1, local_radius_m=1.0)
    voxel_map.integrate(make_geometry(points=((0.04, 0.04, 0.04),)), 100)
    frozen = voxel_map.freeze_snapshot()
    voxel_map.integrate(make_geometry(points=((0.24, 0.04, 0.04),)), 200)

    assert frozen.map_version == 1
    assert frozen.voxel_size_m == 0.1
    assert frozen.source_timestamp_ns == 100
    assert frozen.occupied_voxels == ((0, 0, 0),)


def test_points_outside_local_radius_are_excluded():
    voxel_map = SparseVoxelMap(voxel_size_m=0.1, local_radius_m=1.0)

    update = voxel_map.integrate(make_geometry(points=((1.1, 0, 0),)), 100)

    assert update.changed_voxels == ()
    assert voxel_map.freeze_snapshot().occupied_voxels == ()


def test_tsdf_configuration_is_rejected_until_backend_exists():
    validate_map_config(
        {
            "backend": "sparse_voxel",
            "voxel_size_m": 0.01,
            "local_radius_m": 1.0,
            "tsdf": {"enabled": False, "truncation_distance_m": 0.04},
        }
    )

    with pytest.raises(ValueError, match="TSDF"):
        validate_map_config(
            {
                "backend": "sparse_voxel",
                "voxel_size_m": 0.01,
                "local_radius_m": 1.0,
                "tsdf": {"enabled": True, "truncation_distance_m": 0.04},
            }
        )
