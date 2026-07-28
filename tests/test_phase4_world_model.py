import pytest

from capmas.contracts.scene import ObjectTrack, SceneSnapshot
from capmas.perception.artifacts import InMemoryArtifactStore
from capmas.perception.geometry import ReferenceGeometryEstimator
from capmas.perception.local_map import SparseVoxelMap
from capmas.perception.protocol import ObservationBundle
from capmas.perception.tracking import KnownObjectTracker
from capmas.perception.tracking import ObjectMeasurement
from capmas.perception.world_model import WorldModelService


class EmptyDepthDecoder:
    def decode(self, frame, depth, artifact_store):
        del frame, depth, artifact_store
        return ()


class SequenceClock:
    def __init__(self, value: int) -> None:
        self.value = value

    def __call__(self) -> int:
        return self.value


def make_world_model(*, clock=None) -> WorldModelService:
    artifacts = InMemoryArtifactStore()
    geometry = ReferenceGeometryEstimator(
        artifact_store=artifacts,
        depth_decoder=EmptyDepthDecoder(),
    )
    return WorldModelService(
        geometry=geometry,
        local_map=SparseVoxelMap(voxel_size_m=0.1, local_radius_m=1.0),
        tracker=KnownObjectTracker(max_match_distance_m=0.2),
        clock=clock or (lambda: 200),
    )


def make_observation(*, episode_id: str = "ep", episode_epoch: int = 1) -> ObservationBundle:
    return ObservationBundle(
        timestamp_ns=100,
        frames=(),
        robot_state={"gripper_opening": 1.0},
        episode_id=episode_id,
        episode_epoch=episode_epoch,
        source="test",
        sequence=1,
    )


def make_scene(*, scene_version: int = 4, episode_id: str = "ep", episode_epoch: int = 1, objects=()):
    return SceneSnapshot(
        episode_id,
        episode_epoch,
        scene_version,
        100,
        150,
        {},
        objects=objects,
    )


def test_world_model_process_publishes_new_snapshot_with_latency():
    service = make_world_model(clock=SequenceClock(200))
    previous = make_scene(scene_version=4)

    snapshot = service.process(make_observation(), previous)

    assert snapshot.scene_version == 5
    assert snapshot.processing_latency_ms == 0.0001
    assert snapshot.sensor_timestamp_ns == 100


def test_world_model_process_does_not_mutate_previous_snapshot():
    service = make_world_model(clock=SequenceClock(200))
    previous_track = ObjectTrack(
        "cube-1",
        "cube",
        (1.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0),
        0.9,
        100,
    )
    previous = make_scene(scene_version=4, objects=(previous_track,))

    snapshot = service.process(make_observation(), previous)

    assert previous.scene_version == 4
    assert tuple(previous.objects) == (previous_track,)
    assert snapshot is not previous


def test_world_model_rejects_cross_episode_observation():
    service = make_world_model()
    previous = make_scene(episode_id="episode-a", episode_epoch=1)

    with pytest.raises(ValueError, match="episode"):
        service.process(make_observation(episode_id="episode-b"), previous)


def test_world_model_uses_measurements_embedded_in_observation_without_provider():
    service = make_world_model()
    observation = ObservationBundle(
        timestamp_ns=100,
        frames=(),
        robot_state={},
        episode_id="ep",
        episode_epoch=1,
        object_measurements=(
            ObjectMeasurement(
                track_id="cube",
                label="cube",
                pose_wxyz_xyz=(1.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3),
                confidence=1.0,
                timestamp_ns=100,
            ),
        ),
    )

    snapshot = service.process(observation, previous=None)

    assert snapshot.objects[0].track_id == "cube"
