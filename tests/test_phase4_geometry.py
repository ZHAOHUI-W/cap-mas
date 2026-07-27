from capmas.contracts.core import ArtifactRef
from capmas.perception.artifacts import InMemoryArtifactStore
from capmas.perception.geometry import ReferenceGeometryEstimator
from capmas.perception.protocol import CameraFrame, CameraModel, ObservationBundle


class SinglePixelDepthDecoder:
    def __init__(self, points: tuple[float, float, float]) -> None:
        self.points = points

    def decode(self, frame, depth, artifact_store):
        del frame, depth, artifact_store
        return (self.points,)


class FixedKinematicsBackend:
    def __init__(self, poses):
        self.poses = poses

    def camera_pose(self, robot_state, camera_id):
        del robot_state
        return self.poses.get(camera_id)


def make_frame(*, depth: ArtifactRef | None, pose_world=(1, 0, 0, 0, 0, 0, 0)) -> CameraFrame:
    camera = CameraModel("agentview", (1.0, 0.0, 0.0), tuple(pose_world))
    return CameraFrame(
        camera_id="agentview",
        timestamp_ns=100,
        rgb=None,
        depth=depth,
        camera=camera,
    )


def test_geometry_uses_camera_pose_from_frame_for_world_points(tmp_path):
    artifacts = InMemoryArtifactStore()
    depth = artifacts.put(b"0.5", "image/depth")
    frame = make_frame(depth=depth, pose_world=(1, 0, 0, 0, 1, 0, 0))
    estimator = ReferenceGeometryEstimator(
        artifact_store=artifacts,
        depth_decoder=SinglePixelDepthDecoder((0.0, 0.0, 0.5)),
    )

    update = estimator.estimate(ObservationBundle(100, (frame,), {}))

    assert update.points_world == ((1.0, 0.0, 0.5),)
    assert update.camera_poses[frame.camera_id] == frame.camera.pose_world


def test_geometry_uses_fk_backend_when_camera_pose_is_empty(tmp_path):
    artifacts = InMemoryArtifactStore()
    frame = make_frame(depth=None, pose_world=())
    fk = FixedKinematicsBackend({frame.camera_id: (1, 0, 0, 0, 1, 0, 0)})
    estimator = ReferenceGeometryEstimator(
        artifact_store=artifacts,
        depth_decoder=SinglePixelDepthDecoder((0.0, 0.0, 0.5)),
        kinematics=fk,
    )

    update = estimator.estimate(ObservationBundle(100, (frame,), {}))

    assert update.camera_poses[frame.camera_id] == fk.poses[frame.camera_id]


def test_geometry_does_not_emit_points_for_missing_depth_artifact(tmp_path):
    artifacts = InMemoryArtifactStore()
    frame = make_frame(depth=None)
    estimator = ReferenceGeometryEstimator(
        artifact_store=artifacts,
        depth_decoder=SinglePixelDepthDecoder((0.0, 0.0, 0.5)),
    )

    update = estimator.estimate(ObservationBundle(100, (frame,), {}))

    assert update.points_world == ()
