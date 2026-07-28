import numpy as np

from capmas.backends.capx_libero_factory import CAPXProcessWorldModelFactory
from capmas.contracts.core import ArtifactRef
from capmas.perception.artifact_bridge import EncodedArtifactStore, FileArtifactStore, NumpyArtifactCodec
from capmas.perception.protocol import CameraFrame, CameraModel, ObservationBundle
from capmas.perception.tracking import ObjectMeasurement


def test_process_world_model_factory_reads_shared_npy_depth_and_embedded_measurements(tmp_path):
    artifacts = EncodedArtifactStore(FileArtifactStore(tmp_path), NumpyArtifactCodec())
    depth_reference = artifacts.put(np.ones((2, 2), dtype=np.float32), "image/depth")
    frame = CameraFrame(
        camera_id="agentview",
        timestamp_ns=100,
        rgb=None,
        depth=depth_reference,
        camera=CameraModel(
            camera_id="agentview",
            intrinsics=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            pose_world=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ),
    )
    observation = ObservationBundle(
        timestamp_ns=100,
        frames=(frame,),
        robot_state={},
        episode_id="ep",
        episode_epoch=1,
        sequence=1,
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

    service = CAPXProcessWorldModelFactory(str(tmp_path), depth_subsample=1)()
    snapshot = service.process(observation, previous=None)

    assert snapshot.objects[0].track_id == "cube"
    assert snapshot.local_map is not None
    assert snapshot.source_artifacts == (depth_reference,)
