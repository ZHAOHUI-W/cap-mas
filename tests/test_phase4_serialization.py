import json

import pytest

from capmas.contracts.core import ArtifactRef
from capmas.perception.protocol import CameraFrame, CameraModel, ObservationBundle
from capmas.perception.serialization import (
    observation_from_json,
    observation_to_json,
    snapshot_from_json,
    snapshot_to_json,
)
from capmas.contracts.scene import ObjectTrack, SceneSnapshot


def make_observation_bundle() -> ObservationBundle:
    rgb = ArtifactRef("artifact://sha256/rgb", "image/rgb", "rgb", 3)
    depth = ArtifactRef("artifact://sha256/depth", "image/depth", "depth", 5)
    camera = CameraModel("agentview", (1.0, 0.0, 2.0), (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    frame = CameraFrame("agentview", 100, rgb, depth, camera)
    return ObservationBundle(
        100,
        (frame,),
        {"gripper_opening": 0.2, "joint": rgb},
        "episode-1",
        2,
        "replay",
        7,
    )


def test_observation_envelope_round_trip_preserves_camera_and_artifact_refs():
    bundle = make_observation_bundle()

    encoded = observation_to_json(bundle)
    restored = observation_from_json(encoded)

    assert restored == bundle


def test_observation_envelope_rejects_scene_version_and_unknown_fields():
    payload = json.loads(observation_to_json(make_observation_bundle()))
    payload["scene_version"] = 4
    with pytest.raises(ValueError, match="unknown fields"):
        observation_from_json(json.dumps(payload))

    payload = json.loads(observation_to_json(make_observation_bundle()))
    payload["unknown"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        observation_from_json(json.dumps(payload))


def test_snapshot_envelope_round_trip_preserves_phase4_fields():
    snapshot = SceneSnapshot(
        "episode-1",
        2,
        3,
        100,
        120,
        {"gripper_opening": 0.2},
        objects=(
            ObjectTrack(
                "cube-1",
                "cube",
                (1.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3),
                0.9,
                100,
                velocity_xyz=(0.1, 0.0, 0.0),
                prediction_timestamp_ns=110,
                track_status="predicted",
            ),
        ),
        processing_latency_ms=2.5,
    )

    restored = snapshot_from_json(snapshot_to_json(snapshot))

    assert restored == snapshot
