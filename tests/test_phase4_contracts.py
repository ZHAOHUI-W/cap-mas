from capmas.contracts.scene import ObjectTrack, SceneSnapshot
from capmas.perception.protocol import ObservationBundle


def test_observation_bundle_keeps_legacy_positional_constructor() -> None:
    bundle = ObservationBundle(100, (), {"gripper_opening": 1.0})

    assert bundle.timestamp_ns == 100
    assert bundle.frames == ()
    assert bundle.episode_id is None
    assert bundle.episode_epoch is None
    assert bundle.source == ""
    assert bundle.sequence == 0


def test_scene_and_track_phase4_fields_have_compatibility_defaults() -> None:
    scene = SceneSnapshot("episode", 1, 0, 100, 110, {})
    track = ObjectTrack("obj", "cube", (1, 0, 0, 0, 0, 0, 0), 0.9, 100)

    assert scene.processing_latency_ms == 0.0
    assert track.velocity_xyz is None
    assert track.prediction_timestamp_ns is None
    assert track.track_status == "observed"
