from capmas.backends.capx import CAPXObservationProvider, CAPXRobotBackend, CAPXStreamingObservationSource
from capmas.perception.artifacts import InMemoryArtifactStore
from capmas.perception.metrics import RealTimeMetrics
from capmas.perception.protocol import ObservationBundle


def test_capx_streaming_source_adds_source_sequence_and_episode_metadata():
    provider = CAPXObservationProvider(
        observation_fn=lambda: {"timestamp_ns": 100},
        artifacts=InMemoryArtifactStore(),
    )
    source = CAPXStreamingObservationSource(
        provider,
        source="capx-libero",
        episode_id="ep-1",
        episode_epoch=4,
    )

    bundle = source.capture()

    assert bundle.source == "capx-libero"
    assert bundle.sequence == 1
    assert bundle.episode_id == "ep-1"
    assert bundle.episode_epoch == 4


def test_capx_streaming_source_carries_privileged_object_measurements():
    provider = CAPXObservationProvider(
        observation_fn=lambda: {"timestamp_ns": 100},
        artifacts=InMemoryArtifactStore(),
        object_poses_fn=lambda: {"cube": ([0.1, 0.2, 0.3], [1.0, 0.0, 0.0, 0.0])},
    )
    source = CAPXStreamingObservationSource(
        provider,
        episode_id="ep-1",
        episode_epoch=4,
    )

    bundle = source.capture()

    assert len(bundle.object_measurements) == 1
    assert bundle.object_measurements[0].track_id == "cube"
    assert bundle.object_measurements[0].pose_wxyz_xyz == (1.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3)


def test_capx_snapshot_records_processing_latency(monkeypatch):
    import capmas.backends.capx as capx_module

    class FakeEnv:
        def reset(self, *, seed=None, options=None):
            del seed, options

    class FakeObservationProvider:
        def capture(self) -> ObservationBundle:
            return ObservationBundle(100, (), {})

        def capture_object_tracks(self, *, timestamp_ns, episode_id, episode_epoch):
            del timestamp_ns, episode_id, episode_epoch
            return ()

    clock_values = iter((200,))
    monkeypatch.setattr("capmas.backends.capx.time.time_ns", lambda: next(clock_values))
    backend = CAPXRobotBackend(FakeEnv(), FakeObservationProvider(), task_id="task", suite_name="suite")

    episode = backend.reset(seed=1)

    assert episode.initial_scene.processing_latency_ms == 0.0001


def test_metrics_record_target_and_achieved_rates_and_drops():
    metrics = RealTimeMetrics(target_hz=10.0)
    metrics.record_observation(timestamp_ns=0)
    metrics.record_observation(timestamp_ns=100_000_000)
    metrics.record_drop()

    summary = metrics.summary(now_ns=200_000_000)

    assert summary.target_hz == 10.0
    assert summary.achieved_hz == 10.0
    assert summary.dropped_frames == 1
