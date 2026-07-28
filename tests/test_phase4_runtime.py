from __future__ import annotations

import time

from capmas.contracts.scene import SceneSnapshot
from capmas.perception.protocol import ObservationBundle
from capmas.perception.sensor_sync import BoundedSensorSynchronizer
from capmas.perception.world_model import (
    ProcessWorldModelRuntime,
    SynchronizerConfig,
    TransportAcknowledgement,
    ThreadWorldModelRuntime,
    WorldModelRuntimeConfig,
)
from capmas.perception.artifact_bridge import FileArtifactStore


def make_observation(sequence: int = 1) -> ObservationBundle:
    return ObservationBundle(
        timestamp_ns=100 + sequence,
        frames=(),
        robot_state={},
        episode_id="ep",
        episode_epoch=1,
        source="test",
        sequence=sequence,
    )


def top_level_world_model_factory():
    return SimpleWorldModelService()


def always_crashing_factory():
    raise RuntimeError("factory crashed")


class FailingObservationService:
    def process(self, observation, previous):
        if observation.sequence == 2:
            raise ValueError("bad shared artifact")
        version = 0 if previous is None else previous.scene_version + 1
        return SceneSnapshot(
            "ep",
            1,
            version,
            observation.timestamp_ns,
            observation.timestamp_ns + 1,
            dict(observation.robot_state),
        )


def failing_observation_factory():
    return FailingObservationService()


class SlowFirstObservationService:
    def process(self, observation, previous):
        if observation.sequence == 1:
            time.sleep(0.35)
        version = 0 if previous is None else previous.scene_version + 1
        return SceneSnapshot(
            "ep",
            1,
            version,
            observation.timestamp_ns,
            observation.timestamp_ns + 1,
            dict(observation.robot_state),
        )


def slow_first_observation_factory():
    return SlowFirstObservationService()


class AlwaysFailingObservationService:
    def process(self, observation, previous):
        del observation, previous
        raise ValueError("corrupt shared artifact")


def always_failing_observation_factory():
    return AlwaysFailingObservationService()


class SimpleWorldModelService:
    def process(self, observation, previous):
        version = 0 if previous is None else previous.scene_version + 1
        return SceneSnapshot(
            "ep",
            1,
            version,
            observation.timestamp_ns,
            observation.timestamp_ns + 1,
            dict(observation.robot_state),
        )


def make_thread_runtime() -> ThreadWorldModelRuntime:
    return ThreadWorldModelRuntime(
        service=SimpleWorldModelService(),
        synchronizer=BoundedSensorSynchronizer(capacity=2, episode_id="ep", episode_epoch=1),
        config=WorldModelRuntimeConfig(queue_capacity=2),
        clock=time.time_ns,
    )


def make_process_runtime(tmp_path, *, service_factory=top_level_world_model_factory):
    return ProcessWorldModelRuntime(
        service_factory=service_factory,
        synchronizer_config=SynchronizerConfig(queue_capacity=2),
        config=WorldModelRuntimeConfig(restart_timeout_ms=500),
        artifact_store=FileArtifactStore(tmp_path),
    )


def test_thread_runtime_processes_observation_without_blocking_submit():
    runtime = make_thread_runtime()
    runtime.start()
    assert runtime.submit(make_observation(sequence=1))
    assert runtime.wait_until_processed(timeout_s=1.0)
    runtime.stop()


def test_thread_runtime_exposes_latest_snapshot_and_health():
    runtime = make_thread_runtime()
    runtime.start()
    runtime.submit(make_observation(sequence=1))

    assert runtime.wait_until_processed(timeout_s=1.0)
    assert runtime.latest_snapshot().scene_version == 0
    assert runtime.health().status == "healthy"
    runtime.stop()


def test_process_runtime_uses_picklable_factory_and_metadata_envelope(tmp_path):
    runtime = make_process_runtime(tmp_path)
    runtime.start()
    runtime.submit(make_observation(sequence=1))

    assert runtime.wait_until_processed(timeout_s=2.0)
    assert runtime.last_transport_message().format == "scene_snapshot_v1"
    runtime.stop()


def test_process_runtime_restarts_worker_and_retains_last_snapshot(tmp_path):
    runtime = make_process_runtime(tmp_path)
    runtime.start()
    runtime.submit(make_observation(sequence=1))
    assert runtime.wait_until_processed(timeout_s=2.0)
    previous = runtime.latest_snapshot()
    runtime.force_worker_exit()

    assert runtime.wait_until_healthy(timeout_s=2.0)
    assert runtime.latest_snapshot() == previous
    runtime.stop()


def test_process_runtime_marks_degraded_after_restart_budget_exhaustion(tmp_path):
    runtime = make_process_runtime(tmp_path, service_factory=always_crashing_factory)
    runtime.start()

    assert runtime.wait_until_degraded(timeout_s=5.0)
    assert runtime.health().status == "degraded"
    assert not runtime.episode_completed()
    runtime.stop()


def test_process_runtime_acknowledges_the_submitted_sequence_and_correlation(tmp_path):
    runtime = make_process_runtime(tmp_path)
    runtime.start()
    observation = make_observation(sequence=7)

    assert runtime.submit(observation)
    acknowledgement = runtime.wait_for_sequence(7, timeout_s=2.0)

    assert isinstance(acknowledgement, TransportAcknowledgement)
    assert acknowledgement.kind == "processed"
    assert acknowledgement.episode_id == "ep"
    assert acknowledgement.episode_epoch == 1
    assert acknowledgement.sequence == 7
    assert acknowledgement.scene_version == 0
    runtime.stop()


def test_process_runtime_reports_failed_sequence_without_killing_worker(tmp_path):
    runtime = make_process_runtime(tmp_path, service_factory=failing_observation_factory)
    runtime.start()

    assert runtime.submit(make_observation(sequence=2))
    failed = runtime.wait_for_sequence(2, timeout_s=2.0)
    assert failed is not None
    assert failed.kind == "failed"
    assert failed.error == "ValueError: bad shared artifact"

    assert runtime.submit(make_observation(sequence=3))
    processed = runtime.wait_for_sequence(3, timeout_s=2.0)
    assert processed is not None
    assert processed.kind == "processed"
    assert runtime.latest_snapshot().scene_version == 0
    assert runtime.health().status == "healthy"
    runtime.stop()


def test_process_runtime_drops_oldest_pending_sequence_and_retains_newest(tmp_path):
    runtime = ProcessWorldModelRuntime(
        service_factory=slow_first_observation_factory,
        synchronizer_config=SynchronizerConfig(queue_capacity=1),
        config=WorldModelRuntimeConfig(queue_capacity=1),
        artifact_store=FileArtifactStore(tmp_path),
    )
    runtime.start()
    assert runtime.submit(make_observation(sequence=1))
    time.sleep(0.08)
    assert runtime.submit(make_observation(sequence=2))
    assert runtime.submit(make_observation(sequence=3))

    dropped = runtime.wait_for_sequence(2, timeout_s=2.0)
    processed_first = runtime.wait_for_sequence(1, timeout_s=2.0)
    processed_latest = runtime.wait_for_sequence(3, timeout_s=2.0)

    assert dropped is not None and dropped.kind == "dropped"
    assert processed_first is not None and processed_first.kind == "processed"
    assert processed_latest is not None and processed_latest.kind == "processed"
    assert runtime.latest_snapshot().scene_version == 1
    runtime.stop()


def test_process_runtime_enters_degraded_after_consecutive_failures(tmp_path):
    runtime = ProcessWorldModelRuntime(
        service_factory=always_failing_observation_factory,
        synchronizer_config=SynchronizerConfig(queue_capacity=2),
        config=WorldModelRuntimeConfig(
            queue_capacity=2,
            max_consecutive_artifact_failures=2,
        ),
        artifact_store=FileArtifactStore(tmp_path),
    )
    runtime.start()
    assert runtime.submit(make_observation(sequence=1))
    assert runtime.wait_for_sequence(1, timeout_s=2.0).kind == "failed"
    assert runtime.submit(make_observation(sequence=2))
    assert runtime.wait_for_sequence(2, timeout_s=2.0).kind == "failed"

    assert runtime.wait_until_degraded(timeout_s=2.0)
    assert runtime.health().status == "degraded"
    runtime.stop()
