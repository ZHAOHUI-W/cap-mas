import pytest

from capmas.perception.protocol import ObservationBundle
from capmas.perception.sensor_sync import (
    BoundedSensorSynchronizer,
    JsonlObservationRecorder,
    JsonlReplaySource,
)


def make_observation_bundle(
    *, sequence: int = 0, episode_id: str | None = "ep", episode_epoch: int | None = 1
) -> ObservationBundle:
    return ObservationBundle(
        timestamp_ns=sequence * 100,
        frames=(),
        robot_state={},
        episode_id=episode_id,
        episode_epoch=episode_epoch,
        source="test",
        sequence=sequence,
    )


def test_replay_source_streams_one_bundle_without_full_materialization(tmp_path):
    first = make_observation_bundle(sequence=1)
    second = make_observation_bundle(sequence=2)
    path = tmp_path / "observations.jsonl"
    recorder = JsonlObservationRecorder(path)
    recorder.append(first)
    recorder.append(second)

    replay = JsonlReplaySource(path)

    assert replay.capture() == first
    assert not replay.exhausted()
    assert replay.capture() == second
    assert replay.exhausted()
    with pytest.raises(StopIteration):
        replay.capture()


def test_synchronizer_rejects_out_of_order_sequence():
    synchronizer = BoundedSensorSynchronizer(capacity=2, episode_id="ep", episode_epoch=1)
    synchronizer.push(make_observation_bundle(sequence=2))

    with pytest.raises(ValueError, match="monotonic"):
        synchronizer.push(make_observation_bundle(sequence=1))

    metrics = synchronizer.metrics()
    assert metrics.accepted == 1
    assert metrics.rejected_out_of_order == 1


def test_synchronizer_rejects_wrong_episode():
    synchronizer = BoundedSensorSynchronizer(capacity=2, episode_id="ep", episode_epoch=1)

    with pytest.raises(ValueError, match="episode"):
        synchronizer.push(make_observation_bundle(episode_id="other", sequence=1))

    assert synchronizer.metrics().rejected_episode == 1


def test_synchronizer_drops_oldest_when_capacity_is_full():
    synchronizer = BoundedSensorSynchronizer(capacity=2, episode_id="ep", episode_epoch=1)
    synchronizer.push(make_observation_bundle(sequence=1))
    synchronizer.push(make_observation_bundle(sequence=2))
    synchronizer.push(make_observation_bundle(sequence=3))

    assert synchronizer.pop_ready().sequence == 2
    assert synchronizer.pop_ready().sequence == 3
    assert synchronizer.metrics().dropped_oldest == 1
