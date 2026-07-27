import pytest

from capmas.contracts.scene import SceneSnapshot
from capmas.runtime.state_store import InMemoryStateStore


def make_scene(*, scene_version: int, sensor_timestamp_ns: int = 100) -> SceneSnapshot:
    return SceneSnapshot(
        "ep",
        1,
        scene_version,
        sensor_timestamp_ns,
        sensor_timestamp_ns + 10,
        {},
    )


def test_pending_observations_do_not_replace_committed_snapshot():
    store = InMemoryStateStore(max_pending_observations=3)
    initial = make_scene(scene_version=1, sensor_timestamp_ns=100)
    store.start_episode(initial)
    store.publish_observation(make_scene(scene_version=2, sensor_timestamp_ns=200))

    assert store.latest_committed() == initial
    assert store.latest_observation().scene_version == 2
    assert store.latest().scene_version == 1


def test_latest_observation_returns_highest_published_version():
    store = InMemoryStateStore(max_pending_observations=3)
    store.start_episode(make_scene(scene_version=1))
    store.publish_observation(make_scene(scene_version=2))
    store.publish_observation(make_scene(scene_version=3))

    assert store.latest_observation().scene_version == 3


def test_commit_after_action_accepts_noncontiguous_post_action_version():
    store = InMemoryStateStore(max_pending_observations=3)
    store.start_episode(make_scene(scene_version=1, sensor_timestamp_ns=100))
    store.publish_observation(make_scene(scene_version=2, sensor_timestamp_ns=150))
    after = make_scene(scene_version=3, sensor_timestamp_ns=300)
    store.publish_observation(after)

    assert store.commit_after_action(1, action_finished_at_ns=200, after=after)
    assert store.latest_committed() == after


def test_commit_after_action_rejects_snapshot_captured_before_action_end():
    store = InMemoryStateStore(max_pending_observations=3)
    store.start_episode(make_scene(scene_version=1, sensor_timestamp_ns=100))
    before_action = make_scene(scene_version=2, sensor_timestamp_ns=150)
    store.publish_observation(before_action)

    assert not store.commit_after_action(1, action_finished_at_ns=200, after=before_action)
    assert store.latest_committed().scene_version == 1


def test_pending_observations_drop_oldest_at_configured_capacity():
    store = InMemoryStateStore(max_pending_observations=2)
    store.start_episode(make_scene(scene_version=1))
    store.publish_observation(make_scene(scene_version=2))
    store.publish_observation(make_scene(scene_version=3))
    store.publish_observation(make_scene(scene_version=4))

    assert store.pending_versions() == (3, 4)


def test_legacy_compare_and_commit_stays_contiguous():
    store = InMemoryStateStore()
    store.start_episode(make_scene(scene_version=1))

    with pytest.raises(ValueError, match="increment"):
        store.compare_and_commit(1, make_scene(scene_version=3))
