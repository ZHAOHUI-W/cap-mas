from __future__ import annotations

from collections import OrderedDict

from capmas.contracts.scene import SceneSnapshot


class InMemoryStateStore:
    """State store with a legacy committed view and Phase 4 observations."""

    def __init__(self, *, max_pending_observations: int = 8) -> None:
        if max_pending_observations <= 0:
            raise ValueError("max_pending_observations must be positive")
        self.max_pending_observations = max_pending_observations
        self._episode_id: str | None = None
        self._episode_epoch: int | None = None
        self._snapshots: dict[int, SceneSnapshot] = {}
        self._pending: OrderedDict[int, SceneSnapshot] = OrderedDict()
        self._committed: SceneSnapshot | None = None

    def start_episode(self, snapshot: SceneSnapshot) -> None:
        self._episode_id = snapshot.episode_id
        self._episode_epoch = snapshot.episode_epoch
        self._snapshots = {}
        self._pending = OrderedDict()
        self._committed = None
        self.publish(snapshot)

    def publish(self, snapshot: SceneSnapshot) -> None:
        """Legacy immediate publication and commit."""
        self._validate_episode(snapshot)
        self._store(snapshot)
        self._committed = snapshot
        for version in tuple(self._pending):
            if version <= snapshot.scene_version:
                self._pending.pop(version, None)

    def publish_observation(self, snapshot: SceneSnapshot) -> None:
        self._validate_episode(snapshot)
        self._store(snapshot)
        if self._committed is not None and snapshot.scene_version <= self._committed.scene_version:
            return
        self._pending[snapshot.scene_version] = snapshot
        self._pending = OrderedDict(sorted(self._pending.items()))
        while len(self._pending) > self.max_pending_observations:
            self._pending.popitem(last=False)

    def latest(self) -> SceneSnapshot:
        return self.latest_committed()

    def latest_committed(self) -> SceneSnapshot:
        if self._committed is None:
            raise RuntimeError("state store is empty")
        return self._committed

    def latest_observation(self) -> SceneSnapshot:
        if not self._snapshots:
            raise RuntimeError("state store is empty")
        return self._snapshots[max(self._snapshots)]

    def get(self, version: int) -> SceneSnapshot:
        try:
            return self._snapshots[version]
        except KeyError as exc:
            raise KeyError(f"unknown scene version: {version}") from exc

    def pending_versions(self) -> tuple[int, ...]:
        return tuple(self._pending)

    def commit_after_action(
        self,
        parent_version: int,
        action_finished_at_ns: int,
        after: SceneSnapshot,
    ) -> bool:
        if self._committed is None or self._committed.scene_version != parent_version:
            return False
        self._validate_episode(after)
        if after.scene_version <= parent_version:
            return False
        published = self._snapshots.get(after.scene_version)
        if published != after:
            return False
        if after.sensor_timestamp_ns < action_finished_at_ns:
            return False
        self._committed = after
        for version in tuple(self._pending):
            if version <= after.scene_version:
                self._pending.pop(version, None)
        return True

    def compare_and_commit(self, parent_version: int, snapshot: SceneSnapshot) -> bool:
        if self.latest_committed().scene_version != parent_version:
            return False
        if snapshot.scene_version != parent_version + 1:
            raise ValueError("committed scene must increment version by one")
        self.publish(snapshot)
        return True

    def _validate_episode(self, snapshot: SceneSnapshot) -> None:
        if self._episode_id is None:
            self._episode_id = snapshot.episode_id
            self._episode_epoch = snapshot.episode_epoch
        if snapshot.episode_id != self._episode_id or snapshot.episode_epoch != self._episode_epoch:
            raise ValueError("snapshot belongs to another episode")

    def _store(self, snapshot: SceneSnapshot) -> None:
        current = self._snapshots.get(snapshot.scene_version)
        if current is not None:
            if current != snapshot:
                raise ValueError(f"scene version already published: {snapshot.scene_version}")
            return
        if self._snapshots and snapshot.scene_version < max(self._snapshots):
            raise ValueError("scene versions must be monotonic")
        self._snapshots[snapshot.scene_version] = snapshot
