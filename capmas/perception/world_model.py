from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import multiprocessing as mp
import queue as queue_module
import threading
import time
from typing import Callable, Protocol, Sequence

from capmas.contracts.core import ArtifactRef
from capmas.contracts.scene import ObjectTrack, SceneSnapshot, SceneUncertainty
from capmas.perception.geometry import GeometryEstimator
from capmas.perception.local_map import SparseVoxelMap
from capmas.perception.protocol import ObservationBundle
from capmas.perception.semantic_triggers import SemanticRequestQueue
from capmas.perception.sensor_sync import (
    BoundedSensorSynchronizer,
    SensorSynchronizer,
)
from capmas.perception.tracking import KnownObjectTracker, ObjectMeasurement
from capmas.perception.serialization import (
    observation_from_json,
    observation_to_json,
    snapshot_from_json,
    snapshot_to_json,
)


@dataclass(frozen=True)
class WorldModelHealth:
    status: str
    restart_count: int
    last_error: str | None = None
    last_snapshot_version: int | None = None


@dataclass(frozen=True)
class WorldModelRuntimeConfig:
    queue_capacity: int = 4
    max_restarts_per_episode: int = 3
    restart_backoff_ms: tuple[int, ...] = (50, 100, 200)
    restart_timeout_ms: int = 500


@dataclass(frozen=True)
class SynchronizerConfig:
    queue_capacity: int = 4
    max_age_ms: float = 150.0


class WorldModelServiceProtocol(Protocol):
    def process(
        self,
        observation: ObservationBundle,
        previous: SceneSnapshot | None,
    ) -> SceneSnapshot: ...


class WorldModelService:
    """Pure deterministic fusion service; runtime lifecycle is separate."""

    def __init__(
        self,
        *,
        geometry: GeometryEstimator,
        local_map: SparseVoxelMap,
        tracker: KnownObjectTracker,
        clock: Callable[[], int] = time.time_ns,
        artifact_store: object | None = None,
        semantic_queue: SemanticRequestQueue | None = None,
        measurement_provider: Callable[[ObservationBundle], Sequence[ObjectMeasurement]] | None = None,
    ) -> None:
        self.geometry = geometry
        self.local_map = local_map
        self.tracker = tracker
        self.clock = clock
        self.artifact_store = artifact_store
        self.semantic_queue = semantic_queue
        self.measurement_provider = measurement_provider

    def process(
        self,
        observation: ObservationBundle,
        previous: SceneSnapshot | None,
    ) -> SceneSnapshot:
        episode_id, episode_epoch = self._episode_identity(observation, previous)
        geometry = self.geometry.estimate(observation)
        self.local_map.integrate(geometry, observation.timestamp_ns)
        objects = self._objects(observation, previous)
        publish_timestamp_ns = self.clock()
        processing_latency_ms = max(
            0.0,
            (publish_timestamp_ns - observation.timestamp_ns) / 1_000_000,
        )
        snapshot = SceneSnapshot(
            episode_id=episode_id,
            episode_epoch=episode_epoch,
            scene_version=0 if previous is None else previous.scene_version + 1,
            sensor_timestamp_ns=observation.timestamp_ns,
            publish_timestamp_ns=publish_timestamp_ns,
            robot=dict(observation.robot_state),
            objects=objects,
            local_map=self._publish_map_artifact(),
            freshness_ms=processing_latency_ms,
            source_artifacts=geometry.source_artifacts,
            uncertainty=_uncertainty(objects),
            processing_latency_ms=processing_latency_ms,
        )
        if self.semantic_queue is not None:
            from capmas.perception.semantic_triggers import DeterministicSemanticTrigger

            DeterministicSemanticTrigger(self.semantic_queue).inspect(snapshot)
        return snapshot

    @staticmethod
    def _episode_identity(
        observation: ObservationBundle,
        previous: SceneSnapshot | None,
    ) -> tuple[str, int]:
        if previous is None:
            return observation.episode_id or "unknown", observation.episode_epoch or 0
        if observation.episode_id is not None and observation.episode_id != previous.episode_id:
            raise ValueError("observation belongs to another episode")
        if observation.episode_epoch is not None and observation.episode_epoch != previous.episode_epoch:
            raise ValueError("observation belongs to another episode epoch")
        return previous.episode_id, previous.episode_epoch

    def _objects(
        self,
        observation: ObservationBundle,
        previous: SceneSnapshot | None,
    ) -> tuple[ObjectTrack, ...]:
        measurements = (
            tuple(self.measurement_provider(observation))
            if self.measurement_provider is not None
            else ()
        )
        if measurements:
            return self.tracker.update(measurements)
        if previous is not None:
            return tuple(previous.objects)
        return self.tracker.predict(observation.timestamp_ns)

    def _publish_map_artifact(self) -> ArtifactRef | None:
        if self.artifact_store is None:
            return None
        snapshot = self.local_map.freeze_snapshot()
        payload = json.dumps(asdict(snapshot), sort_keys=True, separators=(",", ":")).encode()
        put = getattr(self.artifact_store, "put")
        return put(payload, "application/x-capmas-sparse-voxel-map")


class WorldModelRuntime(Protocol):
    def start(self, initial_scene: SceneSnapshot | None = None) -> None: ...

    def submit(self, observation: ObservationBundle) -> bool: ...

    def latest_observation(self) -> SceneSnapshot: ...

    def health(self) -> WorldModelHealth: ...

    def stop(self, timeout_ms: int = 500) -> None: ...


class ThreadWorldModelRuntime:
    def __init__(
        self,
        service: WorldModelServiceProtocol,
        synchronizer: SensorSynchronizer,
        config: WorldModelRuntimeConfig,
        clock: Callable[[], int],
    ) -> None:
        self.service = service
        self.synchronizer = synchronizer
        self.config = config
        self.clock = clock
        self._condition = threading.Condition()
        self._running = False
        self._worker: threading.Thread | None = None
        self._previous: SceneSnapshot | None = None
        self._latest: SceneSnapshot | None = None
        self._submitted = 0
        self._processed = 0
        self._health = WorldModelHealth("stopped", 0)

    def start(self, initial_scene: SceneSnapshot | None = None) -> None:
        with self._condition:
            if self._running:
                return
            self._running = True
            self._previous = initial_scene
            self._latest = initial_scene
            self._health = WorldModelHealth(
                "starting", 0, last_snapshot_version=(initial_scene.scene_version if initial_scene else None)
            )
            self._worker = threading.Thread(target=self._run, name="capmas-world-model", daemon=True)
            self._worker.start()

    def submit(self, observation: ObservationBundle) -> bool:
        with self._condition:
            if not self._running:
                return False
            try:
                self.synchronizer.push(observation)
            except ValueError:
                return False
            self._submitted += 1
            self._condition.notify()
            return True

    def latest_observation(self) -> SceneSnapshot:
        with self._condition:
            if self._latest is None:
                raise RuntimeError("world model has not published a snapshot")
            return self._latest

    def latest_snapshot(self) -> SceneSnapshot:
        return self.latest_observation()

    def health(self) -> WorldModelHealth:
        with self._condition:
            return self._health

    def wait_until_processed(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while self._processed < self._submitted and self._running:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(timeout=remaining)
            return self._processed >= self._submitted

    def stop(self, timeout_ms: int = 500) -> None:
        with self._condition:
            self._running = False
            self._condition.notify_all()
        if self._worker is not None:
            self._worker.join(timeout=max(0, timeout_ms) / 1000)
        with self._condition:
            self._health = WorldModelHealth(
                "stopped",
                self._health.restart_count,
                self._health.last_error,
                self._health.last_snapshot_version,
            )

    def _run(self) -> None:
        with self._condition:
            self._health = WorldModelHealth("healthy", 0)
            self._condition.notify_all()
        while True:
            with self._condition:
                if not self._running:
                    return
                observation = self.synchronizer.pop_ready()
                if observation is None:
                    self._condition.wait(timeout=0.05)
                    continue
            try:
                snapshot = self.service.process(observation, self._previous)
            except Exception as exc:
                with self._condition:
                    self._health = WorldModelHealth(
                        "degraded", self._health.restart_count, str(exc),
                        self._health.last_snapshot_version,
                    )
                    self._processed += 1
                    self._condition.notify_all()
                continue
            with self._condition:
                self._previous = snapshot
                self._latest = snapshot
                self._processed += 1
                self._health = WorldModelHealth("healthy", 0, last_snapshot_version=snapshot.scene_version)
                self._condition.notify_all()


@dataclass(frozen=True)
class TransportMessage:
    format: str
    payload: str


class ProcessWorldModelRuntime:
    def __init__(
        self,
        service_factory: Callable[[], WorldModelServiceProtocol],
        synchronizer_config: SynchronizerConfig,
        config: WorldModelRuntimeConfig,
        artifact_store: object,
    ) -> None:
        self.service_factory = service_factory
        self.synchronizer_config = synchronizer_config
        self.config = config
        self.artifact_store = artifact_store
        self._context = mp.get_context("spawn")
        self._input = self._context.Queue(maxsize=synchronizer_config.queue_capacity)
        self._output = self._context.Queue()
        self._synchronizer = BoundedSensorSynchronizer(synchronizer_config.queue_capacity)
        self._process: mp.Process | None = None
        self._monitor: threading.Thread | None = None
        self._running = False
        self._stopping = False
        self._latest: SceneSnapshot | None = None
        self._submitted = 0
        self._processed = 0
        self._restart_count = 0
        self._last_error: str | None = None
        self._last_transport: TransportMessage | None = None
        self._health = WorldModelHealth("stopped", 0)

    def start(self, initial_scene: SceneSnapshot | None = None) -> None:
        if self._running:
            return
        self._running = True
        self._stopping = False
        self._latest = initial_scene
        self._restart_count = 0
        self._health = WorldModelHealth(
            "starting", 0, last_snapshot_version=(initial_scene.scene_version if initial_scene else None)
        )
        self._start_worker(initial_scene)
        self._monitor = threading.Thread(target=self._monitor_loop, name="capmas-world-model-monitor", daemon=True)
        self._monitor.start()

    def submit(self, observation: ObservationBundle) -> bool:
        if not self._running or self._health.status == "degraded":
            return False
        try:
            self._synchronizer.push(observation)
            ready = self._synchronizer.pop_ready()
            assert ready is not None
            self._input.put_nowait(("observation", observation_to_json(ready)))
        except (ValueError, queue_module.Full):
            return False
        self._submitted += 1
        return True

    def latest_observation(self) -> SceneSnapshot:
        if self._latest is None:
            raise RuntimeError("world model has not published a snapshot")
        return self._latest

    def latest_snapshot(self) -> SceneSnapshot:
        return self.latest_observation()

    def health(self) -> WorldModelHealth:
        return self._health

    def last_transport_message(self) -> TransportMessage:
        if self._last_transport is None:
            raise RuntimeError("world model has not sent a transport message")
        return self._last_transport

    def wait_until_processed(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._processed >= self._submitted:
                return True
            time.sleep(0.01)
        return self._processed >= self._submitted

    def wait_until_healthy(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._health.status == "healthy":
                return True
            time.sleep(0.01)
        return self._health.status == "healthy"

    def wait_until_degraded(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._health.status == "degraded":
                return True
            time.sleep(0.01)
        return self._health.status == "degraded"

    def force_worker_exit(self) -> None:
        if self._process is not None and self._process.is_alive():
            self._process.terminate()

    def episode_completed(self) -> bool:
        return False

    def stop(self, timeout_ms: int = 500) -> None:
        self._stopping = True
        self._running = False
        try:
            self._input.put_nowait(None)
        except queue_module.Full:
            pass
        if self._process is not None:
            self._process.join(timeout=max(0, timeout_ms) / 1000)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join()
        if self._monitor is not None:
            self._monitor.join(timeout=max(0, timeout_ms) / 1000)
        self._health = WorldModelHealth(
            "stopped", self._restart_count, self._last_error,
            self._latest.scene_version if self._latest else None,
        )

    def _start_worker(self, initial_scene: SceneSnapshot | None) -> None:
        self._process = self._context.Process(
            target=_process_worker_entry,
            args=(self.service_factory, self._input, self._output),
            daemon=True,
        )
        self._process.start()
        if initial_scene is not None:
            self._input.put(("initial", snapshot_to_json(initial_scene)))

    def _monitor_loop(self) -> None:
        while self._running or (self._process is not None and self._process.is_alive()):
            try:
                message = self._output.get(timeout=0.05)
            except queue_module.Empty:
                message = None
            if message is not None:
                self._handle_message(message)
            if self._process is not None and not self._process.is_alive() and self._running:
                self._restart_or_degrade()
        self._health = WorldModelHealth(
            "stopped" if self._stopping else self._health.status,
            self._restart_count,
            self._last_error,
            self._latest.scene_version if self._latest else None,
        )

    def _handle_message(self, message: dict[str, str]) -> None:
        kind = message.get("type")
        if kind == "ready":
            self._health = WorldModelHealth("healthy", self._restart_count)
        elif kind == "error":
            self._last_error = message.get("error", "worker error")
        elif kind == "snapshot":
            payload = message["payload"]
            self._last_transport = TransportMessage("scene_snapshot_v1", payload)
            self._latest = snapshot_from_json(payload)
            self._processed += 1
            self._health = WorldModelHealth(
                "healthy", self._restart_count, self._last_error, self._latest.scene_version
            )

    def _restart_or_degrade(self) -> None:
        if self._restart_count >= self.config.max_restarts_per_episode:
            self._health = WorldModelHealth(
                "degraded", self._restart_count, self._last_error,
                self._latest.scene_version if self._latest else None,
            )
            self._running = False
            return
        index = min(self._restart_count, len(self.config.restart_backoff_ms) - 1)
        if index >= 0:
            time.sleep(self.config.restart_backoff_ms[index] / 1000)
        self._restart_count += 1
        self._health = WorldModelHealth("starting", self._restart_count, self._last_error)
        self._start_worker(self._latest)


def _process_worker_entry(service_factory, input_queue, output_queue) -> None:
    try:
        service = service_factory()
    except BaseException as exc:
        output_queue.put({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
        return
    previous: SceneSnapshot | None = None
    output_queue.put({"type": "ready"})
    while True:
        message = input_queue.get()
        if message is None:
            return
        kind, payload = message
        if kind == "initial":
            previous = snapshot_from_json(payload)
            continue
        if kind != "observation":
            continue
        try:
            observation = observation_from_json(payload)
            snapshot = service.process(observation, previous)
            previous = snapshot
            output_queue.put({"type": "snapshot", "payload": snapshot_to_json(snapshot)})
        except BaseException as exc:
            output_queue.put({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
            return


def _uncertainty(objects: Sequence[ObjectTrack]) -> SceneUncertainty:
    if not objects:
        return SceneUncertainty()
    confidence = sum(track.confidence for track in objects) / len(objects)
    stale = tuple(track.track_id for track in objects if track.track_status in {"stale", "lost"})
    return SceneUncertainty(
        scene_confidence=confidence,
        stale_track_ids=stale,
    )
