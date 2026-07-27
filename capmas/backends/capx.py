from __future__ import annotations

from dataclasses import dataclass
import inspect
import time
from typing import Any, Callable, Mapping
from uuid import uuid4

from capmas.backends.protocol import RobotBackend, SkillExecutionResult
from capmas.contracts.action import ExecutionBudget
from capmas.contracts.core import ArtifactRef, EpisodeHandle, SkillRef
from capmas.contracts.scene import (
    EpisodeStart,
    EpisodeStatus,
    ObjectTrack,
    SceneSnapshot,
    VisualEvidence,
)
from capmas.perception.artifacts import InMemoryArtifactStore
from capmas.perception.protocol import (
    CameraFrame,
    CameraModel,
    ObservationBundle,
    ObservationProvider,
)


def _flatten_numbers(value: object) -> tuple[float, ...]:
    if hasattr(value, "reshape"):
        value = value.reshape(-1).tolist()  # type: ignore[union-attr]
    if isinstance(value, (list, tuple)):
        result: list[float] = []
        for item in value:
            result.extend(_flatten_numbers(item))
        return tuple(result)
    return (float(value),)


@dataclass
class CAPXObservationProvider(ObservationProvider):
    """Normalize a CAP-X API observation without exposing its environment object."""

    observation_fn: Callable[[], Mapping[str, object]]
    artifacts: InMemoryArtifactStore
    object_poses_fn: Callable[[], Mapping[str, object]] | None = None
    object_pose_fn: Callable[[str], object] | None = None
    object_names: tuple[str, ...] = ()
    object_pose_confidence: float = 1.0

    def capture(self) -> ObservationBundle:
        raw = self.observation_fn()
        timestamp_ns = int(raw.get("timestamp_ns", time.time_ns()))
        frames: list[CameraFrame] = []
        for camera_id in ("agentview", "robot0_eye_in_hand"):
            camera = raw.get(camera_id)
            if not isinstance(camera, Mapping):
                continue
            images = camera.get("images", {})
            if not isinstance(images, Mapping):
                images = {}
            intrinsics = camera.get("intrinsics", ())
            pose = camera.get("pose_mat", camera.get("pose", ()))
            model = CameraModel(
                camera_id=camera_id,
                intrinsics=_flatten_numbers(intrinsics),
                pose_world=_flatten_numbers(pose),
            )
            rgb = self._put(images.get("rgb"), "image/rgb")
            depth = self._put(images.get("depth"), "image/depth")
            frames.append(
                CameraFrame(
                    camera_id=camera_id,
                    timestamp_ns=timestamp_ns,
                    rgb=rgb,
                    depth=depth,
                    camera=model,
                )
            )
        robot_cartesian_pos = raw.get("robot_cartesian_pos")
        robot_state = {
            "joint_position": self._put(raw.get("robot_joint_pos"), "array/joint-position"),
            "ee_pose": self._put(robot_cartesian_pos, "array/ee-pose"),
            "ee_pose_wxyz_xyz": self._as_ee_pose(robot_cartesian_pos),
            "gripper_opening": robot_cartesian_pos[-1] if robot_cartesian_pos is not None else None,
        }
        return ObservationBundle(
            timestamp_ns=timestamp_ns,
            frames=tuple(frames),
            robot_state=robot_state,
        )

    def _put(self, value: object, media_type: str) -> ArtifactRef | None:
        if value is None:
            return None
        return self.artifacts.put(value, media_type)

    @staticmethod
    def _as_float_tuple(value: object) -> tuple[float, ...] | None:
        if value is None:
            return None
        try:
            return _flatten_numbers(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _as_ee_pose(cls, value: object) -> tuple[float, ...] | None:
        values = cls._as_float_tuple(value)
        if values is None or len(values) < 7:
            return None
        return tuple(values[3:7]) + tuple(values[:3])

    def capture_object_tracks(
        self,
        *,
        timestamp_ns: int,
        episode_id: str,
        episode_epoch: int,
    ) -> tuple[ObjectTrack, ...]:
        """Convert CAP-X pose API results to immutable scene object tracks."""
        raw_poses: dict[str, object] = {}
        if self.object_poses_fn is not None:
            value = self.object_poses_fn()
            if isinstance(value, Mapping):
                raw_poses.update(value)
        if self.object_pose_fn is not None:
            for name in self.object_names:
                try:
                    pose = self.object_pose_fn(name)
                    if pose is not None:
                        raw_poses[name] = pose
                except Exception:
                    continue

        tracks: list[ObjectTrack] = []
        for name, value in raw_poses.items():
            pose = _normalize_capx_object_pose(value)
            if pose is None:
                continue
            position, quaternion_wxyz = pose
            label = str(name).replace("_", " ")
            tracks.append(
                ObjectTrack(
                    track_id=str(name),
                    label=label,
                    pose_wxyz_xyz=tuple(quaternion_wxyz) + tuple(position),
                    confidence=self.object_pose_confidence,
                    last_seen_ns=timestamp_ns,
                )
            )
        return tuple(tracks)


class CAPXStreamingObservationSource:
    """Add Phase 4 stream metadata while reusing the CAP-X provider boundary."""

    def __init__(
        self,
        provider: ObservationProvider,
        source: str = "capx",
        episode_id: str | None = None,
        episode_epoch: int | None = None,
    ) -> None:
        self.provider = provider
        self.source = source
        self.episode_id = episode_id
        self.episode_epoch = episode_epoch
        self._sequence = 0

    def capture(self) -> ObservationBundle:
        observation = self.provider.capture()
        self._sequence += 1
        return ObservationBundle(
            timestamp_ns=observation.timestamp_ns,
            frames=observation.frames,
            robot_state=observation.robot_state,
            episode_id=self.episode_id,
            episode_epoch=self.episode_epoch,
            source=self.source,
            sequence=self._sequence,
        )


class CAPXTypedSkill:
    """Adapt one allowlisted CAP-X API callable to the CAP-MAS skill protocol."""

    def __init__(
        self,
        reference: SkillRef,
        function: Callable[..., object],
    ) -> None:
        self.skill_id = reference.skill_id
        self.version = reference.version
        self._function = function
        self._signature = inspect.signature(function)

    def validate_args(self, args: dict[str, object]) -> None:
        try:
            self._signature.bind(**args)
        except TypeError as exc:
            raise ValueError(
                f"invalid arguments for {self.skill_id}: {exc}; "
                f"expected_signature={self._signature}; actual_keys={sorted(args)!r}"
            ) from exc

    def execute(self, args: dict[str, object], budget: ExecutionBudget) -> SkillExecutionResult:
        del budget
        try:
            return SkillExecutionResult(ok=True, output={"result": self._function(**args)})
        except BaseException as exc:
            return SkillExecutionResult(
                ok=False,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )


def build_capx_skills(
    api: object,
    allowed_functions: Mapping[str, str],
    version: str = "capx-compat-1",
    function_overrides: Mapping[str, Callable[..., object]] | None = None,
) -> dict[SkillRef, CAPXTypedSkill]:
    """Build typed bindings from an existing CAP-X ApiBase allowlist.

    `allowed_functions` maps the CAP-MAS skill ID to a function name returned by
    the CAP-X API's `functions()` method. The adapter does not expose the API
    object, environment, or unlisted methods to the runtime.
    """
    functions = api.functions()
    if function_overrides:
        functions = {**functions, **function_overrides}
    bindings: dict[SkillRef, CAPXTypedSkill] = {}
    for skill_id, function_name in allowed_functions.items():
        if function_name not in functions:
            raise ValueError(f"CAP-X API does not expose function: {function_name}")
        reference = SkillRef(skill_id, version)
        bindings[reference] = CAPXTypedSkill(reference, functions[function_name])
    return bindings


class CAPXRobotBackend(RobotBackend):
    """Small backend shell around a CAP-X low-level environment and API."""

    def __init__(
        self,
        env: object,
        observation_provider: CAPXObservationProvider,
        task_id: str,
        suite_name: str,
        backend_id: str = "capx",
    ) -> None:
        self.env = env
        self.observation_provider = observation_provider
        self.task_id = task_id
        self.suite_name = suite_name
        self.backend_id = backend_id
        self._handle: EpisodeHandle | None = None
        self._scene_version = 0

    def reset(self, seed: int | None = None, options: Mapping[str, object] | None = None) -> EpisodeStart:
        reset = getattr(self.env, "reset")
        reset(seed=seed, options=dict(options or {}))
        self._scene_version = 0
        self._handle = EpisodeHandle(
            episode_id=str(uuid4()),
            task_id=self.task_id,
            suite_name=self.suite_name,
            backend_id=self.backend_id,
            seed=seed,
            episode_epoch=1,
            started_at_ns=0,
            status=EpisodeStatus.ACTIVE,
        )
        scene = self._snapshot(self.observation_provider.capture())
        return EpisodeStart(self._handle, scene)

    def observe(self) -> SceneSnapshot:
        if self._handle is None:
            raise RuntimeError("backend episode has not started")
        self._scene_version += 1
        return self._snapshot(self.observation_provider.capture())

    def execute_skill(
        self,
        skill: CAPXTypedSkill,
        args: dict[str, object],
        budget: ExecutionBudget,
    ) -> SkillExecutionResult:
        return skill.execute(args, budget)

    def stop(self, lease: object) -> None:
        del lease
        stop = getattr(self.env, "stop", None)
        if callable(stop):
            stop()

    def evaluator_success(self) -> bool:
        check_success = getattr(self.env, "task_completed", None)
        if not callable(check_success):
            raise RuntimeError("CAP-X evaluator does not expose task_completed")
        return bool(check_success())

    def _snapshot(self, observation: ObservationBundle) -> SceneSnapshot:
        assert self._handle is not None
        object_tracks = self.observation_provider.capture_object_tracks(
            timestamp_ns=observation.timestamp_ns,
            episode_id=self._handle.episode_id,
            episode_epoch=self._handle.episode_epoch,
        )
        visual_evidence = tuple(
            VisualEvidence(
                artifact=artifact,
                evidence_type=artifact.media_type,
                captured_at_ns=frame.timestamp_ns,
                camera_id=frame.camera_id,
            )
            for frame in observation.frames
            for artifact in (frame.rgb, frame.depth)
            if artifact is not None
        )
        publish_timestamp_ns = time.time_ns()
        return SceneSnapshot(
            episode_id=self._handle.episode_id,
            episode_epoch=self._handle.episode_epoch,
            scene_version=self._scene_version,
            sensor_timestamp_ns=observation.timestamp_ns,
            # The sensor timestamp describes when capture started. Publish
            # freshness must describe when perception has finished and the
            # snapshot is available to the planning/runtime planes.
            publish_timestamp_ns=publish_timestamp_ns,
            robot=observation.robot_state,
            objects=object_tracks,
            source_artifacts=tuple(
                artifact
                for frame in observation.frames
                for artifact in (frame.rgb, frame.depth)
                if artifact is not None
            ),
            visual_evidence=visual_evidence,
            processing_latency_ms=max(
                0.0,
                (publish_timestamp_ns - observation.timestamp_ns) / 1_000_000,
            ),
        )


def _normalize_capx_object_pose(value: object) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    try:
        if len(value) != 2:  # type: ignore[arg-type]
            return None
        position_value = value[0]  # type: ignore[index]
        quaternion_value = value[1]  # type: ignore[index]
    except (TypeError, IndexError):
        return None
    try:
        position = _flatten_numbers(position_value)
        quaternion_wxyz = _flatten_numbers(quaternion_value)
    except (TypeError, ValueError):
        return None
    if len(position) != 3 or len(quaternion_wxyz) != 4:
        return None
    return position, quaternion_wxyz
