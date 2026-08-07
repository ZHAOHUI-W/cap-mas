from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from functools import wraps

from capmas.backends.capx import (
    CAPXObservationProvider,
    CAPXRobotBackend,
    CAPXTypedSkill,
    PlacementPoseResult,
    _normalize_capx_object_pose,
    build_capx_skills,
)
from capmas.perception.artifact_bridge import ArtifactSink
from capmas.perception.artifacts import InMemoryArtifactStore
from capmas.skills.registry import SkillRegistry


def build_capx_world_model_enricher(provider, **kwargs):
    """Build the optional live RGB-D World Model bridge for CAP-X scenes."""
    from capmas.perception.capx_world_model import (
        build_capx_world_model_enricher as _build,
    )

    return _build(provider, **kwargs)


DEFAULT_LIBERO_SKILLS: dict[str, str] = {
    "get_observation": "get_observation",
    "get_object_pose": "get_object_pose",
    "sample_grasp_pose": "sample_grasp_pose",
    "goto_pose": "goto_pose",
    "open_gripper": "open_gripper",
    "close_gripper": "close_gripper",
}

OPTIONAL_LIBERO_SKILLS: dict[str, str] = {
    "goto_home_joint_position": "goto_home_joint_position",
}

DEFAULT_LIBERO_POSTCONDITIONS: dict[str, tuple[str, ...]] = {
    "goto_pose": ("scene_fresh(2000)",),
    "sample_grasp_pose": ("scene_fresh(2000)",),
    "lift_after_grasp": ("scene_fresh(2000)",),
    "close_gripper": ("gripper_closed()",),
    "open_gripper": ("gripper_open()",),
}


def estimate_libero_placement_pose(
    points: object,
    *,
    release_clearance_m: float = 0.0,
) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    """Estimate a safe placement reference from a segmented target cloud.

    CAP-X's generic ``get_object_pose`` returns an oriented-box center. That
    center is useful as an object track pose, but it is biased for open
    containers because only visible walls are segmented. A trimmed
    axis-aligned cloud center is a more stable XY entry point for the gripper.
    ``release_clearance_m`` moves the robot-only release reference above the
    visible top of a container. The semantic track keeps the original pose;
    this result is only used by placement grounding.
    """
    try:
        import numpy as np

        values = np.asarray(points, dtype=float)
        if values.ndim != 2 or values.shape[1] != 3:
            return None
        values = values[np.isfinite(values).all(axis=1)]
        if len(values) < 16:
            return None
        lower = np.percentile(values, 2.0, axis=0)
        upper = np.percentile(values, 98.0, axis=0)
        position = (lower + upper) / 2.0
        position[2] = upper[2] + max(float(release_clearance_m), 0.0)
        if not np.isfinite(position).all():
            return None
        return tuple(float(value) for value in position), (0.0, 1.0, 0.0, 0.0)
    except (ImportError, TypeError, ValueError):
        return None


@dataclass(frozen=True)
class CAPXRuntimeBundle:
    """Constructed CAP-X resources exposed to the CAP-MAS runtime."""

    environment: object
    low_level_environment: object
    api: object
    apis: Mapping[str, object]
    observation_provider: CAPXObservationProvider
    backend: CAPXRobotBackend
    skill_registry: SkillRegistry
    task_id: str
    suite_name: str
    task_language: str | None = None


@dataclass(frozen=True)
class CAPXProcessWorldModelFactory:
    """Pickle-safe World Model factory for the spawned CAP-X worker."""

    artifact_root: str
    depth_subsample: int = 16
    fsync: bool = False

    def __call__(self):
        from capmas.perception.artifact_bridge import (
            EncodedArtifactStore,
            FileArtifactStore,
            NumpyArtifactCodec,
        )
        from capmas.perception.capx_depth import CAPXDepthDecoder
        from capmas.perception.geometry import ReferenceGeometryEstimator
        from capmas.perception.local_map import SparseVoxelMap
        from capmas.perception.tracking import KnownObjectTracker
        from capmas.perception.world_model import WorldModelService

        store = EncodedArtifactStore(
            FileArtifactStore(self.artifact_root, fsync=self.fsync),
            NumpyArtifactCodec(),
        )
        return WorldModelService(
            geometry=ReferenceGeometryEstimator(
                artifact_store=store,
                depth_decoder=CAPXDepthDecoder(subsample=self.depth_subsample),
            ),
            local_map=SparseVoxelMap(voxel_size_m=0.01, local_radius_m=1.0),
            tracker=KnownObjectTracker(max_match_distance_m=0.08),
            artifact_store=store,
        )


def _default_loader(path: str | Path) -> Mapping[str, object]:
    from capx.envs.configs.loader import DictLoader

    return DictLoader.load(str(path))


def _default_instantiator(config: object) -> object:
    from capx.envs.configs.instantiate import instantiate

    return instantiate(config)


def _default_api_factory(name: str) -> Callable[[object], object]:
    from capx.integrations.base_api import get_api

    return get_api(name)


def _get_env_config(config: Mapping[str, object]) -> Mapping[str, object]:
    env_config = config.get("env")
    if not isinstance(env_config, Mapping):
        raise ValueError("CAP-X YAML must contain an env mapping")
    return env_config


def _get_nested_mapping(value: object, key: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        return {}
    nested = value.get(key)
    return nested if isinstance(nested, Mapping) else {}


def _extract_task_language(*owners: object) -> str | None:
    """Read the benchmark instruction without coupling CAP-MAS to LIBERO types."""
    for owner in owners:
        handle = getattr(owner, "handle", None)
        language = getattr(handle, "task_language", None)
        if isinstance(language, str) and language.strip():
            return language.strip()
        language = getattr(owner, "task_language", None)
        if isinstance(language, str) and language.strip():
            return language.strip()
    return None


def _normalize_object_name(value: object) -> str:
    return " ".join(str(value).replace("_", " ").lower().split())


def build_capx_runtime_from_yaml(
    config_path: str | Path,
    *,
    loader: Callable[[str | Path], Mapping[str, object]] | None = None,
    instantiator: Callable[[object], object] | None = None,
    api_factory: Callable[[str], Callable[[object], object]] | None = None,
    api_name: str | None = None,
    skill_bindings: Mapping[str, str] | None = None,
    backend_id: str = "capx_libero",
    instantiate_code_env: bool = False,
    object_names: Sequence[str] | None = None,
    artifact_store: ArtifactSink | None = None,
    reset_hook: Callable[[object, int | None, Mapping[str, object]], None] | None = None,
) -> CAPXRuntimeBundle:
    """Build CAP-MAS resources from an existing CAP-X LIBERO YAML.

    The CAP-X environment wrapper and API registry remain the source of truth.
    CAP-MAS only adds an observation adapter, a typed-skill registry, and its
    own runtime boundary around the already constructed low-level environment.
    """
    load = loader or _default_loader
    instantiate = instantiator or _default_instantiator
    get_api_factory = api_factory or _default_api_factory
    config = load(config_path)
    env_config = _get_env_config(config)
    cfg = _get_nested_mapping(env_config, "cfg")
    low_level_cfg = _get_nested_mapping(cfg, "low_level")
    if not low_level_cfg:
        raise ValueError("CAP-X YAML env.cfg.low_level must be a mapping for CAP-MAS")
    if instantiate_code_env:
        env = instantiate(env_config)
        low_level_env = getattr(env, "low_level_env", env)
    else:
        direct_low_level_cfg = dict(low_level_cfg)
        for key in ("privileged", "enable_render", "viser_debug"):
            if key in cfg:
                direct_low_level_cfg[key] = cfg[key]
        env = instantiate(direct_low_level_cfg)
        low_level_env = env

    configured_api_names = cfg.get("apis", ())
    if not isinstance(configured_api_names, (list, tuple)):
        raise ValueError("CAP-X env.cfg.apis must be a list")
    names = tuple(str(name) for name in configured_api_names)
    if not names:
        raise ValueError("CAP-X YAML must configure at least one API")
    selected_api_name = api_name or names[0]
    if selected_api_name not in names:
        raise ValueError(f"requested API is not configured in YAML: {selected_api_name}")

    cached_apis = getattr(env, "_apis", {})
    if not isinstance(cached_apis, Mapping):
        cached_apis = {}
    apis: dict[str, object] = {}
    for name in names:
        api = cached_apis.get(name)
        if api is None:
            api = get_api_factory(name)(low_level_env)
        apis[name] = api
    primary_api = apis[selected_api_name]
    functions = getattr(primary_api, "functions", None)
    if not callable(functions):
        raise TypeError(f"CAP-X API does not expose functions(): {selected_api_name}")
    function_map = functions()
    if not isinstance(function_map, Mapping):
        raise TypeError(f"CAP-X API functions() must return a mapping: {selected_api_name}")

    if skill_bindings is None:
        bindings = dict(DEFAULT_LIBERO_SKILLS)
        for skill_id, function_name in OPTIONAL_LIBERO_SKILLS.items():
            if function_name in function_map:
                bindings[skill_id] = function_name
    else:
        bindings = dict(skill_bindings)

    artifacts = artifact_store if artifact_store is not None else InMemoryArtifactStore()
    observation_fn = function_map.get("get_observation")
    if not callable(observation_fn):
        raise ValueError("selected CAP-X API must expose get_observation")

    def observation_with_gripper_state() -> Mapping[str, object]:
        """Add the commanded state that CAP-X keeps on the low-level env."""
        raw = observation_fn()
        commanded_fraction = getattr(low_level_env, "_gripper_fraction", None)
        if commanded_fraction is None:
            return raw
        enriched = dict(raw)
        enriched["gripper_commanded_fraction"] = commanded_fraction
        return enriched

    get_all_object_poses = function_map.get("get_all_object_poses")
    get_object_pose = function_map.get("get_object_pose")
    get_object_geometry = function_map.get(
        "get_object_3d_points_and_masks_from_language"
    )
    sample_grasp_pose = function_map.get("sample_grasp_pose")
    configured_object_names = tuple(
        str(name)
        for name in _get_nested_mapping(cfg, "scene").get("object_names", ())
    )
    tracked_object_names = tuple(object_names or configured_object_names)
    task_language = _extract_task_language(low_level_env, env)
    source_object_name = tracked_object_names[0] if tracked_object_names else None
    pose_cache: dict[str, object] = {}
    function_overrides: dict[str, Callable[..., object]] = {}
    goto_pose = function_map.get("goto_pose")
    if callable(goto_pose):
        def lift_after_grasp(
            position: list[float],
            quaternion_wxyz: list[float],
            z_lift: float = 0.12,
        ) -> object:
            """Lift a grasped object vertically before lateral placement."""
            if len(position) != 3:
                raise ValueError("position must contain exactly three values")
            lifted_position = [
                float(position[0]),
                float(position[1]),
                float(position[2]) + float(z_lift),
            ]
            return goto_pose(
                position=lifted_position,
                quaternion_wxyz=quaternion_wxyz,
                z_approach=0.0,
            )

        function_overrides["lift_after_grasp"] = lift_after_grasp
        bindings.setdefault("lift_after_grasp", "lift_after_grasp")
    missing = [
        name
        for name in bindings.values()
        if name not in function_map and name not in function_overrides
    ]
    if missing:
        raise ValueError(f"CAP-X API is missing configured functions: {sorted(set(missing))}")

    if callable(get_object_pose):
        def _task_language_query(
            args: Sequence[object], kwargs: Mapping[str, object]
        ) -> tuple[list[object], dict[str, object]] | None:
            if not task_language:
                return None
            rewritten_args = list(args)
            rewritten_kwargs = dict(kwargs)
            if "object_name" in rewritten_kwargs:
                rewritten_kwargs["object_name"] = task_language
            elif rewritten_args:
                rewritten_args[0] = task_language
            else:
                return None
            return rewritten_args, rewritten_kwargs

        @wraps(get_object_pose)
        def tracked_get_object_pose(*args: object, **kwargs: object) -> object:
            name = kwargs.get("object_name", args[0] if args else None)
            is_source_object = (
                name is not None
                and _normalize_object_name(name)
                == _normalize_object_name(source_object_name)
            )
            fallback = _task_language_query(args, kwargs) if is_source_object else None
            try:
                result = get_object_pose(*args, **kwargs)
            except Exception as primary_error:
                if fallback is None:
                    raise
                try:
                    result = get_object_pose(*fallback[0], **fallback[1])
                except Exception:
                    raise primary_error
            else:
                if fallback is not None and _normalize_capx_object_pose(result) is None:
                    fallback_result = get_object_pose(*fallback[0], **fallback[1])
                    if _normalize_capx_object_pose(fallback_result) is not None:
                        result = fallback_result
            if name is not None:
                pose_cache[str(name)] = result
            return result

        function_overrides["get_object_pose"] = tracked_get_object_pose

    if callable(sample_grasp_pose):
        @wraps(sample_grasp_pose)
        def grounded_sample_grasp_pose(*args: object, **kwargs: object) -> object:
            name = kwargs.get("object_name", args[0] if args else None)
            is_source_object = (
                name is not None
                and _normalize_object_name(name)
                == _normalize_object_name(source_object_name)
            )
            if is_source_object and callable(get_object_pose):
                grounded_pose = pose_cache.get(str(name))
                if grounded_pose is None:
                    try:
                        grounded_pose = tracked_get_object_pose(*args, **kwargs)
                    except Exception:
                        grounded_pose = None
                normalized_pose = _normalize_capx_object_pose(grounded_pose)
                if normalized_pose is not None:
                    position, _ = normalized_pose
                    return position, (0.0, 1.0, 0.0, 0.0)
            try:
                return sample_grasp_pose(*args, **kwargs)
            except Exception as primary_error:
                fallback = _task_language_query(args, kwargs) if is_source_object else None
                if fallback is None:
                    raise
                try:
                    return sample_grasp_pose(*fallback[0], **fallback[1])
                except Exception:
                    raise primary_error

        function_overrides["sample_grasp_pose"] = grounded_sample_grasp_pose

    def target_placement_pose(object_name: str) -> PlacementPoseResult:
        if not callable(get_object_geometry):
            return PlacementPoseResult(
                None,
                "semantic_pose_fallback",
                "geometry_api_unavailable",
            )
        try:
            try:
                raw = get_object_geometry(object_name, use_multiview=True)
            except TypeError as exc:
                if "use_multiview" not in str(exc):
                    raise
                raw = get_object_geometry(object_name)
            if not isinstance(raw, Mapping):
                return PlacementPoseResult(
                    None,
                    "semantic_pose_fallback",
                    f"unexpected_geometry_payload:{type(raw).__name__}",
                )
            normalized_name = _normalize_object_name(object_name)
            release_clearance_m = (
                0.02 if "basket" in normalized_name else 0.0
            )
            pose = estimate_libero_placement_pose(
                raw.get("points_3d"),
                release_clearance_m=release_clearance_m,
            )
            if pose is None:
                return PlacementPoseResult(
                    None,
                    "semantic_pose_fallback",
                    "invalid_or_insufficient_pointcloud",
                )
            return PlacementPoseResult(pose, "geometry_pointcloud")
        except Exception as exc:
            return PlacementPoseResult(
                None,
                "semantic_pose_fallback",
                f"{type(exc).__name__}: {exc}",
            )

    def object_poses() -> Mapping[str, object]:
        poses: dict[str, object] = dict(pose_cache)
        if callable(get_all_object_poses):
            value = get_all_object_poses()
            if isinstance(value, Mapping):
                poses.update(value)
        return poses

    observation_provider = CAPXObservationProvider(
        observation_with_gripper_state,
        artifacts,
        object_poses_fn=object_poses,
        object_pose_fn=(
            tracked_get_object_pose
            if callable(get_object_pose) and not callable(get_all_object_poses)
            else None
        ),
        object_names=tracked_object_names,
        placement_pose_fn=(
            target_placement_pose if callable(get_object_geometry) else None
        ),
        placement_object_names=tuple(
            name for name in tracked_object_names if name != source_object_name
        ),
    )
    suite_name = str(low_level_cfg.get("suite_name", "libero"))
    raw_task_id = low_level_cfg.get("task_id", 0)
    task_id = f"{suite_name}_{raw_task_id}"
    backend = CAPXRobotBackend(
        low_level_env,
        observation_provider,
        task_id=task_id,
        suite_name=suite_name,
        backend_id=backend_id,
        reset_hook=reset_hook,
    )
    typed_skills: dict[Any, CAPXTypedSkill] = build_capx_skills(
        primary_api,
        bindings,
        function_overrides=function_overrides,
        default_postconditions=DEFAULT_LIBERO_POSTCONDITIONS,
    )
    registry = SkillRegistry()
    for reference, skill in typed_skills.items():
        registry.register(reference, skill)
    return CAPXRuntimeBundle(
        environment=env,
        low_level_environment=low_level_env,
        api=primary_api,
        apis=apis,
        observation_provider=observation_provider,
        backend=backend,
        skill_registry=registry,
        task_id=task_id,
        suite_name=suite_name,
        task_language=task_language,
    )
