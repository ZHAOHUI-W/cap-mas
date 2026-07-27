from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from functools import wraps

from capmas.backends.capx import (
    CAPXObservationProvider,
    CAPXRobotBackend,
    CAPXTypedSkill,
    build_capx_skills,
)
from capmas.perception.artifacts import InMemoryArtifactStore
from capmas.skills.registry import SkillRegistry


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
    missing = [name for name in bindings.values() if name not in function_map]
    if missing:
        raise ValueError(f"CAP-X API is missing configured functions: {sorted(set(missing))}")

    artifacts = InMemoryArtifactStore()
    observation_fn = function_map.get("get_observation")
    if not callable(observation_fn):
        raise ValueError("selected CAP-X API must expose get_observation")
    get_all_object_poses = function_map.get("get_all_object_poses")
    get_object_pose = function_map.get("get_object_pose")
    configured_object_names = tuple(
        str(name)
        for name in _get_nested_mapping(cfg, "scene").get("object_names", ())
    )
    tracked_object_names = tuple(object_names or configured_object_names)
    pose_cache: dict[str, object] = {}
    function_overrides: dict[str, Callable[..., object]] = {}
    if callable(get_object_pose):
        @wraps(get_object_pose)
        def tracked_get_object_pose(*args: object, **kwargs: object) -> object:
            result = get_object_pose(*args, **kwargs)
            name = kwargs.get("object_name", args[0] if args else None)
            if name is not None:
                pose_cache[str(name)] = result
            return result

        function_overrides["get_object_pose"] = tracked_get_object_pose

    def object_poses() -> Mapping[str, object]:
        poses: dict[str, object] = dict(pose_cache)
        if callable(get_all_object_poses):
            value = get_all_object_poses()
            if isinstance(value, Mapping):
                poses.update(value)
        return poses

    observation_provider = CAPXObservationProvider(
        observation_fn,
        artifacts,
        object_poses_fn=object_poses,
        object_pose_fn=(
            get_object_pose
            if callable(get_object_pose) and not callable(get_all_object_poses)
            else None
        ),
        object_names=tracked_object_names,
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
    )
    typed_skills: dict[Any, CAPXTypedSkill] = build_capx_skills(
        primary_api,
        bindings,
        function_overrides=function_overrides,
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
    )
