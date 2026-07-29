from dataclasses import dataclass

from capmas.backends.capx import CAPXTypedSkill
from capmas.backends.capx_libero_factory import build_capx_runtime_from_yaml
from capmas.contracts.action import ExecutionBudget
from capmas.contracts.core import SkillRef
from capmas.perception.artifact_bridge import EncodedArtifactStore, FileArtifactStore, NumpyArtifactCodec


class FakeApi:
    def get_observation(self):
        return {
            "timestamp_ns": 1,
            "agentview": {
                "intrinsics": [[1]],
                "pose_mat": [[1]],
                "images": {"rgb": [[1]], "depth": [[1]]},
            },
            "robot_joint_pos": [0.0],
            "robot_cartesian_pos": [0.0] * 8,
        }

    def open_gripper(self):
        return None

    def functions(self):
        return {
            "get_observation": self.get_observation,
            "open_gripper": self.open_gripper,
        }


class FakeLowLevelEnv:
    def __init__(self):
        self.reset_calls = []

    def reset(self, seed=None, options=None):
        self.reset_calls.append((seed, options))


class GroundingApi:
    def __init__(self) -> None:
        self.pose_queries: list[str] = []
        self.grasp_queries: list[str] = []

    def get_observation(self):
        return {
            "timestamp_ns": 1,
            "robot_cartesian_pos": [0.0] * 8,
        }

    def get_object_pose(self, object_name, use_multiview=True):
        del use_multiview
        self.pose_queries.append(object_name)
        return ([0.1, 0.2, 0.3], [1.0, 0.0, 0.0, 0.0])

    def sample_grasp_pose(self, object_name, use_multiview=True):
        del use_multiview
        self.grasp_queries.append(object_name)
        return ([0.9, 0.8, 0.7], [0.0, 0.0, 0.0, 1.0])

    def goto_pose(self, position, quaternion_wxyz, z_approach=0.0):
        return position, quaternion_wxyz, z_approach

    def open_gripper(self):
        return None

    def close_gripper(self):
        return None

    def functions(self):
        return {
            "get_observation": self.get_observation,
            "get_object_pose": self.get_object_pose,
            "sample_grasp_pose": self.sample_grasp_pose,
            "goto_pose": self.goto_pose,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
        }


class GroundingLowLevelEnv(FakeLowLevelEnv):
    def __init__(self) -> None:
        super().__init__()
        self.handle = type(
            "Handle",
            (),
            {
                "task_language": (
                    "Pick up the black bowl between the plate and the ramekin "
                    "and place it on the plate"
                )
            },
        )()


class UnavailablePoseGroundingApi(GroundingApi):
    def get_object_pose(self, object_name, use_multiview=True):
        del use_multiview
        self.pose_queries.append(object_name)
        raise ValueError("no grounded pose available")


@dataclass
class FakeHighLevelEnv:
    low_level_env: FakeLowLevelEnv
    _apis: dict[str, FakeApi]


def test_factory_loads_yaml_env_and_reuses_registered_api_instance() -> None:
    low_level = FakeLowLevelEnv()
    api = FakeApi()
    high_level = FakeHighLevelEnv(low_level, {"FrankaLiberoApi": api})
    config = {
        "env": {
            "_target_": "capx.envs.tasks.franka.franka_libero_env.FrankaLiberoCodeEnv",
            "cfg": {
                "low_level": {"suite_name": "libero_spatial", "task_id": 0},
                "apis": ["FrankaLiberoApi"],
            },
        }
    }

    bundle = build_capx_runtime_from_yaml(
        "unused.yaml",
        loader=lambda path: config,
        instantiator=lambda env_config: high_level,
        skill_bindings={
            "get_observation": "get_observation",
            "open_gripper": "open_gripper",
        },
        instantiate_code_env=True,
    )

    assert bundle.api is api
    assert bundle.backend.env is low_level
    assert SkillRef("open_gripper", "capx-compat-1") in bundle.skill_registry._skills
    assert bundle.task_id == "libero_spatial_0"


def test_factory_uses_capx_registered_api_when_wrapper_has_no_api_cache() -> None:
    low_level = FakeLowLevelEnv()
    api = FakeApi()
    high_level = FakeHighLevelEnv(low_level, {})
    config = {
        "env": {
            "cfg": {
                "low_level": {"suite_name": "libero_spatial", "task_id": 0},
                "apis": ["FrankaLiberoApi"],
            }
        }
    }
    calls = []

    def api_factory(name):
        calls.append(name)
        return lambda env: api

    bundle = build_capx_runtime_from_yaml(
        "unused.yaml",
        loader=lambda path: config,
        instantiator=lambda env_config: high_level,
        api_factory=api_factory,
        skill_bindings={
            "get_observation": "get_observation",
            "open_gripper": "open_gripper",
        },
        instantiate_code_env=True,
    )

    assert calls == ["FrankaLiberoApi"]
    assert bundle.api is api


def test_factory_constructs_low_level_directly_by_default() -> None:
    low_level = FakeLowLevelEnv()
    api = FakeApi()
    config = {
        "env": {
            "_target_": "code.executor.MustNotBeConstructed",
            "cfg": {
                "low_level": {"_target_": "low.level.Env", "suite_name": "libero_spatial", "task_id": 0},
                "apis": ["FrankaLiberoApi"],
            },
        }
    }
    instantiated = []

    def instantiator(env_config):
        instantiated.append(env_config)
        return low_level

    bundle = build_capx_runtime_from_yaml(
        "unused.yaml",
        loader=lambda path: config,
        instantiator=instantiator,
        api_factory=lambda name: (lambda env: api),
        skill_bindings={
            "get_observation": "get_observation",
            "open_gripper": "open_gripper",
        },
    )

    assert bundle.environment is low_level
    assert instantiated == [config["env"]["cfg"]["low_level"]]


def test_factory_injects_replaceable_artifact_store_into_capx_provider(tmp_path) -> None:
    low_level = FakeLowLevelEnv()
    api = FakeApi()
    config = {
        "env": {
            "cfg": {
                "low_level": {"suite_name": "libero_spatial", "task_id": 0},
                "apis": ["FrankaLiberoApi"],
            }
        }
    }
    artifacts = EncodedArtifactStore(FileArtifactStore(tmp_path), NumpyArtifactCodec())

    bundle = build_capx_runtime_from_yaml(
        "unused.yaml",
        loader=lambda path: config,
        instantiator=lambda env_config: low_level,
        api_factory=lambda name: (lambda env: api),
        skill_bindings={
            "get_observation": "get_observation",
            "open_gripper": "open_gripper",
        },
        artifact_store=artifacts,
    )

    assert bundle.observation_provider.artifacts is artifacts


def test_factory_preserves_task_language_and_routes_source_grasp_query() -> None:
    low_level = GroundingLowLevelEnv()
    api = GroundingApi()
    config = {
        "env": {
            "cfg": {
                "low_level": {"suite_name": "libero_spatial", "task_id": 0},
                "apis": ["FrankaLiberoApi"],
                "scene": {"object_names": ["akita black bowl", "plate"]},
            }
        }
    }

    bundle = build_capx_runtime_from_yaml(
        "unused.yaml",
        loader=lambda path: config,
        instantiator=lambda env_config: low_level,
        api_factory=lambda name: (lambda env: api),
        object_names=("akita black bowl", "plate"),
    )

    assert bundle.task_language == low_level.handle.task_language
    result = bundle.skill_registry.get(
        SkillRef("sample_grasp_pose", "capx-compat-1")
    ).execute(
        {"object_name": "akita black bowl"},
        ExecutionBudget(max_duration_ms=1000, max_sim_steps=10),
    )

    assert result.ok is True
    assert result.output["result"] == (
        (0.1, 0.2, 0.3), (0.0, 1.0, 0.0, 0.0)
    )
    assert api.pose_queries == [low_level.handle.task_language]
    assert api.grasp_queries == []


def test_factory_observation_uses_same_relation_aware_source_query() -> None:
    low_level = GroundingLowLevelEnv()
    api = GroundingApi()
    config = {
        "env": {
            "cfg": {
                "low_level": {"suite_name": "libero_spatial", "task_id": 0},
                "apis": ["FrankaLiberoApi"],
            }
        }
    }

    bundle = build_capx_runtime_from_yaml(
        "unused.yaml",
        loader=lambda path: config,
        instantiator=lambda env_config: low_level,
        api_factory=lambda name: (lambda env: api),
        object_names=("akita black bowl", "plate"),
    )

    tracks = bundle.observation_provider.capture_object_tracks(
        timestamp_ns=1,
        episode_id="episode",
        episode_epoch=1,
    )

    assert tracks[0].track_id == "akita black bowl"
    assert api.pose_queries == [low_level.handle.task_language, "plate"]


def test_factory_grounded_grasp_prefers_relation_aware_object_pose() -> None:
    low_level = GroundingLowLevelEnv()
    api = GroundingApi()
    config = {
        "env": {
            "cfg": {
                "low_level": {"suite_name": "libero_spatial", "task_id": 0},
                "apis": ["FrankaLiberoApi"],
                "scene": {"object_names": ["akita black bowl", "plate"]},
            }
        }
    }

    bundle = build_capx_runtime_from_yaml(
        "unused.yaml",
        loader=lambda path: config,
        instantiator=lambda env_config: low_level,
        api_factory=lambda name: (lambda env: api),
        object_names=("akita black bowl", "plate"),
    )

    result = bundle.skill_registry.get(
        SkillRef("sample_grasp_pose", "capx-compat-1")
    ).execute(
        {"object_name": "akita black bowl"},
        ExecutionBudget(max_duration_ms=1000, max_sim_steps=10),
    )

    assert result.ok is True
    assert result.output["result"] == (
        ((0.1, 0.2, 0.3), (0.0, 1.0, 0.0, 0.0))
    )
    assert api.pose_queries == [low_level.handle.task_language]
    assert api.grasp_queries == []


def test_factory_grounded_grasp_falls_back_to_raw_sample_on_pose_failure() -> None:
    low_level = GroundingLowLevelEnv()
    api = UnavailablePoseGroundingApi()
    config = {
        "env": {
            "cfg": {
                "low_level": {"suite_name": "libero_spatial", "task_id": 0},
                "apis": ["FrankaLiberoApi"],
                "scene": {"object_names": ["akita black bowl", "plate"]},
            }
        }
    }

    bundle = build_capx_runtime_from_yaml(
        "unused.yaml",
        loader=lambda path: config,
        instantiator=lambda env_config: low_level,
        api_factory=lambda name: (lambda env: api),
        object_names=("akita black bowl", "plate"),
    )

    result = bundle.skill_registry.get(
        SkillRef("sample_grasp_pose", "capx-compat-1")
    ).execute(
        {"object_name": "akita black bowl"},
        ExecutionBudget(max_duration_ms=1000, max_sim_steps=10),
    )

    assert result.ok is True
    assert result.output["result"] == (
        [0.9, 0.8, 0.7], [0.0, 0.0, 0.0, 1.0]
    )
    assert api.pose_queries == [low_level.handle.task_language]
    assert api.grasp_queries == [low_level.handle.task_language]
