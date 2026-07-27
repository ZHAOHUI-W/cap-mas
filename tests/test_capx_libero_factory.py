from dataclasses import dataclass

from capmas.backends.capx import CAPXTypedSkill
from capmas.backends.capx_libero_factory import build_capx_runtime_from_yaml
from capmas.contracts.core import SkillRef


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
