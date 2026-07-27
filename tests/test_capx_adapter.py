import pytest

from capmas.backends.capx import (
    CAPXObservationProvider,
    CAPXRobotBackend,
    CAPXTypedSkill,
    build_capx_skills,
)
from capmas.contracts.action import ActionContract, ExecutionBudget, SkillCall
from capmas.contracts.core import SkillRef
from capmas.perception.artifacts import InMemoryArtifactStore
from capmas.perception.protocol import ObservationBundle
from capmas.skills.registry import SkillRegistry


class FakeAPI:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def move(self, target: str) -> dict[str, str]:
        self.calls.append(target)
        return {"target": target}

    def functions(self) -> dict[str, object]:
        return {"move": self.move}


def test_capx_api_function_becomes_typed_skill() -> None:
    api = FakeAPI()
    skills = build_capx_skills(api, {"goto_pose": "move"})
    skill = skills[SkillRef("goto_pose", "capx-compat-1")]

    skill.validate_args({"target": "box"})
    result = skill.execute({"target": "box"}, ExecutionBudget(1000, 10))

    assert result.ok is True
    assert api.calls == ["box"]


def test_skill_registry_exposes_callable_argument_names_for_strict_schemas() -> None:
    api = FakeAPI()
    skills = build_capx_skills(api, {"goto_pose": "move"})
    registry = SkillRegistry()
    for reference, skill in skills.items():
        registry.register(reference, skill)

    assert registry.argument_names() == ("target",)
    assert registry.argument_schemas() == {"target": {"type": "string"}}


def test_registry_preserves_signature_details_for_invalid_arguments() -> None:
    def goto_pose(position, quaternion_wxyz, z_approach=0.0):
        return position, quaternion_wxyz, z_approach

    registry = SkillRegistry()
    goto_pose_ref = SkillRef("goto_pose", "1.0.0")
    skill = CAPXTypedSkill(goto_pose_ref, goto_pose)
    registry.register(goto_pose_ref, skill)

    contract = ActionContract(
        contract_id="invalid",
        episode_id="episode",
        episode_epoch=1,
        parent_scene_version=1,
        subgoal_id="place",
        skills=(SkillCall(goto_pose_ref, {"approach_z": 0.1}),),
        expected_postconditions=(),
        max_duration_ms=1000,
        max_sim_steps=10,
        proposed_by="test",
    )

    with pytest.raises(ValueError) as error:
        registry.validate_contract(contract)

    message = str(error.value)
    assert "goto_pose" in message
    assert "position" in message
    assert "quaternion_wxyz" in message
    assert "approach_z" in message


def test_capx_observation_is_artifactized_and_not_exposed_as_raw_arrays() -> None:
    provider = CAPXObservationProvider(
        observation_fn=lambda: {
            "timestamp_ns": 42,
            "agentview": {
                "intrinsics": [[1, 0, 2], [0, 1, 2], [0, 0, 1]],
                "pose_mat": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                "images": {"rgb": [[1]], "depth": [[2]]},
            },
            "robot_joint_pos": [0.0, 1.0],
            "robot_cartesian_pos": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        },
        artifacts=InMemoryArtifactStore(),
    )

    observation = provider.capture()

    assert observation.frames[0].rgb is not None
    assert observation.frames[0].rgb.uri.startswith("artifact://")
    assert observation.robot_state["joint_position"].uri.startswith("artifact://")


def test_capx_object_pose_is_aligned_to_scene_track_pose() -> None:
    provider = CAPXObservationProvider(
        observation_fn=lambda: {
            "timestamp_ns": 42,
            "robot_cartesian_pos": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        },
        artifacts=InMemoryArtifactStore(),
        object_pose_fn=lambda name: (
            [0.2, 0.3, 0.4],
            [0.707, 0.0, 0.707, 0.0],
        ) if name == "akita_black_bowl" else (None, None),
        object_names=("akita_black_bowl",),
    )

    tracks = provider.capture_object_tracks(
        timestamp_ns=42,
        episode_id="episode-1",
        episode_epoch=1,
    )

    assert tracks[0].track_id == "akita_black_bowl"
    assert tracks[0].label == "akita black bowl"
    assert tracks[0].pose_wxyz_xyz == (0.707, 0.0, 0.707, 0.0, 0.2, 0.3, 0.4)


def test_capx_robot_pose_is_reordered_for_scene_geometry() -> None:
    provider = CAPXObservationProvider(
        observation_fn=lambda: {
            "timestamp_ns": 42,
            "robot_cartesian_pos": [0.2, 0.3, 0.4, 1.0, 0.0, 0.0, 0.0, 0.25],
        },
        artifacts=InMemoryArtifactStore(),
    )

    observation = provider.capture()

    assert observation.robot_state["ee_pose_wxyz_xyz"] == (1.0, 0.0, 0.0, 0.0, 0.2, 0.3, 0.4)
    assert observation.robot_state["gripper_opening"] == 0.25


def test_capx_snapshot_publish_time_is_after_observation_processing(monkeypatch) -> None:
    import capmas.backends.capx as capx_module

    class FakeEnv:
        def reset(self, *, seed=None, options=None):
            del seed, options

    class FakeObservationProvider:
        def capture(self) -> ObservationBundle:
            return ObservationBundle(100, (), {})

        def capture_object_tracks(self, *, timestamp_ns, episode_id, episode_epoch):
            del timestamp_ns, episode_id, episode_epoch
            assert capx_module.time.time_ns() == 100
            return ()

    clock_values = iter((100, 200))
    monkeypatch.setattr(
        "capmas.backends.capx.time.time_ns",
        lambda: next(clock_values),
    )
    backend = CAPXRobotBackend(
        FakeEnv(),
        FakeObservationProvider(),
        task_id="task",
        suite_name="suite",
    )

    episode = backend.reset(seed=1)

    assert episode.initial_scene.sensor_timestamp_ns == 100
    assert episode.initial_scene.publish_timestamp_ns == 200
