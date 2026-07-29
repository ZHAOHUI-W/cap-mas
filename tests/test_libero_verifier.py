import pytest

from capmas.contracts.action import ActionContract, SkillCall, SkillOutputRef
from capmas.contracts.core import SkillRef
from capmas.contracts.graph import MotionIntent, SubgraphNodeSpec, SubgraphSpec
from capmas.contracts.scene import ObjectTrack, SceneSnapshot
from capmas.contracts.trace import ExecutionTrace
from capmas.verification.libero import (
    LiberoObservableVerifier,
    compile_time_preconditions,
    ground_libero_grasp_subgraph,
    repair_libero_grasp_subgraph,
    validate_libero_grasp_subgraph,
    validate_libero_skill_sequence,
)
from capmas.verification.predicates import PredicateBasedVerifier


def make_contract(**kwargs):
    values = {
        "contract_id": "c",
        "episode_id": "ep",
        "episode_epoch": 1,
        "parent_scene_version": 0,
        "subgoal_id": "sg",
        "skills": (),
        "expected_postconditions": (),
        "max_duration_ms": 100,
        "max_sim_steps": 10,
        "proposed_by": "policy",
    }
    values.update(kwargs)
    return ActionContract(**values)


def test_libero_verifier_uses_scene_facts_for_gripper_postconditions() -> None:
    before = SceneSnapshot("ep", 1, 0, 1, 1, {"gripper_opening": 1.0})
    after = SceneSnapshot("ep", 1, 1, 2, 2, {"gripper_opening": 0.0})
    contract = make_contract(expected_postconditions=("scene_advanced", "gripper_closed"))
    verifier = LiberoObservableVerifier()

    result = verifier.commit(contract, before, after, ExecutionTrace("t", "ep", 1, "c", "l", 0, 0, 1, 1, 2, "completed"))

    assert result.passed is True
    assert result.failure_class is None


def test_libero_skill_sequence_requires_motion_after_grasp_sampling() -> None:
    with pytest.raises(ValueError, match="sample_grasp_pose.*goto_pose"):
        validate_libero_skill_sequence(
            (
                SkillCall(SkillRef("sample_grasp_pose", "1"), {"object_name": "bowl"}),
                SkillCall(SkillRef("close_gripper", "1"), {}),
            )
        )

    with pytest.raises(ValueError, match="must use sample_grasp_pose"):
        validate_libero_skill_sequence(
            (
                SkillCall(SkillRef("goto_pose", "1"), {}),
                SkillCall(SkillRef("close_gripper", "1"), {}),
            )
        )

    validate_libero_skill_sequence(
        (
            SkillCall(SkillRef("sample_grasp_pose", "1"), {"object_name": "bowl"}),
            SkillCall(
                SkillRef("goto_pose", "1"),
                {"position": [0.1, 0.2, 0.3], "quaternion_wxyz": [1, 0, 0, 0]},
            ),
            SkillCall(SkillRef("close_gripper", "1"), {}),
        )
    )


def test_libero_compile_time_preconditions_defer_mutable_robot_state() -> None:
    assert compile_time_preconditions(
        (
            "track_exists:plate",
            "object_in_gripper(akita_black_bowl)",
            "gripper_closed()",
            "scene_fresh(1000)",
        )
    ) == ("track_exists:plate",)


def test_libero_grasp_grounding_uses_sample_output_and_default_approach() -> None:
    subgraph = SubgraphSpec(
        "grasp",
        "grasp",
        "grasp",
        (
            SubgraphNodeSpec(
                "action",
                "grasp",
                (
                    SkillCall(SkillRef("sample_grasp_pose", "1"), {}),
                    SkillCall(
                        SkillRef("goto_pose", "1"),
                        {"position": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0]},
                    ),
                    SkillCall(SkillRef("close_gripper", "1"), {}),
                ),
            ),
        ),
        (),
        "action",
        ("action",),
        ("action",),
    )

    grounded = ground_libero_grasp_subgraph(subgraph)
    args = grounded.nodes[0].skill_calls[1].args

    assert args["position"].call_index == 0
    assert args["position"].path == ("result", 0)
    assert args["quaternion_wxyz"].path == ("result", 1)
    assert args["z_approach"] == 0.10


def test_libero_grasp_grounding_inserts_lift_after_close() -> None:
    subgraph = SubgraphSpec(
        "grasp",
        "grasp",
        "grasp",
        (
            SubgraphNodeSpec(
                "action",
                "grasp",
                (
                    SkillCall(SkillRef("sample_grasp_pose", "1"), {}),
                    SkillCall(
                        SkillRef("goto_pose", "1"),
                        {"position": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0]},
                    ),
                    SkillCall(SkillRef("close_gripper", "1"), {}),
                ),
            ),
        ),
        (),
        "action",
        ("action",),
        ("action",),
    )

    grounded = ground_libero_grasp_subgraph(subgraph)
    calls = grounded.nodes[0].skill_calls

    assert [call.skill.skill_id for call in calls] == [
        "sample_grasp_pose",
        "goto_pose",
        "close_gripper",
        "lift_after_grasp",
    ]
    assert calls[-1].args == {
        "position": SkillOutputRef(0, ("result", 0)),
        "quaternion_wxyz": SkillOutputRef(0, ("result", 1)),
        "z_lift": 0.12,
    }


def test_libero_grasp_repair_inserts_motion_after_sample_before_close() -> None:
    subgraph = SubgraphSpec(
        "grasp",
        "grasp",
        "grasp",
        (
            SubgraphNodeSpec(
                "action",
                "grasp",
                (
                    SkillCall(SkillRef("sample_grasp_pose", "1"), {}),
                    SkillCall(SkillRef("close_gripper", "1"), {}),
                ),
            ),
        ),
        (),
        "action",
        ("action",),
        ("action",),
    )

    repaired = repair_libero_grasp_subgraph(subgraph)

    assert [call.skill.skill_id for call in repaired.nodes[0].skill_calls] == [
        "sample_grasp_pose",
        "goto_pose",
        "close_gripper",
    ]
    assert repaired.nodes[0].skill_calls[1].args["position"].call_index == 0


def test_libero_grasp_repair_inserts_motion_for_any_incomplete_sampled_grasp() -> None:
    subgraph = SubgraphSpec(
        "grasp",
        "grasp",
        "grasp",
        (
            SubgraphNodeSpec(
                "action",
                "grasp",
                (SkillCall(SkillRef("sample_grasp_pose", "1"), {}),),
            ),
        ),
        (),
        "action",
        ("action",),
        ("action",),
    )

    repaired = repair_libero_grasp_subgraph(subgraph)

    assert [call.skill.skill_id for call in repaired.nodes[0].skill_calls] == [
        "sample_grasp_pose",
        "goto_pose",
    ]


def test_libero_grasp_repair_removes_goto_only_args_from_sampler() -> None:
    subgraph = SubgraphSpec(
        "grasp",
        "grasp",
        "grasp",
        (
            SubgraphNodeSpec(
                "action",
                "grasp",
                (
                    SkillCall(
                        SkillRef("sample_grasp_pose", "1"),
                        {"object_name": "bowl", "z_approach": 0.1, "z_lift": 0.12},
                    ),
                    SkillCall(SkillRef("goto_pose", "1"), {}),
                ),
            ),
        ),
        (),
        "action",
        ("action",),
        ("action",),
    )

    repaired = repair_libero_grasp_subgraph(subgraph)

    assert repaired.nodes[0].skill_calls[0].args == {"object_name": "bowl"}


def test_libero_grasp_subgraph_rejects_cross_node_grasp_dataflow() -> None:
    subgraph = SubgraphSpec(
        "grasp",
        "grasp",
        "grasp",
        (
            SubgraphNodeSpec(
                "sample",
                "sample",
                (SkillCall(SkillRef("sample_grasp_pose", "1"), {"object_name": "bowl"}),),
            ),
            SubgraphNodeSpec(
                "close",
                "close",
                (SkillCall(SkillRef("close_gripper", "1"), {}),),
            ),
        ),
        (),
        "sample",
        ("close",),
        ("close",),
    )

    with pytest.raises(ValueError, match="one action node"):
        validate_libero_grasp_subgraph(subgraph)


def test_libero_grasp_subgraph_rejects_natural_language_pose_placeholder() -> None:
    subgraph = SubgraphSpec(
        "grasp",
        "grasp",
        "grasp",
        (
            SubgraphNodeSpec(
                "action",
                "action",
                (
                    SkillCall(SkillRef("sample_grasp_pose", "1"), {"object_name": "bowl"}),
                    SkillCall(
                        SkillRef("goto_pose", "1"),
                        {
                            "position": "sampled_grasp_position(bowl)",
                            "quaternion_wxyz": "sampled_grasp_quaternion_wxyz(bowl)",
                        },
                    ),
                ),
            ),
        ),
        (),
        "action",
        ("action",),
        ("action",),
    )

    with pytest.raises(ValueError, match="natural-language placeholder"):
        validate_libero_grasp_subgraph(subgraph)


def test_libero_placement_grounding_uses_scene_target_pose() -> None:
    subgraph = SubgraphSpec(
        "place",
        "place",
        "place",
        (
            SubgraphNodeSpec(
                "action",
                "place",
                (
                    SkillCall(
                        SkillRef("goto_pose", "1"),
                        {
                            "position": [9, 9, 9],
                            "quaternion_wxyz": [1, 0, 0, 0],
                            "z_approach": 0.06,
                        },
                    ),
                    SkillCall(SkillRef("open_gripper", "1"), {}),
                ),
                postconditions=("object_at_target(bowl,plate)",),
            ),
        ),
        (),
        "action",
        ("action",),
        ("action",),
    )
    scene = SceneSnapshot(
        "ep",
        1,
        0,
        1,
        1,
        {},
        objects=(_track("plate", "plate", (0.4, 0.5, 0.6)),),
    )

    grounded = ground_libero_grasp_subgraph(subgraph, scene)
    args = grounded.nodes[0].skill_calls[0].args

    assert args["position"] == (0.4, 0.5, 0.6)
    assert args["quaternion_wxyz"] == (0.0, 1.0, 0.0, 0.0)
    assert args["z_approach"] == 0.12


def test_libero_placement_grounding_rebinds_motion_intent_to_effective_pose() -> None:
    subgraph = SubgraphSpec(
        "place",
        "place",
        "place",
        (
            SubgraphNodeSpec(
                "action",
                "place",
                (
                    SkillCall(
                        SkillRef("goto_pose", "1"),
                        {
                            "position": [9, 9, 9],
                            "quaternion_wxyz": [1, 0, 0, 0],
                        },
                    ),
                ),
                postconditions=("object_at_target(bowl,plate)",),
                motion_intent=MotionIntent(
                    "place",
                    target_track_id="plate",
                    approach_vector_xyz=(0.0, 0.0, -1.0),
                    target_pose_wxyz_xyz=(1.0, 0.0, 0.0, 0.0, 9.0, 9.0, 9.0),
                ),
            ),
        ),
        (),
        "action",
        ("action",),
        ("action",),
    )
    scene = SceneSnapshot(
        "ep",
        1,
        0,
        1,
        1,
        {},
        objects=(_track("plate", "plate", (0.4, 0.5, 0.6)),),
    )

    grounded = ground_libero_grasp_subgraph(subgraph, scene)
    intent = grounded.nodes[0].motion_intent

    assert intent is not None
    assert intent.target_track_id == "plate"
    assert intent.approach_vector_xyz == (0.0, 0.0, -1.0)
    assert intent.target_pose_wxyz_xyz == (0.0, 1.0, 0.0, 0.0, 0.4, 0.5, 0.6)


def test_libero_placement_grounding_reads_target_predicate_from_subgraph_checkpoint() -> None:
    subgraph = SubgraphSpec(
        "place",
        "place",
        "place",
        (
            SubgraphNodeSpec(
                "move",
                "move to release pose",
                (
                    SkillCall(
                        SkillRef("goto_pose", "1"),
                        {"position": [9, 9, 9], "quaternion_wxyz": [1, 0, 0, 0]},
                    ),
                ),
                postconditions=("scene_fresh(1000)",),
            ),
            SubgraphNodeSpec(
                "release",
                "release object",
                (SkillCall(SkillRef("open_gripper", "1"), {}),),
                postconditions=("object_at_target(bowl,plate)",),
            ),
        ),
        (),
        "move",
        ("release",),
        ("move",),
    )
    scene = SceneSnapshot(
        "ep",
        1,
        0,
        1,
        1,
        {},
        objects=(_track("plate", "plate", (0.4, 0.5, 0.6)),),
    )

    grounded = ground_libero_grasp_subgraph(subgraph, scene)
    args = grounded.nodes[0].skill_calls[0].args

    assert args["position"] == (0.4, 0.5, 0.6)
    assert args["quaternion_wxyz"] == (0.0, 1.0, 0.0, 0.0)
    assert args["z_approach"] == 0.12


def test_libero_verifier_rejects_unknown_precondition_without_evaluator_access() -> None:
    scene = SceneSnapshot("ep", 1, 0, 1, 1, {"gripper_opening": 1.0})
    contract = make_contract(preconditions=("task_completed",))

    result = LiberoObservableVerifier().approve(contract, scene)

    assert result.passed is False
    assert result.failure_class == "PRECONDITION_FAILED"


def _track(track_id: str, label: str, position: tuple[float, float, float]) -> ObjectTrack:
    return ObjectTrack(
        track_id=track_id,
        label=label,
        pose_wxyz_xyz=(1.0, 0.0, 0.0, 0.0, *position),
        confidence=0.9,
        last_seen_ns=1_000_000,
    )


def test_predicate_verifier_checks_object_in_gripper_from_scene_geometry() -> None:
    scene = SceneSnapshot(
        "ep",
        1,
        0,
        1_000_000,
        1_000_000,
        {
            "gripper_opening": 0.0,
            "ee_pose_wxyz_xyz": (1.0, 0.0, 0.0, 0.0, 0.20, 0.20, 0.20),
        },
        objects=(_track("bowl-1", "akita black bowl", (0.21, 0.20, 0.20)),),
    )
    contract = make_contract(preconditions=("object_in_gripper(akita_black_bowl)",))

    result = PredicateBasedVerifier().approve(contract, scene)

    assert result.passed is True
    assert result.predicate_results[0].confidence == 0.9


def test_predicate_verifier_separates_near_gripper_geometry_from_held_state() -> None:
    scene = SceneSnapshot(
        "ep",
        1,
        0,
        1_000_000,
        1_000_000,
        {
            "gripper_opening": 1.0,
            "ee_pose_wxyz_xyz": (1.0, 0.0, 0.0, 0.0, 0.20, 0.20, 0.20),
        },
        objects=(_track("bowl-1", "bowl", (0.21, 0.20, 0.20)),),
    )

    near = PredicateBasedVerifier().approve(
        make_contract(preconditions=("object_near_gripper(bowl)",)),
        scene,
    )
    held = PredicateBasedVerifier().approve(
        make_contract(preconditions=("object_in_gripper(bowl)",)),
        scene,
    )

    assert near.passed is True
    assert held.passed is False


def test_predicate_verifier_accepts_libero_tcp_to_object_center_offset() -> None:
    scene = SceneSnapshot(
        "ep",
        1,
        0,
        1_000_000,
        1_000_000,
        {
            "gripper_opening": 0.0,
            "ee_pose_wxyz_xyz": (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        },
        objects=(_track("bowl-1", "akita black bowl", (0.14, 0.0, 0.0)),),
    )
    contract = make_contract(preconditions=("object_in_gripper(bowl-1)",))

    result = PredicateBasedVerifier().approve(contract, scene)

    assert result.passed is True


def test_predicate_verifier_checks_object_at_target_distance() -> None:
    before = SceneSnapshot(
        "ep",
        1,
        0,
        1_000_000,
        1_000_000,
        {"gripper_opening": 0.0},
        objects=(
            _track("bowl-1", "akita black bowl", (0.20, 0.20, 0.20)),
            _track("plate-1", "plate", (0.40, 0.20, 0.20)),
        ),
    )
    after = SceneSnapshot(
        "ep",
        1,
        1,
        2_000_000,
        2_000_000,
        {"gripper_opening": 1.0},
        objects=(
            _track("bowl-1", "akita black bowl", (0.40, 0.20, 0.20)),
            _track("plate-1", "plate", (0.42, 0.20, 0.20)),
        ),
    )
    contract = make_contract(expected_postconditions=("object_at_target(bowl-1,plate)",))

    result = PredicateBasedVerifier().commit(
        contract,
        before,
        after,
        ExecutionTrace("t", "ep", 1, "c", "l", 0, 0, 1, 1, 2, "completed"),
    )

    assert result.passed is True


def test_predicate_verifier_checks_scene_fresh_threshold() -> None:
    scene = SceneSnapshot("ep", 1, 0, 1_000_000, 1_000_000, {})
    contract = make_contract(preconditions=("scene_fresh(5)",))

    result = PredicateBasedVerifier(clock=lambda: 4_000_000).approve(contract, scene)

    assert result.passed is True

    stale = PredicateBasedVerifier(clock=lambda: 7_000_000).approve(contract, scene)
    assert stale.passed is False
    assert "exceeds" in (stale.predicate_results[0].reason or "")


def test_predicate_verifier_checks_gripper_open_and_closed() -> None:
    open_scene = SceneSnapshot("ep", 1, 0, 1, 1, {"gripper_opening": 1.0})
    closed_scene = SceneSnapshot("ep", 1, 0, 1, 1, {"gripper_opening": 0.0})
    open_contract = make_contract(preconditions=("gripper_open",))
    closed_contract = make_contract(preconditions=("gripper_closed",))

    assert PredicateBasedVerifier().approve(open_contract, open_scene).passed is True
    assert PredicateBasedVerifier().approve(closed_contract, closed_scene).passed is True


def test_predicate_verifier_exposes_task_goal_check_without_evaluator_access() -> None:
    scene = SceneSnapshot("ep", 1, 0, 1, 1, {"gripper_opening": 1.0})

    assert PredicateBasedVerifier().goal_satisfied(("gripper_open",), scene) is True
