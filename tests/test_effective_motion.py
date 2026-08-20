from __future__ import annotations

from dataclasses import replace

import pytest

from capmas.contracts.action import SkillCall, SkillOutputRef
from capmas.contracts.candidates import GraphCandidate, rewrite_report_for
from capmas.contracts.core import SkillRef
from capmas.contracts.graph import (
    MissionEdge,
    MissionGraph,
    MotionIntent,
    SubgraphNodeSpec,
    SubgraphSpec,
)
from capmas.contracts.scene import SceneSnapshot
from capmas.perception.effective_motion import (
    CandidateExecutionContext,
    bind_effective_motion,
    execution_graph_fingerprint,
    materialize_execution_graph,
)


def test_effective_motion_contracts_are_exported_from_perception_package() -> None:
    from capmas.perception import EffectiveMotionProgram

    assert EffectiveMotionProgram.__name__ == "EffectiveMotionProgram"


def _pick_place_context() -> tuple[CandidateExecutionContext, SceneSnapshot]:
    pick = SubgraphSpec(
        subgraph_id="sg_pick",
        subgoal_id="pick_butter",
        description="Pick butter.",
        nodes=(
            SubgraphNodeSpec(
                node_id="pick",
                description="Sample, grasp, and lift butter.",
                skill_calls=(
                    SkillCall(SkillRef("sample_grasp_pose", "capx-compat-1"), {"object_name": "butter"}),
                    SkillCall(
                        SkillRef("goto_pose", "capx-compat-1"),
                        {
                            "position": SkillOutputRef(0, ("result", 0)),
                            "quaternion_wxyz": SkillOutputRef(0, ("result", 1)),
                            "z_approach": 0.05,
                        },
                    ),
                    SkillCall(SkillRef("close_gripper", "capx-compat-1"), {}),
                    SkillCall(
                        SkillRef("lift_after_grasp", "capx-compat-1"),
                        {
                            "position": SkillOutputRef(0, ("result", 0)),
                            "quaternion_wxyz": SkillOutputRef(0, ("result", 1)),
                            "z_lift": 0.12,
                        },
                    ),
                ),
                postconditions=("object_in_gripper(butter)",),
                motion_intent=MotionIntent(
                    "grasp",
                    object_track_id="butter",
                    target_track_id="butter",
                    approach_vector_xyz=(0.0, 0.0, -1.0),
                    standoff_m=0.05,
                    target_pose_wxyz_xyz=(1.0, 0.0, 0.0, 0.0, 0.5, 0.2, 0.1),
                ),
            ),
        ),
        edges=(),
        entry_node="pick",
        success_nodes=("pick",),
        failure_nodes=("pick",),
    )
    place = SubgraphSpec(
        subgraph_id="sg_place",
        subgoal_id="place_butter",
        description="Place butter.",
        nodes=(
            SubgraphNodeSpec(
                node_id="place",
                description="Move and release butter.",
                skill_calls=(
                    SkillCall(
                        SkillRef("goto_pose", "capx-compat-1"),
                        {
                            "position": [0.6, 0.25, 0.04],
                            "quaternion_wxyz": [0.0, 1.0, 0.0, 0.0],
                            "z_approach": 0.08,
                        },
                    ),
                    SkillCall(SkillRef("open_gripper", "capx-compat-1"), {}),
                ),
                postconditions=("object_at_target(butter,basket)", "gripper_open()"),
                motion_intent=MotionIntent(
                    "place",
                    object_track_id="butter",
                    target_track_id="basket",
                    approach_vector_xyz=(0.0, 0.0, -1.0),
                    standoff_m=0.08,
                    target_pose_wxyz_xyz=(0.0, 1.0, 0.0, 0.0, 0.6, 0.25, 0.04),
                ),
            ),
        ),
        edges=(),
        entry_node="place",
        success_nodes=("place",),
        failure_nodes=("place",),
    )
    graph = MissionGraph(
        mission_id="pick-place-butter",
        task="Put the butter in the basket.",
        subgraphs=(pick, place),
        edges=(MissionEdge("sg_pick", "sg_place", "success"),),
        bindings=(),
        entry_subgraph="sg_pick",
        success_subgraphs=("sg_place",),
        failure_subgraphs=(),
        parent_scene_version=1,
    )
    candidate = GraphCandidate(
        candidate_id="pick:policy-0:0",
        subgraph=pick,
        parent_scene_version=1,
        producer_agent="policy-0",
        raw_subgraph=pick,
        rewrite_report=rewrite_report_for(pick, pick),
    )
    scene = SceneSnapshot("episode", 1, 1, 100, 101, {})
    return (
        CandidateExecutionContext(
            candidate=candidate,
            mission_graph=graph,
            selected_subgraph_id="sg_pick",
            execution_graph_fingerprint=execution_graph_fingerprint(graph),
        ),
        scene,
    )


def test_bind_effective_motion_extracts_pick_lift_transfer_place_release() -> None:
    context, scene = _pick_place_context()

    program = bind_effective_motion(context, scene)

    assert [item.kind for item in program.segments] == [
        "grasp_approach",
        "lift",
        "transfer",
        "place_approach",
        "release",
    ]
    assert program.segments[0].start_pose_wxyz_xyz[-1] == pytest.approx(0.15)
    assert program.segments[1].end_pose_wxyz_xyz[-1] == pytest.approx(0.22)
    assert program.segments[3].end_pose_wxyz_xyz[-3:] == pytest.approx((0.60, 0.25, 0.04))


def test_materialized_graph_uses_bound_literal_poses() -> None:
    context, scene = _pick_place_context()

    graph = materialize_execution_graph(bind_effective_motion(context, scene), context.mission_graph)

    pick = graph.subgraph("sg_pick").node("pick")
    place = graph.subgraph("sg_place").node("place")
    assert pick.skill_calls[1].args["position"] == [0.5, 0.2, 0.1]
    assert pick.skill_calls[3].args["position"] == [0.5, 0.2, 0.22]
    assert place.skill_calls[0].args["position"] == [0.6, 0.25, 0.04]


def test_binding_rejects_wrong_scene_or_selected_subgraph() -> None:
    context, scene = _pick_place_context()

    with pytest.raises(ValueError, match="decision scene"):
        bind_effective_motion(context, replace(scene, scene_version=2))
    with pytest.raises(ValueError, match="selected subgraph"):
        bind_effective_motion(replace(context, selected_subgraph_id="sg_place"), scene)
