import json
from pathlib import Path

import pytest

from scripts.debug_libero_grasp_probe import (
    append_phase_snapshot,
    capx_dependency_paths,
    resolve_probe_object_names,
    select_grasp_pose,
)
from scripts.build_libero_p55_candidates import build_candidate_artifact


def test_grasp_probe_keeps_each_physics_boundary_as_a_named_record() -> None:
    snapshots: list[dict[str, object]] = []

    append_phase_snapshot(snapshots, "after_close_gripper", {"bowl": [1, 2, 3]})
    append_phase_snapshot(snapshots, "after_lift", {"bowl": [4, 5, 6]})

    assert [item["phase"] for item in snapshots] == [
        "after_close_gripper",
        "after_lift",
    ]
    assert snapshots[0]["physics"] == {"bowl": [1, 2, 3]}
    assert snapshots[1]["physics"] == {"bowl": [4, 5, 6]}


def test_grasp_probe_accepts_task_specific_source_and_target_names() -> None:
    assert resolve_probe_object_names(" butter ", " basket ") == ("butter", "basket")

    with pytest.raises(ValueError, match="non-empty"):
        resolve_probe_object_names("", "basket")


def test_grasp_probe_includes_nested_libero_python_root() -> None:
    paths = capx_dependency_paths("/workspace/cap-x", "/workspace/cap-mas")

    assert paths[0] == "/workspace/cap-mas"
    assert "/workspace/cap-x/capx/third_party/LIBERO-PRO/libero" in paths


def test_phase5_nonprivileged_configs_declare_capx_perception_servers() -> None:
    config_root = Path(__file__).parents[1] / "configs" / "phase5"
    for name in (
        "capx_libero_goal_1_nonprivileged.yaml",
        "capx_libero_object_6_nonprivileged.yaml",
    ):
        text = (config_root / name).read_text(encoding="utf-8")
        assert "api_servers:" in text
        assert "launch_sam3_server.main" in text
        assert "launch_contact_graspnet_server.main" in text
        assert "launch_pyroki_server.main" in text


def test_probe_can_use_pose_fallback_only_when_explicitly_enabled() -> None:
    sampled = ([1.0, 2.0, 3.0], [1.0, 0.0, 0.0, 0.0])
    fallback = ([4.0, 5.0, 6.0], [0.0, 1.0, 0.0, 0.0])

    def failing_sample() -> object:
        raise RuntimeError("no grasp candidates")

    def pose() -> object:
        return fallback

    with pytest.raises(RuntimeError, match="no grasp candidates"):
        select_grasp_pose(failing_sample, pose, allow_pose_fallback=False)

    result = select_grasp_pose(failing_sample, pose, allow_pose_fallback=True)
    assert result == (*fallback, "object_pose_fallback", "no grasp candidates")

    assert select_grasp_pose(lambda: sampled, pose, allow_pose_fallback=True) == (
        *sampled,
        "sample_grasp_pose",
        None,
    )


def test_candidate_builder_rewrites_phrase_and_snake_case_graph_ids(tmp_path) -> None:
    base = {
        "scene_version": 1,
        "candidates": [
            {
                "candidate_id": "sg_pick_akita_black_bowl:policy-0:0",
                "graph": {
                    "subgraphs": [
                        {
                            "subgraph_id": "sg_pick_akita_black_bowl",
                            "subgoal_id": "pick_akita_black_bowl",
                            "nodes": [
                                {
                                    "node_id": "act_pick_akita_black_bowl",
                                    "description": "Pick the akita black bowl.",
                                    "postconditions": [
                                        "object_in_gripper(akita black bowl)"
                                    ],
                                }
                            ],
                        }
                    ]
                },
            }
        ],
    }
    source = tmp_path / "base.json"
    destination = tmp_path / "object_6.json"
    source.write_text(json.dumps(base), encoding="utf-8")

    build_candidate_artifact(
        base_artifact=source,
        output_path=destination,
        suite_name="libero_object",
        task_id=6,
        object_name="butter",
        target_name="basket",
        target_z_offset=0.08,
        target_position=(0.1, 0.2, 0.3),
    )

    graph = json.loads(destination.read_text(encoding="utf-8"))["candidates"][0]["graph"]
    subgraph = graph["subgraphs"][0]
    node = subgraph["nodes"][0]
    assert subgraph["subgraph_id"] == "sg_pick_butter"
    assert subgraph["subgoal_id"] == "pick_butter"
    assert node["node_id"] == "act_pick_butter"
    assert node["description"] == "Pick the butter."
    assert node["postconditions"] == ["object_in_gripper(butter)"]
