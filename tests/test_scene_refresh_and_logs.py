from __future__ import annotations

from capmas.contracts.action import SkillCall
from capmas.contracts.core import SkillRef
from capmas.contracts.graph import CheckpointSpec, SubgraphNodeSpec, SubgraphSpec, MissionGraph
from capmas.contracts.scene import ObjectTrack, SceneSnapshot
from capmas.runtime.llm_scheduler import LLMGraphScheduler
from scripts.run_libero_b3_llm import reserve_run_log_path
from capmas.verification.libero import ground_libero_grasp_subgraph


def _scene(version: int, target_position: tuple[float, float, float]) -> SceneSnapshot:
    target = ObjectTrack(
        track_id="plate",
        label="plate",
        pose_wxyz_xyz=(1.0, 0.0, 0.0, 0.0, *target_position),
        confidence=1.0,
        last_seen_ns=version + 1,
    )
    return SceneSnapshot(
        "episode",
        1,
        version,
        1,
        1,
        {},
        objects=(target,),
    )


def _place_graph(parent_scene_version: int) -> MissionGraph:
    node = SubgraphNodeSpec(
        node_id="place-action",
        description="place bowl on plate",
        skill_calls=(
            SkillCall(
                SkillRef("goto_pose", "1.0.0"),
                {"position": (9.0, 9.0, 9.0), "quaternion_wxyz": (1.0, 0.0, 0.0, 0.0)},
            ),
        ),
        postconditions=("object_at_target(bowl,plate)",),
    )
    subgraph = SubgraphSpec(
        "place",
        "place",
        "place bowl",
        (node,),
        (),
        "place-action",
        ("place-action",),
        ("place-action",),
        checkpoints=(CheckpointSpec("place-check", ("object_at_target(bowl,plate)",)),),
    )
    return MissionGraph(
        "mission",
        "place bowl",
        (subgraph,),
        (),
        (),
        "place",
        ("place",),
        ("place",),
        parent_scene_version=parent_scene_version,
    )


def test_rebase_graph_updates_scene_version_and_regrounds_scene_dependent_pose() -> None:
    scheduler = LLMGraphScheduler(
        object(),
        {},
        candidate_scene_rewriter=ground_libero_grasp_subgraph,
    )
    rebased = scheduler.rebase_graph(_place_graph(0), _scene(1, (0.4, 0.5, 0.6)))

    assert rebased.parent_scene_version == 1
    assert rebased.subgraphs[0].nodes[0].skill_calls[0].args["position"] == (
        0.4,
        0.5,
        0.6,
    )


def test_reserve_run_log_path_keeps_existing_run_log(tmp_path) -> None:
    requested = tmp_path / "episode.log"
    requested.write_text("first run\n", encoding="utf-8")

    reserved = reserve_run_log_path(requested)

    assert reserved != requested
    assert reserved.suffix == ".log"
    assert reserved.parent == requested.parent
    assert requested.read_text(encoding="utf-8") == "first run\n"
