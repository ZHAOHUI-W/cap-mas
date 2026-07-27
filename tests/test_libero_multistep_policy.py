from capmas.agents.libero import (
    LiberoSpatialTask0MultiStepPolicy,
    build_libero_spatial_task0_mission_graph,
)
from capmas.contracts.agent import AgentContext, CycleHistory
from capmas.contracts.scene import SceneSnapshot
from capmas.contracts.trace import ExecutionTrace
from capmas.contracts.verification import PredicateReport, VerificationResult


def context(version: int, history: CycleHistory) -> AgentContext:
    return AgentContext(
        task_id="libero_spatial_0",
        episode_id="ep",
        episode_epoch=1,
        scene=SceneSnapshot("ep", 1, version, version + 1, version + 1, {}),
        history=history,
    )


def committed_history(subgoal: str, version: int) -> CycleHistory:
    trace = ExecutionTrace("trace", "ep", 1, "contract", "lease", version - 1, version - 1, version, 0, 1, "completed")
    verification = VerificationResult("contract", "commit", version, (PredicateReport("post", True),))
    return CycleHistory((trace,), verification, subgoal, 0)


def test_libero_multistep_policy_advances_one_subgoal_per_commit() -> None:
    policy = LiberoSpatialTask0MultiStepPolicy(include_home=False)

    first = policy.propose(context(0, CycleHistory()))
    second = policy.propose(context(1, committed_history(first.subgoal_id, 1)))

    assert first.subgoal_id == "open_gripper"
    assert second.subgoal_id == "approach_object"
    assert first.parent_scene_version == 0
    assert second.parent_scene_version == 1
    assert tuple(call.skill.skill_id for call in first.skills) == ("open_gripper",)


def test_libero_multistep_policy_recovery_retries_current_subgoal() -> None:
    policy = LiberoSpatialTask0MultiStepPolicy(include_home=False)
    first = policy.propose(context(0, CycleHistory()))
    failed = CycleHistory(
        traces=(),
        last_verification=VerificationResult("contract", "recover", 1, (), failure_class="POSTCONDITION_FAILED"),
        current_subgoal=first.subgoal_id,
        recovery_count=1,
    )

    retry = policy.recover(None, failed.last_verification, context(1, failed))

    assert retry is not None
    assert retry.subgoal_id == "open_gripper"
    assert retry.parent_scene_version == 1


def test_libero_multistep_policy_uses_stable_grasp_orientation_for_target() -> None:
    policy = LiberoSpatialTask0MultiStepPolicy(include_home=False)
    first = policy.propose(context(0, CycleHistory()))
    second = policy.propose(context(1, committed_history(first.subgoal_id, 1)))
    third = policy.propose(context(2, committed_history(second.subgoal_id, 2)))
    target = policy.propose(context(3, committed_history(third.subgoal_id, 3)))

    assert target.subgoal_id == "approach_target"
    assert target.skills[1].args["quaternion_wxyz"] == (0.0, 1.0, 0.0, 0.0)


def test_libero_multistep_policy_compiles_to_valid_fixed_mission_graph() -> None:
    from capmas.graph.validator import GraphValidator

    graph = build_libero_spatial_task0_mission_graph(include_home=False)
    validation = GraphValidator().validate(graph)

    assert validation.valid is True
    assert [subgraph.subgraph_id for subgraph in graph.subgraphs] == [
        "open_gripper",
        "approach_object",
        "close_and_verify_grasp",
        "approach_target",
        "release_and_verify_placement",
    ]
    assert len(graph.edges) == 4
    assert graph.entry_subgraph == "open_gripper"
    assert graph.success_subgraphs == ("release_and_verify_placement",)
    assert all(
        node.resources[0].resource_id == "robot_arm_0"
        for subgraph in graph.subgraphs
        for node in subgraph.nodes
    )
