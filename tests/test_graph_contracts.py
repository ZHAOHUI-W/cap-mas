from __future__ import annotations

from capmas.contracts.action import SkillCall
from capmas.agents.policy import CallableGraphPolicyAgent
from capmas.contracts.agent import AgentArtifact, AgentContext
from capmas.contracts.core import SkillRef
from capmas.contracts.graph import (
    CheckpointSpec,
    GraphEdge,
    MissionGraph,
    MissionEdge,
    MissionBinding,
    PortBinding,
    PortSpec,
    ResourceRequirement,
    SubgraphNodeSpec,
    SubgraphSpec,
    SubgraphOutputBinding,
)
from capmas.contracts.scene import SceneSnapshot
from capmas.graph.validator import GraphValidator, scene_initial_facts


NOOP = SkillRef("noop", "1.0.0")


def _node(
    node_id: str,
    *,
    inputs: tuple[PortSpec, ...] = (),
    outputs: tuple[PortSpec, ...] = (),
    preconditions: tuple[str, ...] = (),
    postconditions: tuple[str, ...] = ("step_done",),
    resources: tuple[ResourceRequirement, ...] = (),
) -> SubgraphNodeSpec:
    return SubgraphNodeSpec(
        node_id=node_id,
        description=node_id,
        skill_calls=(SkillCall(NOOP, {}),),
        inputs=inputs,
        outputs=outputs,
        preconditions=preconditions,
        postconditions=postconditions,
        resources=resources,
        proposed_by="test-policy",
    )


def _subgraph(
    subgraph_id: str,
    *,
    nodes: tuple[SubgraphNodeSpec, ...],
    edges: tuple[GraphEdge, ...],
    inputs: tuple[PortSpec, ...] = (),
    outputs: tuple[PortSpec, ...] = (),
    bindings: tuple[PortBinding, ...] = (),
    output_bindings: tuple[SubgraphOutputBinding, ...] = (),
    success_nodes: tuple[str, ...] = (),
    failure_nodes: tuple[str, ...] = (),
) -> SubgraphSpec:
    return SubgraphSpec(
        subgraph_id=subgraph_id,
        subgoal_id=subgraph_id,
        description=subgraph_id,
        inputs=inputs,
        outputs=outputs,
        nodes=nodes,
        edges=edges,
        bindings=bindings,
        output_bindings=output_bindings,
        entry_node=nodes[0].node_id,
        success_nodes=success_nodes or (nodes[-1].node_id,),
        failure_nodes=failure_nodes or (nodes[-1].node_id,),
        checkpoints=(CheckpointSpec(f"{subgraph_id}_checkpoint", ("step_done",)),),
    )


def test_valid_graph_compiles_a_subgraph_node_to_an_action_contract() -> None:
    grasp = _subgraph(
        "grasp",
        nodes=(
            _node(
                "grasp_action",
                outputs=(PortSpec("ok", "bool"),),
                postconditions=("object_in_gripper(bowl)",),
                resources=(ResourceRequirement("robot_arm_0"),),
            ),
        ),
        edges=(),
        outputs=(PortSpec("ok", "bool"),),
        output_bindings=(SubgraphOutputBinding("grasp_action", "ok", "ok"),),
    )
    graph = MissionGraph(
        mission_id="pick-and-place",
        task="Pick up the bowl",
        subgraphs=(grasp,),
        edges=(),
        bindings=(),
        entry_subgraph="grasp",
        success_subgraphs=("grasp",),
        failure_subgraphs=("grasp",),
    )

    result = GraphValidator().validate(graph)
    assert result.valid is True

    context = AgentContext(
        task_id="task",
        episode_id="episode",
        episode_epoch=3,
        scene=SceneSnapshot("episode", 3, 7, 1, 2, {}),
    )
    contract = grasp.to_action_contract("grasp_action", context)
    assert contract.parent_scene_version == 7
    assert contract.subgoal_id == "grasp"
    assert contract.skills[0].skill == NOOP
    assert contract.expected_postconditions == ("object_in_gripper(bowl)",)


def test_validator_accepts_precondition_established_by_mandatory_predecessor() -> None:
    grasp = _subgraph(
        "grasp",
        nodes=(_node("grasp_action", postconditions=("object_in_gripper(bowl)",)),),
        edges=(),
    )
    place = _subgraph(
        "place",
        nodes=(_node("place_action", preconditions=("object_in_gripper(bowl)",)),),
        edges=(),
    )
    graph = MissionGraph(
        mission_id="pick-and-place",
        task="Place the bowl",
        subgraphs=(grasp, place),
        edges=(MissionEdge("grasp", "place", "success"),),
        bindings=(),
        entry_subgraph="grasp",
        success_subgraphs=("place",),
        failure_subgraphs=("grasp", "place"),
    )

    result = GraphValidator().validate(graph)

    assert result.valid is True


def test_validator_rejects_precondition_without_initial_fact_or_predecessor() -> None:
    place = _subgraph(
        "place",
        nodes=(_node("place_action", preconditions=("object_in_gripper(bowl)",)),),
        edges=(),
    )
    graph = MissionGraph(
        mission_id="invalid-place",
        task="Place the bowl",
        subgraphs=(place,),
        edges=(),
        bindings=(),
        entry_subgraph="place",
        success_subgraphs=("place",),
        failure_subgraphs=("place",),
    )

    result = GraphValidator().validate(graph)

    assert result.valid is False
    assert any(
        diagnostic.code == "UNESTABLISHED_PRECONDITION"
        for diagnostic in result.diagnostics
    )


def test_validator_defers_scene_fresh_precondition_to_dispatch() -> None:
    action = _subgraph(
        "grasp",
        nodes=(_node("grasp_action", preconditions=("scene_fresh(1000)",)),),
        edges=(),
    )
    graph = MissionGraph(
        mission_id="stale-scene",
        task="Place the bowl",
        subgraphs=(action,),
        edges=(),
        bindings=(),
        entry_subgraph="grasp",
        success_subgraphs=("grasp",),
        failure_subgraphs=("grasp",),
    )

    stale_scene = SceneSnapshot("episode", 1, 0, 1, 1, {})
    result = GraphValidator().validate(
        graph,
        initial_facts=scene_initial_facts(graph, stale_scene),
    )

    assert result.valid is True


def test_validator_rejects_precondition_not_guaranteed_on_all_predecessor_paths() -> None:
    start = _subgraph("start", nodes=(_node("start_action"),), edges=())
    grasp = _subgraph(
        "grasp",
        nodes=(_node("grasp_action", postconditions=("object_in_gripper(bowl)",)),),
        edges=(),
    )
    observe = _subgraph("observe", nodes=(_node("observe_action"),), edges=())
    place = _subgraph(
        "place",
        nodes=(_node("place_action", preconditions=("object_in_gripper(bowl)",)),),
        edges=(),
    )
    graph = MissionGraph(
        mission_id="ambiguous-place",
        task="Place the bowl",
        subgraphs=(start, grasp, observe, place),
        edges=(
            MissionEdge("start", "grasp", "success"),
            MissionEdge("start", "observe", "success"),
            MissionEdge("grasp", "place", "success"),
            MissionEdge("observe", "place", "success"),
        ),
        bindings=(),
        entry_subgraph="start",
        success_subgraphs=("place",),
        failure_subgraphs=("start", "grasp", "observe", "place"),
    )

    result = GraphValidator().validate(graph)

    assert result.valid is False
    assert any(
        diagnostic.code == "UNESTABLISHED_PRECONDITION"
        and "place_action" in diagnostic.path
        for diagnostic in result.diagnostics
    )


def test_callable_graph_policy_is_a_replaceable_agent_seam() -> None:
    expected = _subgraph(
        "grasp",
        nodes=(_node("grasp_action"),),
        edges=(),
    )
    agent = CallableGraphPolicyAgent(lambda _subgoal, _scene, _context: expected)
    context = AgentContext(
        task_id="task",
        episode_id="episode",
        episode_epoch=1,
        scene=SceneSnapshot("episode", 1, 0, 1, 2, {}),
    )

    actual = agent.propose_subgraph(
        AgentArtifact("subgoal", "subgoal", {}, "manager"),
        context.scene,
        context,
    )
    assert actual == expected


def test_validator_reports_dangling_edges_and_type_mismatches() -> None:
    source = _node("source", outputs=(PortSpec("pose", "Pose"),))
    target = _node(
        "target",
        inputs=(PortSpec("pose", "PointCloud"),),
    )
    subgraph = _subgraph(
        "bad",
        nodes=(source, target),
        edges=(GraphEdge("source", "target"), GraphEdge("target", "missing")),
        bindings=(PortBinding("source", "pose", "target", "pose"),),
    )
    graph = MissionGraph(
        mission_id="bad-graph",
        task="bad",
        subgraphs=(subgraph,),
        edges=(),
        bindings=(),
        entry_subgraph="bad",
        success_subgraphs=("bad",),
        failure_subgraphs=("bad",),
    )

    result = GraphValidator().validate(graph)
    assert result.valid is False
    codes = {diagnostic.code for diagnostic in result.diagnostics}
    assert "DANGLING_EDGE" in codes
    assert "PORT_TYPE_MISMATCH" in codes


def test_validator_requires_checkpoint_and_reachable_nodes() -> None:
    first = _node("first")
    unreachable = _node("unreachable")
    subgraph = SubgraphSpec(
        subgraph_id="incomplete",
        subgoal_id="incomplete",
        description="incomplete",
        nodes=(first, unreachable),
        edges=(),
        entry_node="first",
        success_nodes=("first",),
        failure_nodes=("first",),
        checkpoints=(),
    )
    graph = MissionGraph(
        mission_id="incomplete-graph",
        task="incomplete",
        subgraphs=(subgraph,),
        edges=(),
        bindings=(),
        entry_subgraph="incomplete",
        success_subgraphs=("incomplete",),
        failure_subgraphs=("incomplete",),
    )

    result = GraphValidator().validate(graph)
    assert result.valid is False
    codes = {diagnostic.code for diagnostic in result.diagnostics}
    assert "UNREACHABLE_NODE" in codes
    assert "MISSING_VALID_CHECKPOINT" in codes


def test_validator_rejects_exclusive_resource_conflicts_on_parallel_branches() -> None:
    left = _subgraph(
        "left",
        nodes=(_node("left_action", resources=(ResourceRequirement("robot_arm_0"),)),),
        edges=(),
    )
    right = _subgraph(
        "right",
        nodes=(_node("right_action", resources=(ResourceRequirement("robot_arm_0"),)),),
        edges=(),
    )
    mission = MissionGraph(
        mission_id="conflict",
        task="conflict",
        subgraphs=(
            _subgraph("start", nodes=(_node("start_action"),), edges=()),
            left,
            right,
        ),
        edges=(MissionEdge("start", "left"), MissionEdge("start", "right")),
        bindings=(),
        entry_subgraph="start",
        success_subgraphs=("left", "right"),
        failure_subgraphs=("start",),
    )

    result = GraphValidator().validate(mission)
    assert result.valid is False
    assert any(diagnostic.code == "PARALLEL_RESOURCE_CONFLICT" for diagnostic in result.diagnostics)


def test_validator_allows_exclusive_resource_reuse_on_conditional_branches() -> None:
    left = _subgraph(
        "left",
        nodes=(_node("left_action", resources=(ResourceRequirement("robot_arm_0"),)),),
        edges=(),
    )
    right = _subgraph(
        "right",
        nodes=(_node("right_action", resources=(ResourceRequirement("robot_arm_0"),)),),
        edges=(),
    )
    mission = MissionGraph(
        mission_id="conditional",
        task="conditional",
        subgraphs=(
            _subgraph("start", nodes=(_node("start_action"),), edges=()),
            left,
            right,
        ),
        edges=(
            MissionEdge("start", "left", condition="success"),
            MissionEdge("start", "right", condition="failure"),
        ),
        bindings=(),
        entry_subgraph="start",
        success_subgraphs=("left", "right"),
        failure_subgraphs=("start",),
    )

    result = GraphValidator().validate(mission)
    assert not any(
        diagnostic.code == "PARALLEL_RESOURCE_CONFLICT"
        for diagnostic in result.diagnostics
    )


def test_validator_requires_every_required_mission_input_to_be_bound() -> None:
    producer = _subgraph(
        "producer",
        nodes=(_node("observe", outputs=(PortSpec("pose", "Pose"),)),),
        edges=(),
        outputs=(PortSpec("pose", "Pose"),),
        output_bindings=(SubgraphOutputBinding("observe", "pose", "pose"),),
    )
    consumer = _subgraph(
        "consumer",
        nodes=(_node("use", inputs=(PortSpec("pose", "Pose"),)),),
        edges=(),
        inputs=(PortSpec("pose", "Pose"),),
    )
    graph = MissionGraph(
        mission_id="missing-mission-input",
        task="missing mission input",
        subgraphs=(producer, consumer),
        edges=(MissionEdge("producer", "consumer", "success"),),
        bindings=(),
        entry_subgraph="producer",
        success_subgraphs=("consumer",),
        failure_subgraphs=("producer",),
    )

    result = GraphValidator().validate(graph)

    assert result.valid is False
    assert any(diagnostic.code == "UNBOUND_MISSION_INPUT" for diagnostic in result.diagnostics)


def test_validator_rejects_mission_binding_without_reachable_exposed_producer() -> None:
    start = _subgraph("start", nodes=(_node("start"),), edges=())
    producer = _subgraph(
        "producer",
        nodes=(_node("observe", outputs=(PortSpec("pose", "Pose"),)),),
        edges=(),
        outputs=(PortSpec("pose", "Pose"),),
        # The subgraph declares the output but does not expose it through a
        # node binding, so the runtime cannot produce a mission value.
    )
    consumer = _subgraph(
        "consumer",
        nodes=(_node("use", inputs=(PortSpec("pose", "Pose"),)),),
        edges=(),
        inputs=(PortSpec("pose", "Pose"),),
    )
    graph = MissionGraph(
        mission_id="bad-mission-binding",
        task="bad mission binding",
        subgraphs=(start, producer, consumer),
        edges=(
            MissionEdge("start", "consumer", "success"),
        ),
        bindings=(MissionBinding("producer", "pose", "consumer", "pose"),),
        entry_subgraph="start",
        success_subgraphs=("consumer",),
        failure_subgraphs=("start",),
    )

    result = GraphValidator().validate(graph)
    codes = {diagnostic.code for diagnostic in result.diagnostics}

    assert result.valid is False
    assert "UNREACHABLE_SUBGRAPH" in codes
    assert "UNREACHABLE_BINDING_SOURCE" in codes
    assert "MISSION_BINDING_SOURCE_NOT_PREDECESSOR" in codes
    assert "UNBOUND_OUTPUT" in codes
