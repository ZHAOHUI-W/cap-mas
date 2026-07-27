from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from capmas.contracts.action import ActionContract, SkillCall, SkillOutputRef
from capmas.contracts.agent import AgentContext
from capmas.contracts.core import SkillRef
from capmas.contracts.graph import (
    CheckpointSpec,
    GraphEdge,
    MissionEdge,
    MissionGraph,
    ResourceRequirement,
    SubgraphNodeSpec,
    SubgraphSpec,
)
from capmas.contracts.scene import SceneSnapshot


@dataclass
class LiberoSpatialTask0Policy:
    """V1 smoke policy for LIBERO Spatial task 0 pick-and-place."""

    object_name: str = "akita black bowl"
    skill_version: str = "capx-compat-1"
    include_home: bool = True
    issued: bool = False

    def propose(self, context: AgentContext) -> ActionContract | None:
        if self.issued:
            return None
        self.issued = True
        sample_call_index = 1
        plate_pose_call_index = 4
        skills = [
            SkillCall(SkillRef("open_gripper", self.skill_version), {}),
            SkillCall(
                SkillRef("sample_grasp_pose", self.skill_version),
                {"object_name": self.object_name},
            ),
            SkillCall(
                SkillRef("goto_pose", self.skill_version),
                {
                    "position": SkillOutputRef(sample_call_index, ("result", 0)),
                    "quaternion_wxyz": SkillOutputRef(sample_call_index, ("result", 1)),
                    "z_approach": 0.10,
                },
            ),
            SkillCall(SkillRef("close_gripper", self.skill_version), {}),
            SkillCall(
                SkillRef("get_object_pose", self.skill_version),
                {"object_name": "plate"},
            ),
            SkillCall(
                SkillRef("goto_pose", self.skill_version),
                {
                    "position": SkillOutputRef(plate_pose_call_index, ("result", 0)),
                    "quaternion_wxyz": SkillOutputRef(sample_call_index, ("result", 1)),
                    "z_approach": 0.12,
                },
            ),
            SkillCall(SkillRef("open_gripper", self.skill_version), {}),
        ]
        if self.include_home:
            skills.append(SkillCall(SkillRef("goto_home_joint_position", self.skill_version), {}))
        return ActionContract(
            contract_id=str(uuid4()),
            episode_id=context.episode_id,
            episode_epoch=context.episode_epoch,
            parent_scene_version=context.scene.scene_version,
            subgoal_id="pick_and_place_target",
            skills=tuple(skills),
            expected_postconditions=(
                f"object_at_target({self.object_name.replace(' ', '_')},plate)",
                "gripper_open",
            ),
            max_duration_ms=120_000,
            max_sim_steps=1_000,
            proposed_by="libero_spatial_task0_smoke_policy",
        )


@dataclass
class LiberoSpatialTask0MultiStepPolicy:
    """Deterministic staged policy for the P2.5 multi-cycle LIBERO loop."""

    object_name: str = "akita black bowl"
    target_name: str = "plate"
    skill_version: str = "capx-compat-1"
    include_home: bool = False
    stage_index: int = 0
    last_subgoal: str | None = None

    def propose(self, context: AgentContext) -> ActionContract | None:
        if (
            context.history.last_verification is not None
            and context.history.last_verification.passed
            and context.history.current_subgoal == self.last_subgoal
        ):
            self.stage_index += 1
        if self.stage_index >= len(self._subgoals()):
            return None
        contract = self._build_contract(context, self.stage_index)
        self.last_subgoal = contract.subgoal_id
        return contract

    def recover(self, trace: object, verification: object, context: AgentContext) -> ActionContract | None:
        del trace, verification
        subgoal = context.history.current_subgoal or self.last_subgoal
        if subgoal is None:
            return None
        try:
            stage = self._subgoals().index(subgoal)
        except ValueError:
            return None
        contract = self._build_contract(context, stage)
        self.last_subgoal = contract.subgoal_id
        return contract

    def _subgoals(self) -> tuple[str, ...]:
        subgoals = [
            "open_gripper",
            "approach_object",
            "close_and_verify_grasp",
            "approach_target",
            "release_and_verify_placement",
        ]
        if self.include_home:
            subgoals.append("return_home")
        return tuple(subgoals)

    def _build_contract(self, context: AgentContext, stage: int) -> ActionContract:
        object_id = self.object_name.replace(" ", "_")
        target_id = self.target_name.replace(" ", "_")
        subgoal = self._subgoals()[stage]
        if subgoal == "open_gripper":
            skills = (SkillCall(SkillRef("open_gripper", self.skill_version), {}),)
            postconditions = ("gripper_open",)
        elif subgoal == "approach_object":
            skills = (
                SkillCall(
                    SkillRef("sample_grasp_pose", self.skill_version),
                    {"object_name": self.object_name},
                ),
                SkillCall(
                    SkillRef("goto_pose", self.skill_version),
                    {
                        "position": SkillOutputRef(0, ("result", 0)),
                        "quaternion_wxyz": SkillOutputRef(0, ("result", 1)),
                        "z_approach": 0.10,
                    },
                ),
            )
            postconditions = (f"object_visible:{object_id}",)
        elif subgoal == "close_and_verify_grasp":
            skills = (SkillCall(SkillRef("close_gripper", self.skill_version), {}),)
            postconditions = (f"object_in_gripper({object_id})",)
        elif subgoal == "approach_target":
            skills = (
                SkillCall(
                    SkillRef("get_object_pose", self.skill_version),
                    {"object_name": self.target_name},
                ),
                SkillCall(
                    SkillRef("goto_pose", self.skill_version),
                    {
                        "position": SkillOutputRef(0, ("result", 0)),
                        "quaternion_wxyz": (0.0, 1.0, 0.0, 0.0),
                        "z_approach": 0.12,
                    },
                ),
            )
            postconditions = (f"object_in_gripper({object_id})",)
        elif subgoal == "release_and_verify_placement":
            skills = (SkillCall(SkillRef("open_gripper", self.skill_version), {}),)
            postconditions = (
                f"object_at_target({object_id},{target_id})",
                "gripper_open",
            )
        else:
            skills = (SkillCall(SkillRef("goto_home_joint_position", self.skill_version), {}),)
            postconditions = ("scene_fresh(1000)",)
        return ActionContract(
            contract_id=str(uuid4()),
            episode_id=context.episode_id,
            episode_epoch=context.episode_epoch,
            parent_scene_version=context.scene.scene_version,
            subgoal_id=subgoal,
            skills=skills,
            expected_postconditions=postconditions,
            max_duration_ms=60_000,
            max_sim_steps=500,
            proposed_by="libero_spatial_task0_multistep_policy",
        )


def build_libero_spatial_task0_mission_graph(
    *,
    object_name: str = "akita black bowl",
    target_name: str = "plate",
    skill_version: str = "capx-compat-1",
    include_home: bool = False,
    parent_scene_version: int | None = None,
) -> MissionGraph:
    """Compile the deterministic LIBERO P2.5 policy into a fixed MissionGraph.

    This is the B3 bridge: it preserves the exact staged skill calls used by
    ``LiberoSpatialTask0MultiStepPolicy`` while moving control flow into the
    typed graph interpreter.  The placeholder context is only used to build
    immutable node metadata; the interpreter creates fresh contracts from the
    live scene before every dispatch.
    """
    policy = LiberoSpatialTask0MultiStepPolicy(
        object_name=object_name,
        target_name=target_name,
        skill_version=skill_version,
        include_home=include_home,
    )
    template_context = AgentContext(
        task_id="libero_spatial_0",
        episode_id="graph-template",
        episode_epoch=1,
        scene=SceneSnapshot("graph-template", 1, 0, 0, 0, {}),
    )
    subgraphs: list[SubgraphSpec] = []
    for stage, subgoal in enumerate(policy._subgoals()):
        contract = policy._build_contract(template_context, stage)
        node_id = f"{subgoal}_action"
        node = SubgraphNodeSpec(
            node_id=node_id,
            description=f"LIBERO task 0: {subgoal}",
            skill_calls=contract.skills,
            preconditions=contract.preconditions,
            postconditions=contract.expected_postconditions,
            resources=(ResourceRequirement("robot_arm_0"),),
            max_duration_ms=contract.max_duration_ms,
            max_sim_steps=contract.max_sim_steps,
            proposed_by=contract.proposed_by,
            recovery_policy=contract.recovery_policy,
        )
        subgraphs.append(
            SubgraphSpec(
                subgraph_id=subgoal,
                subgoal_id=subgoal,
                description=f"LIBERO task 0 subgoal: {subgoal}",
                nodes=(node,),
                edges=(),
                entry_node=node_id,
                success_nodes=(node_id,),
                failure_nodes=(node_id,),
                checkpoints=(CheckpointSpec(f"{subgoal}_verified", node.postconditions),),
                assigned_agent="libero_fixed_graph_policy",
            )
        )

    edges = tuple(
        MissionEdge(left.subgraph_id, right.subgraph_id, "success")
        for left, right in zip(subgraphs, subgraphs[1:])
    )
    return MissionGraph(
        mission_id="libero_spatial_0_fixed_graph",
        task=f"Place {object_name} on {target_name}",
        subgraphs=tuple(subgraphs),
        edges=edges,
        bindings=(),
        entry_subgraph=subgraphs[0].subgraph_id,
        success_subgraphs=(subgraphs[-1].subgraph_id,),
        failure_subgraphs=tuple(subgraph.subgraph_id for subgraph in subgraphs),
        parent_scene_version=parent_scene_version,
        graph_version=1,
    )
