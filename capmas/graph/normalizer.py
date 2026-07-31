"""Canonicalize candidate motion metadata before geometry evidence is computed."""

from __future__ import annotations

from dataclasses import replace
from collections.abc import Iterable
from typing import Callable

from capmas.contracts.candidates import GraphCandidate, rewrite_report_for
from capmas.contracts.graph import MotionIntent, SubgraphNodeSpec, SubgraphSpec
from capmas.contracts.scene import SceneSnapshot
from capmas.skills.registry import SkillRegistry


_GEOMETRY_KEYS = {"approach", "standoff", "approach_vector", "approach_vector_xyz"}


def normalize_motion_intent(node: SubgraphNodeSpec) -> SubgraphNodeSpec:
    """Derive missing typed intent fields from known CAP-X calls.

    This function intentionally does not inspect free-form text.  It only uses
    existing typed arguments and explicit node metadata, so preview evidence is
    bound to the same effective proposal that will be executed.
    """
    intent = node.motion_intent
    for call in node.skill_calls:
        skill_id = call.skill.skill_id
        args = call.args
        if skill_id == "sample_grasp_pose":
            object_id = args.get("object_track_id", args.get("object_name"))
            if object_id is not None:
                intent = _merge_intent(
                    intent,
                    MotionIntent("grasp", object_track_id=str(object_id)),
                )
        elif skill_id == "goto_pose":
            pose = _target_pose(args)
            kind = "place" if _looks_like_place(node) else "move"
            intent = _merge_intent(intent, MotionIntent(kind, target_pose_wxyz_xyz=pose))
    return replace(node, motion_intent=intent)


class CandidateNormalizer:
    """Normalize an effective candidate against the registered skill schema."""

    def __init__(
        self,
        registry: SkillRegistry | None = None,
        condition_enricher: Callable[[SubgraphSpec, SceneSnapshot, str], SubgraphSpec]
        | None = None,
    ) -> None:
        self.registry = registry
        self.condition_enricher = condition_enricher

    def normalize(
        self,
        candidate: GraphCandidate,
        scene: SceneSnapshot | None = None,
    ) -> GraphCandidate:
        raw = candidate.raw_subgraph or candidate.subgraph
        normalized = replace(
            candidate.subgraph,
            nodes=tuple(normalize_motion_intent(node) for node in candidate.subgraph.nodes),
        )
        if self.condition_enricher is not None and scene is not None:
            normalized = self.condition_enricher(
                normalized,
                scene,
                candidate.strategy or "balanced",
            )
        if self.registry is not None:
            self._validate_registered_arguments(normalized.nodes)
        report = rewrite_report_for(raw, normalized)
        return replace(
            candidate,
            subgraph=normalized,
            raw_subgraph=candidate.raw_subgraph or raw,
            rewrite_report=report,
        )

    def _validate_registered_arguments(self, nodes: Iterable[SubgraphNodeSpec]) -> None:
        assert self.registry is not None
        for node in nodes:
            for call_index, call in enumerate(node.skill_calls):
                allowed = set(self.registry.argument_names_for(call.skill))
                unexpected = sorted(set(call.args) - allowed)
                geometry_keys = sorted(set(unexpected) & _GEOMETRY_KEYS)
                if geometry_keys:
                    raise ValueError(
                        f"unregistered geometry arguments for {call.skill.skill_id}"
                        f"[{call_index}]: {geometry_keys}"
                    )


def _target_pose(args: dict[str, object]) -> tuple[float, ...] | None:
    position = _number_tuple(args.get("position"), 3)
    quaternion = _number_tuple(args.get("quaternion_wxyz"), 4)
    if position is None or quaternion is None:
        return None
    return quaternion + position


def _number_tuple(value: object, size: int) -> tuple[float, ...] | None:
    if not isinstance(value, (tuple, list)) or len(value) != size:
        return None
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        return None
    return tuple(float(item) for item in value)


def _looks_like_place(node: SubgraphNodeSpec) -> bool:
    return any(predicate.startswith("object_at_target(") for predicate in node.postconditions)


def _merge_intent(current: MotionIntent | None, derived: MotionIntent) -> MotionIntent:
    if current is None:
        return derived
    if current.kind != derived.kind and derived.kind != "move":
        raise ValueError(
            f"motion intent kind {current.kind!r} conflicts with typed call {derived.kind!r}"
        )
    if (
        current.object_track_id is not None
        and derived.object_track_id is not None
        and current.object_track_id != derived.object_track_id
    ):
        raise ValueError("motion intent object track conflicts with typed call")
    if (
        current.target_pose_wxyz_xyz is not None
        and derived.target_pose_wxyz_xyz is not None
        and current.target_pose_wxyz_xyz != derived.target_pose_wxyz_xyz
    ):
        raise ValueError("motion intent target pose conflicts with typed call")
    return replace(
        current,
        object_track_id=current.object_track_id or derived.object_track_id,
        target_track_id=current.target_track_id or derived.target_track_id,
        approach_vector_xyz=current.approach_vector_xyz or derived.approach_vector_xyz,
        standoff_m=current.standoff_m if current.standoff_m is not None else derived.standoff_m,
        target_pose_wxyz_xyz=(
            current.target_pose_wxyz_xyz
            if current.target_pose_wxyz_xyz is not None
            else derived.target_pose_wxyz_xyz
        ),
    )
