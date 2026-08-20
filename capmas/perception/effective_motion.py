"""Pure decision-time binding for object-6 effective motion programs."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import Literal

from capmas.contracts.action import SkillCall, SkillOutputRef
from capmas.contracts.candidates import GraphCandidate, subgraph_fingerprint
from capmas.contracts.graph import MissionGraph, MotionIntent, SubgraphNodeSpec, SubgraphSpec
from capmas.contracts.scene import SceneSnapshot
from capmas.graph.serialization import mission_graph_to_dict

Pose = tuple[float, float, float, float, float, float, float]
SegmentKind = Literal[
    "grasp_approach",
    "lift",
    "transfer",
    "place_approach",
    "release",
]


@dataclass(frozen=True)
class CandidateExecutionContext:
    """Connect one local Arbiter candidate to the graph it will execute."""

    candidate: GraphCandidate
    mission_graph: MissionGraph
    selected_subgraph_id: str
    execution_graph_fingerprint: str

    def __post_init__(self) -> None:
        if self.selected_subgraph_id != self.candidate.subgraph.subgraph_id:
            raise ValueError("selected subgraph does not match candidate")
        try:
            selected = self.mission_graph.subgraph(self.selected_subgraph_id)
        except KeyError as exc:
            raise ValueError("selected subgraph does not exist in mission graph") from exc
        if selected != self.candidate.subgraph:
            raise ValueError("mission graph selected subgraph does not match candidate")
        if self.mission_graph.parent_scene_version != self.candidate.parent_scene_version:
            raise ValueError("mission graph parent scene does not match candidate")
        if self.execution_graph_fingerprint != execution_graph_fingerprint(self.mission_graph):
            raise ValueError("execution graph fingerprint does not match mission graph")


@dataclass(frozen=True)
class EffectiveMotionSegment:
    """One explicit, read-only segment of the executable successful path."""

    segment_id: str
    kind: SegmentKind
    source_subgraph_id: str
    source_node_id: str
    start_pose_wxyz_xyz: Pose | None
    end_pose_wxyz_xyz: Pose | None
    approach_vector_xyz: tuple[float, float, float] | None
    approach_distance_m: float | None
    payload_track_id: str | None

    def __post_init__(self) -> None:
        if not self.segment_id:
            raise ValueError("motion segment id must not be empty")
        if self.kind not in {
            "grasp_approach",
            "lift",
            "transfer",
            "place_approach",
            "release",
        }:
            raise ValueError("unsupported effective motion segment kind")
        for pose in (self.start_pose_wxyz_xyz, self.end_pose_wxyz_xyz):
            if pose is not None and (
                len(pose) != 7 or not all(math.isfinite(value) for value in pose)
            ):
                raise ValueError("motion segment poses must contain seven finite values")
        if self.approach_vector_xyz is not None and (
            len(self.approach_vector_xyz) != 3
            or not all(math.isfinite(value) for value in self.approach_vector_xyz)
        ):
            raise ValueError("motion segment approach vector must contain three finite values")
        if self.approach_distance_m is not None and (
            self.approach_distance_m < 0 or not math.isfinite(self.approach_distance_m)
        ):
            raise ValueError("motion segment approach distance must be finite and non-negative")


@dataclass(frozen=True)
class EffectiveMotionProgram:
    """Candidate-bound full motion path reused by preview and execution."""

    candidate_fingerprint: str
    execution_graph_fingerprint: str
    program_fingerprint: str
    decision_scene_version: int
    selected_subgraph_id: str
    segments: tuple[EffectiveMotionSegment, ...]
    semantic_signature: str

    def __post_init__(self) -> None:
        if not all(
            value
            for value in (
                self.candidate_fingerprint,
                self.execution_graph_fingerprint,
                self.program_fingerprint,
                self.selected_subgraph_id,
                self.semantic_signature,
            )
        ):
            raise ValueError("effective motion program fingerprints and selected subgraph are required")
        if self.decision_scene_version < 0:
            raise ValueError("effective motion program scene version must not be negative")
        if not self.segments:
            raise ValueError("effective motion program must contain at least one segment")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def execution_graph_fingerprint(graph: MissionGraph) -> str:
    """Return a stable identity for the full graph whose terminal label is used."""

    return _sha256(mission_graph_to_dict(graph))


def bind_effective_motion(
    context: CandidateExecutionContext,
    scene: SceneSnapshot,
) -> EffectiveMotionProgram:
    """Bind the selected successful mission suffix without robot side effects."""

    if scene.scene_version != context.candidate.parent_scene_version:
        raise ValueError("decision scene version does not match candidate")
    route = _success_suffix(context.mission_graph, context.selected_subgraph_id)
    grasp_subgraph, grasp_node, grasp_intent = _single_motion_node(route, "grasp")
    place_subgraph, place_node, place_intent = _single_motion_node(route, "place")
    grasp_pose = _pose_from_intent(grasp_intent, "grasp")
    place_pose = _pose_from_intent(place_intent, "place")
    grasp_approach = _approach_vector(grasp_intent, "grasp")
    place_approach = _approach_vector(place_intent, "place")
    grasp_distance = _skill_distance(grasp_node, "goto_pose", "z_approach")
    lift_distance = _skill_distance(grasp_node, "lift_after_grasp", "z_lift")
    place_distance = _skill_distance(place_node, "goto_pose", "z_approach")
    _require_skill(grasp_node, "sample_grasp_pose")
    _require_pose_call(grasp_node, "goto_pose", grasp_pose, require_symbolic=True)
    _require_pose_call(grasp_node, "lift_after_grasp", grasp_pose, require_symbolic=True)
    _require_pose_call(place_node, "goto_pose", place_pose, require_symbolic=False)

    grasp_start = _offset_pose(grasp_pose, grasp_approach, grasp_distance)
    lift_pose = _lift_pose(grasp_pose, lift_distance)
    place_start = _offset_pose(place_pose, place_approach, place_distance)
    segments = (
        EffectiveMotionSegment(
            "grasp_approach",
            "grasp_approach",
            grasp_subgraph.subgraph_id,
            grasp_node.node_id,
            grasp_start,
            grasp_pose,
            grasp_approach,
            grasp_distance,
            grasp_intent.object_track_id,
        ),
        EffectiveMotionSegment(
            "lift",
            "lift",
            grasp_subgraph.subgraph_id,
            grasp_node.node_id,
            grasp_pose,
            lift_pose,
            (0.0, 0.0, 1.0),
            lift_distance,
            grasp_intent.object_track_id,
        ),
        EffectiveMotionSegment(
            "transfer",
            "transfer",
            place_subgraph.subgraph_id,
            place_node.node_id,
            lift_pose,
            place_start,
            None,
            _position_distance(lift_pose, place_start),
            grasp_intent.object_track_id,
        ),
        EffectiveMotionSegment(
            "place_approach",
            "place_approach",
            place_subgraph.subgraph_id,
            place_node.node_id,
            place_start,
            place_pose,
            place_approach,
            place_distance,
            grasp_intent.object_track_id,
        ),
        EffectiveMotionSegment(
            "release",
            "release",
            place_subgraph.subgraph_id,
            place_node.node_id,
            place_pose,
            place_pose,
            None,
            0.0,
            grasp_intent.object_track_id,
        ),
    )
    candidate_fingerprint = subgraph_fingerprint(context.candidate.subgraph)
    program_payload = {
        "candidate_fingerprint": candidate_fingerprint,
        "execution_graph_fingerprint": context.execution_graph_fingerprint,
        "decision_scene_version": scene.scene_version,
        "selected_subgraph_id": context.selected_subgraph_id,
        "segments": [asdict(segment) for segment in segments],
    }
    return EffectiveMotionProgram(
        candidate_fingerprint=candidate_fingerprint,
        execution_graph_fingerprint=context.execution_graph_fingerprint,
        program_fingerprint=_sha256(program_payload),
        decision_scene_version=scene.scene_version,
        selected_subgraph_id=context.selected_subgraph_id,
        segments=segments,
        semantic_signature=_semantic_signature(segments),
    )


def materialize_execution_graph(
    program: EffectiveMotionProgram,
    graph: MissionGraph,
) -> MissionGraph:
    """Replace symbolic pose references with the exact program-bound literals."""

    if program.execution_graph_fingerprint != execution_graph_fingerprint(graph):
        raise ValueError("program execution graph fingerprint does not match graph")
    by_source = {
        (segment.source_subgraph_id, segment.source_node_id, segment.kind): segment
        for segment in program.segments
    }
    subgraphs = tuple(
        _materialize_subgraph(subgraph, by_source)
        for subgraph in graph.subgraphs
    )
    return replace(graph, subgraphs=subgraphs)


def _success_suffix(graph: MissionGraph, selected_subgraph_id: str) -> tuple[SubgraphSpec, ...]:
    route: list[SubgraphSpec] = []
    current = selected_subgraph_id
    visited: set[str] = set()
    while True:
        if current in visited:
            raise ValueError("successful control-flow path contains a cycle")
        visited.add(current)
        route.append(graph.subgraph(current))
        if current in graph.success_subgraphs:
            return tuple(route)
        successors = tuple(
            edge.target
            for edge in graph.edges
            if edge.source == current and edge.condition == "success"
        )
        if len(successors) != 1:
            raise ValueError("unsupported successful control-flow branch")
        current = successors[0]


def _single_motion_node(
    route: tuple[SubgraphSpec, ...],
    kind: Literal["grasp", "place"],
) -> tuple[SubgraphSpec, SubgraphNodeSpec, MotionIntent]:
    matches = tuple(
        (subgraph, node, node.motion_intent)
        for subgraph in route
        for node in subgraph.nodes
        if node.motion_intent is not None and node.motion_intent.kind == kind
    )
    if len(matches) != 1:
        raise ValueError(f"effective motion requires exactly one {kind} motion intent")
    subgraph, node, intent = matches[0]
    assert intent is not None
    return subgraph, node, intent


def _pose_from_intent(intent: MotionIntent, kind: str) -> Pose:
    pose = intent.target_pose_wxyz_xyz
    if pose is None:
        raise ValueError(f"{kind} motion intent target pose is unavailable")
    return tuple(float(value) for value in pose)  # type: ignore[return-value]


def _approach_vector(intent: MotionIntent, kind: str) -> tuple[float, float, float]:
    vector = intent.approach_vector_xyz
    if vector is None:
        raise ValueError(f"{kind} motion intent approach vector is unavailable")
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1e-9:
        raise ValueError(f"{kind} motion intent approach vector is zero")
    return tuple(value / norm for value in vector)  # type: ignore[return-value]


def _skill_distance(node: SubgraphNodeSpec, skill_id: str, argument: str) -> float:
    call = _require_skill(node, skill_id)
    value = call.args.get(argument)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{skill_id} requires numeric {argument}")
    distance = float(value)
    if not math.isfinite(distance) or distance < 0:
        raise ValueError(f"{skill_id} requires finite non-negative {argument}")
    return distance


def _require_skill(node: SubgraphNodeSpec, skill_id: str) -> SkillCall:
    calls = tuple(call for call in node.skill_calls if call.skill.skill_id == skill_id)
    if len(calls) != 1:
        raise ValueError(f"effective motion requires exactly one {skill_id} call")
    return calls[0]


def _require_pose_call(
    node: SubgraphNodeSpec,
    skill_id: str,
    expected_pose: Pose,
    *,
    require_symbolic: bool,
) -> None:
    call = _require_skill(node, skill_id)
    position = call.args.get("position")
    quaternion = call.args.get("quaternion_wxyz")
    if _is_output_ref(position) or _is_output_ref(quaternion):
        if not (_is_output_ref(position) and _is_output_ref(quaternion)):
            raise ValueError(f"{skill_id} pose references must be complete")
        if not require_symbolic:
            raise ValueError(f"{skill_id} must use a literal pose")
        return
    if require_symbolic:
        raise ValueError(f"{skill_id} requires a resolvable symbolic pose")
    literal = _pose_from_arguments(position, quaternion)
    if literal != expected_pose:
        raise ValueError(f"{skill_id} literal pose does not match motion intent")


def _offset_pose(
    pose: Pose,
    approach: tuple[float, float, float],
    distance: float,
) -> Pose:
    return (
        pose[0],
        pose[1],
        pose[2],
        pose[3],
        pose[4] - approach[0] * distance,
        pose[5] - approach[1] * distance,
        pose[6] - approach[2] * distance,
    )


def _lift_pose(pose: Pose, distance: float) -> Pose:
    return (pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], pose[6] + distance)


def _position_distance(first: Pose, second: Pose) -> float:
    return math.sqrt(sum((first[index] - second[index]) ** 2 for index in range(4, 7)))


def _semantic_signature(segments: tuple[EffectiveMotionSegment, ...]) -> str:
    payload = [
        {
            "kind": segment.kind,
            "payload_track_id": segment.payload_track_id,
            "start_pose_wxyz_xyz": segment.start_pose_wxyz_xyz,
            "end_pose_wxyz_xyz": segment.end_pose_wxyz_xyz,
            "approach_vector_xyz": segment.approach_vector_xyz,
            "approach_distance_m": segment.approach_distance_m,
        }
        for segment in segments
    ]
    return _sha256(payload)


def _materialize_subgraph(
    subgraph: SubgraphSpec,
    by_source: Mapping[tuple[str, str, SegmentKind], EffectiveMotionSegment],
) -> SubgraphSpec:
    nodes = tuple(
        _materialize_node(subgraph.subgraph_id, node, by_source)
        for node in subgraph.nodes
    )
    return replace(subgraph, nodes=nodes)


def _materialize_node(
    subgraph_id: str,
    node: SubgraphNodeSpec,
    by_source: Mapping[tuple[str, str, SegmentKind], EffectiveMotionSegment],
) -> SubgraphNodeSpec:
    grasp = by_source.get((subgraph_id, node.node_id, "grasp_approach"))
    lift = by_source.get((subgraph_id, node.node_id, "lift"))
    place = by_source.get((subgraph_id, node.node_id, "place_approach"))
    if grasp is None and lift is None and place is None:
        return node
    calls: list[SkillCall] = []
    seen: set[str] = set()
    for call in node.skill_calls:
        replacement: EffectiveMotionSegment | None = None
        if call.skill.skill_id == "lift_after_grasp" and lift is not None:
            replacement = lift
        elif call.skill.skill_id == "goto_pose" and grasp is not None:
            replacement = grasp
        elif call.skill.skill_id == "goto_pose" and place is not None:
            replacement = place
        if replacement is None:
            calls.append(call)
            continue
        if replacement.end_pose_wxyz_xyz is None:
            raise ValueError(f"{replacement.segment_id} has no bound end pose")
        seen.add(replacement.kind)
        calls.append(replace(call, args=_materialize_pose_args(call.args, replacement.end_pose_wxyz_xyz)))
    expected = {
        segment.kind
        for segment in (grasp, lift, place)
        if segment is not None
    }
    if seen != expected:
        raise ValueError("motion-bearing call cannot be proven to match a program segment")
    return replace(node, skill_calls=tuple(calls))


def _materialize_pose_args(args: Mapping[str, object], pose: Pose) -> dict[str, object]:
    position = args.get("position")
    quaternion = args.get("quaternion_wxyz")
    if _is_output_ref(position) or _is_output_ref(quaternion):
        if not (_is_output_ref(position) and _is_output_ref(quaternion)):
            raise ValueError("motion call pose references must be complete")
        updated = dict(args)
        updated["position"] = list(pose[4:])
        updated["quaternion_wxyz"] = list(pose[:4])
        return updated
    if _pose_from_arguments(position, quaternion) != pose:
        raise ValueError("literal motion call pose does not match bound program")
    return dict(args)


def _is_output_ref(value: object) -> bool:
    if isinstance(value, SkillOutputRef):
        return True
    if not isinstance(value, Mapping):
        return False
    return set(value) in ({"$skill_output", "path"}, {"call_index", "path"})


def _pose_from_arguments(position: object, quaternion: object) -> Pose:
    position_values = _number_tuple(position, 3, "position")
    quaternion_values = _number_tuple(quaternion, 4, "quaternion_wxyz")
    return quaternion_values + position_values  # type: ignore[return-value]


def _number_tuple(value: object, size: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != size:
        raise ValueError(f"motion call {name} must contain {size} numeric values")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError(f"motion call {name} must contain numeric values")
    numeric = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in numeric):
        raise ValueError(f"motion call {name} must contain finite values")
    return numeric


def _sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "CandidateExecutionContext",
    "EffectiveMotionProgram",
    "EffectiveMotionSegment",
    "bind_effective_motion",
    "execution_graph_fingerprint",
    "materialize_execution_graph",
]
