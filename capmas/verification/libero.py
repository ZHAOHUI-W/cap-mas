from __future__ import annotations

from collections.abc import Sequence

from dataclasses import replace

from capmas.contracts.action import SkillCall, SkillOutputRef
from capmas.contracts.candidates import CandidateEvidence, GraphCandidate, PerceptionEvidence
from capmas.contracts.core import SkillRef
from capmas.contracts.graph import SubgraphSpec
from capmas.contracts.scene import SceneSnapshot
from capmas.verification.predicates import PredicateBasedVerifier


class LiberoObservableVerifier(PredicateBasedVerifier):
    """LIBERO verifier using only observable CAP-MAS scene facts."""


def validate_libero_skill_sequence(skill_calls: Sequence[SkillCall]) -> None:
    """Reject LIBERO grasp candidates that cannot move to a sampled pose."""
    skill_ids = [call.skill.skill_id for call in skill_calls]
    if "close_gripper" in skill_ids and "goto_pose" in skill_ids:
        if "sample_grasp_pose" not in skill_ids:
            raise ValueError(
                "goto_pose followed by close_gripper must use sample_grasp_pose"
            )
    if "sample_grasp_pose" not in skill_ids:
        return
    sample_index = skill_ids.index("sample_grasp_pose")
    try:
        goto_index = skill_ids.index("goto_pose", sample_index + 1)
    except ValueError as exc:
        raise ValueError(
            "sample_grasp_pose must be followed by goto_pose before gripper actuation"
        ) from exc
    if "close_gripper" in skill_ids and skill_ids.index("close_gripper") < goto_index:
        raise ValueError("close_gripper cannot run before goto_pose after sampling a grasp")


def compile_time_preconditions(preconditions: Sequence[str]) -> tuple[str, ...]:
    """Keep only scene facts safe to check before upstream subgoals execute.

    Gripper and object-state predicates are intentionally deferred to the
    runtime dispatch point. A downstream subgraph may depend on a grasp or
    release established by an earlier subgraph, so evaluating those facts
    against the episode's initial scene would reject valid candidates.
    """
    stable_prefixes = ("track_exists:", "object_visible:")
    return tuple(
        predicate for predicate in preconditions if predicate.startswith(stable_prefixes)
    )


def ground_libero_grasp_subgraph(
    subgraph: SubgraphSpec,
    scene: SceneSnapshot | None = None,
) -> SubgraphSpec:
    """Ground grasp and placement poses to typed runtime observations."""
    grounded_nodes = []
    for node in subgraph.nodes:
        calls = list(node.skill_calls)
        sample_index: int | None = None
        changed = False
        for index, call in enumerate(calls):
            if call.skill.skill_id == "sample_grasp_pose":
                sample_index = index
                continue
            if call.skill.skill_id != "goto_pose" or sample_index is None:
                if call.skill.skill_id != "goto_pose" or scene is None:
                    continue
            args = dict(call.args)
            if sample_index is not None:
                args["position"] = SkillOutputRef(sample_index, ("result", 0))
                args["quaternion_wxyz"] = SkillOutputRef(sample_index, ("result", 1))
                args.setdefault("z_approach", 0.10)
            else:
                target_position = _target_position(subgraph, node.postconditions, scene)
                if target_position is None:
                    continue
                args["position"] = target_position
                args["quaternion_wxyz"] = (0.0, 1.0, 0.0, 0.0)
                args["z_approach"] = _placement_approach(args.get("z_approach"))
            calls[index] = SkillCall(call.skill, args)
            changed = True
        grounded_nodes.append(
            replace(node, skill_calls=tuple(calls)) if changed else node
        )
    if tuple(grounded_nodes) == subgraph.nodes:
        return subgraph
    return replace(subgraph, nodes=tuple(grounded_nodes))


def _placement_approach(value: object) -> float:
    """Keep the release approach above the empirically safe LIBERO floor."""
    try:
        approach = float(value) if value is not None else 0.12
    except (TypeError, ValueError):
        approach = 0.12
    return max(0.12, approach)


def repair_libero_grasp_subgraph(subgraph: SubgraphSpec) -> SubgraphSpec:
    """Insert a typed motion step for a sampled grasp missing ``goto_pose``.

    This is a narrow safety canonicalization. It does not invent postconditions
    or target poses; those remain Policy/Verifier responsibilities.
    """
    repaired_nodes = []
    changed = False
    for node in subgraph.nodes:
        calls = list(node.skill_calls)
        for index, call in enumerate(calls):
            if call.skill.skill_id != "sample_grasp_pose" or "z_approach" not in call.args:
                continue
            args = dict(call.args)
            args.pop("z_approach", None)
            calls[index] = SkillCall(call.skill, args)
            changed = True
        skill_ids = [call.skill.skill_id for call in calls]
        if "sample_grasp_pose" in skill_ids:
            sample_index = skill_ids.index("sample_grasp_pose")
            has_motion = "goto_pose" in skill_ids[sample_index + 1 :]
            if not has_motion:
                sample_call = calls[sample_index]
                calls.insert(
                    sample_index + 1,
                    SkillCall(
                        SkillRef("goto_pose", sample_call.skill.version),
                        {
                            "position": SkillOutputRef(sample_index, ("result", 0)),
                            "quaternion_wxyz": SkillOutputRef(sample_index, ("result", 1)),
                            "z_approach": 0.10,
                        },
                    ),
                )
                changed = True
        repaired_nodes.append(replace(node, skill_calls=tuple(calls)) if changed else node)
    return replace(subgraph, nodes=tuple(repaired_nodes)) if changed else subgraph


def libero_candidate_evidence(
    candidate: GraphCandidate,
    scene: SceneSnapshot,
) -> CandidateEvidence:
    """Score candidate perception quality from the committed LIBERO scene.

    This provider is intentionally read-only. It does not call CAP-X or a
    simulator and therefore remains safe to run concurrently with Policy
    inference. Rehearsal and OOD dimensions stay unavailable until their
    separate providers are attached.
    """
    subgraph = candidate.raw_subgraph or candidate.subgraph
    predicates = [
        predicate
        for node in subgraph.nodes
        for predicate in (*node.preconditions, *node.postconditions)
    ]
    predicates.extend(
        predicate
        for checkpoint in subgraph.checkpoints
        if checkpoint.validate
        for predicate in checkpoint.predicates
    )
    target_ids = _candidate_scene_ids(predicates)
    tracks = tuple(
        track
        for identifier in target_ids
        for track in (_scene_track(scene, identifier),)
        if track is not None
    )
    visibility = len(tracks) / len(target_ids) if target_ids else 0.0
    track_confidence = min((track.confidence for track in tracks), default=0.0)
    identity_confidence = (
        0.0
        if any(track.track_id in scene.uncertainty.ambiguous_track_ids for track in tracks)
        else 1.0 if tracks and len(tracks) == len(target_ids) else 0.0
    )
    pose_reliability = (
        min(track.confidence for track in tracks)
        if tracks and all(len(track.pose_wxyz_xyz) >= 7 for track in tracks)
        else 0.0
    )
    freshness = max(0.0, min(1.0, 1.0 - max(scene.freshness_ms, 0.0) / 1000.0))
    evidence_refs = tuple(
        artifact.uri
        for track in tracks
        for artifact in track.evidence
    )
    perception = PerceptionEvidence(
        scene_freshness=freshness,
        scene_confidence=scene.uncertainty.scene_confidence,
        target_visibility=visibility,
        track_confidence=track_confidence,
        identity_confidence=identity_confidence,
        pose_reliability=pose_reliability,
        evidence_refs=evidence_refs,
    )
    return CandidateEvidence(
        evidence_refs=evidence_refs,
        perception=perception,
        available_metrics=("perception",),
        scene_version=scene.scene_version,
        provider="libero_scene_snapshot",
        captured_at_ns=scene.publish_timestamp_ns,
    )


def _target_position(
    subgraph: SubgraphSpec,
    postconditions: Sequence[str],
    scene: SceneSnapshot,
) -> tuple[float, float, float] | None:
    predicates = list(postconditions)
    predicates.extend(
        predicate
        for node in subgraph.nodes
        for predicate in node.postconditions
    )
    predicates.extend(
        predicate
        for checkpoint in subgraph.checkpoints
        if checkpoint.validate
        for predicate in checkpoint.predicates
    )
    for predicate in predicates:
        if not predicate.startswith("object_at_target(") or not predicate.endswith(")"):
            continue
        parts = predicate[len("object_at_target(") : -1].split(",")
        if len(parts) != 2:
            continue
        target_id = parts[1].strip()
        for track in scene.objects:
            normalized_id = track.track_id.replace(" ", "_")
            normalized_label = track.label.replace(" ", "_")
            if target_id in {track.track_id, normalized_id, normalized_label}:
                return tuple(float(value) for value in track.pose_wxyz_xyz[4:7])
    return None


def _candidate_scene_ids(predicates: Sequence[str]) -> tuple[str, ...]:
    identifiers: list[str] = []
    for predicate in predicates:
        if predicate.startswith("object_at_target(") and predicate.endswith(")"):
            identifiers.extend(
                part.strip()
                for part in predicate[len("object_at_target(") : -1].split(",")
            )
        elif predicate.startswith("object_in_gripper(") and predicate.endswith(")"):
            identifiers.append(predicate[len("object_in_gripper(") : -1].strip())
        elif predicate.startswith(("track_exists:", "object_visible:")):
            identifiers.append(predicate.split(":", 1)[1].strip())
    return tuple(dict.fromkeys(identifier for identifier in identifiers if identifier))


def _scene_track(scene: SceneSnapshot, identifier: str):
    normalized = identifier.replace(" ", "_")
    for track in scene.objects:
        if identifier in {track.track_id, track.label}:
            return track
        if normalized in {
            track.track_id.replace(" ", "_"),
            track.label.replace(" ", "_"),
        }:
            return track
    return None
