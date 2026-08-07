from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Callable, Mapping, Sequence

from capmas.contracts.action import ActionContract
from capmas.contracts.failures import FailureClass
from capmas.contracts.scene import ObjectTrack, SceneSnapshot
from capmas.contracts.trace import ExecutionTrace
from capmas.contracts.verification import PredicateReport, VerificationResult


class PredicateRegistry:
    def __init__(self, predicates: Mapping[str, Callable[[SceneSnapshot], bool]] | None = None) -> None:
        self._predicates = dict(predicates or {})

    def register(self, name: str, predicate: Callable[[SceneSnapshot], bool]) -> None:
        self._predicates[name] = predicate

    def evaluate(self, name: str, scene: SceneSnapshot) -> PredicateReport:
        if name not in self._predicates:
            return PredicateReport(name, False, reason="unknown predicate")
        try:
            passed = bool(self._predicates[name](scene))
        except Exception as exc:
            return PredicateReport(name, False, reason=str(exc))
        return PredicateReport(name, passed)


@dataclass
class PredicateBasedVerifier:
    """Deterministic verifier for predicates grounded in a scene snapshot."""

    gripper_open_threshold: float = 0.8
    gripper_closed_threshold: float = 0.2
    object_gripper_distance_threshold_m: float = 0.16
    object_target_distance_threshold_m: float = 0.06
    clock: Callable[[], int] = time.time_ns

    def evaluate_predicates(
        self,
        predicates: Sequence[str],
        scene: SceneSnapshot,
    ) -> tuple[PredicateReport, ...]:
        return tuple(self._evaluate(name, scene) for name in predicates)

    def goal_satisfied(self, predicates: Sequence[str], scene: SceneSnapshot) -> bool:
        return all(report.passed for report in self.evaluate_predicates(predicates, scene))

    def approve(self, contract: ActionContract, scene: SceneSnapshot) -> VerificationResult:
        preconditions = tuple(self._evaluate(name, scene) for name in contract.preconditions)
        invariants = tuple(self._evaluate(name, scene) for name in contract.safety_invariants)
        failed_invariants = tuple(
            name for name, report in zip(contract.safety_invariants, invariants) if not report.passed
        )
        if failed_invariants:
            decision = "reject"
            failure_class = FailureClass.COLLISION_RISK
        elif any(not report.passed for report in preconditions):
            decision = "reject"
            failure_class = FailureClass.PRECONDITION_FAILED
        else:
            decision = "approve"
            failure_class = None
        return VerificationResult(
            contract_id=contract.contract_id,
            decision=decision,
            checked_scene_version=scene.scene_version,
            predicate_results=preconditions + invariants,
            violated_invariants=failed_invariants,
            failure_class=failure_class,
        )

    def commit(
        self,
        contract: ActionContract,
        before: SceneSnapshot,
        after: SceneSnapshot,
        trace: ExecutionTrace,
    ) -> VerificationResult:
        del trace
        reports = tuple(
            self._evaluate(name, after, before=before) for name in contract.expected_postconditions
        )
        passed = all(report.passed for report in reports)
        return VerificationResult(
            contract_id=contract.contract_id,
            decision="commit" if passed else "recover",
            checked_scene_version=after.scene_version,
            predicate_results=reports,
            failure_class=None if passed else FailureClass.POSTCONDITION_FAILED,
        )

    def _evaluate(
        self,
        name: str,
        scene: SceneSnapshot,
        *,
        before: SceneSnapshot | None = None,
    ) -> PredicateReport:
        predicate, arguments = _parse_predicate(name)
        if predicate in {"scene_advanced", "scene_version_advanced"}:
            passed = before is not None and scene.scene_version == before.scene_version + 1
            return PredicateReport(
                name,
                passed,
                reason=None if passed else "scene version did not advance",
            )
        if predicate in {"gripper_open", "gripper.open"}:
            value = _gripper_control_value(scene)
            passed = value is not None and value >= self.gripper_open_threshold
            return PredicateReport(name, passed, reason=None if passed else "gripper is not open")
        if predicate in {"gripper_closed", "gripper.closed"}:
            value = _gripper_control_value(scene)
            passed = value is not None and value <= self.gripper_closed_threshold
            return PredicateReport(name, passed, reason=None if passed else "gripper is not closed")
        if predicate == "scene_fresh":
            if len(arguments) != 1:
                return PredicateReport(name, False, reason="scene_fresh requires threshold_ms")
            try:
                threshold_ms = float(arguments[0])
            except ValueError:
                return PredicateReport(name, False, reason="scene_fresh threshold_ms is not numeric")
            age_ms = max(0.0, (self.clock() - scene.publish_timestamp_ns) / 1_000_000.0)
            passed = age_ms <= threshold_ms
            return PredicateReport(
                name,
                passed,
                confidence=scene.uncertainty.scene_confidence,
                reason=None if passed else f"scene age {age_ms:.3f}ms exceeds {threshold_ms:.3f}ms",
            )
        if predicate in {"object_in_gripper", "object_held", "object_near_gripper"}:
            if len(arguments) != 1:
                return PredicateReport(name, False, reason=f"{predicate} requires obj_id")
            track = _find_track(scene, arguments[0])
            gripper_value = _gripper_control_value(scene)
            ee_position = _ee_position(scene)
            if track is None:
                return PredicateReport(name, False, reason="object track not found")
            if predicate == "object_held" and (
                gripper_value is None or gripper_value > self.gripper_closed_threshold
            ):
                return PredicateReport(name, False, reason="gripper is not closed")
            if ee_position is None:
                return PredicateReport(name, False, reason="end-effector pose is unavailable")
            object_position = _position_from_pose(track.pose_wxyz_xyz)
            if object_position is None:
                return PredicateReport(name, False, reason="object pose is unavailable")
            distance = _distance(ee_position, object_position)
            passed = distance <= self.object_gripper_distance_threshold_m
            return PredicateReport(
                name,
                passed,
                confidence=track.confidence,
                evidence=(track.track_id,),
                reason=None if passed else f"object-gripper distance {distance:.4f}m exceeds threshold",
            )
        if predicate == "object_at_target":
            if len(arguments) != 2:
                return PredicateReport(name, False, reason="object_at_target requires obj_id,target_id")
            object_track = _find_track(scene, arguments[0])
            target_track = _find_track(scene, arguments[1])
            if object_track is None or target_track is None:
                return PredicateReport(name, False, reason="object or target track not found")
            object_position = _position_from_pose(object_track.pose_wxyz_xyz)
            target_position = _position_from_pose(target_track.pose_wxyz_xyz)
            if object_position is None or target_position is None:
                return PredicateReport(name, False, reason="object or target pose is unavailable")
            # A container can be partially occluded after release, which makes
            # the OBB-based semantic pose drift laterally.  The CAP-X adapter's
            # placement pose has a robust XY reference; retain the semantic
            # target height because placement poses may intentionally sit above
            # a container opening for collision-free release.
            placement_pose = target_track.placement_pose_wxyz_xyz
            if placement_pose is not None and len(placement_pose) >= 7:
                target_position = (
                    float(placement_pose[4]),
                    float(placement_pose[5]),
                    target_position[2],
                )
            distance = _distance(object_position, target_position)
            passed = distance <= self.object_target_distance_threshold_m
            return PredicateReport(
                name,
                passed,
                confidence=min(object_track.confidence, target_track.confidence),
                evidence=(object_track.track_id, target_track.track_id),
                reason=None if passed else f"object-target distance {distance:.4f}m exceeds threshold",
            )
        if predicate == "track_exists" and len(arguments) == 1:
            track_id = arguments[0]
            passed = _find_track(scene, track_id) is not None
            return PredicateReport(name, passed, reason=None if passed else "track not found")
        if name.startswith("track_exists:"):
            track_id = name.split(":", 1)[1]
            passed = _find_track(scene, track_id) is not None
            return PredicateReport(name, passed, reason=None if passed else "track not found")
        if predicate == "object_visible" and len(arguments) == 1:
            label = arguments[0]
            passed = _find_track(scene, label) is not None
            return PredicateReport(name, passed, reason=None if passed else "object not visible")
        if name.startswith("object_visible:"):
            label = name.split(":", 1)[1]
            passed = _find_track(scene, label) is not None
            return PredicateReport(name, passed, reason=None if passed else "object not visible")
        if predicate == "relation" and len(arguments) == 3:
            subject, relation, target = arguments
            passed = any(
                item.subject_track_id == subject
                and item.relation == relation
                and item.object_track_id == target
                for item in scene.spatial_relations
            )
            return PredicateReport(name, passed, reason=None if passed else "relation not observed")
        if name.startswith("relation:"):
            _, subject, relation, target = (name.split(":", 3) + [""])[:4]
            passed = any(
                item.subject_track_id == subject
                and item.relation == relation
                and item.object_track_id == target
                for item in scene.spatial_relations
            )
            return PredicateReport(name, passed, reason=None if passed else "relation not observed")
        if name.startswith("scene_confidence>="):
            threshold = float(name.split(">=", 1)[1])
            passed = scene.uncertainty.scene_confidence >= threshold
            return PredicateReport(name, passed, confidence=scene.uncertainty.scene_confidence)
        return PredicateReport(name, False, reason="unknown observable predicate")


def _parse_predicate(name: str) -> tuple[str, tuple[str, ...]]:
    match = re.fullmatch(r"([a-zA-Z_][a-zA-Z0-9_.]*)\((.*)\)", name.strip())
    if match is None:
        return name.strip(), ()
    predicate = match.group(1)
    arguments = tuple(item.strip() for item in match.group(2).split(",") if item.strip())
    return predicate, arguments


def _canonical_identifier(value: str) -> str:
    return " ".join(value.replace("_", " ").lower().split())


def _find_track(scene: SceneSnapshot, identifier: str) -> ObjectTrack | None:
    canonical = _canonical_identifier(identifier)
    for track in scene.objects:
        if canonical in {
            _canonical_identifier(track.track_id),
            _canonical_identifier(track.label),
        }:
            return track
    return None


def _position_from_pose(pose: object) -> tuple[float, float, float] | None:
    try:
        values = tuple(float(item) for item in pose)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if len(values) < 7:
        return None
    return values[4], values[5], values[6]


def _ee_position(scene: SceneSnapshot) -> tuple[float, float, float] | None:
    for key in ("ee_pose_wxyz_xyz", "ee_pose"):
        value = scene.robot.get(key)
        position = _position_from_pose(value)
        if position is not None:
            return position
    value = scene.robot.get("ee_position")
    try:
        values = tuple(float(item) for item in value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return values if len(values) == 3 else None  # type: ignore[return-value]


def _gripper_value(scene: SceneSnapshot) -> float | None:
    value = scene.robot.get("gripper_opening")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _gripper_control_value(scene: SceneSnapshot) -> float | None:
    """Prefer CAP-X's commanded fraction over physical finger opening.

    A held object can keep the physical fingers apart even after the robot has
    received the closed command. Older snapshots do not carry the command, so
    they retain the legacy opening-based behavior.
    """
    value = scene.robot.get("gripper_commanded_fraction")
    try:
        if value is not None:
            return float(value)
    except (TypeError, ValueError):
        pass
    return _gripper_value(scene)


def _distance(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    return sum((left - right) ** 2 for left, right in zip(first, second)) ** 0.5
