"""JSON-safe physical execution telemetry shared by LIBERO runners."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from capmas.contracts.graph import MissionGraph
from capmas.contracts.scene import SceneSnapshot


def predicate_report_payload(report: object) -> dict[str, object]:
    """Project one deterministic predicate result without losing its reason."""

    return {
        "name": getattr(report, "name", None),
        "passed": bool(getattr(report, "passed", False)),
        "confidence": getattr(report, "confidence", None),
        "evidence": list(getattr(report, "evidence", ()) or ()),
        "reason": getattr(report, "reason", None),
    }


def verification_result_payload(result: object | None) -> dict[str, object] | None:
    """Project pre/postcondition results, including every checked predicate."""

    if result is None:
        return None
    return {
        "contract_id": getattr(result, "contract_id", None),
        "decision": getattr(result, "decision", None),
        "checked_scene_version": getattr(result, "checked_scene_version", None),
        "predicate_results": [
            predicate_report_payload(report)
            for report in getattr(result, "predicate_results", ())
        ],
        "violated_invariants": list(getattr(result, "violated_invariants", ()) or ()),
        "failure_class": getattr(result, "failure_class", None),
    }


def _artifact_ref_payload(reference: object | None) -> dict[str, object] | None:
    if reference is None:
        return None
    return {
        "uri": getattr(reference, "uri", None),
        "media_type": getattr(reference, "media_type", None),
        "sha256": getattr(reference, "sha256", None),
        "byte_size": getattr(reference, "byte_size", None),
    }


def skill_trace_payload(trace: object) -> dict[str, object]:
    """Project a typed skill invocation, retaining grounded arguments and output."""

    return {
        "invocation_id": getattr(trace, "invocation_id", None),
        "skill_id": getattr(trace, "skill_id", None),
        "skill_version": getattr(trace, "skill_version", None),
        "args": dict(getattr(trace, "args", {}) or {}),
        "started_at_ns": getattr(trace, "started_at_ns", None),
        "finished_at_ns": getattr(trace, "finished_at_ns", None),
        "status": getattr(trace, "status", None),
        "error_type": getattr(trace, "error_type", None),
        "error_message": getattr(trace, "error_message", None),
        "output": dict(getattr(trace, "output", {}) or {}),
    }


def execution_trace_payload(trace: object) -> dict[str, object]:
    """Project a graph action trace with its verification boundary data."""

    return {
        "trace_id": getattr(trace, "trace_id", None),
        "episode_id": getattr(trace, "episode_id", None),
        "episode_epoch": getattr(trace, "episode_epoch", None),
        "contract_id": getattr(trace, "contract_id", None),
        "lease_id": getattr(trace, "lease_id", None),
        "parent_scene_version": getattr(trace, "parent_scene_version", None),
        "start_scene_version": getattr(trace, "start_scene_version", None),
        "end_scene_version": getattr(trace, "end_scene_version", None),
        "started_at_ns": getattr(trace, "started_at_ns", None),
        "finished_at_ns": getattr(trace, "finished_at_ns", None),
        "status": getattr(trace, "status", None),
        "failure_class": getattr(trace, "failure_class", None),
        "precondition_result": verification_result_payload(
            getattr(trace, "precondition_result", None)
        ),
        "postcondition_result": verification_result_payload(
            getattr(trace, "postcondition_result", None)
        ),
        "observation_before": _artifact_ref_payload(
            getattr(trace, "observation_before", None)
        ),
        "observation_after": _artifact_ref_payload(
            getattr(trace, "observation_after", None)
        ),
        "metadata": dict(getattr(trace, "metadata", {}) or {}),
        "skill_traces": [
            skill_trace_payload(skill_trace)
            for skill_trace in getattr(trace, "skill_traces", ())
        ],
    }


def graph_event_payload(event: object) -> dict[str, object]:
    return {
        "sequence": getattr(event, "sequence", None),
        "kind": getattr(event, "kind", None),
        "subgraph_id": getattr(event, "subgraph_id", None),
        "node_id": getattr(event, "node_id", None),
        "node_type": getattr(event, "node_type", None),
        "attempt": getattr(event, "attempt", None),
        "outcome": getattr(event, "outcome", None),
        "occurred_at_ns": getattr(event, "occurred_at_ns", None),
    }


def scene_snapshot_payload(
    scene: SceneSnapshot,
    *,
    object_ids: Sequence[str] = (),
) -> dict[str, object]:
    """Serialize observable scene state used by action predicates."""

    requested = {str(identifier).strip().lower() for identifier in object_ids}
    tracks = []
    for track in scene.objects:
        identifiers = {
            str(track.track_id).strip().lower(),
            str(track.label).strip().lower(),
        }
        if requested and requested.isdisjoint(identifiers):
            continue
        tracks.append(
            {
                "track_id": track.track_id,
                "label": track.label,
                "pose_wxyz_xyz": tuple(track.pose_wxyz_xyz),
                "placement_pose_wxyz_xyz": (
                    tuple(track.placement_pose_wxyz_xyz)
                    if track.placement_pose_wxyz_xyz is not None
                    else None
                ),
                "placement_pose_source": track.placement_pose_source,
                "placement_pose_reason": track.placement_pose_reason,
                "confidence": track.confidence,
                "last_seen_ns": track.last_seen_ns,
                "track_status": track.track_status,
            }
        )
    return {
        "scene_version": scene.scene_version,
        "sensor_timestamp_ns": scene.sensor_timestamp_ns,
        "publish_timestamp_ns": scene.publish_timestamp_ns,
        "freshness_ms": scene.freshness_ms,
        "processing_latency_ms": scene.processing_latency_ms,
        "robot": {
            key: scene.robot.get(key)
            for key in (
                "ee_pose_wxyz_xyz",
                "gripper_opening",
                "gripper_commanded_fraction",
            )
            if key in scene.robot
        },
        "objects": tracks,
    }


def _failure_payload(failure: object | None) -> dict[str, object] | None:
    if failure is None:
        return None
    metadata = getattr(failure, "metadata", {})
    evidence_refs = getattr(failure, "evidence_refs", ())
    return {
        "failure_id": getattr(failure, "failure_id", None),
        "failure_class": getattr(failure, "failure_class", None),
        "message": getattr(failure, "message", None),
        "scene_version": getattr(failure, "scene_version", None),
        "source_agent": getattr(failure, "source_agent", None),
        "node_id": getattr(failure, "node_id", None),
        "subgraph_id": getattr(failure, "subgraph_id", None),
        "recoverable": bool(getattr(failure, "recoverable", True)),
        "retry_count": int(getattr(failure, "retry_count", 0)),
        "recovery_policy": getattr(failure, "recovery_policy", None),
        "evidence_refs": list(evidence_refs or ()),
        "metadata": dict(metadata or {}),
    }


def physical_result_payload(
    result: object,
    *,
    evaluator_success: bool,
    graph: MissionGraph | None = None,
    scene_before: SceneSnapshot | None = None,
    scene_after: SceneSnapshot | None = None,
    object_ids: Sequence[str] = (),
    layout_report: object | None = None,
    scene_diagnostics: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Serialize a physical graph outcome without discarding diagnostic context."""

    from capmas.evaluation.labels import extract_horizon

    failure = _failure_payload(getattr(result, "failure", None))
    completed = bool(getattr(result, "completed", False))
    diagnostics = dict(scene_diagnostics or {})
    if scene_before is not None:
        diagnostics.setdefault(
            "before", scene_snapshot_payload(scene_before, object_ids=object_ids)
        )
    if scene_after is not None:
        diagnostics.setdefault(
            "after", scene_snapshot_payload(scene_after, object_ids=object_ids)
        )
    payload: dict[str, object] = {
        "completed": completed,
        "graph_completed": completed,
        "evaluator_success": bool(evaluator_success),
        "verifier_success": completed,
        "success": bool(completed and evaluator_success),
        "execution_valid": True,
        "failure_class": failure["failure_class"] if failure is not None else None,
        "failure_reason": failure["message"] if failure is not None else None,
        "failure": failure,
        "trace_count": len(getattr(result, "traces", ()) or ()),
        "traces": [
            execution_trace_payload(trace)
            for trace in getattr(result, "traces", ())
        ],
        "terminal_subgraph": getattr(result, "terminal_subgraph", None),
        "next_subgraph": getattr(result, "next_subgraph", None),
        "layout_application": layout_report,
        "scene_diagnostics": diagnostics,
    }
    if graph is None:
        payload["horizon"] = {
            "planned_critical_path_actions": None,
            "planned_critical_path_subgoals": None,
            "planned_checkpoint_subgraphs": None,
            "attempted_actions": None,
            "completed_actions": None,
            "attempted_subgoals": None,
            "completed_subgoals": None,
            "attempted_checkpoints": None,
            "completed_checkpoints": None,
            "planned_source": "unknown",
            "realized_source": "unknown",
            "planned_valid": False,
            "realized_valid": False,
        }
        return payload
    events = tuple(getattr(result, "events", ()) or ())
    payload["graph_events"] = [graph_event_payload(event) for event in events]
    payload["horizon"] = extract_horizon(graph, events).to_dict()
    return payload


__all__ = [
    "execution_trace_payload",
    "graph_event_payload",
    "physical_result_payload",
    "predicate_report_payload",
    "scene_snapshot_payload",
    "skill_trace_payload",
    "verification_result_payload",
]
