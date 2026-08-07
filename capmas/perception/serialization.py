from __future__ import annotations

from dataclasses import asdict
import json
import math
from typing import Any, Mapping

from capmas.contracts.core import ArtifactRef
from capmas.contracts.scene import (
    ObjectTrack,
    SceneSnapshot,
    SceneUncertainty,
    SpatialRelation,
    VisualEvidence,
)
from capmas.perception.protocol import CameraFrame, CameraModel, ObservationBundle
from capmas.perception.tracking import ObjectMeasurement


_OBSERVATION_FIELDS = {
    "timestamp_ns",
    "episode_id",
    "episode_epoch",
    "source",
    "sequence",
    "robot_state",
    "frames",
    "object_measurements",
}
_SNAPSHOT_FIELDS = {
    "episode_id",
    "episode_epoch",
    "scene_version",
    "sensor_timestamp_ns",
    "publish_timestamp_ns",
    "robot",
    "objects",
    "local_map",
    "freshness_ms",
    "source_artifacts",
    "visual_evidence",
    "spatial_relations",
    "uncertainty",
    "processing_latency_ms",
}


def observation_to_json(bundle: ObservationBundle) -> str:
    payload = {
        "timestamp_ns": bundle.timestamp_ns,
        "episode_id": bundle.episode_id,
        "episode_epoch": bundle.episode_epoch,
        "source": bundle.source,
        "sequence": bundle.sequence,
        "robot_state": _encode(bundle.robot_state),
        "frames": [_frame_to_dict(frame) for frame in bundle.frames],
        "object_measurements": [
            _measurement_to_dict(measurement) for measurement in bundle.object_measurements
        ],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def observation_from_json(encoded: str | bytes) -> ObservationBundle:
    payload = _load_object(encoded)
    _reject_unknown(payload, _OBSERVATION_FIELDS, "observation")
    return ObservationBundle(
        timestamp_ns=_int(payload, "timestamp_ns"),
        frames=tuple(_frame_from_dict(item) for item in _list(payload, "frames")),
        robot_state=_decode_mapping(payload.get("robot_state", {})),
        episode_id=_optional_str(payload.get("episode_id"), "episode_id"),
        episode_epoch=_optional_int(payload.get("episode_epoch"), "episode_epoch"),
        source=_str(payload.get("source", ""), "source"),
        sequence=_int(payload, "sequence", default=0),
        object_measurements=tuple(
            _measurement_from_dict(item)
            for item in _list(payload, "object_measurements")
        ),
    )


def snapshot_to_json(snapshot: SceneSnapshot) -> str:
    payload = {
        "episode_id": snapshot.episode_id,
        "episode_epoch": snapshot.episode_epoch,
        "scene_version": snapshot.scene_version,
        "sensor_timestamp_ns": snapshot.sensor_timestamp_ns,
        "publish_timestamp_ns": snapshot.publish_timestamp_ns,
        "robot": _encode(snapshot.robot),
        "objects": [_object_to_dict(obj) for obj in snapshot.objects],
        "local_map": _artifact_to_dict(snapshot.local_map),
        "freshness_ms": snapshot.freshness_ms,
        "source_artifacts": [_artifact_to_dict(ref) for ref in snapshot.source_artifacts],
        "visual_evidence": [_visual_evidence_to_dict(item) for item in snapshot.visual_evidence],
        "spatial_relations": [_relation_to_dict(item) for item in snapshot.spatial_relations],
        "uncertainty": asdict(snapshot.uncertainty),
        "processing_latency_ms": snapshot.processing_latency_ms,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def snapshot_from_json(encoded: str | bytes) -> SceneSnapshot:
    payload = _load_object(encoded)
    _reject_unknown(payload, _SNAPSHOT_FIELDS, "snapshot")
    uncertainty = _mapping(payload.get("uncertainty", {}), "uncertainty")
    _reject_unknown(uncertainty, {"scene_confidence", "ambiguous_track_ids", "stale_track_ids", "notes"}, "uncertainty")
    return SceneSnapshot(
        episode_id=_str(payload["episode_id"], "episode_id"),
        episode_epoch=_int(payload, "episode_epoch"),
        scene_version=_int(payload, "scene_version"),
        sensor_timestamp_ns=_int(payload, "sensor_timestamp_ns"),
        publish_timestamp_ns=_int(payload, "publish_timestamp_ns"),
        robot=_decode_mapping(payload.get("robot", {})),
        objects=tuple(_object_from_dict(item) for item in _list(payload, "objects")),
        local_map=_artifact_from_dict(payload.get("local_map")),
        freshness_ms=_finite_float(payload.get("freshness_ms", 0.0), "freshness_ms"),
        source_artifacts=tuple(_artifact_from_dict(item) for item in _list(payload, "source_artifacts")),
        visual_evidence=tuple(
            _visual_evidence_from_dict(item) for item in _list(payload, "visual_evidence")
        ),
        spatial_relations=tuple(
            _relation_from_dict(item) for item in _list(payload, "spatial_relations")
        ),
        uncertainty=SceneUncertainty(
            scene_confidence=_finite_float(uncertainty.get("scene_confidence", 1.0), "scene_confidence"),
            ambiguous_track_ids=tuple(_string_list(uncertainty.get("ambiguous_track_ids", ()))),
            stale_track_ids=tuple(_string_list(uncertainty.get("stale_track_ids", ()))),
            notes=tuple(_string_list(uncertainty.get("notes", ()))),
        ),
        processing_latency_ms=_finite_float(
            payload.get("processing_latency_ms", 0.0), "processing_latency_ms"
        ),
    )


def _frame_to_dict(frame: CameraFrame) -> dict[str, object]:
    return {
        "camera_id": frame.camera_id,
        "timestamp_ns": frame.timestamp_ns,
        "rgb": _artifact_to_dict(frame.rgb),
        "depth": _artifact_to_dict(frame.depth),
        "camera": {
            "camera_id": frame.camera.camera_id,
            "intrinsics": list(frame.camera.intrinsics),
            "pose_world": list(frame.camera.pose_world),
        },
    }


def _frame_from_dict(value: object) -> CameraFrame:
    item = _mapping(value, "frame")
    _reject_unknown(item, {"camera_id", "timestamp_ns", "rgb", "depth", "camera"}, "frame")
    camera = _mapping(item["camera"], "camera")
    _reject_unknown(camera, {"camera_id", "intrinsics", "pose_world"}, "camera")
    return CameraFrame(
        camera_id=_str(item["camera_id"], "camera_id"),
        timestamp_ns=_int(item, "timestamp_ns"),
        rgb=_artifact_from_dict(item.get("rgb")),
        depth=_artifact_from_dict(item.get("depth")),
        camera=CameraModel(
            camera_id=_str(camera["camera_id"], "camera.camera_id"),
            intrinsics=tuple(_number_list(camera.get("intrinsics", ()))),
            pose_world=tuple(_number_list(camera.get("pose_world", ()))),
        ),
    )


def _object_to_dict(obj: ObjectTrack) -> dict[str, object]:
    return {
        "track_id": obj.track_id,
        "label": obj.label,
        "pose_wxyz_xyz": list(obj.pose_wxyz_xyz),
        "confidence": obj.confidence,
        "last_seen_ns": obj.last_seen_ns,
        "covariance": _artifact_to_dict(obj.covariance),
        "evidence": [_artifact_to_dict(ref) for ref in obj.evidence],
        "visual_evidence": [_visual_evidence_to_dict(item) for item in obj.visual_evidence],
        "velocity_xyz": list(obj.velocity_xyz) if obj.velocity_xyz is not None else None,
        "prediction_timestamp_ns": obj.prediction_timestamp_ns,
        "track_status": obj.track_status,
        "placement_pose_wxyz_xyz": (
            list(obj.placement_pose_wxyz_xyz)
            if obj.placement_pose_wxyz_xyz is not None
            else None
        ),
        "placement_pose_source": obj.placement_pose_source,
        "placement_pose_reason": obj.placement_pose_reason,
    }


def _measurement_to_dict(measurement: ObjectMeasurement) -> dict[str, object]:
    return {
        "track_id": measurement.track_id,
        "label": measurement.label,
        "pose_wxyz_xyz": list(measurement.pose_wxyz_xyz),
        "confidence": measurement.confidence,
        "timestamp_ns": measurement.timestamp_ns,
        "covariance": _artifact_to_dict(measurement.covariance),
        "evidence": [_artifact_to_dict(ref) for ref in measurement.evidence],
    }


def _measurement_from_dict(value: object) -> ObjectMeasurement:
    item = _mapping(value, "object_measurement")
    _reject_unknown(
        item,
        {"track_id", "label", "pose_wxyz_xyz", "confidence", "timestamp_ns", "covariance", "evidence"},
        "object_measurement",
    )
    return ObjectMeasurement(
        track_id=_optional_str(item.get("track_id"), "object_measurement.track_id"),
        label=_str(item["label"], "object_measurement.label"),
        pose_wxyz_xyz=tuple(_number_list(item.get("pose_wxyz_xyz", ()))),
        confidence=_finite_float(item["confidence"], "object_measurement.confidence"),
        timestamp_ns=_int(item, "timestamp_ns"),
        covariance=_artifact_from_dict(item.get("covariance")),
        evidence=tuple(_artifact_from_dict(ref) for ref in _list(item, "evidence")),
    )


def _object_from_dict(value: object) -> ObjectTrack:
    item = _mapping(value, "object")
    _reject_unknown(
        item,
        {
            "track_id",
            "label",
            "pose_wxyz_xyz",
            "confidence",
            "last_seen_ns",
            "covariance",
            "evidence",
            "visual_evidence",
            "velocity_xyz",
            "prediction_timestamp_ns",
            "track_status",
            "placement_pose_wxyz_xyz",
            "placement_pose_source",
            "placement_pose_reason",
        },
        "object",
    )
    velocity = item.get("velocity_xyz")
    return ObjectTrack(
        track_id=_str(item["track_id"], "track_id"),
        label=_str(item["label"], "label"),
        pose_wxyz_xyz=tuple(_number_list(item.get("pose_wxyz_xyz", ()))),
        confidence=_finite_float(item["confidence"], "confidence"),
        last_seen_ns=_int(item, "last_seen_ns"),
        covariance=_artifact_from_dict(item.get("covariance")),
        evidence=tuple(_artifact_from_dict(ref) for ref in _list(item, "evidence")),
        visual_evidence=tuple(
            _visual_evidence_from_dict(ref) for ref in _list(item, "visual_evidence")
        ),
        velocity_xyz=tuple(_number_list(velocity)) if velocity is not None else None,
        prediction_timestamp_ns=_optional_int(item.get("prediction_timestamp_ns"), "prediction_timestamp_ns"),
        track_status=_str(item.get("track_status", "observed"), "track_status"),
        placement_pose_wxyz_xyz=(
            tuple(_number_list(item["placement_pose_wxyz_xyz"]))
            if item.get("placement_pose_wxyz_xyz") is not None
            else None
        ),
        placement_pose_source=_optional_str(
            item.get("placement_pose_source"), "placement_pose_source"
        ),
        placement_pose_reason=_optional_str(
            item.get("placement_pose_reason"), "placement_pose_reason"
        ),
    )


def _visual_evidence_to_dict(item: VisualEvidence) -> dict[str, object]:
    return {
        "artifact": _artifact_to_dict(item.artifact),
        "evidence_type": item.evidence_type,
        "captured_at_ns": item.captured_at_ns,
        "camera_id": item.camera_id,
        "region_xyxy": list(item.region_xyxy) if item.region_xyxy is not None else None,
        "track_id": item.track_id,
    }


def _visual_evidence_from_dict(value: object) -> VisualEvidence:
    item = _mapping(value, "visual_evidence")
    _reject_unknown(item, {"artifact", "evidence_type", "captured_at_ns", "camera_id", "region_xyxy", "track_id"}, "visual_evidence")
    region = item.get("region_xyxy")
    return VisualEvidence(
        artifact=_artifact_from_dict(item.get("artifact")),
        evidence_type=_str(item["evidence_type"], "evidence_type"),
        captured_at_ns=_int(item, "captured_at_ns"),
        camera_id=_optional_str(item.get("camera_id"), "camera_id"),
        region_xyxy=tuple(_number_list(region)) if region is not None else None,
        track_id=_optional_str(item.get("track_id"), "track_id"),
    )


def _relation_to_dict(item: SpatialRelation) -> dict[str, object]:
    return {
        "subject_track_id": item.subject_track_id,
        "object_track_id": item.object_track_id,
        "relation": item.relation,
        "confidence": item.confidence,
        "evidence": [_artifact_to_dict(ref) for ref in item.evidence],
    }


def _relation_from_dict(value: object) -> SpatialRelation:
    item = _mapping(value, "spatial_relation")
    _reject_unknown(item, {"subject_track_id", "object_track_id", "relation", "confidence", "evidence"}, "spatial_relation")
    return SpatialRelation(
        subject_track_id=_str(item["subject_track_id"], "subject_track_id"),
        object_track_id=_str(item["object_track_id"], "object_track_id"),
        relation=_str(item["relation"], "relation"),
        confidence=_finite_float(item["confidence"], "confidence"),
        evidence=tuple(_artifact_from_dict(ref) for ref in _list(item, "evidence")),
    )


def _artifact_to_dict(reference: ArtifactRef | None) -> dict[str, object] | None:
    if reference is None:
        return None
    return {
        "uri": reference.uri,
        "media_type": reference.media_type,
        "sha256": reference.sha256,
        "byte_size": reference.byte_size,
    }


def _artifact_from_dict(value: object) -> ArtifactRef | None:
    if value is None:
        return None
    item = _mapping(value, "artifact")
    _reject_unknown(item, {"uri", "media_type", "sha256", "byte_size"}, "artifact")
    return ArtifactRef(
        uri=_str(item["uri"], "artifact.uri"),
        media_type=_str(item["media_type"], "artifact.media_type"),
        sha256=_optional_str(item.get("sha256"), "artifact.sha256"),
        byte_size=_optional_int(item.get("byte_size"), "artifact.byte_size"),
    )


def _encode(value: object) -> object:
    if isinstance(value, ArtifactRef):
        return {"$artifact_ref": _artifact_to_dict(value)}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float is not serializable")
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("mapping keys must be strings")
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    raise ValueError(f"unsupported value type: {type(value).__name__}")


def _decode(value: object) -> object:
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {"$artifact_ref"}:
            return _artifact_from_dict(value["$artifact_ref"])
        return {key: _decode(item) for key, item in value.items()}
    return value


def _decode_mapping(value: object) -> dict[str, object]:
    decoded = _decode(value)
    if not isinstance(decoded, dict):
        raise ValueError("expected object mapping")
    return decoded


def _load_object(encoded: str | bytes) -> dict[str, object]:
    try:
        payload = json.loads(encoded)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid JSON envelope") from exc
    return _mapping(payload, "envelope")


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _list(mapping: Mapping[str, object], key: str) -> list[object]:
    value = mapping.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"{key} must be an array")
    return value


def _reject_unknown(mapping: Mapping[str, object], allowed: set[str], name: str) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(f"unknown fields in {name}: {sorted(unknown)}")


def _str(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _optional_str(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _str(value, name)


def _int(mapping: Mapping[str, object], key: str, default: int | None = None) -> int:
    value = mapping.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _optional_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _number_list(value: object) -> list[float]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("expected numeric array")
    return [_finite_float(item, "array item") for item in value]


def _string_list(value: object) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("expected string array")
    return [_str(item, "array item") for item in value]
