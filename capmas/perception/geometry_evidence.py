"""Convert side-effect-free motion previews into candidate-bound evidence."""

from __future__ import annotations

import time

from capmas.contracts.candidates import (
    EvidenceDimension,
    GeometryEvidence,
    GraphCandidate,
    subgraph_fingerprint,
)
from capmas.contracts.graph import MotionIntent
from capmas.contracts.scene import SceneSnapshot
from capmas.graph.normalizer import normalize_motion_intent
from capmas.perception.local_map import LocalMapBackend
from capmas.perception.motion_preview import MotionPreview, MotionPreviewBackend


def candidate_geometry_evidence(
    candidate: GraphCandidate,
    scene: SceneSnapshot,
    local_map: LocalMapBackend | None,
    preview_backend: MotionPreviewBackend,
    deadline_ns: int,
) -> GeometryEvidence:
    started = time.monotonic_ns()
    captured_at_ns = time.time_ns()
    fingerprint = subgraph_fingerprint(candidate.subgraph)
    intent = _candidate_intent(candidate)
    map_version = _map_version(local_map)
    if intent is None or time.monotonic_ns() >= deadline_ns:
        return _unknown_evidence(
            fingerprint,
            scene,
            map_version,
            captured_at_ns,
            _latency_ms(started),
            "motion intent is unavailable or geometry deadline expired",
        )

    preview = preview_backend.preview(intent, scene, local_map)
    latency = _latency_ms(started)
    if time.monotonic_ns() >= deadline_ns:
        return _unknown_evidence(
            fingerprint,
            scene,
            map_version,
            captured_at_ns,
            latency,
            "geometry deadline expired",
        )
    return _evidence_from_preview(
        fingerprint,
        scene,
        map_version,
        captured_at_ns,
        latency,
        preview,
    )


def _candidate_intent(candidate: GraphCandidate) -> MotionIntent | None:
    for node in candidate.subgraph.nodes:
        if node.motion_intent is not None:
            return node.motion_intent
        if node.node_type == "action":
            return normalize_motion_intent(node).motion_intent
    return None


def _evidence_from_preview(
    fingerprint: str,
    scene: SceneSnapshot,
    map_version: int | None,
    captured_at_ns: int,
    latency: float,
    preview: MotionPreview,
) -> GeometryEvidence:
    grasp_quality = EvidenceDimension(
        "grasp_quality",
        "unknown",
        None,
        None,
        "surface normal/contact evidence is unavailable",
    )
    reachability = _bool_dimension(
        "reachability",
        preview.ik_valid,
        0.5,
        "conservative workspace preview",
    )
    if preview.collision_free is None or preview.clearance_m is None:
        clearance = EvidenceDimension("clearance", "unknown", None, 0.5, preview.reason)
        collision = EvidenceDimension("collision_risk", "unknown", None, 0.5, preview.reason)
    else:
        clearance_score = min(preview.clearance_m / 0.1, 1.0)
        clearance = EvidenceDimension(
            "clearance",
            "pass" if clearance_score >= 0.5 else "fail",
            clearance_score,
            0.5,
            f"clearance={preview.clearance_m:.4f}m; {preview.reason}",
        )
        risk_score = 0.0 if preview.collision_free else 1.0
        collision = EvidenceDimension(
            "collision_risk",
            "pass" if preview.collision_free else "fail",
            risk_score,
            0.5,
            preview.reason,
        )
    return GeometryEvidence(
        grasp_quality=grasp_quality,
        reachability=reachability,
        clearance=clearance,
        collision_risk=collision,
        candidate_fingerprint=fingerprint,
        scene_version=scene.scene_version,
        map_version=map_version,
        map_backend="none" if map_version is None else "local_map",
        provider=preview.backend or "motion_preview",
        provider_version=preview.backend_version or "unknown",
        captured_at_ns=captured_at_ns,
        latency_ms=latency,
    )


def _bool_dimension(
    name: str,
    value: bool | None,
    threshold: float,
    reason: str,
) -> EvidenceDimension:
    if value is None:
        return EvidenceDimension(name, "unknown", None, threshold, reason)
    return EvidenceDimension(name, "pass" if value else "fail", 1.0 if value else 0.0, threshold, reason)


def _unknown_evidence(
    fingerprint: str,
    scene: SceneSnapshot,
    map_version: int | None,
    captured_at_ns: int,
    latency: float,
    reason: str,
) -> GeometryEvidence:
    def unknown(name: str) -> EvidenceDimension:
        return EvidenceDimension(name, "unknown", None, None, reason)
    return GeometryEvidence(
        grasp_quality=unknown("grasp_quality"),
        reachability=unknown("reachability"),
        clearance=unknown("clearance"),
        collision_risk=unknown("collision_risk"),
        candidate_fingerprint=fingerprint,
        scene_version=scene.scene_version,
        map_version=map_version,
        map_backend="none" if map_version is None else "local_map",
        provider="motion_preview",
        provider_version="unknown",
        captured_at_ns=captured_at_ns,
        latency_ms=latency,
    )


def _map_version(local_map: LocalMapBackend | None) -> int | None:
    if local_map is None:
        return None
    return local_map.map_version()


def _latency_ms(started_ns: int) -> float:
    return (time.monotonic_ns() - started_ns) / 1_000_000.0
