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
from capmas.perception.effective_motion import EffectiveMotionProgram
from capmas.perception.local_map import LocalMapBackend
from capmas.perception.motion_preview import (
    MotionPreview,
    MotionPreviewBackend,
    ProgramMotionPreview,
)


def candidate_geometry_evidence(
    candidate: GraphCandidate,
    scene: SceneSnapshot,
    local_map: LocalMapBackend | None,
    preview_backend: MotionPreviewBackend,
    deadline_ns: int,
    *,
    program: EffectiveMotionProgram | None = None,
) -> GeometryEvidence:
    started = time.monotonic_ns()
    captured_at_ns = time.time_ns()
    fingerprint = subgraph_fingerprint(candidate.subgraph)
    map_version = _map_version(local_map)
    if program is not None:
        return _program_geometry_evidence(
            candidate,
            scene,
            local_map,
            preview_backend,
            deadline_ns,
            started,
            captured_at_ns,
            fingerprint,
            map_version,
            program,
        )

    intent = _candidate_intent(candidate)
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


def _program_geometry_evidence(
    candidate: GraphCandidate,
    scene: SceneSnapshot,
    local_map: LocalMapBackend | None,
    preview_backend: MotionPreviewBackend,
    deadline_ns: int,
    started_ns: int,
    captured_at_ns: int,
    fingerprint: str,
    map_version: int | None,
    program: EffectiveMotionProgram,
) -> GeometryEvidence:
    if program.candidate_fingerprint != fingerprint:
        raise ValueError("effective motion program candidate fingerprint does not match candidate")
    if program.decision_scene_version != scene.scene_version:
        raise ValueError("effective motion program scene version does not match decision scene")
    if time.monotonic_ns() >= deadline_ns:
        return _unknown_evidence(
            fingerprint,
            scene,
            map_version,
            captured_at_ns,
            _latency_ms(started_ns),
            "geometry deadline expired",
            execution_graph_fingerprint=program.execution_graph_fingerprint,
            program_fingerprint=program.program_fingerprint,
            program_scope="mission_suffix",
        )
    preview_program = getattr(preview_backend, "preview_program", None)
    if not callable(preview_program):
        return _unknown_evidence(
            fingerprint,
            scene,
            map_version,
            captured_at_ns,
            _latency_ms(started_ns),
            "motion preview backend does not support effective motion programs",
            execution_graph_fingerprint=program.execution_graph_fingerprint,
            program_fingerprint=program.program_fingerprint,
            program_scope="mission_suffix",
        )
    preview = preview_program(program, scene, local_map)
    latency = _latency_ms(started_ns)
    if time.monotonic_ns() >= deadline_ns:
        return _unknown_evidence(
            fingerprint,
            scene,
            map_version,
            captured_at_ns,
            latency,
            "geometry deadline expired",
            execution_graph_fingerprint=program.execution_graph_fingerprint,
            program_fingerprint=program.program_fingerprint,
            program_scope="mission_suffix",
        )
    if not isinstance(preview, ProgramMotionPreview):
        raise TypeError("effective motion preview backend must return ProgramMotionPreview")
    return _evidence_from_program_preview(
        fingerprint,
        scene,
        map_version,
        captured_at_ns,
        latency,
        preview,
        program,
        provider=getattr(preview_backend, "backend", "motion_preview"),
        provider_version=getattr(preview_backend, "backend_version", "unknown"),
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


def _evidence_from_program_preview(
    fingerprint: str,
    scene: SceneSnapshot,
    map_version: int | None,
    captured_at_ns: int,
    latency: float,
    preview: ProgramMotionPreview,
    program: EffectiveMotionProgram,
    *,
    provider: str,
    provider_version: str,
) -> GeometryEvidence:
    grasp = next(
        (segment for segment in preview.segments if segment.segment_id == "grasp_approach"),
        None,
    )
    grasp_quality = (
        EvidenceDimension(
            "grasp_quality",
            "pass",
            1.0,
            0.5,
            "bound grasp pose and approach are available",
        )
        if grasp is not None
        and grasp.end_pose_wxyz_xyz is not None
        and next(
            (
                segment.approach_vector_xyz
                for segment in program.segments
                if segment.segment_id == "grasp_approach"
            ),
            None,
        )
        is not None
        else EvidenceDimension(
            "grasp_quality",
            "unknown",
            None,
            None,
            "bound grasp pose or approach is unavailable",
        )
    )
    ik_values = tuple(segment.ik_valid for segment in preview.segments if segment.ik_valid is not None)
    if not ik_values:
        reachability = EvidenceDimension(
            "reachability", "unknown", None, 0.5, "program segments have no IK result"
        )
    else:
        reachable = all(ik_values)
        reachability = EvidenceDimension(
            "reachability",
            "pass" if reachable else "fail",
            1.0 if reachable else 0.0,
            0.5,
            "all measurable program segments are reachable"
            if reachable
            else "at least one program segment is outside conservative workspace",
        )
    clearances = tuple(
        segment.clearance_m
        for segment in preview.segments
        if segment.clearance_m is not None
    )
    if not clearances:
        clearance = EvidenceDimension(
            "clearance", "unknown", None, 0.5, "program segments have no clearance measurement"
        )
    else:
        minimum_clearance = min(clearances)
        clearance_score = min(minimum_clearance / 0.1, 1.0)
        clearance = EvidenceDimension(
            "clearance",
            "pass" if clearance_score >= 0.5 else "fail",
            clearance_score,
            0.5,
            f"minimum segment clearance={minimum_clearance:.4f}m",
        )
    collision_values = tuple(
        segment.collision_free
        for segment in preview.segments
        if segment.collision_free is not None
    )
    if not collision_values:
        collision = EvidenceDimension(
            "collision_risk", "unknown", None, 0.5, "program segments have no collision result"
        )
    else:
        collision_free = all(collision_values)
        collision = EvidenceDimension(
            "collision_risk",
            "pass" if collision_free else "fail",
            0.0 if collision_free else 1.0,
            0.5,
            "all measurable program segments are clear"
            if collision_free
            else "at least one program segment is occupied",
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
        provider=provider or "motion_preview",
        provider_version=provider_version or "unknown",
        captured_at_ns=captured_at_ns,
        latency_ms=latency,
        execution_graph_fingerprint=program.execution_graph_fingerprint,
        program_fingerprint=program.program_fingerprint,
        program_scope="mission_suffix",
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
    *,
    execution_graph_fingerprint: str | None = None,
    program_fingerprint: str | None = None,
    program_scope: str = "subgraph",
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
        execution_graph_fingerprint=execution_graph_fingerprint,
        program_fingerprint=program_fingerprint,
        program_scope=program_scope,  # type: ignore[arg-type]
    )


def _map_version(local_map: LocalMapBackend | None) -> int | None:
    if local_map is None:
        return None
    return local_map.map_version()


def _latency_ms(started_ns: int) -> float:
    return (time.monotonic_ns() - started_ns) / 1_000_000.0
