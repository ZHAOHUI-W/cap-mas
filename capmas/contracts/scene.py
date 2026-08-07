from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from capmas.contracts.core import ArtifactRef, EpisodeHandle


class EpisodeStatus:
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass(frozen=True)
class VisualEvidence:
    """An artifact-backed visual observation that grounds a scene fact."""

    artifact: ArtifactRef
    evidence_type: str
    captured_at_ns: int
    camera_id: str | None = None
    region_xyxy: tuple[float, float, float, float] | None = None
    track_id: str | None = None


@dataclass(frozen=True)
class SpatialRelation:
    subject_track_id: str
    object_track_id: str
    relation: str
    confidence: float
    evidence: tuple[ArtifactRef, ...] = ()


@dataclass(frozen=True)
class SceneUncertainty:
    scene_confidence: float = 1.0
    ambiguous_track_ids: tuple[str, ...] = ()
    stale_track_ids: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ObjectTrack:
    track_id: str
    label: str
    pose_wxyz_xyz: tuple[float, ...]
    confidence: float
    last_seen_ns: int
    covariance: ArtifactRef | None = None
    evidence: tuple[ArtifactRef, ...] = ()
    visual_evidence: tuple[VisualEvidence, ...] = ()
    velocity_xyz: tuple[float, float, float] | None = None
    prediction_timestamp_ns: int | None = None
    track_status: str = "observed"
    # Optional pose used to approach a placement target.  This is deliberately
    # separate from ``pose_wxyz_xyz``: a container's semantic body center is
    # not necessarily a safe point for the gripper to enter.
    placement_pose_wxyz_xyz: tuple[float, ...] | None = None
    placement_pose_source: str | None = None
    placement_pose_reason: str | None = None


@dataclass(frozen=True)
class SceneSnapshot:
    episode_id: str
    episode_epoch: int
    scene_version: int
    sensor_timestamp_ns: int
    publish_timestamp_ns: int
    robot: Mapping[str, object]
    objects: Sequence[ObjectTrack] = ()
    local_map: ArtifactRef | None = None
    freshness_ms: float = 0.0
    source_artifacts: tuple[ArtifactRef, ...] = ()
    visual_evidence: tuple[VisualEvidence, ...] = ()
    spatial_relations: tuple[SpatialRelation, ...] = ()
    uncertainty: SceneUncertainty = field(default_factory=SceneUncertainty)
    processing_latency_ms: float = 0.0


@dataclass(frozen=True)
class EpisodeStart:
    handle: EpisodeHandle
    initial_scene: SceneSnapshot
