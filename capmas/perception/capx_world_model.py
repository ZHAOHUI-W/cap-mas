"""CAP-X observation bridge for the reference World Model path."""

from __future__ import annotations

from dataclasses import replace
import time
from typing import Callable

from capmas.contracts.scene import SceneSnapshot
from capmas.perception.capx_depth import CAPXDepthDecoder
from capmas.perception.geometry import ReferenceGeometryEstimator
from capmas.perception.local_map import SparseVoxelMap
from capmas.perception.protocol import ObservationBundle, ObservationProvider
from capmas.perception.tracking import KnownObjectTracker, ObjectMeasurement
from capmas.perception.world_model import WorldModelService


class CAPXWorldModelEnricher:
    """Update a live local map while preserving CAP-X scene version semantics.

    CAP-X remains the execution authority. This adapter only consumes the same
    observation that was used to build the base snapshot and adds World Model
    products such as ``SceneSnapshot.local_map``. Failures are deliberately
    fail-open: the base CAP-X snapshot is returned and geometry evidence can
    downgrade to ``unknown`` rather than blocking physical execution.
    """

    def __init__(self, service: WorldModelService) -> None:
        self.service = service
        self.local_map = service.local_map
        self.processed_observations = 0
        self.last_error: str | None = None
        self._previous: SceneSnapshot | None = None

    def enrich(
        self,
        observation: ObservationBundle,
        snapshot: SceneSnapshot,
    ) -> SceneSnapshot:
        normalized = replace(
            observation,
            episode_id=snapshot.episode_id,
            episode_epoch=snapshot.episode_epoch,
        )
        try:
            enriched = self.service.process(normalized, self._previous)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return snapshot

        self._previous = enriched
        self.processed_observations += 1
        self.last_error = None
        # The backend owns the externally visible scene version. The World
        # Model's internal chain is used for fusion state only.
        return replace(
            enriched,
            scene_version=snapshot.scene_version,
            publish_timestamp_ns=max(
                snapshot.publish_timestamp_ns,
                enriched.publish_timestamp_ns,
            ),
            # CAP-X still uses freshness_ms as a compatibility control-plane
            # field. Do not overwrite it with sensor-to-publish latency: CAP-X
            # object grounding may legitimately take seconds and is already
            # represented by processing_latency_ms.
            freshness_ms=snapshot.freshness_ms,
        )


def build_capx_world_model_enricher(
    provider: ObservationProvider,
    *,
    depth_subsample: int = 16,
    voxel_size_m: float = 0.01,
    local_radius_m: float = 1.0,
    clock: Callable[[], int] = time.time_ns,
) -> CAPXWorldModelEnricher:
    """Build the reference RGB-D World Model from an existing CAP-X provider."""
    from capmas.perception.world_model import WorldModelService

    capture_tracks = getattr(provider, "capture_object_tracks", None)
    if not callable(capture_tracks):
        raise TypeError("CAP-X provider must expose capture_object_tracks")
    artifact_store = getattr(provider, "artifacts", None)
    if artifact_store is None:
        raise TypeError("CAP-X provider must expose an artifact store")

    def measurements(observation: ObservationBundle) -> tuple[ObjectMeasurement, ...]:
        if observation.object_measurements:
            return observation.object_measurements
        tracks = capture_tracks(
            timestamp_ns=observation.timestamp_ns,
            episode_id=observation.episode_id or "unknown",
            episode_epoch=observation.episode_epoch or 0,
        )
        return tuple(
            ObjectMeasurement(
                track_id=track.track_id,
                label=track.label,
                pose_wxyz_xyz=track.pose_wxyz_xyz,
                confidence=track.confidence,
                timestamp_ns=observation.timestamp_ns,
                covariance=track.covariance,
                evidence=track.evidence,
            )
            for track in tracks
        )

    service = WorldModelService(
        geometry=ReferenceGeometryEstimator(
            artifact_store=artifact_store,
            depth_decoder=CAPXDepthDecoder(subsample=depth_subsample),
        ),
        local_map=SparseVoxelMap(
            voxel_size_m=voxel_size_m,
            local_radius_m=local_radius_m,
        ),
        tracker=KnownObjectTracker(max_match_distance_m=0.08),
        clock=clock,
        artifact_store=artifact_store,
        measurement_provider=measurements,
    )
    return CAPXWorldModelEnricher(service)
