from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from capmas.contracts.scene import ObjectTrack, SceneSnapshot, VisualEvidence
from capmas.perception.protocol import (
    FusedPerceptionBackend,
    ObservationBundle,
    PerceptionRequest,
    PerceptionResult,
)


@dataclass
class PerceptionFacade:
    backend: FusedPerceptionBackend

    def infer(self, request: PerceptionRequest, observation: ObservationBundle) -> PerceptionResult:
        return self.backend.infer(request, observation)

    def publish_scene(
        self,
        request: PerceptionRequest,
        observation: ObservationBundle,
        previous: SceneSnapshot | None,
    ) -> SceneSnapshot:
        result = self.infer(request, observation)
        return self.backend.publish_scene(observation, result, previous)


def tracks_from_result(result: PerceptionResult, scene_version: int, episode_id: str, episode_epoch: int) -> tuple[ObjectTrack, ...]:
    tracks: list[ObjectTrack] = []
    for index, pose in enumerate(result.poses_3d):
        track_id = pose.track_id or f"{pose.label}-{index}"
        visual_evidence = tuple(
            VisualEvidence(
                artifact=artifact,
                evidence_type="pose_support",
                captured_at_ns=result.timestamp_ns,
                track_id=track_id,
            )
            for artifact in pose.evidence
        )
        visual_evidence += tuple(
            evidence
            for evidence in result.visual_evidence
            if evidence.track_id in (None, track_id)
        )
        tracks.append(
            ObjectTrack(
            track_id=track_id,
            label=pose.label,
            pose_wxyz_xyz=pose.pose_wxyz_xyz,
            confidence=pose.confidence,
            last_seen_ns=result.timestamp_ns,
            covariance=pose.covariance,
            evidence=pose.evidence,
            visual_evidence=visual_evidence,
        )
        )
    return tuple(tracks)
