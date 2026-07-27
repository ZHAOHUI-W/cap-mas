from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from capmas.contracts.core import ArtifactRef
from capmas.contracts.scene import ObjectTrack


@dataclass(frozen=True)
class ObjectMeasurement:
    track_id: str | None
    label: str
    pose_wxyz_xyz: tuple[float, ...]
    confidence: float
    timestamp_ns: int
    covariance: ArtifactRef | None = None
    evidence: tuple[ArtifactRef, ...] = ()


@dataclass
class _TrackState:
    track_id: str
    label: str
    pose_wxyz_xyz: tuple[float, ...]
    confidence: float
    last_seen_ns: int
    covariance: ArtifactRef | None
    evidence: tuple[ArtifactRef, ...]
    velocity_xyz: tuple[float, float, float] | None = None


class KnownObjectTracker:
    def __init__(
        self,
        *,
        max_match_distance_m: float = 0.08,
        prediction_timeout_ms: int = 250,
        stale_timeout_ms: int = 500,
        confidence_decay: float = 0.8,
    ) -> None:
        if max_match_distance_m <= 0.0:
            raise ValueError("max_match_distance_m must be positive")
        if prediction_timeout_ms < 0 or stale_timeout_ms < 0:
            raise ValueError("prediction and stale timeouts must be non-negative")
        if not 0.0 < confidence_decay <= 1.0:
            raise ValueError("confidence_decay must be in (0, 1]")
        self.max_match_distance_m = max_match_distance_m
        self.prediction_timeout_ms = prediction_timeout_ms
        self.stale_timeout_ms = stale_timeout_ms
        self.confidence_decay = confidence_decay
        self._tracks: dict[str, _TrackState] = {}
        self._next_index: dict[str, int] = {}

    def update(self, measurements: Sequence[ObjectMeasurement]) -> tuple[ObjectTrack, ...]:
        used: set[str] = set()
        for measurement in measurements:
            _validate_measurement(measurement)
            track_id = measurement.track_id
            if track_id is None:
                track_id = self._associate(measurement, used)
            if track_id is None:
                track_id = self._new_track_id(measurement.label)
            state = self._tracks.get(track_id)
            velocity = None
            if state is not None:
                old_position = _position(state.pose_wxyz_xyz)
                new_position = _position(measurement.pose_wxyz_xyz)
                elapsed_ns = measurement.timestamp_ns - state.last_seen_ns
                if elapsed_ns > 0:
                    seconds = elapsed_ns / 1_000_000_000
                    velocity = tuple(
                        (new_position[index] - old_position[index]) / seconds
                        for index in range(3)
                    )
            self._tracks[track_id] = _TrackState(
                track_id=track_id,
                label=measurement.label,
                pose_wxyz_xyz=measurement.pose_wxyz_xyz,
                confidence=max(0.0, min(1.0, measurement.confidence)),
                last_seen_ns=measurement.timestamp_ns,
                covariance=measurement.covariance,
                evidence=measurement.evidence,
                velocity_xyz=velocity,
            )
            used.add(track_id)
        return tuple(self._to_track(state, "observed") for state in self._sorted_states())

    def predict(self, timestamp_ns: int) -> tuple[ObjectTrack, ...]:
        result: list[ObjectTrack] = []
        for state in self._sorted_states():
            elapsed_ns = max(0, timestamp_ns - state.last_seen_ns)
            elapsed_ms = elapsed_ns / 1_000_000
            if elapsed_ms <= self.prediction_timeout_ms:
                status = "predicted"
                position = _predicted_position(state, elapsed_ns)
            elif elapsed_ms <= self.prediction_timeout_ms + self.stale_timeout_ms:
                status = "stale"
                position = _predicted_position(state, elapsed_ns)
            else:
                status = "lost"
                position = _position(state.pose_wxyz_xyz)
            decay_steps = max(1, math.ceil(elapsed_ms / max(1, self.prediction_timeout_ms or 1)))
            confidence = state.confidence * (self.confidence_decay**decay_steps)
            pose = tuple(state.pose_wxyz_xyz[:4]) + tuple(position)
            result.append(
                ObjectTrack(
                    track_id=state.track_id,
                    label=state.label,
                    pose_wxyz_xyz=pose,
                    confidence=confidence,
                    last_seen_ns=state.last_seen_ns,
                    covariance=state.covariance,
                    evidence=state.evidence,
                    velocity_xyz=state.velocity_xyz,
                    prediction_timestamp_ns=timestamp_ns,
                    track_status=status,
                )
            )
        return tuple(result)

    def _associate(self, measurement: ObjectMeasurement, used: set[str]) -> str | None:
        position = _position(measurement.pose_wxyz_xyz)
        candidates = [
            state
            for state in self._tracks.values()
            if state.track_id not in used and state.label == measurement.label
        ]
        candidates.sort(key=lambda state: _distance(_position(state.pose_wxyz_xyz), position))
        if candidates and _distance(_position(candidates[0].pose_wxyz_xyz), position) <= self.max_match_distance_m:
            return candidates[0].track_id
        return None

    def _new_track_id(self, label: str) -> str:
        index = self._next_index.get(label, 0)
        while f"{label}-{index}" in self._tracks:
            index += 1
        self._next_index[label] = index + 1
        return f"{label}-{index}"

    def _sorted_states(self) -> list[_TrackState]:
        return [self._tracks[key] for key in sorted(self._tracks)]

    @staticmethod
    def _to_track(state: _TrackState, status: str) -> ObjectTrack:
        return ObjectTrack(
            track_id=state.track_id,
            label=state.label,
            pose_wxyz_xyz=state.pose_wxyz_xyz,
            confidence=state.confidence,
            last_seen_ns=state.last_seen_ns,
            covariance=state.covariance,
            evidence=state.evidence,
            velocity_xyz=state.velocity_xyz,
            track_status=status,
        )


def _validate_measurement(measurement: ObjectMeasurement) -> None:
    if len(measurement.pose_wxyz_xyz) != 7:
        raise ValueError("object pose must contain quaternion and position")
    if not all(math.isfinite(value) for value in measurement.pose_wxyz_xyz):
        raise ValueError("object pose contains a non-finite value")
    if not measurement.label:
        raise ValueError("object label must not be empty")


def _position(pose: tuple[float, ...]) -> tuple[float, float, float]:
    return (pose[4], pose[5], pose[6])


def _predicted_position(state: _TrackState, elapsed_ns: int) -> tuple[float, float, float]:
    position = _position(state.pose_wxyz_xyz)
    if state.velocity_xyz is None:
        return position
    seconds = elapsed_ns / 1_000_000_000
    return tuple(position[index] + state.velocity_xyz[index] * seconds for index in range(3))


def _distance(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    return math.sqrt(sum((first[index] - second[index]) ** 2 for index in range(3)))
