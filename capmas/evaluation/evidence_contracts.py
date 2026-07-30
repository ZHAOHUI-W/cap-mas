"""Version and candidate identity checks shared by Phase 5 evidence lanes."""

from __future__ import annotations

from dataclasses import dataclass


class EvidenceCompatibilityError(ValueError):
    """Raised when evidence cannot be used for the current arbitration request."""


@dataclass(frozen=True)
class EvidenceRequestContext:
    candidate_fingerprint: str
    scene_version: int
    map_version: int | None = None

    def __post_init__(self) -> None:
        if not self.candidate_fingerprint:
            raise ValueError("candidate fingerprint must not be empty")
        if self.scene_version < 0:
            raise ValueError("scene version must not be negative")
        if self.map_version is not None and self.map_version < 0:
            raise ValueError("map version must not be negative")


def assert_evidence_compatible(
    context: EvidenceRequestContext,
    *,
    candidate_fingerprint: str,
    scene_version: int,
    map_version: int | None = None,
) -> None:
    """Reject stale or cross-candidate evidence before it reaches Arbiter."""

    if candidate_fingerprint != context.candidate_fingerprint:
        raise EvidenceCompatibilityError(
            "evidence candidate fingerprint does not match request fingerprint"
        )
    if scene_version != context.scene_version:
        raise EvidenceCompatibilityError(
            f"evidence scene version {scene_version} does not match request "
            f"scene version {context.scene_version}"
        )
    if context.map_version is None:
        return
    if map_version != context.map_version:
        raise EvidenceCompatibilityError(
            f"evidence map version {map_version} does not match request "
            f"map version {context.map_version}"
        )
