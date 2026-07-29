"""Typed strategy profiles used by local Policy Agents and the Arbiter."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyProfile:
    """Executable trade-offs for one Policy specialization."""

    name: str
    min_scene_freshness: float = 0.0
    min_scene_confidence: float = 0.0
    min_target_visibility: float = 0.0
    min_track_confidence: float = 0.0
    min_identity_confidence: float = 0.0
    min_pose_reliability: float = 0.0
    confidence_weight: float = 0.25
    perception_weight: float = 0.0
    geometry_weight: float = 0.0
    min_reachability: float = 0.0
    max_collision_risk: float = 1.0
    verifier_weight: float = 0.20
    rehearsal_weight: float = 0.25
    ood_weight: float = 0.25
    latency_penalty: float = 0.03
    recovery_penalty: float = 0.02

    def __post_init__(self) -> None:
        if self.name not in {"balanced", "safety", "robust", "efficient"}:
            raise ValueError(f"unsupported strategy profile: {self.name}")
        for field_name in (
            "min_scene_freshness",
            "min_scene_confidence",
            "min_target_visibility",
            "min_track_confidence",
            "min_identity_confidence",
            "min_pose_reliability",
            "confidence_weight",
            "perception_weight",
            "geometry_weight",
            "min_reachability",
            "max_collision_risk",
            "verifier_weight",
            "rehearsal_weight",
            "ood_weight",
        ):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1]")
        if self.latency_penalty < 0 or self.recovery_penalty < 0:
            raise ValueError("strategy penalties must not be negative")

    @classmethod
    def for_name(cls, name: str | None) -> "StrategyProfile":
        """Return a stable profile while keeping P3.1 names compatible."""
        normalized = (name or "balanced").split(":", 1)[-1]
        profiles = {
            "balanced": cls(
                "balanced", perception_weight=0.25, geometry_weight=0.40,
                min_reachability=0.50, max_collision_risk=0.50,
            ),
            "safety": cls(
                "safety",
                min_scene_freshness=0.75,
                min_scene_confidence=0.70,
                min_target_visibility=0.80,
                min_track_confidence=0.70,
                min_identity_confidence=0.70,
                min_pose_reliability=0.70,
                confidence_weight=0.10,
                perception_weight=0.55,
                geometry_weight=0.50,
                min_reachability=0.70,
                max_collision_risk=0.35,
                verifier_weight=0.20,
                rehearsal_weight=0.05,
                ood_weight=0.10,
                latency_penalty=0.01,
                recovery_penalty=0.04,
            ),
            "robust": cls(
                "robust",
                min_scene_confidence=0.60,
                min_target_visibility=0.70,
                min_track_confidence=0.60,
                min_identity_confidence=0.60,
                min_pose_reliability=0.60,
                confidence_weight=0.15,
                perception_weight=0.30,
                geometry_weight=0.45,
                min_reachability=0.60,
                max_collision_risk=0.45,
                verifier_weight=0.20,
                rehearsal_weight=0.20,
                ood_weight=0.30,
                latency_penalty=0.02,
                recovery_penalty=0.03,
            ),
            "efficient": cls(
                "efficient",
                min_scene_confidence=0.50,
                min_target_visibility=0.60,
                min_track_confidence=0.50,
                min_identity_confidence=0.50,
                min_pose_reliability=0.50,
                confidence_weight=0.35,
                perception_weight=0.15,
                geometry_weight=0.25,
                min_reachability=0.50,
                max_collision_risk=0.65,
                verifier_weight=0.20,
                rehearsal_weight=0.15,
                ood_weight=0.15,
                latency_penalty=0.08,
                recovery_penalty=0.01,
            ),
        }
        try:
            return profiles[normalized]
        except KeyError as exc:
            raise ValueError(f"unsupported strategy profile: {name}") from exc


__all__ = ["StrategyProfile"]
