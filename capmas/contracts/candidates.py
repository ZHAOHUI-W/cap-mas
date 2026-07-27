from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Sequence

from capmas.contracts.graph import SubgraphSpec
from capmas.graph.serialization import local_subgraph_to_dict


@dataclass(frozen=True)
class PerceptionEvidence:
    """Scene-backed quality dimensions for one candidate's target geometry."""

    scene_freshness: float | None = None
    scene_confidence: float | None = None
    target_visibility: float | None = None
    track_confidence: float | None = None
    identity_confidence: float | None = None
    pose_reliability: float | None = None
    evidence_refs: Sequence[str] = ()

    def __post_init__(self) -> None:
        for name in (
            "scene_freshness",
            "scene_confidence",
            "target_visibility",
            "track_confidence",
            "identity_confidence",
            "pose_reliability",
        ):
            value = getattr(self, name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1] when available")

    @property
    def available(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in (
                "scene_freshness",
                "scene_confidence",
                "target_visibility",
                "track_confidence",
                "identity_confidence",
                "pose_reliability",
            )
            if getattr(self, name) is not None
        )

    def score(self) -> float:
        values = [getattr(self, name) for name in self.available]
        return sum(values) / len(values) if values else 0.0


@dataclass(frozen=True)
class CandidateRewriteReport:
    """Auditable before/after metadata for scene grounding and repair."""

    raw_fingerprint: str
    normalized_fingerprint: str
    changed: bool = False
    rewrite_count: int = 0
    operations: Sequence[str] = ()


def subgraph_fingerprint(subgraph: SubgraphSpec) -> str:
    """Return a stable fingerprint for typed candidate comparison."""
    encoded = json.dumps(
        local_subgraph_to_dict(subgraph),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def rewrite_report_for(raw: SubgraphSpec, normalized: SubgraphSpec) -> CandidateRewriteReport:
    raw_fingerprint = subgraph_fingerprint(raw)
    normalized_fingerprint = subgraph_fingerprint(normalized)
    changed = raw_fingerprint != normalized_fingerprint
    return CandidateRewriteReport(
        raw_fingerprint=raw_fingerprint,
        normalized_fingerprint=normalized_fingerprint,
        changed=changed,
        rewrite_count=1 if changed else 0,
        operations=("scene_or_safety_normalization",) if changed else (),
    )


@dataclass(frozen=True)
class GraphCandidate:
    """A bounded local graph proposal exchanged between Policy Agents and Arbiter."""

    candidate_id: str
    subgraph: SubgraphSpec
    parent_scene_version: int
    producer_agent: str
    confidence: float | None = None
    rationale: str = ""
    evidence: "CandidateEvidence" | None = None
    strategy: str = "balanced"
    raw_subgraph: SubgraphSpec | None = None
    rewrite_report: CandidateRewriteReport = field(
        default_factory=lambda: CandidateRewriteReport("", "")
    )

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate id must not be empty")
        if not self.producer_agent:
            raise ValueError("candidate producer must not be empty")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("candidate confidence must be in [0, 1]")
        from capmas.contracts.strategy import StrategyProfile

        StrategyProfile.for_name(self.strategy)


@dataclass(frozen=True)
class CandidateEvidence:
    """Offline and online evidence used to rank a valid candidate."""

    verifier_pass_rate: float = 0.0
    rehearsal_success_rate: float = 0.0
    ood_success_rate: float = 0.0
    expected_latency_ms: float = 0.0
    recovery_cost: float = 0.0
    evidence_refs: Sequence[str] = ()
    perception: PerceptionEvidence | None = None
    available_metrics: Sequence[str] = ()
    scene_version: int | None = None
    provider: str | None = None
    captured_at_ns: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "verifier_pass_rate",
            "rehearsal_success_rate",
            "ood_success_rate",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.expected_latency_ms < 0 or self.recovery_cost < 0:
            raise ValueError("latency and recovery cost must not be negative")
        if self.scene_version is not None and self.scene_version < 0:
            raise ValueError("evidence scene version must not be negative")
        if self.captured_at_ns is not None and self.captured_at_ns < 0:
            raise ValueError("evidence capture timestamp must not be negative")
        allowed = {"verifier", "rehearsal", "ood", "latency", "recovery", "perception"}
        unknown = set(self.available_metrics) - allowed
        if unknown:
            raise ValueError(f"unknown evidence metrics: {', '.join(sorted(unknown))}")


@dataclass(frozen=True)
class CandidateRejection:
    candidate_id: str
    code: str
    reason: str


@dataclass(frozen=True)
class ArbitrationResult:
    selected: GraphCandidate | None
    considered: tuple[GraphCandidate, ...] = ()
    rejections: tuple[CandidateRejection, ...] = ()
    selection_basis: str = "none"
    tie_broken: bool = False
    score_breakdowns: dict[str, dict[str, float]] = field(default_factory=dict)


__all__ = [
    "ArbitrationResult",
    "CandidateEvidence",
    "CandidateRejection",
    "CandidateRewriteReport",
    "GraphCandidate",
    "PerceptionEvidence",
    "rewrite_report_for",
    "subgraph_fingerprint",
]
