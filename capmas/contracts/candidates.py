from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import TYPE_CHECKING, Literal, Sequence

from capmas.contracts.core import ArtifactRef
from capmas.contracts.graph import SubgraphSpec
from capmas.graph.serialization import local_subgraph_to_dict

if TYPE_CHECKING:
    from capmas.verification.evidence import VerifierEvidence


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
class EvidenceDimension:
    """A measurable geometry dimension with explicit three-state semantics."""

    name: str
    status: Literal["pass", "fail", "unknown"]
    score: float | None
    threshold: float | None
    reason: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("evidence dimension name must not be empty")
        if self.status not in {"pass", "fail", "unknown"}:
            raise ValueError("evidence dimension status must be pass, fail, or unknown")
        if self.status == "unknown" and self.score is not None:
            raise ValueError("unknown dimension score must be None")
        if self.status in {"pass", "fail"} and self.score is None:
            raise ValueError("pass or fail dimension score must be available")
        if self.score is not None and not 0.0 <= self.score <= 1.0:
            raise ValueError("evidence dimension score must be in [0, 1]")
        if self.threshold is not None and not 0.0 <= self.threshold <= 1.0:
            raise ValueError("evidence dimension threshold must be in [0, 1]")


@dataclass(frozen=True)
class GeometryEvidence:
    """Candidate-conditioned geometry evidence and its reproducibility context."""

    grasp_quality: EvidenceDimension
    reachability: EvidenceDimension
    clearance: EvidenceDimension
    collision_risk: EvidenceDimension
    candidate_fingerprint: str
    scene_version: int
    map_version: int | None
    map_backend: str
    provider: str
    provider_version: str
    captured_at_ns: int
    latency_ms: float
    used_privileged_state: bool = False
    artifact_refs: tuple[ArtifactRef, ...] = ()
    execution_graph_fingerprint: str | None = None
    program_fingerprint: str | None = None
    program_scope: Literal["subgraph", "mission_suffix"] = "subgraph"
    segment_artifact_refs: tuple[ArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        if not self.candidate_fingerprint:
            raise ValueError("geometry candidate fingerprint must not be empty")
        if self.scene_version < 0:
            raise ValueError("geometry scene version must not be negative")
        if self.map_version is not None and self.map_version < 0:
            raise ValueError("geometry map version must not be negative")
        if not self.map_backend:
            raise ValueError("geometry map backend must not be empty")
        if not self.provider or not self.provider_version:
            raise ValueError("geometry provider and version must not be empty")
        if self.captured_at_ns < 0:
            raise ValueError("geometry capture timestamp must not be negative")
        if self.latency_ms < 0:
            raise ValueError("geometry latency must not be negative")
        if self.execution_graph_fingerprint is not None and not self.execution_graph_fingerprint:
            raise ValueError("geometry execution graph fingerprint must not be empty")
        if self.program_scope not in {"subgraph", "mission_suffix"}:
            raise ValueError("geometry program scope must be subgraph or mission_suffix")
        if self.program_scope == "mission_suffix" and not self.program_fingerprint:
            raise ValueError("mission-suffix geometry program fingerprint is required")
        if self.program_fingerprint is not None and not self.program_fingerprint:
            raise ValueError("geometry program fingerprint must not be empty")
        expected = {
            "grasp_quality",
            "reachability",
            "clearance",
            "collision_risk",
        }
        dimensions = (
            self.grasp_quality,
            self.reachability,
            self.clearance,
            self.collision_risk,
        )
        if {dimension.name for dimension in dimensions} != expected:
            raise ValueError("geometry evidence dimensions have unexpected names")

    @property
    def measurable(self) -> bool:
        return any(
            dimension.status in {"pass", "fail"}
            for dimension in (
                self.grasp_quality,
                self.reachability,
                self.clearance,
                self.collision_risk,
            )
        )


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
    geometry: GeometryEvidence | None = None
    available_metrics: Sequence[str] = ()
    scene_version: int | None = None
    provider: str | None = None
    captured_at_ns: int | None = None
    verifier: "VerifierEvidence | None" = None

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
        allowed = {
            "verifier",
            "rehearsal",
            "ood",
            "latency",
            "recovery",
            "perception",
            "geometry",
        }
        unknown = set(self.available_metrics) - allowed
        if unknown:
            raise ValueError(f"unknown evidence metrics: {', '.join(sorted(unknown))}")
        if "geometry" in self.available_metrics and self.geometry is None:
            raise ValueError("geometry evidence metric requires geometry evidence")
        if self.verifier is not None:
            if self.verifier_pass_rate != self.verifier.pass_rate:
                raise ValueError("verifier_pass_rate must match typed verifier evidence")
            if self.scene_version is not None and self.scene_version != self.verifier.scene_version:
                raise ValueError("evidence scene version must match typed verifier evidence")
            # The outer provider/timestamp identify the aggregate evidence
            # envelope. Each typed evidence lane retains its own provenance.
            if "verifier" in self.available_metrics and self.verifier.coverage <= 0.0:
                raise ValueError("typed verifier metric requires positive coverage")


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
    "EvidenceDimension",
    "GeometryEvidence",
    "GraphCandidate",
    "PerceptionEvidence",
    "rewrite_report_for",
    "subgraph_fingerprint",
]
