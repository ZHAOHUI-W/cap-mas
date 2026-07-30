"""Typed, provenance-aware evidence produced by observable predicates."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

from capmas.contracts.verification import PredicateReport


VerifierPhase = Literal["static", "dynamic"]
VerifierStatus = Literal["pass", "fail", "unknown"]

_UNAVAILABLE_REASON_TOKENS = (
    "unknown",
    "not found",
    "unavailable",
    "not observed",
    "not available",
)


@dataclass(frozen=True)
class VerifierPredicateEvidence:
    """One observable predicate result with explicit three-state semantics."""

    predicate: str
    phase: VerifierPhase
    status: VerifierStatus
    confidence: float | None
    reason: str | None
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.predicate:
            raise ValueError("verifier predicate must not be empty")
        if self.phase not in {"static", "dynamic"}:
            raise ValueError("verifier evidence phase must be static or dynamic")
        if self.status not in {"pass", "fail", "unknown"}:
            raise ValueError("verifier evidence status must be pass, fail, or unknown")
        if self.confidence is not None and (
            not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("verifier evidence confidence must be in [0, 1]")
        if self.status in {"pass", "fail"} and self.confidence is None:
            raise ValueError("pass or fail verifier evidence requires confidence")
        if any(not isinstance(ref, str) for ref in self.evidence_refs):
            raise ValueError("verifier evidence references must be strings")

    def to_dict(self) -> dict[str, object]:
        return {
            "predicate": self.predicate,
            "phase": self.phase,
            "status": self.status,
            "confidence": self.confidence,
            "reason": self.reason,
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(frozen=True)
class VerifierEvidence:
    """A candidate- and scene-bound collection of predicate evidence."""

    candidate_fingerprint: str
    scene_version: int
    pass_rate: float
    coverage: float
    provider: str
    captured_at_ns: int
    static_results: tuple[VerifierPredicateEvidence, ...] = ()
    dynamic_results: tuple[VerifierPredicateEvidence, ...] = ()
    source_verification: str | None = None

    def __post_init__(self) -> None:
        if not self.candidate_fingerprint:
            raise ValueError("verifier candidate fingerprint must not be empty")
        if self.scene_version < 0:
            raise ValueError("verifier scene version must not be negative")
        if self.captured_at_ns < 0:
            raise ValueError("verifier capture timestamp must not be negative")
        if not self.provider:
            raise ValueError("verifier provider must not be empty")
        for name in ("pass_rate", "coverage"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"verifier {name} must be in [0, 1]")
        _validate_results(self.static_results, "static")
        _validate_results(self.dynamic_results, "dynamic")
        expected = summarize_verifier_results((*self.static_results, *self.dynamic_results))
        if (self.pass_rate, self.coverage) != expected:
            raise ValueError(
                "verifier evidence summary does not match predicate results: "
                f"expected {expected}, got {(self.pass_rate, self.coverage)}"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_fingerprint": self.candidate_fingerprint,
            "scene_version": self.scene_version,
            "pass_rate": self.pass_rate,
            "coverage": self.coverage,
            "provider": self.provider,
            "captured_at_ns": self.captured_at_ns,
            "static_results": [item.to_dict() for item in self.static_results],
            "dynamic_results": [item.to_dict() for item in self.dynamic_results],
            "source_verification": self.source_verification,
        }


def summarize_verifier_results(
    results: tuple[VerifierPredicateEvidence, ...] | list[VerifierPredicateEvidence],
) -> tuple[float, float]:
    """Return pass rate over determined results and deterministic coverage."""

    total = len(results)
    determined = sum(item.status in {"pass", "fail"} for item in results)
    passed = sum(item.status == "pass" for item in results)
    return (
        passed / determined if determined else 0.0,
        determined / total if total else 0.0,
    )


def predicate_report_to_evidence(
    report: PredicateReport,
    *,
    phase: VerifierPhase,
) -> VerifierPredicateEvidence:
    """Convert an observable verifier report to conservative typed evidence."""

    if report.passed:
        status: VerifierStatus = "pass"
        confidence = report.confidence
    elif _is_unavailable_reason(report.reason):
        status = "unknown"
        confidence = None
    else:
        status = "fail"
        confidence = report.confidence
    return VerifierPredicateEvidence(
        predicate=report.name,
        phase=phase,
        status=status,
        confidence=confidence,
        reason=report.reason,
        evidence_refs=tuple(report.evidence),
    )


def _validate_results(
    results: tuple[VerifierPredicateEvidence, ...],
    expected_phase: VerifierPhase,
) -> None:
    names: set[str] = set()
    for item in results:
        if item.phase != expected_phase:
            raise ValueError(
                f"{expected_phase} verifier results must have phase {expected_phase}"
            )
        if item.predicate in names:
            raise ValueError(f"duplicate verifier predicate in {expected_phase} results")
        names.add(item.predicate)


def _is_unavailable_reason(reason: str | None) -> bool:
    if reason is None:
        return False
    normalized = reason.lower()
    return any(token in normalized for token in _UNAVAILABLE_REASON_TOKENS)


__all__ = [
    "VerifierEvidence",
    "VerifierPhase",
    "VerifierPredicateEvidence",
    "VerifierStatus",
    "predicate_report_to_evidence",
    "summarize_verifier_results",
]
