"""Typed candidate diversity checks for physical-execution proposal waves."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from capmas.contracts.candidates import (
    CandidateEvidence,
    CandidateIdentifiability,
    GraphCandidate,
    subgraph_fingerprint,
)
from capmas.perception.effective_motion import EffectiveMotionProgram

_REQUIRED_DIFFERENCE_FIELDS = (
    "grasp_pose_or_offset",
    "approach",
    "lift",
    "transfer",
    "place_or_release",
)


@dataclass(frozen=True)
class CandidateDiversityDecision:
    """Per-candidate audit values plus a bounded regeneration directive."""

    identifiability: tuple[CandidateIdentifiability, ...]
    requires_regeneration: bool
    required_difference_fields: tuple[str, ...] = ()
    reason: str | None = None

    def __post_init__(self) -> None:
        if not self.identifiability:
            raise ValueError("candidate diversity decision requires at least one candidate")
        if self.requires_regeneration and not self.required_difference_fields:
            raise ValueError("regeneration requires explicit difference fields")
        if not self.requires_regeneration and self.required_difference_fields:
            raise ValueError("difference fields require regeneration")
        if self.reason is not None and not self.reason:
            raise ValueError("candidate diversity reason must not be empty")


class CandidateDiversityValidator:
    """Compare candidate-bound motion semantics and admissible evidence only."""

    def inspect(
        self,
        programs: Sequence[EffectiveMotionProgram],
        candidates: Sequence[GraphCandidate],
    ) -> CandidateDiversityDecision:
        proposals = tuple(candidates)
        bound_programs = tuple(programs)
        if not proposals:
            raise ValueError("candidate diversity inspection requires at least one candidate")
        if len(bound_programs) != len(proposals):
            raise ValueError("candidate diversity programs must align one-to-one with candidates")

        programs_by_fingerprint: dict[str, EffectiveMotionProgram] = {}
        for program in bound_programs:
            if program.candidate_fingerprint in programs_by_fingerprint:
                raise ValueError("effective motion programs must have unique candidate fingerprints")
            programs_by_fingerprint[program.candidate_fingerprint] = program

        candidate_fingerprints = tuple(subgraph_fingerprint(candidate.subgraph) for candidate in proposals)
        if len(set(candidate_fingerprints)) != len(candidate_fingerprints):
            raise ValueError("candidate diversity requires unique candidate fingerprints")
        if set(programs_by_fingerprint) != set(candidate_fingerprints):
            raise ValueError("effective motion programs do not match candidate fingerprints")

        ordered_programs = tuple(programs_by_fingerprint[fingerprint] for fingerprint in candidate_fingerprints)
        semantic_counts = Counter(program.semantic_signature for program in ordered_programs)
        evidence_signatures = tuple(_evidence_signature(candidate.evidence) for candidate in proposals)
        evidence_counts = Counter(evidence_signatures)
        all_semantically_equivalent = len(semantic_counts) == 1 and len(proposals) > 1
        identifiability = tuple(
            _identifiability(
                candidate,
                program,
                semantic_equivalent=semantic_counts[program.semantic_signature] > 1,
                evidence_identical=evidence_counts[evidence_signature] > 1,
                has_peer=len(proposals) > 1,
            )
            for candidate, program, evidence_signature in zip(
                proposals,
                ordered_programs,
                evidence_signatures,
                strict=True,
            )
        )
        if all_semantically_equivalent:
            return CandidateDiversityDecision(
                identifiability=identifiability,
                requires_regeneration=True,
                required_difference_fields=_REQUIRED_DIFFERENCE_FIELDS,
                reason="all candidate programs have equivalent motion semantics",
            )
        return CandidateDiversityDecision(
            identifiability=identifiability,
            requires_regeneration=False,
            reason=(
                "candidate evidence does not distinguish all motion programs"
                if any(item.candidate_evidence_identical for item in identifiability)
                else None
            ),
        )


def _identifiability(
    candidate: GraphCandidate,
    program: EffectiveMotionProgram,
    *,
    semantic_equivalent: bool,
    evidence_identical: bool,
    has_peer: bool,
) -> CandidateIdentifiability:
    selection_identifiable = not semantic_equivalent and (not evidence_identical or not has_peer)
    if semantic_equivalent:
        reason = "candidate motion program is semantically equivalent to a peer"
    elif evidence_identical:
        reason = "candidate evidence is identical to a semantically distinct peer"
    else:
        reason = None
    return CandidateIdentifiability(
        candidate_id=candidate.candidate_id,
        semantic_signature=program.semantic_signature,
        program_fingerprint=program.program_fingerprint,
        candidate_semantic_equivalent=semantic_equivalent,
        candidate_evidence_identical=evidence_identical,
        selection_identifiable=selection_identifiable,
        abstention_reason=reason,
    )


def _evidence_signature(evidence: CandidateEvidence | None) -> str:
    """Hash only selection-relevant values, excluding identity and provenance."""

    if evidence is None:
        return _stable_hash({"available_metrics": ()})
    available = tuple(sorted(set(evidence.available_metrics)))
    payload: dict[str, object] = {"available_metrics": available}
    if "verifier" in available:
        payload["verifier_pass_rate"] = evidence.verifier_pass_rate
    if "rehearsal" in available:
        payload["rehearsal_success_rate"] = evidence.rehearsal_success_rate
    if "ood" in available:
        payload["ood_success_rate"] = evidence.ood_success_rate
    if "latency" in available:
        payload["expected_latency_ms"] = evidence.expected_latency_ms
    if "recovery" in available:
        payload["recovery_cost"] = evidence.recovery_cost
    if "perception" in available:
        payload["perception"] = (
            None
            if evidence.perception is None
            else {
                name: getattr(evidence.perception, name)
                for name in evidence.perception.available
            }
        )
    if "geometry" in available:
        payload["geometry"] = (
            None
            if evidence.geometry is None
            else {
                dimension.name: {
                    "status": dimension.status,
                    "score": dimension.score,
                }
                for dimension in (
                    evidence.geometry.grasp_quality,
                    evidence.geometry.reachability,
                    evidence.geometry.clearance,
                    evidence.geometry.collision_risk,
                )
            }
        )
    return _stable_hash(payload)


def _stable_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = ["CandidateDiversityDecision", "CandidateDiversityValidator"]
