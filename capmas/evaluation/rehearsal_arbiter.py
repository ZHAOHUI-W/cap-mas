"""Optional attachment of rehearsal results to Arbiter candidate evidence."""

from __future__ import annotations

from dataclasses import replace

from capmas.contracts.candidates import CandidateEvidence, GraphCandidate, subgraph_fingerprint
from capmas.evaluation.rehearsal_evidence import RehearsalEvidence


def merge_rehearsal_evidence(
    candidate: GraphCandidate,
    rehearsal: RehearsalEvidence,
    *,
    include_in_arbiter: bool,
) -> GraphCandidate:
    """Attach one result only when it identifies this exact candidate and scene.

    Shadow mode intentionally returns the original candidate. The caller can
    persist the returned rehearsal record while keeping the physical selection
    baseline unchanged.
    """

    if rehearsal.candidate_id != candidate.candidate_id:
        raise ValueError("rehearsal candidate id does not match GraphCandidate")
    if rehearsal.fingerprint_scope == "graph":
        if rehearsal.arbiter_subgraph_id != candidate.subgraph.subgraph_id:
            raise ValueError("rehearsal target subgraph does not match GraphCandidate")
        effective_fingerprint = rehearsal.arbiter_fingerprint
    else:
        effective_fingerprint = rehearsal.candidate_fingerprint
    if effective_fingerprint != subgraph_fingerprint(candidate.subgraph):
        raise ValueError("rehearsal candidate fingerprint does not match GraphCandidate")
    if rehearsal.scene_version != candidate.parent_scene_version:
        raise ValueError("rehearsal scene version does not match GraphCandidate")
    if not include_in_arbiter:
        return candidate

    current = candidate.evidence or CandidateEvidence()
    available = tuple(dict.fromkeys((*current.available_metrics, "rehearsal")))
    refs = tuple(current.evidence_refs) + (
        f"rehearsal://{rehearsal.candidate_id}/{rehearsal.seed}",
    )
    evidence = replace(
        current,
        rehearsal_success_rate=rehearsal.score,
        evidence_refs=refs,
        available_metrics=available,
        scene_version=rehearsal.scene_version,
        provider="libero_process_rehearsal",
    )
    return replace(candidate, evidence=evidence)
