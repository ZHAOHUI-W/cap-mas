"""Pure, non-physical comparison of baseline and evidence-enriched arbitration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from capmas.agents.arbiter import CandidateArbiter
from capmas.contracts.candidates import ArbitrationResult, GraphCandidate
from capmas.contracts.scene import SceneSnapshot
from capmas.evaluation.rehearsal_arbiter import merge_rehearsal_evidence
from capmas.evaluation.rehearsal_evidence import RehearsalEvidence


@dataclass(frozen=True)
class ShadowArbitrationReport:
    """The live decision and a hypothetical decision using rehearsal evidence.

    ``shadow`` is diagnostic only.  In particular, a changed shadow winner does
    not grant it execution ownership and does not mutate the candidates passed
    to :func:`run_shadow_arbitration`.
    """

    baseline: ArbitrationResult
    shadow: ArbitrationResult
    baseline_winner: str | None
    shadow_winner: str | None
    would_change_selection: bool
    physical_execution_required: bool = False
    evidence_rejections: tuple[str, ...] = ()


def run_shadow_arbitration(
    candidates: Sequence[GraphCandidate],
    rehearsals: Mapping[str, RehearsalEvidence],
    scene: SceneSnapshot,
    arbiter: CandidateArbiter | None = None,
) -> ShadowArbitrationReport:
    """Compare baseline selection with a hypothetical evidence-aware selection.

    The mapping is keyed by the exact ``GraphCandidate.candidate_id``.  Each
    rehearsal is attached through the normal identity and scene-version gate,
    so stale or mismatched evidence is rejected rather than interpreted as a
    zero-quality result.  Missing evidence leaves that candidate unchanged.

    This function deliberately has no dependency on a backend, ``ActionLease``,
    or executor.  It may therefore be called alongside live arbitration without
    causing a second physical action.
    """

    arbiter = arbiter or CandidateArbiter()
    live_candidates = tuple(candidates)
    baseline = arbiter.select(live_candidates, scene)

    shadow_candidates: list[GraphCandidate] = []
    evidence_rejections: list[str] = []
    for candidate in live_candidates:
        rehearsal = rehearsals.get(candidate.candidate_id)
        if rehearsal is None:
            shadow_candidates.append(candidate)
            continue
        try:
            shadow_candidates.append(
                merge_rehearsal_evidence(
                    candidate,
                    rehearsal,
                    include_in_arbiter=True,
                )
            )
        except ValueError as exc:
            evidence_rejections.append(f"{candidate.candidate_id}: {exc}")
            shadow_candidates.append(candidate)

    shadow = arbiter.select(tuple(shadow_candidates), scene)
    baseline_winner = baseline.selected.candidate_id if baseline.selected else None
    shadow_winner = shadow.selected.candidate_id if shadow.selected else None
    return ShadowArbitrationReport(
        baseline=baseline,
        shadow=shadow,
        baseline_winner=baseline_winner,
        shadow_winner=shadow_winner,
        would_change_selection=baseline_winner != shadow_winner,
        physical_execution_required=False,
        evidence_rejections=tuple(evidence_rejections),
    )


__all__ = ["ShadowArbitrationReport", "run_shadow_arbitration"]
