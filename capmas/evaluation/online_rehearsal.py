"""Online, bounded attachment of process-rehearsal evidence to arbitration."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import time
from typing import Literal

from capmas.agents.arbiter import CandidateArbiter
from capmas.contracts.candidates import ArbitrationResult, GraphCandidate
from capmas.contracts.scene import SceneSnapshot
from capmas.evaluation.rehearsal_arbiter import merge_rehearsal_evidence
from capmas.evaluation.rehearsal_evidence import RehearsalEvidence


RehearsalMode = Literal["disabled", "shadow", "online_bounded"]
RehearsalEvidenceProvider = Callable[
    [Sequence[GraphCandidate], SceneSnapshot],
    Mapping[str, RehearsalEvidence],
]


@dataclass(frozen=True)
class RehearsalArbitrationReport:
    """Baseline, hypothetical evidence-aware, and executable decisions."""

    mode: RehearsalMode
    baseline: ArbitrationResult
    evidence_aware: ArbitrationResult | None
    live: ArbitrationResult
    attached_candidate_ids: tuple[str, ...] = ()
    evidence_rejections: tuple[str, ...] = ()
    provider_latency_ms: float = 0.0
    fallback_reason: str | None = None

    @property
    def would_change_selection(self) -> bool:
        """Whether accepted rehearsal evidence changes the winner."""

        if self.evidence_aware is None:
            return False
        return _winner_id(self.baseline) != _winner_id(self.evidence_aware)


def select_with_rehearsal(
    candidates: Sequence[GraphCandidate],
    scene: SceneSnapshot,
    arbiter: CandidateArbiter,
    *,
    mode: RehearsalMode = "disabled",
    provider: RehearsalEvidenceProvider | None = None,
    expected_subgoal: str | None = None,
) -> RehearsalArbitrationReport:
    """Select one candidate with optional version-bound rehearsal evidence.

    The provider is called at most once for a selection decision. It has no
    access to the live executor or ``ActionLease``. Invalid or missing
    evidence remains unavailable; it is never represented as a zero score.
    ``shadow`` computes a hypothetical result but keeps the baseline result
    live, while ``online_bounded`` promotes the evidence-aware result only
    when that result contains a selected candidate.
    """

    _validate_mode(mode)
    live_candidates = tuple(candidates)
    baseline = arbiter.select(
        live_candidates,
        scene,
        expected_subgoal=expected_subgoal,
    )
    if mode == "disabled":
        return RehearsalArbitrationReport(
            mode=mode,
            baseline=baseline,
            evidence_aware=None,
            live=baseline,
        )

    if provider is None:
        fallback = "provider_missing"
        return RehearsalArbitrationReport(
            mode=mode,
            baseline=baseline,
            evidence_aware=None,
            live=baseline,
            fallback_reason=fallback,
        )

    started = time.perf_counter()
    try:
        provided = provider(live_candidates, scene)
        if not isinstance(provided, Mapping):
            raise TypeError("rehearsal provider must return a mapping")
    except Exception as exc:
        return RehearsalArbitrationReport(
            mode=mode,
            baseline=baseline,
            evidence_aware=None,
            live=baseline,
            provider_latency_ms=_elapsed_ms(started),
            fallback_reason=f"provider_error: {type(exc).__name__}: {exc}",
        )

    enriched: list[GraphCandidate] = []
    attached: list[str] = []
    rejections: list[str] = []
    candidate_ids = {candidate.candidate_id for candidate in live_candidates}
    for candidate in live_candidates:
        rehearsal = provided.get(candidate.candidate_id)
        if rehearsal is None:
            enriched.append(candidate)
            continue
        try:
            enriched.append(
                merge_rehearsal_evidence(
                    candidate,
                    rehearsal,
                    include_in_arbiter=True,
                )
            )
        except Exception as exc:
            rejections.append(f"{candidate.candidate_id}: {exc}")
            enriched.append(candidate)
        else:
            attached.append(candidate.candidate_id)

    for unknown_id in sorted(set(provided) - candidate_ids):
        rejections.append(f"{unknown_id}: evidence has no matching candidate")

    evidence_aware = arbiter.select(
        tuple(enriched),
        scene,
        expected_subgoal=expected_subgoal,
    )
    if mode == "shadow":
        live = baseline
        fallback_reason = None
    elif evidence_aware.selected is not None:
        live = evidence_aware
        fallback_reason = None
    else:
        live = baseline
        fallback_reason = "evidence_aware_arbiter_selected_none"

    return RehearsalArbitrationReport(
        mode=mode,
        baseline=baseline,
        evidence_aware=evidence_aware,
        live=live,
        attached_candidate_ids=tuple(attached),
        evidence_rejections=tuple(rejections),
        provider_latency_ms=_elapsed_ms(started),
        fallback_reason=fallback_reason,
    )


def _validate_mode(mode: str) -> None:
    if mode not in {"disabled", "shadow", "online_bounded"}:
        raise ValueError(
            "rehearsal mode must be disabled, shadow, or online_bounded"
        )


def _winner_id(result: ArbitrationResult) -> str | None:
    return result.selected.candidate_id if result.selected is not None else None


def _elapsed_ms(started: float) -> float:
    return max(0.0, (time.perf_counter() - started) * 1000.0)


__all__ = [
    "RehearsalArbitrationReport",
    "RehearsalEvidenceProvider",
    "RehearsalMode",
    "select_with_rehearsal",
]
