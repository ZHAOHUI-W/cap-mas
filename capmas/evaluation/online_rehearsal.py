"""Online, bounded attachment of process-rehearsal evidence to arbitration."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from capmas.agents.arbiter import CandidateArbiter
from capmas.contracts.candidates import (
    ArbitrationResult,
    GraphCandidate,
    subgraph_fingerprint,
)
from capmas.contracts.scene import SceneSnapshot
from capmas.evaluation.evidence_cache import (
    EvidenceCacheKey,
    EvidenceCacheStats,
    VersionedEvidenceCache,
)
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
    cache_stats: EvidenceCacheStats | None = None
    evidence_candidates: tuple[GraphCandidate, ...] = ()

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
    evidence_cache: VersionedEvidenceCache | None = None,
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
            cache_stats=None,
            evidence_candidates=live_candidates,
        )

    if provider is None:
        fallback = "provider_missing"
        return RehearsalArbitrationReport(
            mode=mode,
            baseline=baseline,
            evidence_aware=None,
            live=baseline,
            fallback_reason=fallback,
            cache_stats=evidence_cache.stats() if evidence_cache is not None else None,
            evidence_candidates=live_candidates,
        )

    started = time.perf_counter()
    effective_provider = _cached_provider(provider, evidence_cache)
    try:
        provided = effective_provider(live_candidates, scene)
        if not isinstance(provided, Mapping):
            raise TypeError("rehearsal provider must return a mapping")
    except Exception as exc:  # noqa: BLE001
        return RehearsalArbitrationReport(
            mode=mode,
            baseline=baseline,
            evidence_aware=None,
            live=baseline,
            provider_latency_ms=_elapsed_ms(started),
            fallback_reason=f"provider_error: {type(exc).__name__}: {exc}",
            cache_stats=evidence_cache.stats() if evidence_cache is not None else None,
            evidence_candidates=live_candidates,
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
        except Exception as exc:  # noqa: BLE001
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
        cache_stats=evidence_cache.stats() if evidence_cache is not None else None,
        evidence_candidates=tuple(enriched),
    )


def _cached_provider(
    provider: RehearsalEvidenceProvider,
    cache: VersionedEvidenceCache | None,
) -> RehearsalEvidenceProvider:
    if cache is None:
        return provider

    def provide(
        candidates: Sequence[GraphCandidate],
        scene: SceneSnapshot,
    ) -> Mapping[str, RehearsalEvidence]:
        cache.advance_scene(scene.scene_version)
        cached: dict[str, RehearsalEvidence] = {}
        missing: list[GraphCandidate] = []
        for candidate in candidates:
            key = EvidenceCacheKey(
                subgraph_fingerprint(candidate.subgraph),
                scene.scene_version,
            )
            value = cache.get(key)
            if isinstance(value, RehearsalEvidence) and _cache_identity_matches(
                candidate, value, scene
            ):
                cached[candidate.candidate_id] = value
            else:
                if value is not None:
                    cache.invalidate_candidate(key.candidate_fingerprint)
                missing.append(candidate)

        if not missing:
            return cached

        fresh = provider(tuple(missing), scene)
        if not isinstance(fresh, Mapping):
            raise TypeError("rehearsal provider must return a mapping")
        result = dict(cached)
        for candidate in missing:
            evidence = fresh.get(candidate.candidate_id)
            if isinstance(evidence, RehearsalEvidence):
                result[candidate.candidate_id] = evidence
                if _cache_identity_matches(candidate, evidence, scene):
                    cache.put(
                        EvidenceCacheKey(
                            subgraph_fingerprint(candidate.subgraph),
                            scene.scene_version,
                        ),
                        evidence,  # type: ignore[arg-type]
                    )
        for candidate_id, evidence in fresh.items():
            if candidate_id not in result:
                result[candidate_id] = evidence
        return result

    return provide


def _cache_identity_matches(
    candidate: GraphCandidate,
    evidence: RehearsalEvidence,
    scene: SceneSnapshot,
) -> bool:
    if evidence.candidate_id != candidate.candidate_id:
        return False
    if evidence.scene_version != scene.scene_version:
        return False
    effective_fingerprint = (
        evidence.arbiter_fingerprint
        if evidence.fingerprint_scope == "graph"
        else evidence.candidate_fingerprint
    )
    return effective_fingerprint == subgraph_fingerprint(candidate.subgraph)


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
