"""Thread-safe, scene-versioned cache for candidate-specific evidence."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from threading import RLock

from capmas.contracts.candidates import CandidateEvidence, GraphCandidate, subgraph_fingerprint
from capmas.contracts.scene import SceneSnapshot
from capmas.evaluation.evidence_contracts import EvidenceCompatibilityError


@dataclass(frozen=True)
class EvidenceCacheKey:
    candidate_fingerprint: str
    scene_version: int

    def __post_init__(self) -> None:
        if not self.candidate_fingerprint:
            raise ValueError("cache candidate fingerprint must not be empty")
        if self.scene_version < 0:
            raise ValueError("cache scene version must not be negative")


@dataclass(frozen=True)
class EvidenceCacheStats:
    hits: int
    misses: int
    stale_rejections: int
    stores: int
    invalidations: int
    evictions: int
    current_scene_version: int | None
    size: int


@dataclass(frozen=True)
class EvidenceCacheEvent:
    kind: str
    candidate_fingerprint: str | None
    scene_version: int | None


class VersionedEvidenceCache:
    """Bounded process-local cache with fail-closed scene versioning."""

    def __init__(self, *, max_entries: int = 256, event_limit: int = 512) -> None:
        if max_entries <= 0:
            raise ValueError("cache max entries must be positive")
        if event_limit <= 0:
            raise ValueError("cache event limit must be positive")
        self._max_entries = max_entries
        self._event_limit = event_limit
        self._lock = RLock()
        self._entries: OrderedDict[EvidenceCacheKey, CandidateEvidence] = OrderedDict()
        self._current_scene_version: int | None = None
        self._hits = 0
        self._misses = 0
        self._stale_rejections = 0
        self._stores = 0
        self._invalidations = 0
        self._evictions = 0
        self._events: list[EvidenceCacheEvent] = []

    def stats(self) -> EvidenceCacheStats:
        with self._lock:
            return EvidenceCacheStats(
                hits=self._hits,
                misses=self._misses,
                stale_rejections=self._stale_rejections,
                stores=self._stores,
                invalidations=self._invalidations,
                evictions=self._evictions,
                current_scene_version=self._current_scene_version,
                size=len(self._entries),
            )

    def events(self) -> tuple[EvidenceCacheEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def advance_scene(self, scene_version: int) -> None:
        """Publish a scene version and invalidate all older entries."""

        if scene_version < 0:
            raise ValueError("cache scene version must not be negative")
        with self._lock:
            if (
                self._current_scene_version is not None
                and scene_version < self._current_scene_version
            ):
                raise EvidenceCompatibilityError(
                    "cache scene versions must be monotonic"
                )
            if self._current_scene_version == scene_version:
                return
            self._advance_scene_locked(scene_version)

    def get(self, key: EvidenceCacheKey) -> CandidateEvidence | None:
        """Return evidence only for the exact current-or-unseen key."""

        with self._lock:
            if (
                self._current_scene_version is not None
                and key.scene_version < self._current_scene_version
            ):
                self._stale_rejections += 1
                self._record("stale_rejection", key.candidate_fingerprint, key.scene_version)
                return None
            evidence = self._entries.get(key)
            if evidence is None:
                self._misses += 1
                self._record("miss", key.candidate_fingerprint, key.scene_version)
                return None
            self._entries.move_to_end(key)
            self._hits += 1
            self._record("hit", key.candidate_fingerprint, key.scene_version)
            return evidence

    def put(self, key: EvidenceCacheKey, evidence: CandidateEvidence) -> None:
        """Store evidence after enforcing its source scene version."""

        if evidence.scene_version is None:
            raise EvidenceCompatibilityError(
                "evidence scene version must be available for cache storage"
            )
        if evidence.scene_version != key.scene_version:
            raise EvidenceCompatibilityError(
                f"evidence scene version {evidence.scene_version} does not match "
                f"cache key scene version {key.scene_version}"
            )
        with self._lock:
            if (
                self._current_scene_version is not None
                and key.scene_version < self._current_scene_version
            ):
                self._stale_rejections += 1
                self._record("stale_rejection", key.candidate_fingerprint, key.scene_version)
                raise EvidenceCompatibilityError(
                    "cannot store evidence for a stale scene version"
                )
            if (
                self._current_scene_version is None
                or key.scene_version > self._current_scene_version
            ):
                self._advance_scene_locked(key.scene_version)
            self._entries[key] = evidence
            self._entries.move_to_end(key)
            self._stores += 1
            self._record("store", key.candidate_fingerprint, key.scene_version)
            while len(self._entries) > self._max_entries:
                evicted_key, _ = self._entries.popitem(last=False)
                self._evictions += 1
                self._record(
                    "eviction",
                    evicted_key.candidate_fingerprint,
                    evicted_key.scene_version,
                )

    def _advance_scene_locked(self, scene_version: int) -> None:
        stale_keys = [
            key
            for key in self._entries
            if key.scene_version < scene_version
        ]
        for key in stale_keys:
            del self._entries[key]
            self._invalidations += 1
            self._record("invalidation", key.candidate_fingerprint, key.scene_version)
        self._current_scene_version = scene_version

    def get_for_candidate(
        self,
        candidate: GraphCandidate,
        scene: SceneSnapshot,
    ) -> CandidateEvidence | None:
        """Look up evidence for a candidate only in its parent scene."""

        fingerprint = subgraph_fingerprint(candidate.subgraph)
        if candidate.parent_scene_version != scene.scene_version:
            with self._lock:
                self._stale_rejections += 1
                self._record("stale_rejection", fingerprint, scene.scene_version)
            return None
        return self.get(EvidenceCacheKey(fingerprint, scene.scene_version))

    def put_for_candidate(
        self,
        candidate: GraphCandidate,
        scene: SceneSnapshot,
        evidence: CandidateEvidence,
    ) -> None:
        """Store evidence after checking candidate and source scene identity."""

        if candidate.parent_scene_version != scene.scene_version:
            raise EvidenceCompatibilityError(
                f"candidate parent scene version {candidate.parent_scene_version} does not "
                f"match scene version {scene.scene_version}"
            )
        self.put(
            EvidenceCacheKey(
                subgraph_fingerprint(candidate.subgraph),
                scene.scene_version,
            ),
            evidence,
        )

    def attach_for_candidate(
        self,
        candidate: GraphCandidate,
        scene: SceneSnapshot,
    ) -> GraphCandidate:
        """Return a copied candidate with exact cached evidence, if present."""

        evidence = self.get_for_candidate(candidate, scene)
        if evidence is None:
            return candidate
        return replace(candidate, evidence=evidence)

    def invalidate_candidate(self, candidate_fingerprint: str) -> None:
        """Remove every scene-versioned entry for one effective candidate."""

        if not candidate_fingerprint:
            raise ValueError("cache candidate fingerprint must not be empty")
        with self._lock:
            keys = [
                key
                for key in self._entries
                if key.candidate_fingerprint == candidate_fingerprint
            ]
            for key in keys:
                del self._entries[key]
                self._invalidations += 1
                self._record("invalidation", key.candidate_fingerprint, key.scene_version)

    def _record(
        self,
        kind: str,
        candidate_fingerprint: str | None,
        scene_version: int | None,
    ) -> None:
        self._events.append(EvidenceCacheEvent(kind, candidate_fingerprint, scene_version))
        if len(self._events) > self._event_limit:
            del self._events[: len(self._events) - self._event_limit]


__all__ = [
    "EvidenceCacheEvent",
    "EvidenceCacheKey",
    "EvidenceCacheStats",
    "VersionedEvidenceCache",
]
