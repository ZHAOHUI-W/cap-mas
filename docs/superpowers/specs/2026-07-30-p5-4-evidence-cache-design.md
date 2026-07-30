# P5.4 Versioned Evidence Cache Design

## Status

Approved direction for the next Phase 5 increment. This design covers the
minimal in-process cache and does not introduce persistent storage or online
physical promotion.

## Goal

Reuse candidate-specific evidence only when it belongs to the exact local
Arbiter candidate fingerprint and the requested source `SceneSnapshot` scene
version. A rolling scene refresh must make older entries unavailable before
they can reach the Arbiter.

## Non-goals

- No cross-process or persistent cache in P5.4. The process rehearsal artifacts
  remain the durable source of evidence.
- No LLM call, semantic inference, geometry computation, or robot execution in
  the cache.
- No automatic online promotion of a shadow winner.
- No change to the existing `CandidateArbiter` scoring weights.

## Design choices

### Cache key

The cache key is the effective local candidate identity, not a candidate ID or
source graph fingerprint:

```python
@dataclass(frozen=True)
class EvidenceCacheKey:
    candidate_fingerprint: str
    scene_version: int
```

For a `GraphCandidate`, the key uses the existing canonical
`subgraph_fingerprint(candidate.subgraph)`. Graph-scoped rehearsal evidence
must use its already-derived `arbiter_fingerprint` from P5.3 before entering
the cache.

### Version semantics

`VersionedEvidenceCache` tracks one current scene version.

- `advance_scene(version)` requires a non-decreasing version.
- Advancing to a newer version removes all entries from older scene versions.
- A lookup for an older version returns no evidence, increments
  `stale_rejections`, and records a `stale_rejection` event.
- A lookup for the current version can hit only an exact key.
- A lookup for a future version is a miss; it does not silently advance the
  cache.
- `put()` rejects evidence whose declared `CandidateEvidence.scene_version`
  is missing or differs from the key version.
- A write for a newer version advances the cache and invalidates older entries.

Unknown or stale evidence is unavailable. It is never converted into a zero
score and never attached to a `GraphCandidate`.

### Thread and memory boundary

The cache is an in-process, thread-safe bounded LRU store. A lock protects
entries, current version, counters, and event history. `max_entries` bounds
memory use; evictions are observable. Cross-process callers must exchange
serialized evidence or artifact URIs and create a cache in the receiving
process; this increment does not share Python memory across processes.

### Public interface

```python
class VersionedEvidenceCache:
    def advance_scene(self, scene_version: int) -> None: ...
    def get(self, key: EvidenceCacheKey) -> CandidateEvidence | None: ...
    def put(self, key: EvidenceCacheKey, evidence: CandidateEvidence) -> None: ...
    def get_for_candidate(
        self, candidate: GraphCandidate, scene: SceneSnapshot
    ) -> CandidateEvidence | None: ...
    def put_for_candidate(
        self, candidate: GraphCandidate, scene: SceneSnapshot,
        evidence: CandidateEvidence,
    ) -> None: ...
    def attach_for_candidate(
        self, candidate: GraphCandidate, scene: SceneSnapshot
    ) -> GraphCandidate: ...
    def invalidate_candidate(self, candidate_fingerprint: str) -> None: ...
    def stats(self) -> EvidenceCacheStats: ...
    def events(self) -> tuple[EvidenceCacheEvent, ...]: ...
```

`attach_for_candidate()` returns a copied candidate with exact cached evidence
or the original candidate when no valid entry exists. It does not mutate the
input and does not attach stale evidence.

### Observability

The cache exposes immutable counters and bounded events:

```python
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
```

Event kinds include `hit`, `miss`, `stale_rejection`, `store`,
`invalidation`, and `eviction`. The experiment driver can serialize
`stats()` and `events()` into its existing run artifact without exposing raw
provider secrets.

## Failure handling

- Empty fingerprints and negative scene versions raise `ValueError`.
- Evidence with no scene version or a mismatched scene version raises
  `EvidenceCompatibilityError` before storage.
- Non-monotonic `advance_scene()` raises `EvidenceCompatibilityError` and
  leaves the cache unchanged.
- Stale lookup is fail-closed and returns `None`.
- Capacity overflow evicts the least-recently-used entry and records it.

## Test and acceptance gates

Unit tests must prove:

1. exact fingerprint/version hits and misses;
2. stale lookup rejection after a scene advance;
3. old entries are invalidated when a newer scene is published;
4. mismatched or missing evidence scene versions cannot be stored;
5. bounded LRU eviction and counters/events;
6. candidate attach returns a copied candidate and never mutates the input;
7. the cache is safe under concurrent get/put calls.

P5.4 code closure requires the local suite, compile check, and diff check. A
real multi-task cache hit-rate experiment is a separate evaluation gate and
must use a new run-scoped artifact directory.
