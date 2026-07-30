from __future__ import annotations

from dataclasses import FrozenInstanceError
from concurrent.futures import ThreadPoolExecutor

import pytest

from capmas.contracts.action import SkillCall
from capmas.contracts.candidates import GraphCandidate, subgraph_fingerprint
from capmas.contracts.core import SkillRef
from capmas.contracts.graph import CheckpointSpec, SubgraphNodeSpec, SubgraphSpec
from capmas.contracts.scene import SceneSnapshot
from capmas.evaluation.evidence_cache import (
    EvidenceCacheEvent,
    EvidenceCacheKey,
    EvidenceCacheStats,
    VersionedEvidenceCache,
)
from capmas.contracts.candidates import CandidateEvidence
from capmas.evaluation.evidence_contracts import EvidenceCompatibilityError


def _evidence(scene_version: int | None, score: float = 1.0) -> CandidateEvidence:
    return CandidateEvidence(
        rehearsal_success_rate=score,
        available_metrics=("rehearsal",),
        scene_version=scene_version,
        provider="test",
    )


def _candidate(candidate_id: str, scene_version: int = 1) -> GraphCandidate:
    node = SubgraphNodeSpec(
        node_id="act",
        description=candidate_id,
        skill_calls=(SkillCall(SkillRef("noop", "1.0.0"), {}),),
        postconditions=("scene_advanced",),
        proposed_by="policy",
    )
    subgraph = SubgraphSpec(
        subgraph_id="pick",
        subgoal_id="pick",
        description="pick",
        nodes=(node,),
        edges=(),
        entry_node="act",
        success_nodes=("act",),
        failure_nodes=("act",),
        checkpoints=(CheckpointSpec("check", ("scene_advanced",)),),
    )
    return GraphCandidate(candidate_id, subgraph, scene_version, "policy")


def test_cache_key_rejects_empty_fingerprint_and_negative_scene_version() -> None:
    with pytest.raises(ValueError, match="fingerprint"):
        EvidenceCacheKey("", 0)
    with pytest.raises(ValueError, match="scene version"):
        EvidenceCacheKey("candidate", -1)


def test_cache_constructor_rejects_invalid_bounds_and_contracts_are_frozen() -> None:
    with pytest.raises(ValueError, match="max entries"):
        VersionedEvidenceCache(max_entries=0)
    with pytest.raises(ValueError, match="event limit"):
        VersionedEvidenceCache(event_limit=0)

    key = EvidenceCacheKey("candidate", 1)
    stats = EvidenceCacheStats(0, 0, 0, 0, 0, 0, None, 0)
    event = EvidenceCacheEvent("miss", "candidate", 1)

    with pytest.raises(FrozenInstanceError):
        key.scene_version = 2  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        stats.hits = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        event.kind = "hit"  # type: ignore[misc]


def test_cache_event_history_is_bounded() -> None:
    cache = VersionedEvidenceCache(event_limit=2)
    key = EvidenceCacheKey("candidate-a", 1)
    cache.put(key, _evidence(1))
    cache.get(key)
    cache.get(EvidenceCacheKey("candidate-b", 1))

    assert len(cache.events()) == 2
    assert [event.kind for event in cache.events()] == ["hit", "miss"]


def test_empty_cache_exposes_zero_stats_and_no_events() -> None:
    cache = VersionedEvidenceCache()

    assert cache.stats() == EvidenceCacheStats(0, 0, 0, 0, 0, 0, None, 0)
    assert cache.events() == ()


def test_cache_returns_exact_versioned_hits_and_records_misses() -> None:
    cache = VersionedEvidenceCache()
    key = EvidenceCacheKey("candidate-a", 1)
    evidence = _evidence(1)

    cache.put(key, evidence)

    assert cache.get(key) == evidence
    assert cache.get(EvidenceCacheKey("candidate-a", 0)) is None
    assert cache.get(EvidenceCacheKey("candidate-a", 2)) is None
    assert cache.get(EvidenceCacheKey("candidate-b", 1)) is None
    assert cache.stats().hits == 1
    assert cache.stats().misses == 2
    assert cache.stats().stale_rejections == 1
    assert [event.kind for event in cache.events()] == [
        "store",
        "hit",
        "stale_rejection",
        "miss",
        "miss",
    ]


def test_scene_advance_invalidates_old_entries_and_rejects_stale_reads() -> None:
    cache = VersionedEvidenceCache()
    old_key = EvidenceCacheKey("candidate-a", 1)
    cache.put(old_key, _evidence(1))

    cache.advance_scene(2)

    assert cache.get(old_key) is None
    stats = cache.stats()
    assert stats.current_scene_version == 2
    assert stats.size == 0
    assert stats.invalidations == 1
    assert stats.stale_rejections == 1
    assert [event.kind for event in cache.events()] == [
        "store",
        "invalidation",
        "stale_rejection",
    ]


def test_newer_write_advances_scene_and_rejects_non_monotonic_refresh() -> None:
    cache = VersionedEvidenceCache()
    cache.put(EvidenceCacheKey("candidate-a", 1), _evidence(1))
    cache.put(EvidenceCacheKey("candidate-a", 2), _evidence(2, score=0.5))

    assert cache.stats().current_scene_version == 2
    assert cache.get(EvidenceCacheKey("candidate-a", 2)) is not None
    with pytest.raises(EvidenceCompatibilityError, match="monotonic"):
        cache.advance_scene(1)
    assert cache.stats().current_scene_version == 2


def test_cache_rejects_missing_or_mismatched_evidence_scene_version() -> None:
    cache = VersionedEvidenceCache()

    with pytest.raises(EvidenceCompatibilityError, match="scene version"):
        cache.put(EvidenceCacheKey("candidate-a", 1), _evidence(None))
    with pytest.raises(EvidenceCompatibilityError, match="scene version"):
        cache.put(EvidenceCacheKey("candidate-a", 1), _evidence(2))


def test_cache_evicts_least_recently_used_entry_with_bounded_capacity() -> None:
    cache = VersionedEvidenceCache(max_entries=2)
    key_a = EvidenceCacheKey("candidate-a", 1)
    key_b = EvidenceCacheKey("candidate-b", 1)
    key_c = EvidenceCacheKey("candidate-c", 1)

    cache.put(key_a, _evidence(1))
    cache.put(key_b, _evidence(1, score=0.5))
    assert cache.get(key_a) is not None
    cache.put(key_c, _evidence(1, score=0.25))

    assert cache.get(key_b) is None
    assert cache.get(key_a) is not None
    assert cache.get(key_c) is not None
    assert cache.stats().evictions == 1
    assert any(event.kind == "eviction" for event in cache.events())


def test_candidate_helpers_use_local_fingerprint_and_do_not_mutate_input() -> None:
    cache = VersionedEvidenceCache()
    candidate = _candidate("candidate-a")
    scene = SceneSnapshot("episode", 1, 1, 1, 2, {})
    evidence = _evidence(1)

    cache.put_for_candidate(candidate, scene, evidence)

    assert cache.get_for_candidate(candidate, scene) == evidence
    assert candidate.evidence is None
    attached = cache.attach_for_candidate(candidate, scene)
    assert attached is not candidate
    assert attached.evidence == evidence
    fingerprint = subgraph_fingerprint(candidate.subgraph)
    assert any(
        event.kind == "store" and event.candidate_fingerprint == fingerprint
        for event in cache.events()
    )


def test_candidate_helpers_fail_closed_for_stale_parent_scene() -> None:
    cache = VersionedEvidenceCache()
    candidate = _candidate("candidate-a", scene_version=1)
    fresh_scene = SceneSnapshot("episode", 1, 2, 1, 2, {})

    assert cache.get_for_candidate(candidate, fresh_scene) is None
    with pytest.raises(EvidenceCompatibilityError, match="parent scene"):
        cache.put_for_candidate(candidate, fresh_scene, _evidence(2))


def test_candidate_invalidation_removes_all_matching_entries() -> None:
    cache = VersionedEvidenceCache()
    candidate = _candidate("candidate-a")
    scene = SceneSnapshot("episode", 1, 1, 1, 2, {})
    cache.put_for_candidate(candidate, scene, _evidence(1))

    fingerprint = subgraph_fingerprint(candidate.subgraph)
    cache.invalidate_candidate(fingerprint)

    assert cache.get_for_candidate(candidate, scene) is None
    assert cache.stats().invalidations == 1


def test_cache_supports_concurrent_put_and_get_on_one_scene() -> None:
    cache = VersionedEvidenceCache(max_entries=32)
    keys = tuple(EvidenceCacheKey(f"candidate-{index}", 1) for index in range(8))

    def store_and_load(key: EvidenceCacheKey) -> CandidateEvidence | None:
        evidence = _evidence(1, score=0.5)
        cache.put(key, evidence)
        return cache.get(key)

    with ThreadPoolExecutor(max_workers=8) as executor:
        loaded = tuple(executor.map(store_and_load, keys))

    assert all(item is not None for item in loaded)
    assert cache.stats().stores == len(keys)
    assert cache.stats().hits == len(keys)
    assert cache.stats().size == len(keys)


def test_evaluation_package_exports_all_cache_interfaces() -> None:
    import capmas.evaluation as evaluation

    namespace: dict[str, object] = {}
    exec("from capmas.evaluation import *", namespace)
    for name in (
        "EvidenceCacheKey",
        "EvidenceCacheStats",
        "EvidenceCacheEvent",
        "VersionedEvidenceCache",
    ):
        assert name in evaluation.__all__
        assert namespace[name] is getattr(evaluation, name)
