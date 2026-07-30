"""Benchmark and learning metrics."""

from capmas.evaluation.parity import (
    NormalizedEpisode,
    ParityComparison,
    compare_artifacts,
    load_capmas_episode,
    load_capx_trial,
)
from capmas.evaluation.candidate_identity import (
    CandidateIdentity,
    candidate_identity_from_raw_graph,
    raw_graph_fingerprint,
)

from capmas.evaluation.rehearsal import (
    ProcessRehearsalPool,
    RehearsalFailureClass,
    RehearsalJob,
    RehearsalResult,
    RehearsalTimeout,
)
from capmas.evaluation.evidence_contracts import (
    EvidenceCompatibilityError,
    EvidenceRequestContext,
    assert_evidence_compatible,
)
from capmas.evaluation.evidence_cache import (
    EvidenceCacheEvent,
    EvidenceCacheKey,
    EvidenceCacheStats,
    VersionedEvidenceCache,
)
from capmas.evaluation.libero_rehearsal import (
    LiberoRehearsalConfig,
    LiberoRehearsalWorker,
    run_libero_rehearsal_job,
)
from capmas.evaluation.phase5_artifacts import Phase5RunDirectory
from capmas.evaluation.rehearsal_evidence import (
    RehearsalEvidence,
    RehearsalPoolConfig,
    rehearsal_result_to_evidence,
    run_with_respawn,
)
from capmas.evaluation.shadow_arbiter import (
    ShadowArbitrationReport,
    run_shadow_arbitration,
)

__all__ = [
    "CandidateIdentity",
    "candidate_identity_from_raw_graph",
    "raw_graph_fingerprint",
    "NormalizedEpisode",
    "ParityComparison",
    "compare_artifacts",
    "load_capmas_episode",
    "load_capx_trial",
    "ProcessRehearsalPool",
    "RehearsalFailureClass",
    "RehearsalJob",
    "RehearsalResult",
    "RehearsalTimeout",
    "EvidenceCompatibilityError",
    "EvidenceRequestContext",
    "assert_evidence_compatible",
    "EvidenceCacheEvent",
    "EvidenceCacheKey",
    "EvidenceCacheStats",
    "VersionedEvidenceCache",
    "LiberoRehearsalConfig",
    "LiberoRehearsalWorker",
    "run_libero_rehearsal_job",
    "Phase5RunDirectory",
    "RehearsalEvidence",
    "RehearsalPoolConfig",
    "rehearsal_result_to_evidence",
    "run_with_respawn",
    "ShadowArbitrationReport",
    "run_shadow_arbitration",
]
