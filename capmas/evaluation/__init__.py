"""Benchmark and learning metrics."""

# ruff: noqa: I001
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
from capmas.evaluation.labels import extract_horizon, planned_horizon, realized_horizon
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
from capmas.evaluation.online_rehearsal import (
    RehearsalArbitrationReport,
    RehearsalEvidenceProvider,
    RehearsalMode,
    select_with_rehearsal,
)
from capmas.evaluation.feature_snapshots import FEATURE_GROUPS_V1, capture_feature_snapshot
from capmas.evaluation.dataset import (
    DatasetAudit,
    LeakageFinding,
    assert_dataset_eligible,
    assign_lineage_splits,
    audit_calibration_dataset,
    build_calibration_dataset,
    normalize_physical_outcomes,
)
from capmas.evaluation.verifier_artifacts import (
    DynamicVerifierArtifact,
    collect_dynamic_verifier_artifacts,
    static_verifier_artifacts_from_arbitrations,
)
from capmas.evaluation.ood import (
    Condition,
    LeakageAudit,
    OODCase,
    OODReplayEvidence,
    OODSplitManifest,
    canonical_manifest_payload,
    dump_ood_manifest,
    load_ood_manifest,
    manifest_sha256,
    validate_ood_manifest,
)
from capmas.evaluation.ood_statistics import (
    ConfidenceInterval,
    OODAggregateReport,
    aggregate_ood_pairs,
    exact_mcnemar_pvalue,
    paired_success_delta,
    wilson_interval,
)

__all__ = [  # noqa: RUF022
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
    "extract_horizon",
    "planned_horizon",
    "realized_horizon",
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
    "RehearsalArbitrationReport",
    "RehearsalEvidenceProvider",
    "RehearsalMode",
    "select_with_rehearsal",
    "FEATURE_GROUPS_V1",
    "capture_feature_snapshot",
    "DatasetAudit",
    "LeakageFinding",
    "assert_dataset_eligible",
    "assign_lineage_splits",
    "audit_calibration_dataset",
    "build_calibration_dataset",
    "normalize_physical_outcomes",
    "DynamicVerifierArtifact",
    "collect_dynamic_verifier_artifacts",
    "static_verifier_artifacts_from_arbitrations",
    "Condition",
    "LeakageAudit",
    "OODCase",
    "OODReplayEvidence",
    "OODSplitManifest",
    "canonical_manifest_payload",
    "dump_ood_manifest",
    "load_ood_manifest",
    "manifest_sha256",
    "validate_ood_manifest",
    "ConfidenceInterval",
    "OODAggregateReport",
    "aggregate_ood_pairs",
    "exact_mcnemar_pvalue",
    "paired_success_delta",
    "wilson_interval",
]
