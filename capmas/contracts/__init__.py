"""Public CAP-MAS value objects."""

from capmas.contracts.action import ActionContract, ExecutionBudget, SkillCall, SkillOutputRef
from capmas.contracts.agent import (
    CycleHistory,
    GraphPolicyAgent,
    GroundedPolicyAgent,
    MissionGraphManager,
    MissionTopologyManager,
    PolicyDecision,
)
from capmas.contracts.calibration import (
    CAPABILITY_SCHEMA_VERSION,
    COLLECTION_SCHEMA_VERSION,
    DATASET_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION,
    HORIZON_SCHEMA_VERSION,
    CalibrationCollectionContext,
    CalibrationOutcome,
    CalibrationPrediction,
    CandidateFeatureSnapshot,
    CollectionLane,
    DatasetSplit,
    ExecutionStatus,
    FeatureStatus,
    HorizonLabel,
    horizon_bucket,
)
from capmas.contracts.candidates import (
    ArbitrationResult,
    CandidateEvidence,
    CandidateRejection,
    CandidateRewriteReport,
    EvidenceDimension,
    GeometryEvidence,
    GraphCandidate,
    PerceptionEvidence,
)
from capmas.contracts.core import ArtifactRef, EpisodeHandle, SkillRef
from capmas.contracts.experiment import ExperimentRunConfig
from capmas.contracts.failures import FailureArtifact, FailureClass
from capmas.contracts.graph import (
    CheckpointSpec,
    GraphEdge,
    LoopSpec,
    MissionBinding,
    MissionEdge,
    MissionGraph,
    MotionIntent,
    PortBinding,
    PortSpec,
    ResourceRequirement,
    SubgraphNodeSpec,
    SubgraphOutputBinding,
    SubgraphSpec,
)
from capmas.contracts.scene import (
    EpisodeStart,
    EpisodeStatus,
    ObjectTrack,
    SceneSnapshot,
    SceneUncertainty,
    SpatialRelation,
    VisualEvidence,
)
from capmas.contracts.staged import (
    STAGED_TOPOLOGY_SCHEMA_VERSION,
    MissionTopology,
    TopologySubgoal,
)
from capmas.contracts.strategy import StrategyProfile
from capmas.contracts.trace import (
    ExecutionTrace,
    GraphEventKind,
    GraphExecutionEvent,
    SkillTrace,
)
from capmas.contracts.verification import PredicateReport, VerificationResult

__all__ = [
    "CAPABILITY_SCHEMA_VERSION",
    "COLLECTION_SCHEMA_VERSION",
    "DATASET_SCHEMA_VERSION",
    "FEATURE_SCHEMA_VERSION",
    "HORIZON_SCHEMA_VERSION",
    "STAGED_TOPOLOGY_SCHEMA_VERSION",
    "ActionContract",
    "ArbitrationResult",
    "ArtifactRef",
    "CalibrationCollectionContext",
    "CalibrationOutcome",
    "CalibrationPrediction",
    "CandidateEvidence",
    "CandidateFeatureSnapshot",
    "CandidateRejection",
    "CandidateRewriteReport",
    "CheckpointSpec",
    "CollectionLane",
    "CycleHistory",
    "DatasetSplit",
    "EpisodeHandle",
    "EpisodeStart",
    "EpisodeStatus",
    "EvidenceDimension",
    "ExecutionBudget",
    "ExecutionStatus",
    "ExecutionTrace",
    "ExperimentRunConfig",
    "FailureArtifact",
    "FailureClass",
    "FeatureStatus",
    "GeometryEvidence",
    "GraphCandidate",
    "GraphEdge",
    "GraphEventKind",
    "GraphExecutionEvent",
    "GraphPolicyAgent",
    "GroundedPolicyAgent",
    "HorizonLabel",
    "LoopSpec",
    "MissionBinding",
    "MissionEdge",
    "MissionGraph",
    "MissionGraphManager",
    "MissionTopology",
    "MissionTopologyManager",
    "MotionIntent",
    "ObjectTrack",
    "PerceptionEvidence",
    "PolicyDecision",
    "PortBinding",
    "PortSpec",
    "PredicateReport",
    "ResourceRequirement",
    "SceneSnapshot",
    "SceneUncertainty",
    "SkillCall",
    "SkillOutputRef",
    "SkillRef",
    "SkillTrace",
    "SpatialRelation",
    "StrategyProfile",
    "SubgraphNodeSpec",
    "SubgraphOutputBinding",
    "SubgraphSpec",
    "TopologySubgoal",
    "VerificationResult",
    "VerifierEvidence",
    "VisualEvidence",
    "horizon_bucket",
]


def __getattr__(name: str) -> object:
    if name == "VerifierEvidence":
        from capmas.verification.evidence import VerifierEvidence

        return VerifierEvidence
    raise AttributeError(name)
