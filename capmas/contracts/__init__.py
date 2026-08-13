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
from capmas.contracts.candidates import (
    ArbitrationResult,
    CandidateEvidence,
    CandidateRejection,
    CandidateRewriteReport,
    EvidenceDimension,
    GraphCandidate,
    GeometryEvidence,
    PerceptionEvidence,
)
from capmas.contracts.core import ArtifactRef, EpisodeHandle, SkillRef
from capmas.contracts.failures import FailureArtifact, FailureClass
from capmas.contracts.experiment import ExperimentRunConfig
from capmas.contracts.strategy import StrategyProfile
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
from capmas.contracts.staged import (
    MissionTopology,
    STAGED_TOPOLOGY_SCHEMA_VERSION,
    TopologySubgoal,
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
from capmas.contracts.trace import ExecutionTrace, GraphExecutionEvent, SkillTrace
from capmas.contracts.verification import PredicateReport, VerificationResult

__all__ = [
    "ActionContract",
    "ArbitrationResult",
    "CandidateEvidence",
    "ArtifactRef",
    "EpisodeHandle",
    "EpisodeStart",
    "EpisodeStatus",
    "ExecutionBudget",
    "ExecutionTrace",
    "CycleHistory",
    "FailureClass",
    "FailureArtifact",
    "ExperimentRunConfig",
    "CheckpointSpec",
    "CandidateRejection",
    "CandidateRewriteReport",
    "EvidenceDimension",
    "GraphEdge",
    "GraphCandidate",
    "GraphExecutionEvent",
    "GeometryEvidence",
    "PerceptionEvidence",
    "VerifierEvidence",
    "GroundedPolicyAgent",
    "GraphPolicyAgent",
    "MissionGraphManager",
    "MissionTopology",
    "MissionTopologyManager",
    "MissionBinding",
    "MissionEdge",
    "MissionGraph",
    "MotionIntent",
    "LoopSpec",
    "ObjectTrack",
    "PredicateReport",
    "PolicyDecision",
    "PortBinding",
    "PortSpec",
    "ResourceRequirement",
    "SceneSnapshot",
    "SceneUncertainty",
    "SkillCall",
    "SkillOutputRef",
    "SkillRef",
    "StrategyProfile",
    "SkillTrace",
    "SubgraphNodeSpec",
    "SubgraphOutputBinding",
    "SubgraphSpec",
    "STAGED_TOPOLOGY_SCHEMA_VERSION",
    "TopologySubgoal",
    "SpatialRelation",
    "VerificationResult",
    "VisualEvidence",
]


def __getattr__(name: str) -> object:
    if name == "VerifierEvidence":
        from capmas.verification.evidence import VerifierEvidence

        return VerifierEvidence
    raise AttributeError(name)
