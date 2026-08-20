"""Unified 2D/3D perception boundaries."""

from capmas.perception.effective_motion import (
    CandidateExecutionContext,
    EffectiveMotionProgram,
    EffectiveMotionSegment,
    bind_effective_motion,
    execution_graph_fingerprint,
    materialize_execution_graph,
)

__all__ = [
    "CandidateExecutionContext",
    "EffectiveMotionProgram",
    "EffectiveMotionSegment",
    "bind_effective_motion",
    "execution_graph_fingerprint",
    "materialize_execution_graph",
]
