from __future__ import annotations

from dataclasses import dataclass

from capmas.contracts.scene import SceneSnapshot
from capmas.contracts.trace import EpisodeTrace, ExecutionTrace


@dataclass(frozen=True)
class VerifiedTransition:
    before: SceneSnapshot
    after: SceneSnapshot
    trace: ExecutionTrace
    progress_before: float
    progress_after: float
    safety_violation: bool = False
    human_intervention: bool = False


@dataclass(frozen=True)
class LearningReturn:
    value: float
    progress_delta: float
    terminal: float
    cost: float
    constrained: bool


class CAPXBinaryReward:
    def benchmark(self, episode: EpisodeTrace, evaluator_success: bool) -> float:
        del episode
        return 1.0 if evaluator_success else 0.0

    def learning_return(self, transition: VerifiedTransition) -> LearningReturn:
        progress_delta = transition.progress_after - transition.progress_before
        cost = 1.0 if transition.human_intervention else 0.0
        constrained = transition.safety_violation
        value = -1.0 if constrained else progress_delta - cost
        return LearningReturn(value, progress_delta, 0.0, cost, constrained)
