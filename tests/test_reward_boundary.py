from capmas.contracts.scene import SceneSnapshot
from capmas.contracts.trace import ExecutionTrace
from capmas.evaluation.reward import CAPXBinaryReward, VerifiedTransition


def test_benchmark_reward_remains_binary_and_separate_from_learning_return() -> None:
    scene = SceneSnapshot("ep", 1, 0, 1, 1, {}, ())
    trace = ExecutionTrace("trace", "ep", 1, "contract", "lease", 0, 0, 1, 1, 2, "completed")
    transition = VerifiedTransition(scene, scene, trace, 0.2, 0.7)
    reward = CAPXBinaryReward()

    assert reward.benchmark(None, evaluator_success=False) == 0.0
    assert reward.benchmark(None, evaluator_success=True) == 1.0
    assert abs(reward.learning_return(transition).value - 0.5) < 1e-9


def test_safety_violation_is_constrained_not_compensated_by_progress() -> None:
    scene = SceneSnapshot("ep", 1, 0, 1, 1, {}, ())
    trace = ExecutionTrace("trace", "ep", 1, "contract", "lease", 0, 0, 1, 1, 2, "failed")
    transition = VerifiedTransition(scene, scene, trace, 0.0, 1.0, safety_violation=True)

    result = CAPXBinaryReward().learning_return(transition)

    assert result.constrained is True
    assert result.value == -1.0
