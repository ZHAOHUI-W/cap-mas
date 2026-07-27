from dataclasses import dataclass

from capmas.backends.protocol import RobotBackend, SkillExecutionResult
from capmas.contracts.action import ActionContract, SkillCall
from capmas.contracts.core import EpisodeHandle, SkillRef
from capmas.contracts.scene import EpisodeStart, EpisodeStatus, SceneSnapshot
from capmas.contracts.verification import PredicateReport, VerificationResult
from capmas.runtime.action_lease import ActionLeaseManager
from capmas.runtime.orchestrator import RuntimeOrchestrator
from capmas.runtime.state_store import InMemoryStateStore
from capmas.skills.registry import SkillRegistry


@dataclass
class FailingSkill:
    skill_id: str = "fail"
    version: str = "1.0.0"

    def validate_args(self, args: dict[str, object]) -> None:
        assert args == {}

    def execute(self, args: dict[str, object], budget: object) -> SkillExecutionResult:
        return SkillExecutionResult(ok=False, error_type="MotionError", error_message="blocked")


class Backend:
    def __init__(self) -> None:
        self.scene = SceneSnapshot("ep", 1, 0, 1, 1, {}, ())

    def reset(self, seed=None, options=None):
        return EpisodeStart(EpisodeHandle("ep", "task", "mock", "mock", seed, 1, 1), self.scene)

    def observe(self):
        return self.scene

    def execute_skill(self, skill, args, budget):
        return skill.execute(args, budget)

    def stop(self, lease):
        return None

    def evaluator_success(self):
        return False


class Verifier:
    def approve(self, contract, scene):
        return VerificationResult(contract.contract_id, "approve", scene.scene_version, (PredicateReport("ok", True),))

    def commit(self, contract, before, after, trace):
        return VerificationResult(contract.contract_id, "commit", after.scene_version, (PredicateReport("post", True),))


def test_failed_skill_returns_trace_for_recovery_and_releases_lease() -> None:
    backend = Backend()
    registry = SkillRegistry()
    registry.register(SkillRef("fail", "1.0.0"), FailingSkill())
    runtime = RuntimeOrchestrator(
        backend,
        InMemoryStateStore(),
        registry,
        ActionLeaseManager(clock=lambda: 10),
        Verifier(),
        clock=lambda: 10,
    )
    runtime.start_episode(backend.reset())
    contract = ActionContract("c", "ep", 1, 0, "sg", (SkillCall(SkillRef("fail", "1.0.0"), {}),), (), 1000, 10, "policy")

    result = runtime.run_cycle(contract)

    assert result.committed is False
    assert result.trace.status == "failed"
    assert result.trace.failure_class == "EXECUTION_ERROR"
    assert result.verification.decision == "recover"
    assert runtime.lease_manager.active() is None
