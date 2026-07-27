from dataclasses import dataclass

from capmas.backends.protocol import RobotBackend, SkillExecutionResult
from capmas.contracts.action import ActionContract, SkillCall
from capmas.contracts.core import EpisodeHandle, SkillRef
from capmas.contracts.scene import EpisodeStart, EpisodeStatus, SceneSnapshot
from capmas.contracts.verification import PredicateReport, VerificationResult
from capmas.runtime.action_lease import ActionLeaseManager
from capmas.runtime.episode_runner import EpisodeRunner, to_jsonable
from capmas.runtime.orchestrator import RuntimeOrchestrator
from capmas.runtime.state_store import InMemoryStateStore
from capmas.skills.registry import SkillRegistry


@dataclass
class NoopSkill:
    skill_id: str = "noop"
    version: str = "1.0.0"

    def validate_args(self, args):
        assert args == {}

    def execute(self, args, budget):
        return SkillExecutionResult(ok=True, output={"ok": True})


class Backend(RobotBackend):
    def __init__(self):
        self.snapshot = SceneSnapshot("ep", 1, 0, 1, 1, {"gripper_opening": 1.0})

    def reset(self, seed=None, options=None):
        return EpisodeStart(
            EpisodeHandle("ep", "libero_spatial_0", "libero_spatial", "mock", seed, 1, 1, EpisodeStatus.ACTIVE),
            self.snapshot,
        )

    def observe(self):
        self.snapshot = SceneSnapshot("ep", 1, 1, 2, 2, {"gripper_opening": 1.0})
        return self.snapshot

    def execute_skill(self, skill, args, budget):
        return skill.execute(args, budget)

    def stop(self, lease):
        return None

    def evaluator_success(self):
        return True


class Verifier:
    def approve(self, contract, scene):
        return VerificationResult(contract.contract_id, "approve", scene.scene_version, (PredicateReport("ok", True),))

    def commit(self, contract, before, after, trace):
        return VerificationResult(contract.contract_id, "commit", after.scene_version, (PredicateReport("ok", True),))


def test_episode_runner_emits_capmas_episode_result() -> None:
    backend = Backend()
    registry = SkillRegistry()
    registry.register(SkillRef("noop", "1.0.0"), NoopSkill())
    runtime = RuntimeOrchestrator(
        backend,
        InMemoryStateStore(),
        registry,
        ActionLeaseManager(clock=lambda: 10),
        Verifier(),
        clock=lambda: 10,
    )

    def policy(context):
        return ActionContract(
            "c", context.episode_id, context.episode_epoch, context.scene.scene_version,
            "sg", (SkillCall(SkillRef("noop", "1.0.0"), {}),), (), 1000, 10, "policy",
        )

    result = EpisodeRunner(runtime).run(
        task_id="libero_spatial_0", seed=1, policy_step=policy, max_cycles=1
    )

    assert result.evaluator_success is True
    assert result.committed_cycles == 1
    assert result.episode_trace.traces[0].status == "completed"
    assert to_jsonable(result)["stop_reason"] == "evaluator_success"
