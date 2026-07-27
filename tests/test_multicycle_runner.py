from dataclasses import dataclass

from capmas.backends.protocol import RobotBackend, SkillExecutionResult
from capmas.contracts.action import ActionContract, SkillCall
from capmas.contracts.agent import AgentContext
from capmas.contracts.core import EpisodeHandle, SkillRef
from capmas.contracts.scene import EpisodeStart, EpisodeStatus, SceneSnapshot
from capmas.contracts.verification import PredicateReport, VerificationResult
from capmas.runtime.action_lease import ActionLeaseManager
from capmas.runtime.episode_runner import MultiCycleEpisodeRunner
from capmas.runtime.orchestrator import RuntimeOrchestrator
from capmas.runtime.state_store import InMemoryStateStore
from capmas.skills.registry import SkillRegistry


@dataclass
class NoopSkill:
    skill_id: str = "noop"
    version: str = "1.0.0"

    def validate_args(self, args: dict[str, object]) -> None:
        assert args == {}

    def execute(self, args: dict[str, object], budget: object) -> SkillExecutionResult:
        return SkillExecutionResult(ok=True, output={"ok": True})


class AdvancingBackend(RobotBackend):
    def __init__(self) -> None:
        self.handle = EpisodeHandle("ep", "task", "mock", "mock", 1, 1, 1, EpisodeStatus.ACTIVE)
        self.scene = SceneSnapshot("ep", 1, 0, 1, 1, {"step": 0})

    def reset(self, seed=None, options=None):
        return EpisodeStart(self.handle, self.scene)

    def observe(self):
        return self.scene

    def execute_skill(self, skill, args, budget):
        self.scene = SceneSnapshot(
            "ep",
            1,
            self.scene.scene_version + 1,
            self.scene.scene_version + 2,
            self.scene.scene_version + 2,
            {"step": self.scene.scene_version + 1},
        )
        return skill.execute(args, budget)

    def stop(self, lease):
        return None

    def evaluator_success(self):
        return False


class AllowVerifier:
    def approve(self, contract, scene):
        return VerificationResult(contract.contract_id, "approve", scene.scene_version, (PredicateReport("pre", True),))

    def commit(self, contract, before, after, trace):
        return VerificationResult(contract.contract_id, "commit", after.scene_version, (PredicateReport("post", True),))


class FailFirstVerifier(AllowVerifier):
    def __init__(self) -> None:
        self.commit_count = 0

    def commit(self, contract, before, after, trace):
        self.commit_count += 1
        if self.commit_count == 1:
            return VerificationResult(
                contract.contract_id,
                "recover",
                after.scene_version,
                (PredicateReport("post", False),),
                failure_class="POSTCONDITION_FAILED",
            )
        return super().commit(contract, before, after, trace)


class AlwaysFailVerifier(AllowVerifier):
    def commit(self, contract, before, after, trace):
        return VerificationResult(
            contract.contract_id,
            "recover",
            after.scene_version,
            (PredicateReport("post", False),),
            failure_class="POSTCONDITION_FAILED",
        )


def make_runtime(backend: AdvancingBackend) -> RuntimeOrchestrator:
    registry = SkillRegistry()
    registry.register(SkillRef("noop", "1.0.0"), NoopSkill())
    runtime = RuntimeOrchestrator(
        backend,
        InMemoryStateStore(),
        registry,
        ActionLeaseManager(clock=lambda: 10),
        AllowVerifier(),
        clock=lambda: 10,
    )
    return runtime


def make_contract(context: AgentContext, stage: int) -> ActionContract:
    return ActionContract(
        contract_id=f"contract-{stage}",
        episode_id=context.episode_id,
        episode_epoch=context.episode_epoch,
        parent_scene_version=context.scene.scene_version,
        subgoal_id=f"stage-{stage}",
        skills=(SkillCall(SkillRef("noop", "1.0.0"), {}),),
        expected_postconditions=("post",),
        max_duration_ms=1000,
        max_sim_steps=10,
        proposed_by="policy",
    )


def test_multicycle_runner_replans_from_each_committed_scene() -> None:
    backend = AdvancingBackend()
    runtime = make_runtime(backend)
    seen: list[tuple[int, int]] = []

    def policy(context: AgentContext) -> ActionContract:
        seen.append((context.scene.scene_version, len(context.history.traces)))
        return make_contract(context, len(seen))

    result = MultiCycleEpisodeRunner(runtime).run(
        task_id="task",
        seed=1,
        policy_step=policy,
        goal_check=lambda scene: scene.scene_version >= 2,
        max_cycles=4,
    )

    assert result.goal_reached is True
    assert result.stop_reason == "task_goal_reached"
    assert result.total_cycles == 2
    assert result.committed_cycles == 2
    assert seen == [(0, 0), (1, 1)]
    assert len(result.episode_trace.traces) == 2


def test_multicycle_runner_recovers_without_deleting_failed_trace() -> None:
    backend = AdvancingBackend()
    verifier = FailFirstVerifier()
    registry = SkillRegistry()
    registry.register(SkillRef("noop", "1.0.0"), NoopSkill())
    runtime = RuntimeOrchestrator(
        backend,
        InMemoryStateStore(),
        registry,
        ActionLeaseManager(clock=lambda: 10),
        verifier,
        clock=lambda: 10,
    )
    recovery_contexts: list[AgentContext] = []

    def recovery(trace, verification, context):
        recovery_contexts.append(context)
        return make_contract(context, 2)

    result = MultiCycleEpisodeRunner(runtime).run(
        task_id="task",
        seed=1,
        policy_step=lambda context: make_contract(context, 1),
        goal_check=lambda scene: scene.scene_version >= 2,
        recovery_step=recovery,
        max_cycles=3,
        max_recoveries=1,
    )

    assert result.goal_reached is True
    assert result.stop_reason == "task_goal_reached"
    assert result.recovery_attempts == 1
    assert result.committed_cycles == 1
    assert result.episode_trace.traces[0].status == "failed"
    assert result.episode_trace.traces[1].status == "completed"
    assert len(recovery_contexts) == 1
    assert len(recovery_contexts[0].history.traces) == 1


def test_multicycle_runner_stops_when_recovery_budget_is_exhausted() -> None:
    backend = AdvancingBackend()
    registry = SkillRegistry()
    registry.register(SkillRef("noop", "1.0.0"), NoopSkill())
    runtime = RuntimeOrchestrator(
        backend,
        InMemoryStateStore(),
        registry,
        ActionLeaseManager(clock=lambda: 10),
        AlwaysFailVerifier(),
        clock=lambda: 10,
    )

    result = MultiCycleEpisodeRunner(runtime).run(
        task_id="task",
        seed=1,
        policy_step=lambda context: make_contract(context, 1),
        goal_check=lambda scene: False,
        recovery_step=lambda trace, verification, context: make_contract(context, 2),
        max_cycles=3,
        max_recoveries=1,
    )

    assert result.goal_reached is False
    assert result.stop_reason == "recovery_exhausted"
    assert result.total_cycles == 2
    assert result.recovery_attempts == 1


def test_multicycle_runner_reports_policy_finished() -> None:
    backend = AdvancingBackend()
    result = MultiCycleEpisodeRunner(make_runtime(backend)).run(
        task_id="task",
        seed=1,
        policy_step=lambda context: None,
        goal_check=lambda scene: False,
        max_cycles=2,
    )

    assert result.stop_reason == "policy_finished"
    assert result.total_cycles == 0
    assert result.goal_reached is False
