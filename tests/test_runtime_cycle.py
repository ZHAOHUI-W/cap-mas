from dataclasses import dataclass

import pytest

from capmas.backends.protocol import RobotBackend, SkillExecutionResult
from capmas.contracts.action import ActionContract, SkillCall, SkillOutputRef
from capmas.contracts.scene import EpisodeStart, EpisodeStatus, SceneSnapshot
from capmas.contracts.verification import PredicateReport, VerificationResult
from capmas.contracts.core import EpisodeHandle, SkillRef
from capmas.runtime.orchestrator import RuntimeOrchestrator
from capmas.runtime.state_store import InMemoryStateStore
from capmas.runtime.action_lease import ActionLeaseManager
from capmas.skills.registry import SkillRegistry


@dataclass
class FakeSkill:
    skill_id: str = "move"
    version: str = "1.0.0"

    def validate_args(self, args: dict[str, object]) -> None:
        assert args["target"] == "box"

    def execute(self, args: dict[str, object], budget: object) -> SkillExecutionResult:
        return SkillExecutionResult(ok=True, output={"target": args["target"]})


@dataclass
class PoseSkill:
    skill_id: str = "sample"
    version: str = "1.0.0"

    def validate_args(self, args: dict[str, object]) -> None:
        assert args == {"object_name": "bowl"}

    def execute(self, args: dict[str, object], budget: object) -> SkillExecutionResult:
        return SkillExecutionResult(ok=True, output={"result": ((0.1, 0.2, 0.3), (1.0, 0.0, 0.0, 0.0))})


@dataclass
class GotoSkill:
    skill_id: str = "goto"
    version: str = "1.0.0"

    def validate_args(self, args: dict[str, object]) -> None:
        assert "position" in args and "quaternion_wxyz" in args

    def execute(self, args: dict[str, object], budget: object) -> SkillExecutionResult:
        return SkillExecutionResult(ok=True, output={"position": args["position"]})


class FakeBackend(RobotBackend):
    def __init__(self) -> None:
        self.executions: list[str] = []
        self.handle = EpisodeHandle(
            episode_id="episode-1",
            task_id="task-1",
            suite_name="mock",
            backend_id="mock",
            seed=7,
            episode_epoch=1,
            started_at_ns=1,
            status=EpisodeStatus.ACTIVE,
        )
        self.snapshot = SceneSnapshot(
            episode_id="episode-1",
            episode_epoch=1,
            scene_version=0,
            sensor_timestamp_ns=1,
            publish_timestamp_ns=1,
            robot={"joint_position": (0.0,), "gripper_opening": 1.0},
            objects=(),
        )

    def reset(self, seed: int | None = None, options: dict[str, object] | None = None) -> EpisodeStart:
        return EpisodeStart(handle=self.handle, initial_scene=self.snapshot)

    def observe(self) -> SceneSnapshot:
        return self.snapshot

    def execute_skill(self, skill: object, args: dict[str, object], budget: object) -> SkillExecutionResult:
        self.executions.append("move")
        self.snapshot = SceneSnapshot(
            episode_id="episode-1",
            episode_epoch=1,
            scene_version=1,
            sensor_timestamp_ns=2,
            publish_timestamp_ns=2,
            robot={"joint_position": (1.0,), "gripper_opening": 1.0},
            objects=(),
        )
        return skill.execute(args, budget)

    def stop(self, lease: object) -> None:
        return None

    def evaluator_success(self) -> bool:
        return False


class AllowVerifier:
    def approve(self, contract: ActionContract, scene: SceneSnapshot) -> VerificationResult:
        return VerificationResult(
            contract_id=contract.contract_id,
            decision="approve",
            checked_scene_version=scene.scene_version,
            predicate_results=(PredicateReport(name="preconditions", passed=True),),
        )

    def commit(
        self,
        contract: ActionContract,
        before: SceneSnapshot,
        after: SceneSnapshot,
        trace: object,
    ) -> VerificationResult:
        return VerificationResult(
            contract_id=contract.contract_id,
            decision="commit",
            checked_scene_version=after.scene_version,
            predicate_results=(PredicateReport(name="postconditions", passed=True),),
        )


class RejectPreconditionVerifier(AllowVerifier):
    def approve(self, contract: ActionContract, scene: SceneSnapshot) -> VerificationResult:
        return VerificationResult(
            contract_id=contract.contract_id,
            decision="reject",
            checked_scene_version=scene.scene_version,
            predicate_results=(
                PredicateReport(
                    name="object_in_gripper(bowl)",
                    passed=False,
                    confidence=0.0,
                    reason="gripper is not closed",
                ),
            ),
            failure_class="PRECONDITION_FAILED",
        )


def make_contract(scene_version: int = 0) -> ActionContract:
    return ActionContract(
        contract_id="contract-1",
        episode_id="episode-1",
        episode_epoch=1,
        parent_scene_version=scene_version,
        subgoal_id="subgoal-1",
        skills=(
            SkillCall(
                skill=SkillRef("move", "1.0.0"),
                args={"target": "box"},
            ),
        ),
        expected_postconditions=("moved",),
        max_duration_ms=1000,
        max_sim_steps=10,
        proposed_by="policy_agent",
    )


def test_valid_contract_commits_new_scene_and_trace() -> None:
    backend = FakeBackend()
    registry = SkillRegistry()
    registry.register(SkillRef("move", "1.0.0"), FakeSkill())
    runtime = RuntimeOrchestrator(
        backend=backend,
        state_store=InMemoryStateStore(),
        skill_registry=registry,
        lease_manager=ActionLeaseManager(clock=lambda: 10),
        verifier=AllowVerifier(),
        clock=lambda: 10,
    )
    runtime.start_episode(backend.reset())

    result = runtime.run_cycle(make_contract())

    assert result.committed is True
    assert result.trace.status == "completed"
    assert result.after_scene.scene_version == 1
    assert backend.executions == ["move"]


def test_failed_precondition_returns_structured_rejection_without_execution() -> None:
    backend = FakeBackend()
    registry = SkillRegistry()
    registry.register(SkillRef("move", "1.0.0"), FakeSkill())
    runtime = RuntimeOrchestrator(
        backend=backend,
        state_store=InMemoryStateStore(),
        skill_registry=registry,
        lease_manager=ActionLeaseManager(clock=lambda: 10),
        verifier=RejectPreconditionVerifier(),
        clock=lambda: 10,
    )
    runtime.start_episode(backend.reset())

    result = runtime.run_cycle(make_contract())

    assert result.committed is False
    assert result.rejected is True
    assert result.reason == "object_in_gripper(bowl): gripper is not closed"
    assert result.trace.status == "rejected"
    assert result.trace.failure_class == "PRECONDITION_FAILED"
    assert result.trace.precondition_result == result.verification
    assert backend.executions == []
    assert runtime.state_store.latest().scene_version == 0


def test_stale_contract_is_rejected_before_backend_execution() -> None:
    backend = FakeBackend()
    registry = SkillRegistry()
    registry.register(SkillRef("move", "1.0.0"), FakeSkill())
    runtime = RuntimeOrchestrator(
        backend=backend,
        state_store=InMemoryStateStore(),
        skill_registry=registry,
        lease_manager=ActionLeaseManager(clock=lambda: 10),
        verifier=AllowVerifier(),
        clock=lambda: 10,
    )
    runtime.start_episode(backend.reset())

    with pytest.raises(ValueError, match="stale scene"):
        runtime.run_cycle(make_contract(scene_version=99))

    assert backend.executions == []


def test_skill_output_can_feed_the_next_typed_skill() -> None:
    backend = FakeBackend()
    registry = SkillRegistry()
    registry.register(SkillRef("sample", "1.0.0"), PoseSkill())
    registry.register(SkillRef("goto", "1.0.0"), GotoSkill())
    runtime = RuntimeOrchestrator(
        backend=backend,
        state_store=InMemoryStateStore(),
        skill_registry=registry,
        lease_manager=ActionLeaseManager(clock=lambda: 10),
        verifier=AllowVerifier(),
        clock=lambda: 10,
    )
    runtime.start_episode(backend.reset())
    contract = ActionContract(
        "contract-chain",
        "episode-1",
        1,
        0,
        "subgoal-1",
        (
            SkillCall(SkillRef("sample", "1.0.0"), {"object_name": "bowl"}),
            SkillCall(
                SkillRef("goto", "1.0.0"),
                {
                    "position": SkillOutputRef(0, ("result", 0)),
                    "quaternion_wxyz": SkillOutputRef(0, ("result", 1)),
                },
            ),
        ),
        ("moved",),
        1000,
        10,
        "policy_agent",
    )

    result = runtime.run_cycle(contract)

    assert result.committed is True
    assert result.trace.skill_traces[1].args["position"] == (0.1, 0.2, 0.3)
