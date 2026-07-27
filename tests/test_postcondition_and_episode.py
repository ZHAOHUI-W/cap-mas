from capmas.contracts.action import ActionContract, SkillCall
from capmas.contracts.core import EpisodeHandle, SkillRef
from capmas.contracts.scene import EpisodeStart, SceneSnapshot
from capmas.contracts.verification import PredicateReport, VerificationResult
from capmas.runtime.action_lease import ActionLeaseManager
from capmas.runtime.orchestrator import RuntimeOrchestrator
from capmas.runtime.state_store import InMemoryStateStore
from capmas.skills.registry import SkillRegistry
from tests.test_runtime_cycle import FakeBackend, FakeSkill


class RejectPostcondition:
    def approve(self, contract, scene):
        return VerificationResult(contract.contract_id, "approve", scene.scene_version, (PredicateReport("pre", True),))

    def commit(self, contract, before, after, trace):
        return VerificationResult(contract.contract_id, "recover", after.scene_version, (PredicateReport("post", False),), failure_class="POSTCONDITION_FAILED")


def test_postcondition_failure_commits_observation_but_returns_recovery() -> None:
    backend = FakeBackend()
    registry = SkillRegistry()
    registry.register(SkillRef("move", "1.0.0"), FakeSkill())
    runtime = RuntimeOrchestrator(backend, InMemoryStateStore(), registry, ActionLeaseManager(clock=lambda: 10), RejectPostcondition(), clock=lambda: 10)
    runtime.start_episode(backend.reset())
    contract = ActionContract("c", "episode-1", 1, 0, "sg", (SkillCall(SkillRef("move", "1.0.0"), {"target": "box"}),), ("moved",), 1000, 10, "policy")

    result = runtime.run_cycle(contract)

    assert result.committed is False
    assert result.trace.status == "failed"
    assert result.trace.failure_class == "POSTCONDITION_FAILED"
    assert runtime.state_store.latest().scene_version == 1
