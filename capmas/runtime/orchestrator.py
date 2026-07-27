from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Callable, Protocol
from uuid import uuid4

from capmas.backends.protocol import RobotBackend
from capmas.contracts.action import ActionContract
from capmas.contracts.action import SkillOutputRef
from capmas.contracts.scene import EpisodeStart, SceneSnapshot
from capmas.contracts.trace import ExecutionTrace, SkillTrace
from capmas.contracts.verification import VerificationResult
from capmas.contracts.failures import FailureClass
from capmas.runtime.action_lease import ActionLease, ActionLeaseManager
from capmas.runtime.state_store import InMemoryStateStore
from capmas.skills.registry import SkillRegistry


class Verifier(Protocol):
    def approve(self, contract: ActionContract, scene: SceneSnapshot) -> VerificationResult: ...

    def commit(
        self,
        contract: ActionContract,
        before: SceneSnapshot,
        after: SceneSnapshot,
        trace: ExecutionTrace,
    ) -> VerificationResult: ...


@dataclass(frozen=True)
class CycleResult:
    committed: bool
    before_scene: SceneSnapshot
    after_scene: SceneSnapshot
    trace: ExecutionTrace
    verification: VerificationResult
    rejected: bool = False
    reason: str | None = None


class RuntimeOrchestrator:
    def __init__(
        self,
        backend: RobotBackend,
        state_store: InMemoryStateStore,
        skill_registry: SkillRegistry,
        lease_manager: ActionLeaseManager,
        verifier: Verifier,
        clock: Callable[[], int] | None = None,
    ) -> None:
        self.backend = backend
        self.state_store = state_store
        self.skill_registry = skill_registry
        self.lease_manager = lease_manager
        self.verifier = verifier
        self.clock = clock or time.time_ns
        self._episode: EpisodeStart | None = None

    def start_episode(self, episode: EpisodeStart) -> None:
        self._episode = episode
        self.state_store.start_episode(episode.initial_scene)

    def run_cycle(self, contract: ActionContract) -> CycleResult:
        if self._episode is None:
            raise RuntimeError("episode has not started")
        current = self.state_store.latest()
        handle = self._episode.handle
        if contract.episode_id != handle.episode_id or contract.episode_epoch != handle.episode_epoch:
            raise ValueError("contract belongs to another episode epoch")
        if contract.parent_scene_version != current.scene_version:
            raise ValueError(
                f"stale scene version: expected {current.scene_version}, "
                f"got {contract.parent_scene_version}"
            )
        self.skill_registry.validate_contract(contract)
        started_at = int(self.clock())
        approval = self.verifier.approve(contract, current)
        if not approval.passed:
            details = "; ".join(
                f"{report.name}: {report.reason or 'failed'}"
                for report in approval.predicate_results
                if not report.passed
            ) or "verification approval failed"
            trace = ExecutionTrace(
                trace_id=str(uuid4()),
                episode_id=handle.episode_id,
                episode_epoch=handle.episode_epoch,
                contract_id=contract.contract_id,
                lease_id="not_acquired",
                parent_scene_version=contract.parent_scene_version,
                start_scene_version=current.scene_version,
                end_scene_version=current.scene_version,
                started_at_ns=started_at,
                finished_at_ns=int(self.clock()),
                status="rejected",
                precondition_result=approval,
                failure_class=approval.failure_class or FailureClass.PRECONDITION_FAILED,
                metadata={
                    "rejected": True,
                    "reason": details,
                    "failed_predicates": tuple(
                        report.name
                        for report in approval.predicate_results
                        if not report.passed
                    ),
                },
            )
            rejection = replace(
                approval,
                decision="reject",
                failure_class=approval.failure_class or FailureClass.PRECONDITION_FAILED,
            )
            return CycleResult(
                committed=False,
                before_scene=current,
                after_scene=current,
                trace=trace,
                verification=rejection,
                rejected=True,
                reason=details,
            )

        lease = self.lease_manager.acquire(
            holder=contract.proposed_by,
            contract_id=contract.contract_id,
            duration_ms=contract.max_duration_ms,
        )
        skill_traces: list[SkillTrace] = []
        skill_outputs: list[dict[str, object]] = []
        try:
            for call in contract.skills:
                skill = self.skill_registry.get(call.skill)
                skill_started = int(self.clock())
                resolved_args = call.args
                try:
                    resolved_args = _resolve_skill_args(call.args, skill_outputs)
                    result = self.backend.execute_skill(skill, resolved_args, contract.budget)
                except BaseException as exc:
                    from capmas.backends.protocol import SkillExecutionResult

                    result = SkillExecutionResult(
                        ok=False,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                skill_finished = int(self.clock())
                status = "completed" if result.ok else "failed"
                skill_traces.append(
                    SkillTrace(
                        invocation_id=str(uuid4()),
                        skill_id=call.skill.skill_id,
                        skill_version=call.skill.version,
                        args=resolved_args,
                        started_at_ns=skill_started,
                        finished_at_ns=skill_finished,
                        status=status,
                        error_type=result.error_type,
                        error_message=result.error_message,
                        output=result.output,
                    )
                )
                skill_outputs.append(result.output)
                if not result.ok:
                    after = self.backend.observe()
                    trace = ExecutionTrace(
                        trace_id=str(uuid4()),
                        episode_id=handle.episode_id,
                        episode_epoch=handle.episode_epoch,
                        contract_id=contract.contract_id,
                        lease_id=lease.lease_id,
                        parent_scene_version=contract.parent_scene_version,
                        start_scene_version=current.scene_version,
                        end_scene_version=after.scene_version,
                        started_at_ns=started_at,
                        finished_at_ns=int(self.clock()),
                        status="failed",
                        skill_traces=tuple(skill_traces),
                        precondition_result=approval,
                        failure_class=FailureClass.EXECUTION_ERROR,
                    )
                    if after.scene_version == current.scene_version + 1:
                        self.state_store.compare_and_commit(current.scene_version, after)
                    failure = VerificationResult(
                        contract_id=contract.contract_id,
                        decision="recover",
                        checked_scene_version=after.scene_version,
                        failure_class=FailureClass.EXECUTION_ERROR,
                    )
                    return CycleResult(False, current, after, trace, failure)

            after = self.backend.observe()
            trace = ExecutionTrace(
                trace_id=str(uuid4()),
                episode_id=handle.episode_id,
                episode_epoch=handle.episode_epoch,
                contract_id=contract.contract_id,
                lease_id=lease.lease_id,
                parent_scene_version=contract.parent_scene_version,
                start_scene_version=current.scene_version,
                end_scene_version=after.scene_version,
                started_at_ns=started_at,
                finished_at_ns=int(self.clock()),
                status="completed",
                skill_traces=tuple(skill_traces),
                precondition_result=approval,
            )
            if not self.state_store.compare_and_commit(current.scene_version, after):
                raise ValueError("scene changed while action was executing")
            verification = self.verifier.commit(contract, current, after, trace)
            if not verification.passed:
                failed_trace = replace(
                    trace,
                    status="failed",
                    postcondition_result=verification,
                    failure_class=FailureClass.POSTCONDITION_FAILED,
                )
                recovery = replace(
                    verification,
                    decision="recover",
                    failure_class=FailureClass.POSTCONDITION_FAILED,
                )
                return CycleResult(False, current, after, failed_trace, recovery)
            trace = replace(trace, postcondition_result=verification)
            return CycleResult(True, current, after, trace, verification)
        finally:
            self.lease_manager.release(lease.lease_id)


def _resolve_skill_args(
    args: dict[str, object],
    outputs: list[dict[str, object]],
) -> dict[str, object]:
    def resolve(value: object) -> object:
        if isinstance(value, SkillOutputRef):
            if value.call_index < 0 or value.call_index >= len(outputs):
                raise ValueError(f"skill output reference is not available: {value.call_index}")
            current: object = outputs[value.call_index]
            for key in value.path:
                if isinstance(key, int):
                    if not isinstance(current, (list, tuple)):
                        raise ValueError("skill output reference expected a sequence")
                    current = current[key]
                else:
                    if not isinstance(current, dict):
                        raise ValueError("skill output reference expected a mapping")
                    current = current[key]
            return current
        if isinstance(value, dict):
            return {key: resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve(item) for item in value]
        if isinstance(value, tuple):
            return tuple(resolve(item) for item in value)
        return value

    return {key: resolve(value) for key, value in args.items()}
