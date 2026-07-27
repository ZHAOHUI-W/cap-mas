from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import json
from pathlib import Path
from typing import Callable, Mapping

from capmas.contracts.action import ActionContract
from capmas.contracts.agent import AgentContext, CycleHistory
from capmas.contracts.scene import SceneSnapshot
from capmas.contracts.trace import EpisodeTrace, ExecutionTrace
from capmas.contracts.verification import VerificationResult
from capmas.runtime.orchestrator import CycleResult, RuntimeOrchestrator


PolicyStep = Callable[[AgentContext], ActionContract | None]
GoalCheck = Callable[[SceneSnapshot], bool]
RecoveryStep = Callable[[ExecutionTrace, VerificationResult, AgentContext], ActionContract | None]


@dataclass(frozen=True)
class EpisodeRunResult:
    episode_trace: EpisodeTrace
    evaluator_success: bool
    committed_cycles: int
    stop_reason: str
    total_cycles: int = 0
    recovery_attempts: int = 0
    goal_reached: bool = False


class EpisodeRunner:
    """Run a CAP-MAS policy and emit only CAP-MAS episode artifacts."""

    def __init__(self, runtime: RuntimeOrchestrator) -> None:
        self.runtime = runtime

    def run(
        self,
        *,
        task_id: str,
        seed: int | None,
        policy_step: PolicyStep,
        max_cycles: int = 1,
    ) -> EpisodeRunResult:
        episode = self.runtime.backend.reset(seed=seed)
        self.runtime.start_episode(episode)
        traces = []
        committed_cycles = 0
        stop_reason = "max_cycles"

        for _ in range(max_cycles):
            scene = self.runtime.state_store.latest()
            context = AgentContext(
                task_id=task_id,
                episode_id=episode.handle.episode_id,
                episode_epoch=episode.handle.episode_epoch,
                scene=scene,
            )
            contract = policy_step(context)
            if contract is None:
                stop_reason = "policy_finished"
                break
            result: CycleResult = self.runtime.run_cycle(contract)
            traces.append(result.trace)
            if result.committed:
                committed_cycles += 1
            if self.runtime.backend.evaluator_success():
                stop_reason = "evaluator_success"
                break
            if not result.committed:
                stop_reason = "cycle_failed"
                break

        return EpisodeRunResult(
            episode_trace=EpisodeTrace(
                episode_id=episode.handle.episode_id,
                episode_epoch=episode.handle.episode_epoch,
                traces=tuple(traces),
            ),
            evaluator_success=self.runtime.backend.evaluator_success(),
            committed_cycles=committed_cycles,
            stop_reason=stop_reason,
            total_cycles=len(traces),
        )


class MultiCycleEpisodeRunner:
    """Run bounded, replanning contract cycles above the atomic orchestrator."""

    def __init__(self, runtime: RuntimeOrchestrator) -> None:
        self.runtime = runtime

    def run(
        self,
        *,
        task_id: str,
        seed: int | None,
        policy_step: PolicyStep,
        goal_check: GoalCheck,
        recovery_step: RecoveryStep | object | None = None,
        max_cycles: int = 8,
        max_recoveries: int = 2,
    ) -> EpisodeRunResult:
        episode = self.runtime.backend.reset(seed=seed)
        self.runtime.start_episode(episode)
        traces = []
        committed_cycles = 0
        recovery_attempts = 0
        history = CycleHistory()
        pending_contract: ActionContract | None = None
        stop_reason = "max_cycles"
        goal_reached = bool(goal_check(episode.initial_scene))

        if goal_reached:
            stop_reason = "task_goal_reached"
        else:
            for _ in range(max_cycles):
                scene = self.runtime.state_store.latest()
                context = AgentContext(
                    task_id=task_id,
                    episode_id=episode.handle.episode_id,
                    episode_epoch=episode.handle.episode_epoch,
                    scene=scene,
                    history=history,
                )
                contract = pending_contract or policy_step(context)
                pending_contract = None
                if contract is None:
                    stop_reason = "policy_finished"
                    break

                result = self.runtime.run_cycle(contract)
                traces.append(result.trace)
                history = CycleHistory(
                    traces=tuple(traces),
                    last_verification=result.verification,
                    current_subgoal=contract.subgoal_id,
                    recovery_count=recovery_attempts,
                )
                if result.committed:
                    committed_cycles += 1
                    if goal_check(result.after_scene):
                        goal_reached = True
                        stop_reason = "task_goal_reached"
                        break
                    continue

                if recovery_step is None or recovery_attempts >= max_recoveries:
                    stop_reason = "recovery_exhausted" if recovery_step is not None else "cycle_failed"
                    break
                recovery_attempts += 1
                recovery_context = AgentContext(
                    task_id=task_id,
                    episode_id=episode.handle.episode_id,
                    episode_epoch=episode.handle.episode_epoch,
                    scene=result.after_scene,
                    history=CycleHistory(
                        traces=tuple(traces),
                        last_verification=result.verification,
                        current_subgoal=contract.subgoal_id,
                        recovery_count=recovery_attempts,
                    ),
                )
                pending_contract = _recover(
                    recovery_step,
                    result.trace,
                    result.verification,
                    recovery_context,
                )
                if pending_contract is None:
                    stop_reason = "recovery_declined"
                    break

        return EpisodeRunResult(
            episode_trace=EpisodeTrace(
                episode_id=episode.handle.episode_id,
                episode_epoch=episode.handle.episode_epoch,
                traces=tuple(traces),
            ),
            evaluator_success=self.runtime.backend.evaluator_success(),
            committed_cycles=committed_cycles,
            stop_reason=stop_reason,
            total_cycles=len(traces),
            recovery_attempts=recovery_attempts,
            goal_reached=goal_reached,
        )


def _recover(
    recovery_step: RecoveryStep | object,
    trace: ExecutionTrace,
    verification: VerificationResult,
    context: AgentContext,
) -> ActionContract | None:
    recover = getattr(recovery_step, "recover", None)
    if callable(recover):
        return recover(trace, verification, context)
    if callable(recovery_step):
        return recovery_step(trace, verification, context)
    raise TypeError("recovery_step must be callable or expose recover()")


def to_jsonable(value: object) -> object:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return to_jsonable(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_episode_result(result: EpisodeRunResult, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(result), indent=2, sort_keys=True))
