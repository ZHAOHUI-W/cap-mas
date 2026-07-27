from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from capmas.contracts.action import ActionContract
from capmas.contracts.scene import SceneSnapshot
from capmas.runtime.orchestrator import CycleResult, RuntimeOrchestrator


class Scheduler(Protocol):
    def dispatch(self, contract: ActionContract, scene: SceneSnapshot) -> CycleResult: ...


@dataclass
class FixedGraphScheduler:
    orchestrator: RuntimeOrchestrator

    def dispatch(self, contract: ActionContract, scene: SceneSnapshot) -> CycleResult:
        if scene.scene_version != self.orchestrator.state_store.latest().scene_version:
            raise ValueError("scheduler received a stale scene")
        return self.orchestrator.run_cycle(contract)
