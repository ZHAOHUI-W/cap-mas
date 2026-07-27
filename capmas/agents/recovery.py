from __future__ import annotations

from typing import Callable

from capmas.contracts.action import ActionContract
from capmas.contracts.agent import AgentContext, RecoveryAgent
from capmas.contracts.verification import VerificationResult


class CallableRecoveryAgent(RecoveryAgent):
    def __init__(self, replanner: Callable[[object, VerificationResult, AgentContext], ActionContract | None]) -> None:
        self._replanner = replanner

    def recover(self, trace: object, verification: VerificationResult, context: AgentContext) -> ActionContract | None:
        return self._replanner(trace, verification, context)
