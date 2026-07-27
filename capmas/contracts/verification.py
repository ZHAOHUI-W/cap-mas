from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PredicateReport:
    name: str
    passed: bool
    confidence: float = 1.0
    evidence: Sequence[str] = ()
    reason: str | None = None


@dataclass(frozen=True)
class VerificationResult:
    contract_id: str
    decision: str
    checked_scene_version: int
    predicate_results: tuple[PredicateReport, ...] = ()
    violated_invariants: tuple[str, ...] = ()
    failure_class: str | None = None

    @property
    def passed(self) -> bool:
        return self.decision in {"approve", "commit"} and all(
            report.passed for report in self.predicate_results
        )
