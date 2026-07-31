"""Serializable verifier artifacts for post-execution evidence."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any

from capmas.contracts.trace import ExecutionTrace
from capmas.verification.evidence import VerifierEvidence, verifier_evidence_from_result


@dataclass(frozen=True)
class DynamicVerifierArtifact:
    """One postcondition result bound to the action trace that produced it."""

    trace_id: str
    subgraph_id: str
    node_id: str
    execution_order: int
    candidate_fingerprint: str
    evidence: VerifierEvidence

    def __post_init__(self) -> None:
        if not self.trace_id:
            raise ValueError("dynamic verifier trace id must not be empty")
        if not self.subgraph_id:
            raise ValueError("dynamic verifier subgraph id must not be empty")
        if not self.node_id:
            raise ValueError("dynamic verifier node id must not be empty")
        if self.execution_order < 0:
            raise ValueError("dynamic verifier execution order must not be negative")
        if self.candidate_fingerprint != self.evidence.candidate_fingerprint:
            raise ValueError(
                "dynamic verifier artifact fingerprint does not match evidence"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "trace_id": self.trace_id,
            "subgraph_id": self.subgraph_id,
            "node_id": self.node_id,
            "execution_order": self.execution_order,
            "candidate_fingerprint": self.candidate_fingerprint,
            "evidence": self.evidence.to_dict(),
        }


def collect_dynamic_verifier_artifacts(
    traces: Sequence[ExecutionTrace],
    *,
    provider: str = "predicate_verifier.dynamic",
    clock: Any = None,
) -> tuple[DynamicVerifierArtifact, ...]:
    """Extract dynamic verifier evidence in the exact execution-trace order.

    Traces without a postcondition result are intentionally omitted, but their
    original trace index is retained as ``execution_order``. This preserves the
    ordering relationship between evidence and action execution in replay.
    """

    kwargs = {"provider": provider}
    if clock is not None:
        kwargs["clock"] = clock
    artifacts: list[DynamicVerifierArtifact] = []
    for execution_order, trace in enumerate(traces):
        result = trace.postcondition_result
        if result is None:
            continue
        metadata = trace.metadata
        subgraph_id = _required_metadata(metadata, "subgraph_id", trace.trace_id)
        node_id = _required_metadata(metadata, "node_id", trace.trace_id)
        candidate_fingerprint = _required_metadata(
            metadata,
            "candidate_fingerprint",
            trace.trace_id,
        )
        evidence = verifier_evidence_from_result(
            candidate_fingerprint,
            result,
            **kwargs,
        )
        artifacts.append(
            DynamicVerifierArtifact(
                trace_id=trace.trace_id,
                subgraph_id=subgraph_id,
                node_id=node_id,
                execution_order=execution_order,
                candidate_fingerprint=candidate_fingerprint,
                evidence=evidence,
            )
        )
    return tuple(artifacts)


def static_verifier_artifacts_from_arbitrations(
    arbitrations: object,
) -> tuple[dict[str, object], ...]:
    """Project selected candidate-bound static evidence from arbitration objects."""

    artifacts: list[dict[str, object]] = []
    _collect_static(arbitrations, artifacts)
    return tuple(artifacts)


def _collect_static(value: object, output: list[dict[str, object]]) -> None:
    if isinstance(value, Mapping):
        for nested in value.values():
            _collect_static(nested, output)
        return
    selected = getattr(value, "selected", None)
    if selected is None:
        return
    evidence = getattr(getattr(selected, "evidence", None), "verifier", None)
    if evidence is None:
        return
    output.append(
        {
            "candidate_id": selected.candidate_id,
            "subgraph_id": selected.subgraph.subgraph_id,
            "candidate_fingerprint": evidence.candidate_fingerprint,
            "evidence": evidence.to_dict(),
        }
    )


def _required_metadata(metadata: dict[str, object], key: str, trace_id: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value:
        label = key.replace("_", " ")
        raise ValueError(
            f"dynamic verifier trace {trace_id!r} is missing candidate {label} metadata"
        )
    return value


__all__ = [
    "DynamicVerifierArtifact",
    "collect_dynamic_verifier_artifacts",
    "static_verifier_artifacts_from_arbitrations",
]
