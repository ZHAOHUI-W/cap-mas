from __future__ import annotations

from types import SimpleNamespace

from capmas.contracts.trace import ExecutionTrace
from capmas.contracts.verification import PredicateReport, VerificationResult
from capmas.evaluation.verifier_artifacts import (
    VerifierEvidence,
    collect_dynamic_verifier_artifacts,
    static_verifier_artifacts_from_arbitrations,
)
from capmas.verification.evidence import VerifierPredicateEvidence


def _trace(
    trace_id: str,
    *,
    order: int,
    fingerprint: str = "candidate-fingerprint",
    with_postcondition: bool = True,
) -> ExecutionTrace:
    result = (
        VerificationResult(
            contract_id=f"contract-{trace_id}",
            decision="commit",
            checked_scene_version=order + 10,
            predicate_results=(PredicateReport("step_done", True),),
        )
        if with_postcondition
        else None
    )
    return ExecutionTrace(
        trace_id=trace_id,
        episode_id="episode",
        episode_epoch=1,
        contract_id=f"contract-{trace_id}",
        lease_id=f"lease-{trace_id}",
        parent_scene_version=order + 9,
        start_scene_version=order + 9,
        end_scene_version=order + 10,
        started_at_ns=order,
        finished_at_ns=order + 1,
        status="completed",
        postcondition_result=result,
        metadata={
            "subgraph_id": f"subgraph-{order}",
            "node_id": f"node-{order}",
            "candidate_fingerprint": fingerprint,
        },
    )


def test_dynamic_artifacts_are_ordered_and_bound_to_trace_identity() -> None:
    artifacts = collect_dynamic_verifier_artifacts(
        (
            _trace("trace-2", order=2),
            _trace("trace-without-result", order=3, with_postcondition=False),
            _trace("trace-1", order=1),
        ),
        clock=lambda: 99,
    )

    assert [item.trace_id for item in artifacts] == ["trace-2", "trace-1"]
    assert [item.execution_order for item in artifacts] == [0, 2]
    assert artifacts[0].candidate_fingerprint == "candidate-fingerprint"
    assert artifacts[0].subgraph_id == "subgraph-2"
    assert artifacts[0].evidence.source_verification == "contract-trace-2"
    assert artifacts[0].evidence.scene_version == 12
    assert artifacts[0].evidence.captured_at_ns == 99
    assert artifacts[0].to_dict()["evidence"]["dynamic_results"][0]["status"] == "pass"


def test_dynamic_artifacts_reject_trace_without_effective_candidate_identity() -> None:
    trace = _trace("trace-1", order=1)
    trace = ExecutionTrace(
        **{
            **trace.__dict__,
            "metadata": {"subgraph_id": "subgraph-1", "node_id": "node-1"},
        }
    )

    try:
        collect_dynamic_verifier_artifacts((trace,))
    except ValueError as exc:
        assert "candidate fingerprint" in str(exc)
    else:
        raise AssertionError("missing candidate identity must not produce evidence")


def test_static_projection_and_runner_payload_keep_both_evidence_phases() -> None:
    from scripts.run_libero_b3_llm import _verifier_evidence_payload

    static = VerifierEvidence(
        candidate_fingerprint="candidate-fingerprint",
        scene_version=4,
        pass_rate=1.0,
        coverage=1.0,
        provider="predicate_verifier.static",
        captured_at_ns=10,
        static_results=(
            VerifierPredicateEvidence(
                "track_exists:bowl",
                "static",
                "pass",
                1.0,
                None,
            ),
        ),
    )
    arbitration = SimpleNamespace(
        selected=SimpleNamespace(
            candidate_id="candidate-1",
            subgraph=SimpleNamespace(subgraph_id="subgraph-1"),
            evidence=SimpleNamespace(verifier=static),
        )
    )

    projected = static_verifier_artifacts_from_arbitrations({"subgraph-1": arbitration})

    assert projected[0]["candidate_id"] == "candidate-1"
    assert projected[0]["evidence"]["static_results"][0]["status"] == "pass"

    payload = _verifier_evidence_payload(
        {"subgraph-1": arbitration},
        SimpleNamespace(traces=(_trace("trace-1", order=1),)),
    )
    assert len(payload["static"]) == 1
    assert payload["dynamic"][0]["evidence"]["source_verification"] == "contract-trace-1"
