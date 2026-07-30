from __future__ import annotations

from capmas.contracts.action import SkillCall
from capmas.contracts.candidates import GraphCandidate, subgraph_fingerprint
from capmas.contracts.core import SkillRef
from capmas.contracts.graph import CheckpointSpec, SubgraphNodeSpec, SubgraphSpec
from capmas.contracts.scene import SceneSnapshot
from capmas.evaluation.rehearsal_evidence import RehearsalEvidence
from capmas.evaluation.shadow_arbiter import run_shadow_arbitration


def _candidate(candidate_id: str, description: str) -> GraphCandidate:
    node = SubgraphNodeSpec(
        node_id="act",
        description=description,
        skill_calls=(SkillCall(SkillRef("noop", "1.0.0"), {}),),
        postconditions=("scene_advanced",),
        proposed_by="policy",
    )
    subgraph = SubgraphSpec(
        subgraph_id="pick",
        subgoal_id="pick",
        description=description,
        nodes=(node,),
        edges=(),
        entry_node="act",
        success_nodes=("act",),
        failure_nodes=("act",),
        checkpoints=(CheckpointSpec("check", ("scene_advanced",)),),
    )
    return GraphCandidate(candidate_id, subgraph, 0, "policy")


def _graph_evidence(
    candidate: GraphCandidate,
    score: float,
    *,
    fingerprint: str | None = None,
) -> RehearsalEvidence:
    return RehearsalEvidence(
        candidate_id=candidate.candidate_id,
        candidate_fingerprint="full-graph-fingerprint",
        seed=1,
        scene_version=0,
        success=score == 1.0,
        score=score,
        latency_ms=3.0,
        fingerprint_scope="graph",
        arbiter_subgraph_id=candidate.subgraph.subgraph_id,
        arbiter_fingerprint=fingerprint or subgraph_fingerprint(candidate.subgraph),
    )


def test_shadow_arbitration_reports_hypothetical_winner_without_mutating_live_candidates():
    candidate_a = _candidate("candidate-a", "a")
    candidate_b = _candidate("candidate-b", "b")
    scene = SceneSnapshot("episode", 1, 0, 1, 2, {})

    report = run_shadow_arbitration(
        (candidate_a, candidate_b),
        {
            "candidate-a": _graph_evidence(candidate_a, 1.0),
            "candidate-b": _graph_evidence(candidate_b, 0.0),
        },
        scene,
    )

    assert report.baseline.selected is not None
    assert report.shadow.selected is not None
    assert report.baseline.selected.candidate_id == "candidate-b"
    assert report.shadow.selected.candidate_id == "candidate-a"
    assert report.baseline.selection_basis == "confidence_fallback"
    assert report.shadow.selection_basis == "evidence_score"
    assert report.would_change_selection is True
    assert report.physical_execution_required is False
    assert candidate_a.evidence is None
    assert candidate_b.evidence is None


def test_shadow_arbitration_keeps_baseline_on_unavailable_or_mismatched_evidence():
    candidate_a = _candidate("candidate-a", "a")
    candidate_b = _candidate("candidate-b", "b")
    scene = SceneSnapshot("episode", 1, 0, 1, 2, {})
    mismatched = _graph_evidence(candidate_a, 1.0, fingerprint="wrong")

    report = run_shadow_arbitration(
        (candidate_a, candidate_b),
        {"candidate-a": mismatched},
        scene,
    )

    assert report.baseline.selected is not None
    assert report.baseline.selected.candidate_id == "candidate-b"
    assert report.evidence_rejections
    assert report.physical_execution_required is False
    assert candidate_a.evidence is None
    assert candidate_b.evidence is None
