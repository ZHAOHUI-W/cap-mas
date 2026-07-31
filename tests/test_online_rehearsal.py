from __future__ import annotations

from capmas.agents.arbiter import CandidateArbiter
from capmas.contracts.action import SkillCall
from capmas.contracts.core import SkillRef
from capmas.contracts.graph import CheckpointSpec, SubgraphNodeSpec, SubgraphSpec
from capmas.contracts.scene import SceneSnapshot
from capmas.contracts.candidates import GraphCandidate, subgraph_fingerprint
from capmas.evaluation.rehearsal_evidence import RehearsalEvidence
from capmas.evaluation.online_rehearsal import select_with_rehearsal


def _scene() -> SceneSnapshot:
    return SceneSnapshot("episode", 1, 4, 1, 1, {})


def _candidate(candidate_id: str, description: str) -> GraphCandidate:
    node_id = f"{candidate_id}-action"
    node = SubgraphNodeSpec(
        node_id=node_id,
        description=description,
        skill_calls=(SkillCall(SkillRef("noop", "1.0.0"), {}),),
        postconditions=(f"done({candidate_id})",),
        proposed_by=description,
    )
    subgraph = SubgraphSpec(
        subgraph_id="first",
        subgoal_id="first",
        description=description,
        nodes=(node,),
        edges=(),
        entry_node=node_id,
        success_nodes=(node_id,),
        failure_nodes=(node_id,),
        checkpoints=(CheckpointSpec(f"{candidate_id}-checkpoint", node.postconditions),),
    )
    return GraphCandidate(
        candidate_id=candidate_id,
        subgraph=subgraph,
        parent_scene_version=4,
        producer_agent=description,
    )


def _evidence(candidate: GraphCandidate, score: float) -> RehearsalEvidence:
    return RehearsalEvidence(
        candidate_id=candidate.candidate_id,
        candidate_fingerprint=subgraph_fingerprint(candidate.subgraph),
        seed=1,
        scene_version=4,
        success=score == 1.0,
        score=score,
        latency_ms=10.0,
    )


def test_disabled_mode_does_not_call_provider() -> None:
    candidates = (_candidate("candidate-a", "a"), _candidate("candidate-b", "b"))
    calls = 0

    def provider(_candidates, _scene):
        nonlocal calls
        calls += 1
        return {}

    report = select_with_rehearsal(
        candidates,
        _scene(),
        CandidateArbiter(),
        mode="disabled",
        provider=provider,
    )

    assert calls == 0
    assert report.evidence_aware is None
    assert report.live == report.baseline


def test_shadow_mode_keeps_baseline_live_and_reports_hypothetical_change() -> None:
    candidates = (_candidate("candidate-a", "a"), _candidate("candidate-b", "b"))
    baseline = CandidateArbiter().select(candidates, _scene())
    preferred = candidates[0] if baseline.selected is candidates[1] else candidates[1]

    report = select_with_rehearsal(
        candidates,
        _scene(),
        CandidateArbiter(),
        mode="shadow",
        provider=lambda items, _scene: {
            candidate.candidate_id: _evidence(candidate, 1.0 if candidate is preferred else 0.0)
            for candidate in items
        },
    )

    assert report.live == report.baseline
    assert report.evidence_aware is not None
    assert report.would_change_selection is True
    assert report.live.selected is report.baseline.selected


def test_online_mode_uses_evidence_aware_winner_once() -> None:
    candidates = (_candidate("candidate-a", "a"), _candidate("candidate-b", "b"))
    baseline = CandidateArbiter().select(candidates, _scene())
    preferred = candidates[0] if baseline.selected is candidates[1] else candidates[1]
    calls = 0

    def provider(items, _scene):
        nonlocal calls
        calls += 1
        return {
            candidate.candidate_id: _evidence(candidate, 1.0 if candidate is preferred else 0.0)
            for candidate in items
        }

    report = select_with_rehearsal(
        candidates,
        _scene(),
        CandidateArbiter(),
        mode="online_bounded",
        provider=provider,
    )

    assert calls == 1
    assert report.evidence_aware is not None
    assert report.live == report.evidence_aware
    assert report.live.selected is not None
    assert report.live.selected.candidate_id == preferred.candidate_id
    assert report.would_change_selection is True


def test_invalid_evidence_is_rejected_and_online_falls_back() -> None:
    candidate = _candidate("candidate-a", "a")

    report = select_with_rehearsal(
        (candidate,),
        _scene(),
        CandidateArbiter(),
        mode="online_bounded",
        provider=lambda items, _scene: {
            items[0].candidate_id: RehearsalEvidence(
                candidate_id=items[0].candidate_id,
                candidate_fingerprint="wrong",
                seed=1,
                scene_version=4,
                success=True,
                score=1.0,
                latency_ms=1.0,
            )
        },
    )

    assert report.live == report.baseline
    assert report.fallback_reason is None
    assert report.attached_candidate_ids == ()
    assert report.evidence_rejections


def test_provider_failure_falls_back_without_aborting() -> None:
    candidates = (_candidate("candidate-a", "a"),)

    def provider(_candidates, _scene):
        raise RuntimeError("rehearsal unavailable")

    report = select_with_rehearsal(
        candidates,
        _scene(),
        CandidateArbiter(),
        mode="online_bounded",
        provider=provider,
    )

    assert report.live == report.baseline
    assert report.evidence_aware is None
    assert report.fallback_reason == "provider_error: RuntimeError: rehearsal unavailable"

