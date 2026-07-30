from capmas.agents.arbiter import CandidateArbiter
from capmas.contracts.action import SkillCall
from capmas.contracts.candidates import GraphCandidate, subgraph_fingerprint
from capmas.contracts.core import SkillRef
from capmas.contracts.graph import CheckpointSpec, SubgraphNodeSpec, SubgraphSpec
from capmas.contracts.scene import SceneSnapshot
from capmas.evaluation.rehearsal_arbiter import merge_rehearsal_evidence
from capmas.evaluation.rehearsal_evidence import RehearsalEvidence


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


def _evidence(candidate: GraphCandidate, score: float) -> RehearsalEvidence:
    return RehearsalEvidence(
        candidate_id=candidate.candidate_id,
        candidate_fingerprint=subgraph_fingerprint(candidate.subgraph),
        seed=1,
        scene_version=0,
        success=score == 1.0,
        score=score,
        latency_ms=3.0,
    )


def test_shadow_rehearsal_is_recorded_without_changing_arbiter_input():
    candidate = _candidate("candidate-a", "a")

    shadow = merge_rehearsal_evidence(
        candidate,
        _evidence(candidate, 1.0),
        include_in_arbiter=False,
    )

    assert shadow.evidence is None


def test_online_rehearsal_evidence_changes_arbiter_score_and_winner():
    weaker = _candidate("candidate-a", "a")
    stronger = _candidate("candidate-b", "b")
    weaker = merge_rehearsal_evidence(weaker, _evidence(weaker, 0.0), include_in_arbiter=True)
    stronger = merge_rehearsal_evidence(stronger, _evidence(stronger, 1.0), include_in_arbiter=True)

    result = CandidateArbiter().select(
        (weaker, stronger),
        SceneSnapshot("episode", 1, 0, 1, 2, {}),
    )

    assert result.selected is not None
    assert result.selected.candidate_id == "candidate-b"
    assert result.selection_basis == "evidence_score"
    assert result.score_breakdowns["candidate-b"]["rehearsal"] > 0.0
