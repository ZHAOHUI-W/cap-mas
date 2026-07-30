from dataclasses import replace

from capmas.agents.arbiter import CandidateArbiter
from capmas.contracts.action import SkillCall
from capmas.contracts.candidates import CandidateEvidence, GraphCandidate, subgraph_fingerprint
from capmas.contracts.core import SkillRef
from capmas.contracts.graph import CheckpointSpec, SubgraphNodeSpec, SubgraphSpec
from capmas.contracts.scene import SceneSnapshot
from capmas.verification.evidence import VerifierEvidence, VerifierPredicateEvidence


def _scene() -> SceneSnapshot:
    return SceneSnapshot("episode", 1, 4, 100, 101, {})


def _candidate() -> GraphCandidate:
    node = SubgraphNodeSpec(
        "action",
        "noop",
        skill_calls=(SkillCall(SkillRef("noop", "1.0.0"), {}),),
        postconditions=("done",),
    )
    subgraph = SubgraphSpec(
        "pick",
        "pick",
        "noop",
        (node,),
        (),
        "action",
        ("action",),
        ("action",),
        checkpoints=(CheckpointSpec("done-check", ("done",)),),
    )
    return GraphCandidate("candidate", subgraph, 4, "policy")


def _verifier(candidate: GraphCandidate, *, scene_version: int = 4, fingerprint: str | None = None) -> VerifierEvidence:
    result = VerifierPredicateEvidence("track_exists:bowl", "static", "pass", 1.0, None)
    return VerifierEvidence(
        fingerprint or subgraph_fingerprint(candidate.subgraph),
        scene_version,
        1.0,
        1.0,
        "test.verifier",
        10,
        static_results=(result,),
    )


def _with_verifier(
    candidate: GraphCandidate,
    verifier: VerifierEvidence,
    *,
    scene_version: int | None = None,
) -> GraphCandidate:
    return replace(
        candidate,
        evidence=CandidateEvidence(
            verifier_pass_rate=verifier.pass_rate,
            available_metrics=("verifier",),
            scene_version=scene_version,
            provider="aggregate",
            captured_at_ns=11,
            verifier=verifier,
        ),
    )


def test_stale_typed_verifier_scene_is_rejected_before_scoring() -> None:
    candidate = _candidate()
    result = CandidateArbiter().select(
        (_with_verifier(candidate, _verifier(candidate, scene_version=3)),),
        _scene(),
    )

    assert result.selected is None
    assert result.rejections[0].code == "STALE_EVIDENCE"
    assert "verifier" in result.rejections[0].reason


def test_cross_candidate_typed_verifier_fingerprint_is_rejected() -> None:
    candidate = _candidate()
    result = CandidateArbiter().select(
        (_with_verifier(candidate, _verifier(candidate, fingerprint="wrong")),),
        _scene(),
    )

    assert result.selected is None
    assert result.rejections[0].code == "STALE_EVIDENCE"
    assert "fingerprint" in result.rejections[0].reason


def test_bound_typed_verifier_uses_existing_scalar_score() -> None:
    candidate = _candidate()
    result = CandidateArbiter().select(
        (_with_verifier(candidate, _verifier(candidate)),),
        _scene(),
    )

    assert result.selected is not None
    assert result.score_breakdowns["candidate"]["verifier"] > 0.0
