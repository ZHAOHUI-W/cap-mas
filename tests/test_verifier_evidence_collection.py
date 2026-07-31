from dataclasses import replace

import pytest

from capmas.contracts.action import SkillCall
from capmas.contracts.candidates import GraphCandidate, subgraph_fingerprint
from capmas.contracts.core import SkillRef
from capmas.contracts.graph import CheckpointSpec, SubgraphNodeSpec, SubgraphSpec
from capmas.contracts.scene import SceneSnapshot
from capmas.contracts.verification import PredicateReport, VerificationResult
from capmas.evaluation.evidence_contracts import EvidenceCompatibilityError
from capmas.verification.evidence import (
    collect_static_verifier_evidence,
    verifier_evidence_from_result,
)
from capmas.verification.libero import compile_time_preconditions
from capmas.verification.predicates import PredicateBasedVerifier


def _scene(version: int = 4) -> SceneSnapshot:
    return SceneSnapshot("episode", 1, version, 100, 101, {})


def _candidate(parent_scene_version: int = 4) -> GraphCandidate:
    node = SubgraphNodeSpec(
        "action",
        "pick bowl",
        skill_calls=(SkillCall(SkillRef("noop", "1.0.0"), {}),),
        preconditions=("track_exists:bowl", "object_in_gripper(bowl)"),
        postconditions=("object_at_target(bowl,plate)",),
    )
    subgraph = SubgraphSpec(
        "pick",
        "pick",
        "pick bowl",
        (node,),
        (),
        "action",
        ("action",),
        ("action",),
        checkpoints=(CheckpointSpec("post-check", node.postconditions),),
    )
    return GraphCandidate(
        "candidate",
        subgraph,
        parent_scene_version,
        "policy",
    )


def test_static_collection_only_reads_selected_preconditions() -> None:
    candidate = _candidate()
    evidence = collect_static_verifier_evidence(
        candidate,
        _scene(),
        PredicateBasedVerifier(),
        predicate_selector=lambda name: name.startswith("track_exists:"),
        clock=lambda: 999,
    )

    assert [item.predicate for item in evidence.static_results] == ["track_exists:bowl"]
    assert evidence.dynamic_results == ()
    assert evidence.candidate_fingerprint == subgraph_fingerprint(candidate.subgraph)
    assert evidence.scene_version == 4
    assert evidence.captured_at_ns == 999


def test_static_collection_rejects_stale_candidate_scene() -> None:
    with pytest.raises(EvidenceCompatibilityError):
        collect_static_verifier_evidence(
            _candidate(parent_scene_version=3),
            _scene(),
            PredicateBasedVerifier(),
        )


def test_dynamic_conversion_preserves_result_provenance_and_three_state() -> None:
    result = VerificationResult(
        "contract-7",
        "recover",
        5,
        (
            PredicateReport("gripper_closed()", False, 0.8, (), "gripper is not closed"),
            PredicateReport(
                "object_at_target(bowl,plate)",
                False,
                0.0,
                (),
                "track not found",
            ),
        ),
    )

    evidence = verifier_evidence_from_result("fp", result, clock=lambda: 20)

    assert evidence.source_verification == "contract-7"
    assert evidence.scene_version == 5
    assert [item.status for item in evidence.dynamic_results] == ["fail", "unknown"]
    assert evidence.pass_rate == 0.0
    assert evidence.coverage == 0.5


def test_static_collection_reports_scene_freshness_coverage() -> None:
    original = _candidate()
    node = original.subgraph.node("action")
    candidate = replace(
        original,
        subgraph=replace(
            original.subgraph,
            nodes=(replace(node, preconditions=("scene_fresh(2000)",)),),
        ),
    )
    scene = _scene()
    evidence = collect_static_verifier_evidence(
        candidate,
        scene,
        PredicateBasedVerifier(clock=lambda: scene.publish_timestamp_ns),
        predicate_selector=lambda predicate: predicate in compile_time_preconditions(
            (predicate,)
        ),
        clock=lambda: 999,
    )

    assert [item.predicate for item in evidence.static_results] == [
        "scene_fresh(2000)"
    ]
    assert evidence.coverage == 1.0
