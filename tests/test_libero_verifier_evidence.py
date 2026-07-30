from capmas.contracts.action import SkillCall
from capmas.contracts.candidates import GraphCandidate
from capmas.contracts.core import SkillRef
from capmas.contracts.graph import SubgraphNodeSpec, SubgraphSpec
from capmas.contracts.scene import ObjectTrack, SceneSnapshot
from capmas.verification.libero import libero_candidate_evidence


def _scene() -> SceneSnapshot:
    return SceneSnapshot(
        "episode",
        1,
        7,
        100,
        101,
        {},
        objects=(
            ObjectTrack(
                "bowl",
                "bowl",
                (1.0, 0.0, 0.0, 0.0, 0.40, 0.00, 0.30),
                0.95,
                100,
            ),
        ),
        freshness_ms=1.0,
    )


def _candidate() -> GraphCandidate:
    node = SubgraphNodeSpec(
        "pick-action",
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
        "pick-action",
        ("pick-action",),
        ("pick-action",),
    )
    return GraphCandidate("candidate", subgraph, 7, "policy")


def test_libero_provider_composes_static_verifier_evidence() -> None:
    evidence = libero_candidate_evidence(_candidate(), _scene())

    assert evidence.verifier is not None
    assert [item.predicate for item in evidence.verifier.static_results] == [
        "track_exists:bowl"
    ]
    assert all(
        item.predicate != "object_in_gripper(bowl)"
        for item in evidence.verifier.static_results
    )
    assert all(
        item.predicate != "object_at_target(bowl,plate)"
        for item in evidence.verifier.static_results
    )
    assert "verifier" in evidence.available_metrics
    assert evidence.scene_version == 7
