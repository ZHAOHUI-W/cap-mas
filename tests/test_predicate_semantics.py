from capmas.contracts.scene import ObjectTrack, SceneSnapshot
from capmas.verification.predicates import PredicateBasedVerifier


def _scene(gripper_opening: float = 0.9) -> SceneSnapshot:
    return SceneSnapshot(
        "episode",
        1,
        2,
        100,
        101,
        {
            "ee_position": (0.40, 0.00, 0.30),
            "gripper_opening": gripper_opening,
        },
        objects=(
            ObjectTrack(
                "bowl",
                "bowl",
                (1.0, 0.0, 0.0, 0.0, 0.40, 0.00, 0.30),
                0.95,
                100,
            ),
        ),
    )


def test_object_in_gripper_is_independent_from_gripper_closure() -> None:
    reports = PredicateBasedVerifier().evaluate_predicates(
        ("object_in_gripper(bowl)", "object_held(bowl)", "gripper_open()"),
        _scene(),
    )

    assert reports[0].passed
    assert not reports[1].passed
    assert reports[2].passed


def test_object_held_requires_closed_gripper() -> None:
    report = PredicateBasedVerifier().evaluate_predicates(
        ("object_held(bowl)",),
        _scene(gripper_opening=0.1),
    )[0]

    assert report.passed
