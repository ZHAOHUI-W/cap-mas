from capmas.backends.capx import CAPXTypedSkill
from capmas.contracts.action import SkillCall
from capmas.contracts.core import SkillRef
from capmas.contracts.graph import MotionIntent, SubgraphNodeSpec, SubgraphSpec
from capmas.contracts.scene import ObjectTrack, SceneSnapshot
from capmas.graph.condition_defaults import SkillConditionEnricher
from capmas.skills.registry import SkillRegistry


def _scene() -> SceneSnapshot:
    return SceneSnapshot(
        "episode",
        1,
        4,
        100,
        100,
        {"gripper_opening": 0.8},
        objects=(
            ObjectTrack("bowl", "bowl", (1.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3), 0.9, 100),
        ),
    )


def _registry() -> SkillRegistry:
    registry = SkillRegistry()
    registry.register(
        SkillRef("sample_grasp_pose", "1.0.0"),
        CAPXTypedSkill(
            SkillRef("sample_grasp_pose", "1.0.0"),
            lambda object_name: object_name,
            default_postconditions=("scene_fresh(2000)",),
        ),
    )
    registry.register(
        SkillRef("close_gripper", "1.0.0"),
        CAPXTypedSkill(
            SkillRef("close_gripper", "1.0.0"),
            lambda: None,
            default_postconditions=("gripper_closed()",),
        ),
    )
    return registry


def _subgraph(
    *,
    preconditions: tuple[str, ...] = (),
    postconditions: tuple[str, ...] = (),
    skill_id: str = "sample_grasp_pose",
    args: dict[str, object] | None = None,
    motion_intent: MotionIntent | None = None,
) -> SubgraphSpec:
    node = SubgraphNodeSpec(
        "action",
        "pick bowl",
        skill_calls=(SkillCall(SkillRef(skill_id, "1.0.0"), args or {"object_name": "bowl"}),),
        preconditions=preconditions,
        postconditions=postconditions,
        motion_intent=motion_intent,
    )
    return SubgraphSpec(
        "pick",
        "pick",
        "pick bowl",
        (node,),
        (),
        "action",
        ("action",),
        ("action",),
    )


def test_balanced_injects_freshness_and_skill_postcondition_only() -> None:
    enriched = SkillConditionEnricher(_registry()).enrich(
        _subgraph(), _scene(), "balanced"
    )

    node = enriched.node("action")
    assert node.preconditions == ("scene_fresh(2000)",)
    assert node.postconditions == ("scene_fresh(2000)",)


def test_safety_adds_exact_grounding_track_and_close_predicate() -> None:
    subgraph = _subgraph(
        skill_id="close_gripper",
        args={},
        motion_intent=MotionIntent("grasp", object_track_id="bowl"),
    )

    enriched = SkillConditionEnricher(_registry()).enrich(subgraph, _scene(), "safety")

    node = enriched.node("action")
    assert node.preconditions == ("scene_fresh(2000)", "track_exists:bowl")
    assert node.postconditions == ("gripper_closed()",)


def test_explicit_predicates_survive_and_enrichment_is_idempotent() -> None:
    subgraph = _subgraph(
        preconditions=("track_exists:bowl",),
        postconditions=("object_in_gripper(bowl)",),
    )
    enricher = SkillConditionEnricher(_registry())

    enriched = enricher.enrich(subgraph, _scene(), "safety")

    assert enriched.node("action").postconditions == (
        "object_in_gripper(bowl)",
        "scene_fresh(2000)",
    )
    assert enricher.enrich(enriched, _scene(), "safety") == enriched


def test_unresolved_object_name_does_not_create_guessed_track_fact() -> None:
    enriched = SkillConditionEnricher(_registry()).enrich(
        _subgraph(args={"object_name": "the bowl on the left"}),
        _scene(),
        "safety",
    )

    assert enriched.node("action").preconditions == ("scene_fresh(2000)",)
