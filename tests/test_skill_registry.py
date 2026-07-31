from capmas.backends.capx import CAPXTypedSkill
from capmas.contracts.core import SkillRef
from capmas.skills.registry import SkillRegistry


def test_registry_returns_declared_default_postconditions() -> None:
    reference = SkillRef("close_gripper", "1.0.0")
    registry = SkillRegistry()
    registry.register(
        reference,
        CAPXTypedSkill(
            reference,
            lambda: None,
            default_postconditions=("gripper_closed()",),
        ),
    )

    assert registry.default_postconditions(reference) == ("gripper_closed()",)


def test_registry_keeps_custom_skills_without_metadata_compatible() -> None:
    class CustomSkill:
        skill_id = "custom"
        version = "1.0.0"

        def validate_args(self, args: dict[str, object]) -> None:
            del args

        def execute(self, args, budget):
            del args, budget
            raise NotImplementedError

    reference = SkillRef("custom", "1.0.0")
    registry = SkillRegistry()
    registry.register(reference, CustomSkill())

    assert registry.default_postconditions(reference) == ()
