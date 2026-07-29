from __future__ import annotations

import inspect
from typing import Any

from capmas.contracts.action import ActionContract
from capmas.contracts.core import SkillRef
from capmas.skills.protocol import TypedSkill


class SkillValidationError(ValueError):
    """A contract argument error with enough context for candidate diagnostics."""

    def __init__(
        self,
        message: str,
        *,
        skill_id: str,
        call_index: int,
        actual_args: dict[str, object],
        expected_signature: str | None = None,
        missing_arguments: tuple[str, ...] = (),
        unexpected_arguments: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.skill_id = skill_id
        self.call_index = call_index
        self.actual_args = dict(actual_args)
        self.expected_signature = expected_signature
        self.missing_arguments = missing_arguments
        self.unexpected_arguments = unexpected_arguments


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[SkillRef, TypedSkill] = {}

    def register(self, reference: SkillRef, skill: TypedSkill) -> None:
        if reference != SkillRef(skill.skill_id, skill.version):
            raise ValueError("skill reference does not match implementation")
        self._skills[reference] = skill

    def get(self, reference: SkillRef) -> TypedSkill:
        try:
            return self._skills[reference]
        except KeyError as exc:
            raise ValueError(f"unknown skill: {reference.skill_id}@{reference.version}") from exc

    def has(self, reference: SkillRef) -> bool:
        return reference in self._skills

    def validate_contract(self, contract: ActionContract) -> None:
        if not contract.skills:
            raise ValueError("action contract has no skills")
        for call_index, call in enumerate(contract.skills):
            skill = self.get(call.skill)
            try:
                skill.validate_args(call.args)
            except Exception as exc:
                signature = getattr(skill, "_signature", None)
                expected_signature = str(signature) if isinstance(signature, inspect.Signature) else None
                details, missing, unexpected = _signature_error_details(signature, call.args)
                message = f"invalid arguments for {call.skill.skill_id}"
                if details:
                    message += f": {details}"
                raise SkillValidationError(
                    message,
                    skill_id=call.skill.skill_id,
                    call_index=call_index,
                    actual_args=call.args,
                    expected_signature=expected_signature,
                    missing_arguments=missing,
                    unexpected_arguments=unexpected,
                ) from exc

    def snapshot_version(self) -> str:
        refs = sorted(f"{ref.skill_id}@{ref.version}" for ref in self._skills)
        return ",".join(refs)

    def argument_names(self) -> tuple[str, ...]:
        """Return the union of registered callable parameter names.

        CAP-MAS uses this allowlist when constructing strict provider schemas.
        Skills remain responsible for validating the values and required
        parameters at execution time; this method only describes the wire
        field names needed by structured output providers.
        """
        names: set[str] = set()
        for skill in self._skills.values():
            signature = getattr(skill, "_signature", None)
            if isinstance(signature, inspect.Signature):
                names.update(
                    parameter.name
                    for parameter in signature.parameters.values()
                    if parameter.kind
                    in {
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    }
                )
            declared = getattr(skill, "argument_names", ())
            if callable(declared):
                declared = declared()
            if isinstance(declared, (tuple, list, set, frozenset)):
                names.update(str(name) for name in declared if name)
        return tuple(sorted(names))

    def argument_names_for(self, reference: SkillRef) -> tuple[str, ...]:
        """Return the callable argument allowlist for one registered skill."""
        skill = self.get(reference)
        signature = getattr(skill, "_signature", None)
        if isinstance(signature, inspect.Signature):
            return tuple(
                parameter.name
                for parameter in signature.parameters.values()
                if parameter.kind
                in {
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                }
            )
        declared = getattr(skill, "argument_names", ())
        if callable(declared):
            declared = declared()
        if isinstance(declared, (tuple, list, set, frozenset)):
            return tuple(str(name) for name in declared if name)
        return ()

    def argument_schemas(self) -> dict[str, dict[str, object]]:
        """Return conservative JSON schemas inferred from skill signatures."""
        schemas: dict[str, dict[str, object]] = {}
        for skill in self._skills.values():
            signature = getattr(skill, "_signature", None)
            if not isinstance(signature, inspect.Signature):
                continue
            for parameter in signature.parameters.values():
                if parameter.kind not in {
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                }:
                    continue
                inferred = _annotation_schema(parameter.annotation)
                previous = schemas.get(parameter.name)
                if previous is None:
                    schemas[parameter.name] = inferred
                elif previous != inferred:
                    # A shared parameter name with conflicting signatures is
                    # deliberately widened only to primitive values. Runtime
                    # TypedSkill validation remains authoritative.
                    schemas[parameter.name] = {"type": "string"}
        return schemas


def _annotation_schema(annotation: object) -> dict[str, object]:
    """Map common CAP-X callable annotations to provider-safe JSON schemas."""
    if annotation is inspect.Parameter.empty:
        return {"type": "string"}
    if annotation is str or _annotation_name(annotation) == "str":
        return {"type": "string"}
    if annotation is bool or _annotation_name(annotation) == "bool":
        return {"type": "boolean"}
    if annotation is int or _annotation_name(annotation) == "int":
        return {"type": "integer"}
    if annotation is float or _annotation_name(annotation) == "float":
        return {"type": "number"}
    if _annotation_name(annotation) in {"ndarray", "array", "list", "tuple"}:
        return {"type": "array", "items": {"type": "number"}}
    return {"type": "string"}


def _signature_error_details(
    signature: inspect.Signature | None,
    args: dict[str, object],
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    if not isinstance(signature, inspect.Signature):
        return "actual_keys=" + repr(sorted(args)), (), ()
    parameters = signature.parameters
    required = [
        parameter.name
        for parameter in parameters.values()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    ]
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    missing = sorted(name for name in required if name not in args)
    unexpected = (
        []
        if accepts_kwargs
        else sorted(name for name in args if name not in parameters)
    )
    return (
        f"expected_signature={signature}; "
        f"missing={missing!r}; unexpected={unexpected!r}; "
        f"actual_keys={sorted(args)!r}"
    ), tuple(missing), tuple(unexpected)


def _annotation_name(annotation: object) -> str:
    name = getattr(annotation, "__name__", None)
    if isinstance(name, str):
        return name.lower()
    text = str(annotation).lower()
    if "ndarray" in text:
        return "ndarray"
    return text.rsplit(".", 1)[-1]
