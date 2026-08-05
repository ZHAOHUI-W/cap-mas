"""Auditable, reset-time MuJoCo layout variants for frozen evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

import numpy as np


def _vector(value: object, *, name: str, size: int) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{name} must contain exactly {size} numbers")
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only numbers") from exc


@dataclass(frozen=True)
class LayoutBodyTransform:
    """A translation applied to a free-jointed MuJoCo body after reset."""

    body_name: str
    translation_delta_xyz: tuple[float, float, float]
    joint_name: str | None = None

    def __post_init__(self) -> None:
        if not self.body_name.strip():
            raise ValueError("layout body_name must not be empty")
        if len(self.translation_delta_xyz) != 3:
            raise ValueError("layout translation delta must contain three values")
        if self.joint_name is not None and not self.joint_name.strip():
            raise ValueError("layout joint_name must not be empty")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "LayoutBodyTransform":
        body_name = value.get("body_name")
        if not isinstance(body_name, str):
            raise ValueError("layout transform body_name must be a string")
        joint_name = value.get("joint_name")
        if joint_name is not None and not isinstance(joint_name, str):
            raise ValueError("layout transform joint_name must be a string")
        return cls(
            body_name=body_name,
            translation_delta_xyz=_vector(
                value.get("translation_delta_xyz"),
                name="layout translation_delta_xyz",
                size=3,
            ),
            joint_name=joint_name,
        )


@dataclass(frozen=True)
class LayoutVariant:
    """One immutable layout perturbation declared by an OOD case manifest."""

    variant_id: str
    layout_family: str
    transforms: tuple[LayoutBodyTransform, ...]
    generator_version: str = "capmas-layout-v1"

    def __post_init__(self) -> None:
        for name in ("variant_id", "layout_family", "generator_version"):
            if not getattr(self, name).strip():
                raise ValueError(f"layout {name} must not be empty")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "LayoutVariant":
        variant_id = value.get("variant_id")
        layout_family = value.get("layout_family")
        generator_version = value.get("generator_version", "capmas-layout-v1")
        transforms = value.get("transforms", ())
        if not isinstance(variant_id, str) or not variant_id.strip():
            raise ValueError("layout variant_id must be a non-empty string")
        if not isinstance(layout_family, str) or not layout_family.strip():
            raise ValueError("layout layout_family must be a non-empty string")
        if not isinstance(generator_version, str) or not generator_version.strip():
            raise ValueError("layout generator_version must be a non-empty string")
        if not isinstance(transforms, (list, tuple)):
            raise ValueError("layout transforms must be a list")
        parsed: list[LayoutBodyTransform] = []
        for index, transform in enumerate(transforms):
            if not isinstance(transform, Mapping):
                raise ValueError(f"layout transforms[{index}] must be an object")
            parsed.append(LayoutBodyTransform.from_mapping(transform))
        if len({item.body_name for item in parsed}) != len(parsed):
            raise ValueError("layout transforms must not repeat a body")
        return cls(
            variant_id=variant_id,
            layout_family=layout_family,
            transforms=tuple(parsed),
            generator_version=generator_version,
        )


@dataclass(frozen=True)
class LayoutTransformReport:
    body_name: str
    joint_name: str
    before_position: tuple[float, float, float]
    after_position: tuple[float, float, float]
    translation_delta_xyz: tuple[float, float, float]


@dataclass(frozen=True)
class LayoutApplicationReport:
    variant_id: str
    layout_family: str
    generator_version: str
    applied: bool
    state_fingerprint: str
    transforms: tuple[LayoutTransformReport, ...]


def layout_variant_from_mapping(value: Mapping[str, object]) -> LayoutVariant:
    return LayoutVariant.from_mapping(value)


def layout_variant_sha256(variant: LayoutVariant) -> str:
    payload = asdict(variant)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_sim(environment: object) -> object:
    if hasattr(environment, "model") and hasattr(environment, "data"):
        return environment
    handle = getattr(environment, "handle", None)
    nested = getattr(handle, "env", None)
    sim = getattr(nested, "sim", None)
    if sim is not None:
        return sim
    sim = getattr(environment, "sim", None)
    if sim is not None:
        return sim
    raise ValueError("layout environment does not expose a MuJoCo sim")


def _state_fingerprint(
    sim: object,
    reports: Sequence[LayoutTransformReport],
) -> str:
    data = getattr(sim, "data")
    body_state = [
        {
            "body_name": item.body_name,
            "position": list(item.after_position),
            "quaternion_wxyz": [float(value) for value in data.xquat[
                getattr(sim, "model").body_name2id(item.body_name)
            ]],
        }
        for item in reports
    ]
    encoded = json.dumps(body_state, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def apply_layout_variant(environment: object, variant: LayoutVariant) -> LayoutApplicationReport:
    """Apply a declared layout after reset and return before/after evidence.

    Only free-jointed bodies are accepted. Static bodies cannot be moved
    without rewriting the task XML, which would invalidate the frozen task
    contract and make the variant impossible to audit at the reset boundary.
    """

    sim = _resolve_sim(environment)
    model = getattr(sim, "model")
    data = getattr(sim, "data")
    reports: list[LayoutTransformReport] = []
    for transform in variant.transforms:
        try:
            body_id = model.body_name2id(transform.body_name)
        except Exception as exc:
            raise ValueError(f"layout body not found: {transform.body_name}") from exc
        joint_name = transform.joint_name or transform.body_name.replace("_main", "_joint0")
        try:
            joint_id = model.joint_name2id(joint_name)
            joint = model.joint(joint_id)
        except Exception as exc:
            raise ValueError(
                f"layout body {transform.body_name} has no free joint; "
                "static bodies are not supported"
            ) from exc
        joint_type = int(np.asarray(getattr(joint, "type")).reshape(-1)[0])
        if joint_type != 0:
            raise ValueError(
                f"layout body {transform.body_name} joint {joint_name} is not a free joint"
            )
        qpos_address = int(np.asarray(getattr(joint, "qposadr")).reshape(-1)[0])
        if len(data.qpos) < qpos_address + 7:
            raise ValueError(f"layout free joint is too short: {joint_name}")
        before = tuple(float(value) for value in data.xpos[body_id])
        after = tuple(
            before[index] + transform.translation_delta_xyz[index]
            for index in range(3)
        )
        data.qpos[qpos_address : qpos_address + 3] = np.asarray(after, dtype=float)
        reports.append(
            LayoutTransformReport(
                body_name=transform.body_name,
                joint_name=joint_name,
                before_position=before,
                after_position=after,
                translation_delta_xyz=transform.translation_delta_xyz,
            )
        )
    forward = getattr(sim, "forward", None)
    if callable(forward):
        forward()
    return LayoutApplicationReport(
        variant_id=variant.variant_id,
        layout_family=variant.layout_family,
        generator_version=variant.generator_version,
        applied=bool(variant.transforms),
        state_fingerprint=_state_fingerprint(sim, reports),
        transforms=tuple(reports),
    )


@dataclass(frozen=True)
class LayoutResetHook:
    """Pickle-safe reset hook used by CAP-X runtime construction."""

    variant_spec: Mapping[str, object]

    def __call__(
        self,
        environment: object,
        seed: int | None,
        options: Mapping[str, object],
    ) -> None:
        del seed, options
        variant = layout_variant_from_mapping(self.variant_spec)
        report = apply_layout_variant(environment, variant)
        # Native cases encode the reset layout explicitly with zero deltas.
        # They must not invalidate CAP-X's already-settled observation cache.
        state_changed = any(
            abs(delta) > 1e-12
            for transform in variant.transforms
            for delta in transform.translation_delta_xyz
        )
        if state_changed:
            handle = getattr(environment, "handle", None)
            libero_env = getattr(handle, "env", None)
            refresh_owner = getattr(libero_env, "env", libero_env)
            refresh = getattr(refresh_owner, "_get_observations", None)
            if callable(refresh):
                try:
                    refreshed = refresh(force_update=True)
                except TypeError:
                    refreshed = refresh()
                if isinstance(refreshed, Mapping):
                    setattr(environment, "_current_obs", refreshed)
        setattr(environment, "_capmas_layout_report", asdict(report))


__all__ = [
    "LayoutApplicationReport",
    "LayoutBodyTransform",
    "LayoutResetHook",
    "LayoutTransformReport",
    "LayoutVariant",
    "apply_layout_variant",
    "layout_variant_from_mapping",
    "layout_variant_sha256",
]
