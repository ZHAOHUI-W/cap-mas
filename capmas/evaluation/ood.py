"""Frozen OOD replay contracts and fail-closed split validation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path
from typing import Literal


Split = Literal["id", "ood"]
OODType = Literal["none", "layout", "task_object", "instruction"]
Condition = Literal["capx", "capmas"]


@dataclass(frozen=True)
class OODCase:
    """One immutable member of an ID/OOD replay split."""

    case_id: str
    split: Split
    ood_type: OODType
    task_id: str
    task_goal: str
    task_family: str
    layout_family: str
    object_name: str
    target_name: str
    seed: int
    pair_id: str
    config_path: str
    candidate_artifact: str
    candidate_artifact_sha256: str
    environment_version: str
    generator_version: str
    parent_case_id: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)
    layout_variant: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "case_id",
            "task_id",
            "task_goal",
            "task_family",
            "layout_family",
            "object_name",
            "target_name",
            "pair_id",
            "config_path",
            "candidate_artifact",
            "candidate_artifact_sha256",
            "environment_version",
            "generator_version",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"OOD case {name} must not be empty")
        if self.split not in {"id", "ood"}:
            raise ValueError(f"invalid OOD split: {self.split!r}")
        if self.ood_type not in {"none", "layout", "task_object", "instruction"}:
            raise ValueError(f"invalid OOD type: {self.ood_type!r}")
        if self.split == "id" and self.ood_type != "none":
            raise ValueError("ID cases must use ood_type='none'")
        if self.split == "ood" and self.ood_type == "none":
            raise ValueError("OOD cases must declare an OOD type")
        if self.seed < 0:
            raise ValueError("OOD case seed must be non-negative")
        if self.parent_case_id == self.case_id:
            raise ValueError("OOD case cannot parent itself")
        if len(self.candidate_artifact_sha256) != 64:
            raise ValueError("candidate artifact digest must be a SHA-256 hex digest")
        try:
            int(self.candidate_artifact_sha256, 16)
        except ValueError as exc:
            raise ValueError("candidate artifact digest must be a SHA-256 hex digest") from exc
        if any(not isinstance(key, str) or not key.strip() for key in self.metadata):
            raise ValueError("OOD case metadata keys must be non-empty strings")
        if any(not isinstance(value, str) for value in self.metadata.values()):
            raise ValueError("OOD case metadata values must be strings")
        if not isinstance(self.layout_variant, Mapping):
            raise ValueError("OOD case layout_variant must be an object")


@dataclass(frozen=True)
class OODSplitManifest:
    """The immutable, auditable membership of one OOD replay suite."""

    suite_id: str
    manifest_version: str
    cases: tuple[OODCase, ...]
    id_task_families: tuple[str, ...]
    ood_task_families: tuple[str, ...]
    id_layout_families: tuple[str, ...]
    ood_layout_families: tuple[str, ...]
    memory_snapshot_version: str
    robot_skill_snapshot_version: str
    prompt_version: str
    code_revision: str
    created_at_utc: str
    manifest_sha256: str = ""

    def __post_init__(self) -> None:
        for name in (
            "suite_id",
            "manifest_version",
            "memory_snapshot_version",
            "robot_skill_snapshot_version",
            "prompt_version",
            "code_revision",
            "created_at_utc",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"OOD manifest {name} must not be empty")
        if not self.cases:
            raise ValueError("OOD manifest must contain at least one case")
        if self.manifest_sha256:
            if len(self.manifest_sha256) != 64:
                raise ValueError("manifest_sha256 must be a SHA-256 hex digest")
            try:
                int(self.manifest_sha256, 16)
            except ValueError as exc:
                raise ValueError("manifest_sha256 must be a SHA-256 hex digest") from exc


@dataclass(frozen=True)
class OODReplayEvidence:
    """One candidate result from a frozen replay; never a live Arbiter weight."""

    case_id: str
    pair_id: str
    condition: Condition
    candidate_id: str
    split: Split
    ood_type: OODType
    source_scene_version: int
    candidate_fingerprint: str
    evaluator_success: bool | None
    verifier_success: bool | None
    graph_completed: bool
    failure_class: str | None
    recovery_count: int
    human_intervention_count: int
    latency_ms: float
    provider_call_count: int
    cache_hit_count: int
    selection_basis: str | None = None
    shadow_only: bool = True
    layout_state_fingerprint: str | None = None

    def __post_init__(self) -> None:
        for name in ("case_id", "pair_id", "candidate_id", "candidate_fingerprint"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"OOD replay {name} must not be empty")
        if self.condition not in {"capx", "capmas"}:
            raise ValueError(f"invalid OOD replay condition: {self.condition!r}")
        if self.split not in {"id", "ood"}:
            raise ValueError(f"invalid OOD replay split: {self.split!r}")
        if self.ood_type not in {"none", "layout", "task_object", "instruction"}:
            raise ValueError(f"invalid OOD replay type: {self.ood_type!r}")
        if self.split == "id" and self.ood_type != "none":
            raise ValueError("ID replay evidence must use ood_type='none'")
        if self.split == "ood" and self.ood_type == "none":
            raise ValueError("OOD replay evidence must declare an OOD type")
        if self.source_scene_version < 0:
            raise ValueError("OOD replay scene version must be non-negative")
        if self.recovery_count < 0 or self.human_intervention_count < 0:
            raise ValueError("OOD replay counts must be non-negative")
        if self.latency_ms < 0 or self.provider_call_count < 0 or self.cache_hit_count < 0:
            raise ValueError("OOD replay metrics must be non-negative")
        if not self.shadow_only:
            raise ValueError("P5.5 OOD replay evidence must remain shadow-only")


@dataclass(frozen=True)
class LeakageAudit:
    """Deterministic result of the pre-replay leakage checks."""

    passed: bool
    duplicate_case_ids: tuple[str, ...] = ()
    id_ood_family_overlap: tuple[str, ...] = ()
    candidate_digest_mismatches: tuple[str, ...] = ()
    forbidden_memory_versions: tuple[str, ...] = ()
    forbidden_skill_versions: tuple[str, ...] = ()
    cross_case_cache_keys: tuple[str, ...] = ()
    details: tuple[str, ...] = ()


def canonical_manifest_payload(manifest: OODSplitManifest) -> dict[str, object]:
    """Return the canonical digest payload without the self-referential digest."""

    payload = asdict(manifest)
    payload.pop("manifest_sha256", None)
    return payload


def manifest_sha256(manifest: OODSplitManifest) -> str:
    encoded = json.dumps(
        canonical_manifest_payload(manifest),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_ood_manifest(
    manifest: OODSplitManifest,
    candidate_digest_resolver: Callable[[OODCase], str] | None = None,
) -> None:
    """Validate membership, pairing, family isolation, and artifact digests."""

    case_ids = [case.case_id for case in manifest.cases]
    duplicate_case_ids = sorted({case_id for case_id in case_ids if case_ids.count(case_id) > 1})
    if duplicate_case_ids:
        raise ValueError(f"duplicate OOD case ids: {duplicate_case_ids}")

    seen_task_seed_split: set[tuple[str, int, str]] = set()
    cases_by_id = {case.case_id: case for case in manifest.cases}
    cases_by_pair: dict[str, list[OODCase]] = {}
    for case in manifest.cases:
        key = (case.task_id, case.seed, case.split)
        if key in seen_task_seed_split:
            raise ValueError(f"duplicate task/seed/split entry: {key}")
        seen_task_seed_split.add(key)
        cases_by_pair.setdefault(case.pair_id, []).append(case)
        if case.parent_case_id is not None:
            parent = cases_by_id.get(case.parent_case_id)
            if parent is None:
                raise ValueError(f"missing OOD parent case: {case.parent_case_id}")
            if parent.split != "id":
                raise ValueError("OOD parent case must be an ID case")
            if case.ood_type == "layout":
                if case.task_goal != parent.task_goal:
                    raise ValueError("layout OOD parent and child must share task goal")
                if case.candidate_artifact_sha256 != parent.candidate_artifact_sha256:
                    raise ValueError("layout OOD parent and child must share candidate digest")
        if candidate_digest_resolver is not None:
            actual = candidate_digest_resolver(case)
            if actual != case.candidate_artifact_sha256:
                raise ValueError(f"candidate digest mismatch for case {case.case_id}")

    for pair_id, pair_cases in cases_by_pair.items():
        splits = [case.split for case in pair_cases]
        if splits.count("id") != 1 or splits.count("ood") != 1:
            raise ValueError(f"pair {pair_id!r} must contain one ID and one OOD case")

    id_task_families = {case.task_family for case in manifest.cases if case.split == "id"}
    id_layout_families = {case.layout_family for case in manifest.cases if case.split == "id"}
    for case in manifest.cases:
        if case.split != "ood":
            continue
        if case.ood_type == "task_object" and case.task_family in id_task_families:
            raise ValueError(f"OOD task family overlaps ID task family: {case.task_family}")
        if case.ood_type == "layout" and case.layout_family in id_layout_families:
            raise ValueError(f"OOD layout family overlaps ID layout family: {case.layout_family}")

    declared_id_tasks = set(manifest.id_task_families)
    declared_ood_tasks = set(manifest.ood_task_families)
    declared_id_layouts = set(manifest.id_layout_families)
    declared_ood_layouts = set(manifest.ood_layout_families)
    if declared_id_tasks != id_task_families:
        raise ValueError("manifest ID task families do not match cases")
    if declared_id_layouts != id_layout_families:
        raise ValueError("manifest ID layout families do not match cases")
    expected_ood_tasks = {case.task_family for case in manifest.cases if case.split == "ood"}
    expected_ood_layouts = {case.layout_family for case in manifest.cases if case.split == "ood"}
    if declared_ood_tasks != expected_ood_tasks or declared_ood_layouts != expected_ood_layouts:
        raise ValueError("manifest OOD families do not match cases")


def audit_leakage(
    manifest: OODSplitManifest,
    *,
    observed_memory_versions: tuple[str, ...] = (),
    observed_skill_versions: tuple[str, ...] = (),
    observed_cache_case_ids: tuple[str, ...] = (),
) -> LeakageAudit:
    """Collect leakage findings without starting replay."""

    case_ids = [case.case_id for case in manifest.cases]
    duplicate_case_ids = tuple(
        sorted({case_id for case_id in case_ids if case_ids.count(case_id) > 1})
    )
    id_task_families = {case.task_family for case in manifest.cases if case.split == "id"}
    id_layout_families = {case.layout_family for case in manifest.cases if case.split == "id"}
    overlap: set[str] = set()
    for case in manifest.cases:
        if case.split != "ood":
            continue
        if case.ood_type == "task_object" and case.task_family in id_task_families:
            overlap.add(f"task:{case.task_family}")
        if case.ood_type == "layout" and case.layout_family in id_layout_families:
            overlap.add(f"layout:{case.layout_family}")
    forbidden_memory_versions = tuple(
        sorted({version for version in observed_memory_versions if version != manifest.memory_snapshot_version})
    )
    forbidden_skill_versions = tuple(
        sorted(
            {
                version
                for version in observed_skill_versions
                if version != manifest.robot_skill_snapshot_version
            }
        )
    )
    allowed_case_ids = set(case_ids)
    cross_case_cache_keys = tuple(
        sorted({case_id for case_id in observed_cache_case_ids if case_id not in allowed_case_ids})
    )
    details: list[str] = []
    if duplicate_case_ids:
        details.append("duplicate case ids")
    if overlap:
        details.append("ID/OOD family overlap")
    if forbidden_memory_versions:
        details.append("non-frozen memory version")
    if forbidden_skill_versions:
        details.append("non-frozen robot skill version")
    if cross_case_cache_keys:
        details.append("cross-case cache namespace")
    return LeakageAudit(
        passed=not any(
            (
                duplicate_case_ids,
                overlap,
                forbidden_memory_versions,
                forbidden_skill_versions,
                cross_case_cache_keys,
            )
        ),
        duplicate_case_ids=duplicate_case_ids,
        id_ood_family_overlap=tuple(sorted(overlap)),
        forbidden_memory_versions=forbidden_memory_versions,
        forbidden_skill_versions=forbidden_skill_versions,
        cross_case_cache_keys=cross_case_cache_keys,
        details=tuple(details),
    )


def assert_leakage_free(audit: LeakageAudit) -> None:
    if not audit.passed:
        details = ", ".join(audit.details) or "unspecified leakage finding"
        raise ValueError(f"leakage audit failed: {details}")


def load_ood_manifest(path: str | Path) -> OODSplitManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("OOD manifest must contain a JSON object")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("OOD manifest cases must be a list")
    cases = tuple(OODCase(**case) for case in raw_cases if isinstance(case, Mapping))
    if len(cases) != len(raw_cases):
        raise ValueError("OOD manifest cases must contain objects")
    manifest = OODSplitManifest(
        suite_id=str(payload.get("suite_id", "")),
        manifest_version=str(payload.get("manifest_version", "")),
        cases=cases,
        id_task_families=tuple(str(item) for item in payload.get("id_task_families", ())),
        ood_task_families=tuple(str(item) for item in payload.get("ood_task_families", ())),
        id_layout_families=tuple(str(item) for item in payload.get("id_layout_families", ())),
        ood_layout_families=tuple(str(item) for item in payload.get("ood_layout_families", ())),
        memory_snapshot_version=str(payload.get("memory_snapshot_version", "")),
        robot_skill_snapshot_version=str(payload.get("robot_skill_snapshot_version", "")),
        prompt_version=str(payload.get("prompt_version", "")),
        code_revision=str(payload.get("code_revision", "")),
        created_at_utc=str(payload.get("created_at_utc", "")),
        manifest_sha256=str(payload.get("manifest_sha256", "")),
    )
    validate_ood_manifest(manifest)
    expected_digest = manifest_sha256(manifest)
    if manifest.manifest_sha256 and manifest.manifest_sha256 != expected_digest:
        raise ValueError("OOD manifest digest mismatch")
    return replace(manifest, manifest_sha256=expected_digest)


def dump_ood_manifest(path: str | Path, manifest: OODSplitManifest) -> None:
    validate_ood_manifest(manifest)
    finalized = replace(manifest, manifest_sha256=manifest_sha256(manifest))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(asdict(finalized), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "Condition",
    "LeakageAudit",
    "OODCase",
    "OODReplayEvidence",
    "OODSplitManifest",
    "OODType",
    "Split",
    "assert_leakage_free",
    "audit_leakage",
    "canonical_manifest_payload",
    "dump_ood_manifest",
    "load_ood_manifest",
    "manifest_sha256",
    "validate_ood_manifest",
]
