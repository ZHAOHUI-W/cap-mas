from __future__ import annotations

import pytest

from capmas.evaluation.ood import LeakageAudit, OODCase
from capmas.evaluation.ood import audit_leakage, assert_leakage_free
from tests.test_ood_contracts import _case, _manifest


def test_audit_rejects_non_frozen_memory_version() -> None:
    manifest = _manifest((_case(),), memory_snapshot_version="memory-v1")
    audit = audit_leakage(manifest, observed_memory_versions=("memory-v2",))
    assert audit.passed is False
    assert audit.forbidden_memory_versions == ("memory-v2",)
    with pytest.raises(ValueError, match="leakage"):
        assert_leakage_free(audit)


def test_audit_rejects_non_frozen_skill_version() -> None:
    manifest = _manifest((_case(),), robot_skill_snapshot_version="robot-v1")
    audit = audit_leakage(manifest, observed_skill_versions=("robot-v2",))
    assert audit.passed is False
    assert audit.forbidden_skill_versions == ("robot-v2",)


def test_audit_rejects_cache_namespace_owned_by_another_case() -> None:
    audit = audit_leakage(
        _manifest((_case(),)), observed_cache_case_ids=("case-from-another-suite",)
    )
    assert audit.passed is False
    assert audit.cross_case_cache_keys == ("case-from-another-suite",)


def test_audit_reports_id_ood_family_overlap_without_starting_replay() -> None:
    id_case = _case(case_id="id-1", task_family="family-a")
    ood_case = _case(
        case_id="ood-1",
        split="ood",
        ood_type="task_object",
        pair_id="pair-1",
        parent_case_id="id-1",
        task_family="family-a",
        layout_family="layout-b",
    )
    manifest = _manifest(
        (id_case, ood_case),
        id_task_families=("family-a",),
        ood_task_families=("family-a",),
        id_layout_families=("layout-a",),
        ood_layout_families=("layout-b",),
    )
    audit = audit_leakage(manifest)
    assert audit.passed is False
    assert audit.id_ood_family_overlap == ("task:family-a",)


def test_passed_audit_is_accepted() -> None:
    audit = LeakageAudit(passed=True)
    assert_leakage_free(audit)
