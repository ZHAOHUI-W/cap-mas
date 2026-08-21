from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_phase5_docs_distinguish_contract_closure_from_runtime_foundation():
    phase5 = (ROOT / "docs/phase5-evidence-evolution.md").read_text(encoding="utf-8")
    roadmap = (ROOT / "docs/implementation-roadmap.md").read_text(encoding="utf-8")

    assert "P5.0-contract" in phase5
    assert "P5.3 may proceed" in phase5
    assert "TSDF and real semantic adapters remain open" in phase5
    assert "P5.0 contract-level closure" in roadmap


def test_phase5_docs_record_p54_cache_boundary():
    phase5 = (ROOT / "docs/phase5-evidence-evolution.md").read_text(encoding="utf-8")
    roadmap = (ROOT / "docs/implementation-roadmap.md").read_text(encoding="utf-8")
    experiments = (ROOT / "docs/experiments.md").read_text(encoding="utf-8")

    assert "P5.4 evidence cache status" in phase5
    assert "process-local LRU" in roadmap
    assert "--selection-repeats" in experiments
    assert "no real\nCAP-X multi-seed artifact" in experiments
    assert "does not establish a downstream" in experiments
    assert "run_libero_p54_matched.py" in phase5
    assert "never share a cache" in experiments


def test_phase5_docs_record_typed_condition_default_boundary():
    phase5 = (ROOT / "docs/phase5-evidence-evolution.md").read_text(encoding="utf-8")
    roadmap = (ROOT / "docs/implementation-roadmap.md").read_text(encoding="utf-8")
    experiments = (ROOT / "docs/experiments.md").read_text(encoding="utf-8")

    assert "SkillConditionEnricher" in phase5
    assert "balanced" in phase5 and "safety" in phase5
    assert "scene_fresh(2000)" in roadmap
    assert "coverage is reported" in experiments


def test_phase5_docs_record_p55_ood_boundary():
    phase5 = (ROOT / "docs/phase5-evidence-evolution.md").read_text(encoding="utf-8")
    roadmap = (ROOT / "docs/implementation-roadmap.md").read_text(encoding="utf-8")
    experiments = (ROOT / "docs/experiments.md").read_text(encoding="utf-8")

    assert "P5.5 frozen OOD replay" in phase5
    assert "shadow-only" in phase5
    assert "OOD gap" in roadmap
    assert "run_libero_p55_ood.py" in experiments


def test_phase5_docs_record_p55_ten_seed_formal_boundary():
    phase5 = (ROOT / "docs/phase5-evidence-evolution.md").read_text(encoding="utf-8")
    roadmap = (ROOT / "docs/implementation-roadmap.md").read_text(encoding="utf-8")
    experiments = (ROOT / "docs/experiments.md").read_text(encoding="utf-8")
    plan = (
        ROOT / "docs/superpowers/plans/2026-08-03-p5-5-ood-replay.md"
    ).read_text(encoding="utf-8")

    for document in (phase5, roadmap, experiments):
        assert "p55_real_layout_3family_10seed.json" in document
        assert "0/30" in document
        assert "0.1135" in document
        assert "evidence_tie_break" in document
        assert "P5.6" in document
    assert "horizon bucket" in phase5
    assert "max_steps=32" in experiments
    assert "- [x] **Step 1: Record the final formal gate**" in plan
    assert "- [x] **Step 2: Verify the P5.6 boundary**" in plan


def test_phase5_docs_record_execution_grounding_smoke():
    phase5 = (ROOT / "docs/phase5-evidence-evolution.md").read_text(encoding="utf-8")
    roadmap = (ROOT / "docs/implementation-roadmap.md").read_text(encoding="utf-8")
    experiments = (ROOT / "docs/experiments.md").read_text(encoding="utf-8")

    for document in (phase5, roadmap, experiments):
        assert "P5.5 execution-grounding smoke" in document
        assert "P5.5_grounding_smoke_venv_20260805" in document
        assert "1/3" in document
        assert "0/3" in document
        assert "0.72409" in document
    assert "0.81099" in document
    assert "not a multi-seed quality result" in experiments


def test_phase5_docs_record_p56a_data_foundation_gate():
    phase5 = (ROOT / "docs/phase5-evidence-evolution.md").read_text(encoding="utf-8")
    roadmap = (ROOT / "docs/implementation-roadmap.md").read_text(encoding="utf-8")
    experiments = (ROOT / "docs/experiments.md").read_text(encoding="utf-8")
    glossary = (ROOT / "docs/glossary.md").read_text(encoding="utf-8")

    for document in (phase5, roadmap, experiments):
        assert "P5.6A data foundation" in document
        assert "p56.feature.v1" in document
        assert "P5.3.2 Task-Family Capability Repair" in document
        assert "max_steps=32 is not a horizon" in document
        assert "manifest_verification.json" in document
    assert "20 Tier A" in experiments
    assert "decision-time feature snapshot" in glossary


def test_phase5_docs_record_gripper_state_semantic_correction():
    phase5 = (ROOT / "docs/phase5-evidence-evolution.md").read_text(encoding="utf-8")
    roadmap = (ROOT / "docs/implementation-roadmap.md").read_text(encoding="utf-8")
    experiments = (ROOT / "docs/experiments.md").read_text(encoding="utf-8")

    for document in (phase5, roadmap, experiments):
        assert "gripper_commanded_fraction" in document
        assert "P5.5" in document
    assert "P5.5_grasp_probe_object6_commanded_20260805" in phase5
    assert "task_completed=false" in experiments


def test_phase5_docs_record_p55_matched_provenance_failure_semantics():
    phase5 = (ROOT / "docs/phase5-evidence-evolution.md").read_text(encoding="utf-8")
    roadmap = (ROOT / "docs/implementation-roadmap.md").read_text(encoding="utf-8")
    experiments = (ROOT / "docs/experiments.md").read_text(encoding="utf-8")
    specification = (
        ROOT / "docs/superpowers/specs/2026-08-03-p5-5-ood-replay-design.md"
    ).read_text(encoding="utf-8")

    for document in (phase5, roadmap, experiments):
        assert "20260806_091429_suite_e169a480" in document
        assert "3/15" in document
        assert "5/15" in document
        assert "McNemar `p=0.5`" in document
        assert "22" in document
        assert "2 verifier false negatives" in document
    assert "task_failure_classes" in specification
    assert "graph_failure_classes" in specification
    assert "verifier_false_negative_classes" in specification
    assert "must not be described as downstream task failures" in specification


def test_phase5_docs_record_effective_motion_capability_boundary():
    documents = [
        (ROOT / name).read_text(encoding="utf-8")
        for name in (
            "docs/phase5-evidence-evolution.md",
            "docs/implementation-roadmap.md",
            "docs/experiments.md",
        )
    ]

    for document in documents:
        assert "EffectiveMotionProgram" in document
        assert "candidate_semantic_equivalence" in document
        assert "P5.6D" in document and "immutable" in document
        assert "20260820_081203_3c028a8f" in document
        assert "P5.3.2 ten-seed capability gate is closed" in document
        assert "2/10 infrastructure-unknown" in document
        assert "4/10 physical-execution reach" in document


def test_phase5_docs_record_p532_diagnostic_only_promotion_boundary():
    documents = [
        (ROOT / name).read_text(encoding="utf-8")
        for name in (
            "docs/phase5-evidence-evolution.md",
            "docs/implementation-roadmap.md",
            "docs/experiments.md",
            "docs/superpowers/specs/2026-08-20-p5-3-2-object6-effective-motion-design.md",
        )
    ]

    for document in documents:
        assert "P5.3.2.1 Diagnostic Observability" in document
        assert "diagnostic_only=true" in document
        assert "eligible_for_evaluation=false" in document
        assert "confirmed single root cause" in document
    assert "seed 53" in documents[3]
    assert "seeds 54 and 58" in documents[3]
    assert "seed 57" in documents[3]
