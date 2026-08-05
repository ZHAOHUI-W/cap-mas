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
