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
    assert "No experiment artifact has yet enabled the cache" in experiments
