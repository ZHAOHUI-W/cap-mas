from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_phase5_docs_distinguish_contract_closure_from_runtime_foundation():
    phase5 = (ROOT / "docs/phase5-evidence-evolution.md").read_text(encoding="utf-8")
    roadmap = (ROOT / "docs/implementation-roadmap.md").read_text(encoding="utf-8")

    assert "P5.0-contract" in phase5
    assert "P5.3 may proceed" in phase5
    assert "TSDF and real semantic adapters remain open" in phase5
    assert "P5.0 contract-level closure" in roadmap

