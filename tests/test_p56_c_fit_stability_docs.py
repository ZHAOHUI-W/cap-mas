from pathlib import Path

_ROOT = Path(__file__).parents[1]


def test_p56c_docs_preserve_offline_only_and_unverified_real_data_boundary() -> None:
    spec = (_ROOT / "docs/superpowers/specs/2026-08-18-p5-6c-fit-stability-design.md").read_text()
    roadmap = (_ROOT / "docs/implementation-roadmap.md").read_text()
    evidence = (_ROOT / "docs/phase5-evidence-evolution.md").read_text()
    experiments = (_ROOT / "docs/experiments.md").read_text()

    assert "p56b.constrained_logistic.v1" in spec
    assert "p56b.constrained_logistic.v2" in spec
    assert "P5.6C fit stability" in roadmap
    assert "P5.6C fit stability" in evidence
    assert "P5.6C fit stability" in experiments
    assert "no verified real 12-row calibration result" in experiments
    assert "collection artifacts are unavailable in this checkout" in roadmap
