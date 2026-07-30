from pathlib import Path


def test_p51_docs_define_typed_and_dynamic_evidence() -> None:
    text = Path("docs/phase5-evidence-evolution.md").read_text()

    assert "VerifierEvidence" in text
    assert "dynamic" in text
    assert "post-execution" in text
    assert "candidate fingerprint" in text
    assert "LIBERO" in text
