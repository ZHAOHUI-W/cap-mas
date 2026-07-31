from __future__ import annotations

import json

import pytest


def test_cache_evaluation_proves_hits_and_scene_invalidation(tmp_path):
    from scripts.run_p54_evidence_cache import run_cache_evaluation

    report = run_cache_evaluation(output_root=tmp_path, seed=1)

    assert report.control.metrics.provider_calls == 9
    assert report.enabled.metrics.provider_calls == 5
    assert report.enabled.metrics.hits == 3
    assert report.enabled.metrics.invalidations == 2
    assert report.enabled.metrics.stale_rejections == 1
    assert report.enabled.metrics.stale_attachments == 0
    assert report.provider_call_reduction == pytest.approx(4 / 9)
    assert report.same_trace is True
    assert report.enabled.metrics.current_scene_version == 2

    control_trace = [
        (entry.operation, entry.candidate_id, entry.scene_version)
        for entry in report.control.trace
    ]
    enabled_trace = [
        (entry.operation, entry.candidate_id, entry.scene_version)
        for entry in report.enabled.trace
    ]
    assert control_trace == enabled_trace


def test_cache_evaluation_emits_trace_and_manifests(tmp_path):
    from scripts.run_p54_evidence_cache import run_cache_evaluation

    report = run_cache_evaluation(output_root=tmp_path, seed=7)

    for result in (report.control, report.enabled):
        assert result.trace[0].operation == "publish"
        assert result.trace[-1].candidate_id == "candidate-c"
        assert result.run_dir.exists()
        assert (result.run_dir / "run_config.json").exists()
        assert (result.run_dir / "logs" / "runner.log").exists()
        assert (result.run_dir / "results" / "cache_trace.json").exists()
        assert (result.run_dir / "results" / "summary.json").exists()
        assert (result.run_dir / "summary.md").exists()
        assert (result.run_dir / "manifest.json").exists()
        assert (result.run_dir / "results" / "paired_comparison.json").exists()

    assert report.control.run_dir != report.enabled.run_dir


def _test_evidence(scene_version):
    from capmas.contracts.candidates import CandidateEvidence

    return CandidateEvidence(
        rehearsal_success_rate=0.5,
        available_metrics=("rehearsal",),
        scene_version=scene_version,
        provider="p5.4-test",
    )


class _FailingProvider:
    def __init__(self, message="provider failure"):
        self.calls = 0
        self.message = message

    def call(self, candidate, scene):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError(self.message)
        return _test_evidence(scene.scene_version)


def test_cache_mode_retains_partial_failure_artifacts(tmp_path):
    from scripts.run_p54_evidence_cache import run_cache_mode

    with pytest.raises(RuntimeError, match="provider failure"):
        run_cache_mode(
            output_root=tmp_path,
            mode="cache_enabled",
            seed=1,
            provider=_FailingProvider(),
        )

    run_dirs = list((tmp_path / "P5.4_cache_evaluation").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert json.loads((run_dir / "run_config.json").read_text())["status"] == "failed"
    assert (run_dir / "failure.json").exists()
    assert (run_dir / "results" / "cache_trace.json").exists()
    assert (run_dir / "logs" / "runner.log").exists()
    assert (run_dir / "manifest.json").exists()


def test_cache_failure_artifacts_redact_provider_secret(tmp_path):
    from scripts.run_p54_evidence_cache import run_cache_mode

    secret = "sk-test-provider-secret"
    with pytest.raises(RuntimeError, match="provider failure"):
        run_cache_mode(
            output_root=tmp_path,
            mode="cache_enabled",
            seed=1,
            provider=_FailingProvider(f"provider failure {secret}"),
        )

    run_dir = next((tmp_path / "P5.4_cache_evaluation").iterdir())
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in run_dir.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    assert secret not in text
