from __future__ import annotations

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
