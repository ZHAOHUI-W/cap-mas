from __future__ import annotations

import json
from pathlib import Path

from scripts.run_libero_p54_matched import (
    MatchedCacheTaskSpec,
    run_matched_cache_evaluation,
)


def _candidate_artifact() -> Path:
    return (
        Path(__file__).parents[1]
        / "outputs"
        / "phase5"
        / "P5.3_process_rehearsal_input_20260730"
        / "matched_candidates.json"
    )


def test_matched_cache_runner_pairs_independent_lanes_per_seed(tmp_path) -> None:
    calls = []

    def fake_runner(**kwargs):
        calls.append(
            {
                "mode": kwargs["mode"],
                "cache_mode": kwargs["cache_mode"],
                "selection_repeats": kwargs["selection_repeats"],
                "seed": kwargs["seed"],
                "candidate_ids": tuple(
                    candidate.candidate_id for candidate in kwargs["candidates"]
                ),
            }
        )
        enabled = kwargs["cache_mode"] == "enabled"
        return {
            "mode": kwargs["mode"],
            "cache_mode": kwargs["cache_mode"],
            "physical_candidate_id": "candidate-a",
            "physical_result": {
                "completed": True,
                "evaluator_success": True,
                "success": True,
            },
            "provider_call_count": 1 if enabled else 2,
            "selection_latency_ms": 10.0 if enabled else 20.0,
            "cache_stats": {"hits": 2 if enabled else 0, "misses": 2 if enabled else 0},
            "run_dir": f"fake/{kwargs['cache_mode']}/seed{kwargs['seed']}",
        }

    report = run_matched_cache_evaluation(
        tasks=(
            MatchedCacheTaskSpec(
                task_id="libero_spatial_0",
                config_path="spatial.yaml",
                candidate_artifact=_candidate_artifact(),
                object_name="akita black bowl",
                target_name="plate",
                seeds=(1, 2),
            ),
        ),
        output_root=tmp_path,
        selection_repeats=2,
        online_runner=fake_runner,
    )

    assert [
        (call["mode"], call["cache_mode"], call["seed"])
        for call in calls
    ] == [
        ("online_bounded", "disabled", 1),
        ("online_bounded", "enabled", 1),
        ("online_bounded", "disabled", 2),
        ("online_bounded", "enabled", 2),
    ]
    assert all(call["selection_repeats"] == 2 for call in calls)
    assert all(call["candidate_ids"] for call in calls)
    assert report.completed_pairs == 2
    assert report.failed_pairs == 0
    assert report.disabled_provider_calls == 4
    assert report.enabled_provider_calls == 2
    assert report.enabled_cache_hits == 4
    assert report.provider_call_reduction == 0.5

    aggregate = json.loads(
        (report.suite_dir / "results" / "aggregate.json").read_text()
    )
    assert aggregate["pair_count"] == 2
    assert aggregate["same_candidate_artifact_count"] == 2
    assert aggregate["disabled_provider_calls"] == 4
    assert aggregate["enabled_provider_calls"] == 2
    assert aggregate["enabled_cache_hits"] == 4
    assert aggregate["provider_call_reduction"] == 0.5

    pair_dirs = list((report.suite_dir / "pairs").iterdir())
    assert len(pair_dirs) == 2
    pair_payload = json.loads(
        (pair_dirs[0] / "results" / "pair.json").read_text()
    )
    assert pair_payload["disabled"]["cache_mode"] == "disabled"
    assert pair_payload["enabled"]["cache_mode"] == "enabled"
    assert pair_payload["same_candidate_artifact"] is True
    assert (report.suite_dir / "run_config.json").exists()
    assert (report.suite_dir / "summary.md").exists()
    assert (report.suite_dir / "manifest.json").exists()


def test_matched_cache_runner_retains_failed_pairs(tmp_path) -> None:
    def fake_runner(**kwargs):
        if kwargs["seed"] == 1 and kwargs["cache_mode"] == "enabled":
            raise RuntimeError("enabled lane failed")
        return {
            "physical_candidate_id": "candidate-a",
            "physical_result": {"success": True},
            "provider_call_count": 1,
            "selection_latency_ms": 1.0,
            "cache_stats": {"hits": 1},
        }

    report = run_matched_cache_evaluation(
        tasks=(
            MatchedCacheTaskSpec(
                task_id="libero_spatial_0",
                config_path="spatial.yaml",
                candidate_artifact=_candidate_artifact(),
                object_name="bowl",
                target_name="plate",
                seeds=(1, 2),
            ),
        ),
        output_root=tmp_path,
        online_runner=fake_runner,
    )

    assert report.completed_pairs == 1
    assert report.failed_pairs == 1
    failed = next(pair for pair in report.pairs if pair.seed == 1)
    assert failed.status == "failed"
    assert (failed.pair_dir / "failure.json").exists()
    assert (failed.pair_dir / "manifest.json").exists()

