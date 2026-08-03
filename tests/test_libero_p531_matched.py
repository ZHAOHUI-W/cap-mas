from __future__ import annotations

import json
from pathlib import Path

from scripts.run_libero_p531_matched import (
    MatchedTaskSpec,
    load_task_manifest,
    run_matched_evaluation,
)


def _candidate_artifact() -> Path:
    return (
        Path(__file__).parents[1]
        / "outputs"
        / "phase5"
        / "P5.3_process_rehearsal_input_20260730"
        / "matched_candidates.json"
    )


def test_matched_runner_pairs_baseline_and_online_per_seed(tmp_path) -> None:
    calls = []

    def fake_runner(**kwargs):
        calls.append(
            {
                "mode": kwargs["mode"],
                "seed": kwargs["seed"],
                "candidate_ids": tuple(
                    candidate.candidate_id for candidate in kwargs["candidates"]
                ),
            }
        )
        online = kwargs["mode"] == "online_bounded"
        return {
            "mode": kwargs["mode"],
            "physical_candidate_id": "candidate-b" if online else "candidate-a",
            "physical_result": {
                "completed": True,
                "evaluator_success": online,
                "success": online,
            },
            "report": {
                "baseline_winner": "candidate-a",
                "live_winner": "candidate-b" if online else "candidate-a",
            },
        }

    report = run_matched_evaluation(
        tasks=(
            MatchedTaskSpec(
                task_id="libero_spatial_0",
                config_path="spatial.yaml",
                candidate_artifact=_candidate_artifact(),
                object_name="akita black bowl",
                target_name="plate",
                seeds=(1, 2),
            ),
        ),
        output_root=tmp_path,
        online_runner=fake_runner,
    )

    assert [(call["mode"], call["seed"]) for call in calls] == [
        ("disabled", 1),
        ("online_bounded", 1),
        ("disabled", 2),
        ("online_bounded", 2),
    ]
    assert all(call["candidate_ids"] for call in calls)
    assert report.completed_pairs == 2
    assert report.baseline_successes == 0
    assert report.online_successes == 2
    assert report.online_minus_baseline == 2
    assert len(report.pairs) == 2

    suite_dir = report.suite_dir
    assert (suite_dir / "run_config.json").exists()
    assert (suite_dir / "results" / "aggregate.json").exists()
    assert (suite_dir / "summary.md").exists()
    assert (suite_dir / "manifest.json").exists()
    pair_dirs = list((suite_dir / "pairs").iterdir())
    assert len(pair_dirs) == 2
    for pair_dir in pair_dirs:
        assert (pair_dir / "results" / "pair.json").exists()
        assert (pair_dir / "manifest.json").exists()

    aggregate = json.loads(
        (suite_dir / "results" / "aggregate.json").read_text(encoding="utf-8")
    )
    assert aggregate["same_candidate_artifact_count"] == 2
    assert aggregate["complete_pair_count"] == 2
    assert aggregate["winner_change_count"] == 2
    assert aggregate["online_selection_basis_counts"] == {"unknown": 2}
    pair_payload = json.loads(
        (pair_dirs[0] / "results" / "pair.json").read_text(encoding="utf-8")
    )
    assert pair_payload["baseline_winner"] == "candidate-a"
    assert pair_payload["online_winner"] == "candidate-b"
    assert pair_payload["winner_changed"] is True


def test_matched_runner_retains_failed_pair_and_continues(tmp_path) -> None:
    calls = []

    def fake_runner(**kwargs):
        calls.append((kwargs["mode"], kwargs["seed"]))
        if kwargs["seed"] == 1 and kwargs["mode"] == "online_bounded":
            raise RuntimeError("online episode failed")
        return {
            "physical_candidate_id": "candidate-a",
            "physical_result": {"completed": True, "evaluator_success": True, "success": True},
        }

    report = run_matched_evaluation(
        tasks=(
            MatchedTaskSpec(
                task_id="libero_spatial_0",
                config_path="spatial.yaml",
                candidate_artifact=_candidate_artifact(),
                object_name="akita black bowl",
                target_name="plate",
                seeds=(1, 2),
            ),
        ),
        output_root=tmp_path,
        online_runner=fake_runner,
    )

    assert calls == [
        ("disabled", 1),
        ("online_bounded", 1),
        ("disabled", 2),
        ("online_bounded", 2),
    ]
    assert report.completed_pairs == 1
    assert report.failed_pairs == 1
    failed = next(pair for pair in report.pairs if pair.seed == 1)
    assert failed.status == "failed"
    assert (failed.pair_dir / "results" / "pair.json").exists()
    assert (failed.pair_dir / "manifest.json").exists()


def test_load_task_manifest_preserves_task_and_seed_controls(tmp_path) -> None:
    manifest = tmp_path / "tasks.json"
    manifest.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": "task-a",
                        "config_path": "a.yaml",
                        "candidate_artifact": "a.json",
                        "object_name": "bowl",
                        "target_name": "plate",
                        "seeds": [3, 1],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    tasks = load_task_manifest(manifest)

    assert tasks == (
        MatchedTaskSpec(
            task_id="task-a",
            config_path="a.yaml",
            candidate_artifact="a.json",
            object_name="bowl",
            target_name="plate",
            seeds=(3, 1),
        ),
    )
