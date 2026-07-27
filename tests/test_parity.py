from __future__ import annotations

import json

import pytest

from capmas.evaluation.parity import compare_artifacts, load_capmas_episode, load_capx_trial


def test_parity_normalizes_capx_trial_and_capmas_episode(tmp_path) -> None:
    capx = tmp_path / "trial_01_sandboxrc_0_reward_0.000_taskcompleted_0"
    capx.mkdir()
    (capx / "summary.txt").write_text(
        "Environment response:\nTraceback: failed\n\n"
        "Reward: 0.0\nTask Completed: False\n"
    )
    capmas = tmp_path / "episode.json"
    capmas.write_text(
        json.dumps(
            {
                "evaluator_success": True,
                "episode_trace": {"traces": [{"id": 1}, {"id": 2}]},
            }
        )
    )

    comparison = compare_artifacts(capx, capmas, task_id="libero_spatial_0", seed=1)

    assert comparison.capx.success is False
    assert comparison.capx.reward == 0.0
    assert comparison.capx.action_count is None
    assert comparison.capmas.success is True
    assert comparison.capmas.action_count == 2
    assert comparison.success_delta == 1
    assert comparison.to_dict()["task_id"] == "libero_spatial_0"


def test_parity_reads_real_b3_artifact_if_present() -> None:
    from pathlib import Path

    artifact = Path("outputs/capmas_libero_b3/episode.json")
    if not artifact.exists():
        pytest.skip("B3 smoke artifact is not available")
    episode = load_capmas_episode(artifact, task_id="libero_spatial_0", seed=1)
    assert episode.system == "capmas"
    assert episode.success is True
    assert episode.action_count == 5


def test_capx_trial_parser_rejects_untrusted_directory_name(tmp_path) -> None:
    trial = tmp_path / "trial-not-a-result"
    trial.mkdir()
    (trial / "summary.txt").write_text("Task Completed: True")

    with pytest.raises(ValueError, match="must match"):
        load_capx_trial(trial, task_id="task")
