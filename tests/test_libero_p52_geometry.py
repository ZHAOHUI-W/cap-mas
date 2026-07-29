from __future__ import annotations

from scripts.run_libero_p52_geometry import (
    P52_MODES,
    build_pilot_specs,
    pilot_summary_markdown,
)


def test_p52_pilot_builds_independent_seed_and_mode_artifacts(tmp_path) -> None:
    specs = build_pilot_specs(tmp_path, seeds=(1, 2), modes=P52_MODES)

    assert len(specs) == 6
    assert {spec.mode for spec in specs} == set(P52_MODES)
    assert {spec.seed for spec in specs} == {1, 2}
    assert len({spec.run_dir.path for spec in specs}) == 6
    assert all(spec.used_privileged_state is False for spec in specs)
    assert all((spec.run_dir.path / "logs").is_dir() for spec in specs)


def test_p52_pilot_summary_markdown_records_result_and_artifact_paths() -> None:
    markdown = pilot_summary_markdown(
        {
            "mode": "geometry_shadow",
            "seed": 3,
            "return_code": 0,
            "evaluator_success": True,
            "output": "results/episode.json",
            "log": "logs/runner.log",
        }
    )

    assert "# CAP-MAS P5.2 pilot run" in markdown
    assert "- mode: geometry_shadow" in markdown
    assert "- seed: 3" in markdown
    assert "- evaluator_success: True" in markdown
    assert "- output: results/episode.json" in markdown
    assert "- log: logs/runner.log" in markdown
