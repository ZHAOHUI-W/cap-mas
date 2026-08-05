from __future__ import annotations

import json

from scripts.create_p55_real_layout_manifest import build_real_layout_manifest


def test_manifest_suite_id_and_case_count_follow_seed_count(tmp_path) -> None:
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    for name in ("spatial_0.json", "goal_1.json", "object_6.json"):
        (candidates / name).write_text(json.dumps({"candidate": name}))

    manifest = build_real_layout_manifest(
        output_path=tmp_path / "manifest.json",
        assets_root=tmp_path,
        seeds=tuple(range(1, 11)),
    )

    assert manifest.suite_id == "p55-real-layout-3family-10seed"
    assert len(manifest.cases) == 60
    assert len({case.pair_id for case in manifest.cases}) == 30
    assert {case.seed for case in manifest.cases} == set(range(1, 11))
