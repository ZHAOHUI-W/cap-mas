from scripts.debug_libero_grasp_probe import append_phase_snapshot


def test_grasp_probe_keeps_each_physics_boundary_as_a_named_record() -> None:
    snapshots: list[dict[str, object]] = []

    append_phase_snapshot(snapshots, "after_close_gripper", {"bowl": [1, 2, 3]})
    append_phase_snapshot(snapshots, "after_lift", {"bowl": [4, 5, 6]})

    assert [item["phase"] for item in snapshots] == [
        "after_close_gripper",
        "after_lift",
    ]
    assert snapshots[0]["physics"] == {"bowl": [1, 2, 3]}
    assert snapshots[1]["physics"] == {"bowl": [4, 5, 6]}
