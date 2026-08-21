from __future__ import annotations

import numpy as np


class _FakeModel:
    vis = type("Visual", (), {"map": type("Map", (), {"znear": 0.01, "zfar": 50.0})()})()
    cam_fovy = np.array([45.0, 60.0])

    @staticmethod
    def camera_name2id(name: str) -> int:
        return {"agentview": 0, "robot0_eye_in_hand": 1}[name]


class _FakeEnvironment:
    def __init__(self) -> None:
        self._current_obs = {
            "agentview_depth": np.full((2, 2), 1.0),
            "robot0_eye_in_hand_depth": np.full((2, 2), 0.5),
        }
        self.handle = type(
            "Handle",
            (), {"env": type("Environment", (), {"sim": type("Sim", (), {"model": _FakeModel()})()})()},
        )()
        self._render_width = 128
        self._render_height = 96


def test_capture_depth_snapshot_records_raw_metric_and_render_ranges() -> None:
    from capmas.evaluation.libero_depth_probe import capture_depth_snapshot

    environment = _FakeEnvironment()

    record = capture_depth_snapshot(
        environment,
        converter=lambda _sim, depth: np.asarray(depth) * 529.771,
    )

    agentview = record["cameras"]["agentview"]
    assert agentview["raw"]["maximum"] == 1.0
    assert agentview["raw"]["uniform"] is True
    assert agentview["metric"]["minimum"] == 529.771
    assert agentview["metric"]["valid_count"] == 0
    assert record["render"] == {
        "height": 96,
        "width": 128,
        "znear": 0.01,
        "zfar": 50.0,
        "camera_fovy_degrees": {"agentview": 45.0, "robot0_eye_in_hand": 60.0},
    }


def test_installed_depth_probe_restores_environment_method_after_capture() -> None:
    from capmas.evaluation.libero_depth_probe import installed_depth_probe

    class _ProbeEnvironment(_FakeEnvironment):
        def _depth_health_stats(self):
            return {"agentview": {"valid": 4}}

    class _Module:
        get_real_depth_map = staticmethod(lambda _sim, depth: depth)
        FrankaLiberoEnv = _ProbeEnvironment

    original = _Module.FrankaLiberoEnv._depth_health_stats
    environment = _Module.FrankaLiberoEnv()
    with installed_depth_probe(module=_Module) as records:
        assert environment._depth_health_stats() == {"agentview": {"valid": 4}}

    assert len(records) == 1
    assert records[0]["health_stats"] == {"agentview": {"valid": 4}}
    assert _Module.FrankaLiberoEnv._depth_health_stats is original
