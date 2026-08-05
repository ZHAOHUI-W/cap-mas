from __future__ import annotations

import numpy as np
import pytest

from capmas.evaluation.layout_variants import (
    LayoutVariant,
    apply_layout_variant,
    layout_variant_from_mapping,
)


class _Joint:
    type = np.asarray([0])
    qposadr = np.asarray([0])


class _Model:
    def __init__(self) -> None:
        self._body_ids = {"bowl_1_main": 0}
        self._joint_ids = {"bowl_1_joint0": 0}

    def body_name2id(self, name: str) -> int:
        return self._body_ids[name]

    def joint_name2id(self, name: str) -> int:
        return self._joint_ids[name]

    def joint(self, _joint_id: int) -> _Joint:
        return _Joint()


class _Data:
    def __init__(self) -> None:
        self.qpos = np.asarray([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0], dtype=float)
        self.xpos = np.asarray([[0.1, 0.2, 0.3]], dtype=float)
        self.xquat = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=float)


class _Sim:
    def __init__(self) -> None:
        self.model = _Model()
        self.data = _Data()

    def forward(self) -> None:
        self.data.xpos[0] = self.data.qpos[:3]
        self.data.xquat[0] = self.data.qpos[3:7]


def test_layout_variant_moves_free_body_and_records_state_fingerprint() -> None:
    sim = _Sim()
    variant = layout_variant_from_mapping(
        {
            "variant_id": "translated-v1",
            "layout_family": "layout-translated",
            "transforms": [
                {
                    "body_name": "bowl_1_main",
                    "translation_delta_xyz": [0.08, -0.03, 0.0],
                }
            ],
        }
    )

    report = apply_layout_variant(sim, variant)

    assert report.applied is True
    assert report.variant_id == "translated-v1"
    assert report.state_fingerprint
    assert report.transforms[0].before_position == (0.1, 0.2, 0.3)
    assert report.transforms[0].after_position == (0.18, 0.17, 0.3)
    np.testing.assert_allclose(sim.data.qpos[:3], [0.18, 0.17, 0.3])


def test_layout_variant_rejects_static_body_without_free_joint() -> None:
    class StaticModel(_Model):
        def joint_name2id(self, name: str) -> int:
            raise KeyError(name)

    sim = _Sim()
    sim.model = StaticModel()
    variant = LayoutVariant.from_mapping(
        {
            "variant_id": "invalid",
            "layout_family": "layout-invalid",
            "transforms": [{"body_name": "bowl_1_main", "translation_delta_xyz": [0.1, 0.0, 0.0]}],
        }
    )

    with pytest.raises(ValueError, match="free joint"):
        apply_layout_variant(sim, variant)


def test_layout_reset_hook_refreshes_capx_observation_cache() -> None:
    from capmas.evaluation.layout_variants import LayoutResetHook

    sim = _Sim()

    class LiberoEnv:
        def __init__(self) -> None:
            self.sim = sim

        def _get_observations(self):
            return {"bowl_1_pos": [0.18, 0.17, 0.3]}

    class Environment:
        def __init__(self) -> None:
            self.handle = type("Handle", (), {"env": LiberoEnv()})()
            self._current_obs = {"bowl_1_pos": [0.1, 0.2, 0.3]}

    environment = Environment()
    LayoutResetHook(
        {
            "variant_id": "refresh-v1",
            "layout_family": "layout-refresh",
            "transforms": [
                {"body_name": "bowl_1_main", "translation_delta_xyz": [0.08, -0.03, 0.0]}
            ],
        }
    )(environment, 1, {})

    assert environment._current_obs["bowl_1_pos"] == [0.18, 0.17, 0.3]


def test_layout_reset_hook_does_not_refresh_for_zero_delta_variant() -> None:
    from capmas.evaluation.layout_variants import LayoutResetHook

    sim = _Sim()
    refresh_calls: list[bool] = []

    class LiberoEnv:
        def __init__(self) -> None:
            self.sim = sim

        def _get_observations(self, force_update=False):
            refresh_calls.append(bool(force_update))
            return {"bowl_1_pos": [0.1, 0.2, 0.3]}

    class Environment:
        def __init__(self) -> None:
            self.handle = type("Handle", (), {"env": LiberoEnv()})()
            self._current_obs = {"bowl_1_pos": [0.1, 0.2, 0.3]}

    environment = Environment()
    LayoutResetHook(
        {
            "variant_id": "native-v1",
            "layout_family": "layout-native",
            "transforms": [
                {"body_name": "bowl_1_main", "translation_delta_xyz": [0.0, 0.0, 0.0]}
            ],
        }
    )(environment, 1, {})

    assert refresh_calls == []
