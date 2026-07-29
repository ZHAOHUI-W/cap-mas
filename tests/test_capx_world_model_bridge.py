from __future__ import annotations

from dataclasses import replace

from capmas.backends.capx_libero_factory import build_capx_world_model_enricher
from capmas.backends.capx_libero_factory import build_capx_runtime_from_yaml
from capmas.contracts.core import ArtifactRef


class _Api:
    def __init__(self) -> None:
        self.step = 0

    def get_observation(self):
        self.step += 1
        return {
            "timestamp_ns": self.step * 100,
            "agentview": {
                "intrinsics": [
                    [10.0, 0.0, 0.0],
                    [0.0, 10.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                "pose_mat": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "images": {"depth": [[0.5, 0.5], [0.5, 0.5]]},
            },
            "robot_cartesian_pos": [0.0] * 8,
        }

    def functions(self):
        return {"get_observation": self.get_observation}


class _Env:
    def reset(self, seed=None, options=None):
        del seed, options


def _bundle(api: _Api):
    config = {
        "env": {
            "cfg": {
                "low_level": {"suite_name": "libero_spatial", "task_id": 0},
                "apis": ["FrankaLiberoApi"],
            }
        }
    }
    return build_capx_runtime_from_yaml(
        "unused.yaml",
        loader=lambda _: config,
        instantiator=lambda _: _Env(),
        api_factory=lambda _: (lambda _env: api),
        skill_bindings={"get_observation": "get_observation"},
    )


def test_capx_backend_applies_scene_enricher_on_reset_and_observe() -> None:
    bundle = _bundle(_Api())

    class Enricher:
        def enrich(self, observation, snapshot):
            del observation
            return replace(
                snapshot,
                local_map=ArtifactRef(
                    uri="artifact://test/map",
                    media_type="application/x-capmas-sparse-voxel-map",
                ),
            )

    bundle.backend.set_scene_enricher(Enricher())
    initial = bundle.backend.reset(seed=1).initial_scene
    observed = bundle.backend.observe()

    assert initial.local_map is not None
    assert observed.local_map is not None
    assert observed.scene_version == initial.scene_version + 1


def test_capx_world_model_enricher_updates_local_map_from_live_observation() -> None:
    bundle = _bundle(_Api())
    enricher = build_capx_world_model_enricher(
        bundle.observation_provider,
        depth_subsample=1,
    )
    bundle.backend.set_scene_enricher(enricher)

    initial = bundle.backend.reset(seed=1).initial_scene
    observed = bundle.backend.observe()

    assert enricher.local_map.map_version() >= 1
    assert initial.local_map is not None
    assert observed.local_map is not None
    assert enricher.last_error is None
