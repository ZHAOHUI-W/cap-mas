from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from capmas.backends.capx import CAPXObservationProvider, CAPXStreamingObservationSource
from capmas.perception.artifacts import InMemoryArtifactStore


def _load_runner_module():
    path = Path(__file__).parents[1] / "scripts" / "run_libero_b5.py"
    spec = importlib.util.spec_from_file_location("run_libero_b5", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_live_world_model_uses_capx_artifacts_depth_and_object_tracks() -> None:
    np = pytest.importorskip("numpy")
    runner = _load_runner_module()
    store = InMemoryArtifactStore()
    depth = np.ones((4, 4), dtype=np.float32)

    def observation():
        return {
            "timestamp_ns": 100,
            "agentview": {
                "intrinsics": np.eye(3),
                "pose_mat": np.eye(4),
                "images": {"rgb": np.zeros((4, 4, 3), dtype=np.uint8), "depth": depth},
            },
            "robot_joint_pos": np.zeros(7),
            "robot_cartesian_pos": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
        }

    provider = CAPXObservationProvider(
        observation_fn=observation,
        artifacts=store,
        object_poses_fn=lambda: {"cube": (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))},
    )
    source = CAPXStreamingObservationSource(provider, episode_id="ep", episode_epoch=1)
    service = runner.build_live_world_model(provider, depth_subsample=2)

    snapshot = service.process(source.capture(), previous=None)

    assert snapshot.objects[0].track_id == "cube"
    assert snapshot.local_map is not None
    assert len(snapshot.source_artifacts) == 2
