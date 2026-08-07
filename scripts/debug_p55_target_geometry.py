"""Record CAP-X container geometry used for LIBERO placement diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from capmas.evaluation.phase5_artifacts import Phase5RunDirectory
from scripts.run_libero_p53_online import _setup_capx_paths
from scripts.run_libero_p55_ood import _start_capx_api_servers


def _jsonable(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())  # type: ignore[union-attr]
    return repr(value)


def _point_stats(points: object) -> dict[str, object]:
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        return {"count": 0, "error": f"unexpected point shape: {values.shape}"}
    values = values[np.isfinite(values).all(axis=1)]
    if len(values) == 0:
        return {"count": 0}
    return {
        "count": int(len(values)),
        "min": np.min(values, axis=0),
        "max": np.max(values, axis=0),
        "mean": np.mean(values, axis=0),
        "median": np.median(values, axis=0),
        "p90": np.percentile(values, 90, axis=0),
        "p97": np.percentile(values, 97, axis=0),
        "p99": np.percentile(values, 99, axis=0),
    }


def main() -> None:
    _setup_capx_paths()
    config_path = PROJECT_ROOT / "configs/phase5/capx_libero_object_6_nonprivileged.yaml"
    from capx.envs.configs.loader import DictLoader

    config = DictLoader.load(str(config_path))
    run = Phase5RunDirectory.create(
        "outputs/phase5/P5.5_target_geometry_probe_20260806",
        "probe",
        "object6_seed1",
    )
    servers = _start_capx_api_servers(str(config_path))
    bundle = None
    payload: dict[str, object] = {
        "config_path": str(config_path),
        "seed": 1,
        "target_name": "basket",
    }
    try:
        from capmas.backends.capx_libero_factory import build_capx_runtime_from_yaml

        bundle = build_capx_runtime_from_yaml(
            config_path,
            loader=lambda _: config,
            object_names=("butter", "basket"),
        )
        bundle.backend.reset(seed=1)
        functions = bundle.api.functions()
        point_fn = functions["get_object_3d_points_and_masks_from_language"]
        pose_fn = functions["get_object_pose"]
        obb_fn = functions["get_oriented_bounding_box_from_3d_points"]
        for use_multiview in (False, True):
            raw = point_fn("basket", use_multiview=use_multiview)
            stats = _point_stats(raw.get("points_3d"))
            points = np.asarray(raw.get("points_3d"), dtype=float)
            points = points[np.isfinite(points).all(axis=1)]
            if len(points):
                obb = obb_fn(points)
                stats["obb_center"] = obb["center"]
                stats["obb_extent"] = obb["extent"]
            pose = pose_fn("basket", use_multiview=use_multiview)
            stats["get_object_pose"] = pose
            payload[f"multiview_{str(use_multiview).lower()}"] = stats

        sim = bundle.low_level_environment.handle.env.sim
        body_id = sim.model.body_name2id("basket_1_main")
        payload["physical_body_position_robot_base"] = sim.data.xpos[body_id]
    except Exception as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if bundle is not None:
            try:
                bundle.backend.stop(None)
            except Exception:
                pass
        for process in servers:
            process.terminate()
            process.join(timeout=5)

    run.write_json("results/target_geometry.json", _jsonable(payload))
    run.write_text("logs/runner.log", json.dumps(_jsonable(payload), indent=2) + "\n")
    run.finalize_manifest()
    print(f"run_dir={run.path}")
    print(json.dumps(_jsonable(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
