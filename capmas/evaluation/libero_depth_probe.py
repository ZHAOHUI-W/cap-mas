"""Reversible CAP-X/LIBERO depth diagnostics for isolated investigations."""

from __future__ import annotations

import importlib
import math
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager

import numpy as np

DEFAULT_CAMERAS = ("agentview", "robot0_eye_in_hand")
DEFAULT_VALID_DEPTH_RANGE_M = (0.015, 20.0)


def _number(value: object) -> float | None:
    if isinstance(value, (int, float, np.number)) and math.isfinite(float(value)):
        return float(value)
    return None


def _depth_summary(
    value: object | None,
    *,
    valid_range_m: tuple[float, float] = DEFAULT_VALID_DEPTH_RANGE_M,
) -> dict[str, object] | None:
    if value is None:
        return None
    array = np.asarray(value).squeeze()
    finite = np.isfinite(array)
    finite_values = array[finite]
    valid = finite & (array >= valid_range_m[0]) & (array <= valid_range_m[1])
    minimum = float(np.min(finite_values)) if finite_values.size else None
    maximum = float(np.max(finite_values)) if finite_values.size else None
    return {
        "shape": tuple(int(size) for size in array.shape),
        "finite_count": int(np.count_nonzero(finite)),
        "valid_count": int(np.count_nonzero(valid)),
        "minimum": minimum,
        "maximum": maximum,
        "mean": float(np.mean(finite_values)) if finite_values.size else None,
        "uniform": bool(finite_values.size and minimum == maximum),
    }


def _render_metadata(environment: object, cameras: Sequence[str]) -> dict[str, object]:
    handle = getattr(environment, "handle", None)
    low_level = getattr(handle, "env", None)
    sim = getattr(low_level, "sim", None)
    model = getattr(sim, "model", None)
    visual = getattr(model, "vis", None)
    visual_map = getattr(visual, "map", None)
    fovy: dict[str, float | None] = {}
    for camera in cameras:
        try:
            camera_id = model.camera_name2id(camera)
            fovy[camera] = _number(model.cam_fovy[camera_id])
        except (AttributeError, KeyError, TypeError, ValueError):
            fovy[camera] = None
    return {
        "width": _number(getattr(environment, "_render_width", None)),
        "height": _number(getattr(environment, "_render_height", None)),
        "znear": _number(getattr(visual_map, "znear", None)),
        "zfar": _number(getattr(visual_map, "zfar", None)),
        "camera_fovy_degrees": fovy,
    }


def _converter_name(converter: Callable[..., object]) -> str:
    module = getattr(converter, "__module__", type(converter).__module__)
    name = getattr(converter, "__qualname__", type(converter).__qualname__)
    return f"{module}.{name}"


def capture_depth_snapshot(
    environment: object,
    converter: Callable[[object, object], object],
    *,
    cameras: Sequence[str] = DEFAULT_CAMERAS,
) -> dict[str, object]:
    """Capture raw and converted depth without changing the environment state."""

    current_obs = getattr(environment, "_current_obs", None)
    if not isinstance(current_obs, Mapping):
        raise TypeError("CAP-X environment does not expose a current observation mapping")
    handle = getattr(environment, "handle", None)
    low_level = getattr(handle, "env", None)
    sim = getattr(low_level, "sim", None)
    records: dict[str, object] = {}
    for camera in cameras:
        raw = current_obs.get(f"{camera}_depth")
        record: dict[str, object] = {"raw": _depth_summary(raw)}
        if raw is None:
            record["metric"] = None
            record["conversion_error"] = "depth observation is unavailable"
        else:
            try:
                raw_array = np.asarray(raw)
                converted = converter(sim, raw_array[::-1])
                record["metric"] = _depth_summary(converted)
                record["conversion_error"] = None
            except Exception as exc:  # noqa: BLE001 - diagnostic collection must not alter reset behavior.
                record["metric"] = None
                record["conversion_error"] = f"{type(exc).__name__}: {exc}"
        records[camera] = record
    return {
        "captured_at_ns": time.time_ns(),
        "conversion_source": _converter_name(converter),
        "valid_depth_range_m": list(DEFAULT_VALID_DEPTH_RANGE_M),
        "render": _render_metadata(environment, cameras),
        "cameras": records,
    }


@contextmanager
def installed_depth_probe(
    *,
    module: object | None = None,
    cameras: Sequence[str] = DEFAULT_CAMERAS,
) -> Iterator[list[dict[str, object]]]:
    """Record CAP-X reset depth checks and restore the original method on exit."""

    target = module or importlib.import_module("capx.envs.simulators.libero")
    environment_type = getattr(target, "FrankaLiberoEnv", None)
    converter = getattr(target, "get_real_depth_map", None)
    original = getattr(environment_type, "_depth_health_stats", None)
    if environment_type is None or not callable(converter) or not callable(original):
        raise TypeError("CAP-X LIBERO module does not expose depth probe boundaries")
    records: list[dict[str, object]] = []

    def instrumented(environment: object) -> object:
        health = original(environment)
        try:
            record = capture_depth_snapshot(environment, converter, cameras=cameras)
        except Exception as exc:  # noqa: BLE001 - preserve the production reset result.
            record = {
                "captured_at_ns": time.time_ns(),
                "capture_error": f"{type(exc).__name__}: {exc}",
            }
        record["health_stats"] = health
        records.append(record)
        return health

    environment_type._depth_health_stats = instrumented
    try:
        yield records
    finally:
        environment_type._depth_health_stats = original


__all__ = [
    "DEFAULT_CAMERAS",
    "DEFAULT_VALID_DEPTH_RANGE_M",
    "capture_depth_snapshot",
    "installed_depth_probe",
]
