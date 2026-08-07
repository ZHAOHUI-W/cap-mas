"""Probe CAP-X LIBERO grasp execution without the LLM scheduler."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any
from uuid import uuid4


def append_phase_snapshot(
    snapshots: list[dict[str, object]],
    phase: str,
    physics: dict[str, object],
) -> None:
    """Append an immutable phase-labelled physics observation to a probe run."""
    snapshots.append({"phase": phase, "physics": physics})


def resolve_probe_object_names(object_query: str, target_query: str) -> tuple[str, str]:
    """Return the source and target names used by the CAP-X runtime factory."""
    object_name = " ".join(str(object_query).split())
    target_name = " ".join(str(target_query).split())
    if not object_name or not target_name:
        raise ValueError("object_query and target_query must be non-empty")
    return object_name, target_name


def capx_dependency_paths(capx_root: str | Path, project_root: str | Path) -> tuple[str, ...]:
    """Return import roots for CAP-X and its vendored LIBERO package."""
    capx_path = Path(capx_root)
    project_path = Path(project_root)
    return (
        str(project_path),
        str(capx_path / "capx" / "third_party" / "LIBERO-PRO" / "libero"),
        str(capx_path / "capx" / "third_party" / "libero_dependencies" / "robosuite"),
        str(capx_path / "capx" / "third_party" / "LIBERO-PRO"),
        str(capx_path),
    )


def select_grasp_pose(
    sample_fn: object,
    pose_fn: object,
    *,
    allow_pose_fallback: bool,
) -> tuple[object, object, str, str | None]:
    """Select a grasp pose, preserving raw sampler failure for diagnostics."""
    if not callable(sample_fn) or not callable(pose_fn):
        raise TypeError("sample_fn and pose_fn must be callable")
    try:
        position, quaternion = sample_fn()
        return position, quaternion, "sample_grasp_pose", None
    except Exception as exc:
        if not allow_pose_fallback:
            raise
        position, quaternion = pose_fn()
        if position is None or quaternion is None:
            raise ValueError("object pose fallback returned no pose") from exc
        return position, quaternion, "object_pose_fallback", str(exc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--use-multiview", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--object-query", default="akita black bowl")
    parser.add_argument("--target-query", default="plate")
    parser.add_argument(
        "--output-root",
        default="outputs/debug/grasp_probe",
        help="root for a unique structured probe directory",
    )
    parser.add_argument(
        "--z-approaches",
        default="0.0,0.05,0.08,0.10",
        help="comma-separated approach offsets to test",
    )
    parser.add_argument(
        "--privileged-api",
        action="store_true",
        help="use the CAP-X privileged pose API for diagnostic isolation only",
    )
    parser.add_argument(
        "--pose-fallback",
        action="store_true",
        help="use get_object_pose if sample_grasp_pose has no candidate; diagnostic only",
    )
    parser.add_argument(
        "--capmas-grounded",
        action="store_true",
        help="invoke CAP-MAS grounded skill bindings instead of raw CAP-X functions",
    )
    parser.add_argument(
        "--place-after-grasp",
        action="store_true",
        help="after lifting, move to the visual plate pose and release for physics diagnostics",
    )
    args = parser.parse_args()

    run_dir = (
        Path(args.output_root)
        / f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path = run_dir / "probe.log"
    results_path = run_dir / "results.json"
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_handle = log_path.open("w", encoding="utf-8", buffering=1)

    class Tee:
        def __init__(self, *streams: object) -> None:
            self.streams = streams

        def write(self, value: str) -> int:
            for stream in self.streams:
                stream.write(value)  # type: ignore[attr-defined]
            return len(value)

        def flush(self) -> None:
            for stream in self.streams:
                stream.flush()  # type: ignore[attr-defined]

        def isatty(self) -> bool:
            return False

    sys.stdout = Tee(original_stdout, log_handle)  # type: ignore[assignment]
    sys.stderr = Tee(original_stderr, log_handle)  # type: ignore[assignment]
    print(f"probe_run_dir={run_dir}")

    project_root = Path(__file__).resolve().parents[1]
    capx_root = project_root.parent / "cap-x"
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/capmas-mpl")
    for path in capx_dependency_paths(capx_root, project_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from capx.envs.configs.loader import DictLoader
    from capx.envs.runner import _start_api_servers
    from capmas.backends.capx_libero_factory import build_capx_runtime_from_yaml
    from capmas.verification.libero import LiberoObservableVerifier

    def jsonable(value: object) -> object:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, dict):
            return {str(key): jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [jsonable(item) for item in value]
        if hasattr(value, "tolist"):
            return jsonable(value.tolist())  # type: ignore[union-attr]
        return repr(value)

    def physics_diagnostics(low_level: object) -> dict[str, Any]:
        candidates = [low_level, getattr(low_level, "handle", None)]
        handle = getattr(low_level, "handle", None)
        candidates.extend((getattr(handle, "env", None), getattr(handle, "sim", None)))
        sim = None
        for candidate in candidates:
            possible = getattr(candidate, "sim", candidate)
            if getattr(possible, "model", None) is not None and getattr(possible, "data", None) is not None:
                sim = possible
                break
        if sim is None:
            return {"error": "mujoco sim was not discoverable", "candidate_types": [type(item).__name__ for item in candidates]}
        model = sim.model
        data = sim.data
        body_count = int(getattr(model, "nbody", 0))
        body_entries: dict[str, object] = {}
        for body_id in range(body_count):
            try:
                name = model.body_id2name(body_id)
            except Exception:
                name = None
            if isinstance(name, bytes):
                name = name.decode(errors="replace")
            name = str(name or body_id)
            lowered = name.lower()
            if not any(token in lowered for token in ("akita", "bowl", "plate", "gripper", "panda_hand", "eef", "robot0_base")):
                continue
            entry: dict[str, object] = {"body_id": body_id}
            try:
                entry["position_xyz"] = [float(item) for item in data.xpos[body_id]]
            except Exception as exc:
                entry["position_error"] = str(exc)
            try:
                entry["quaternion_wxyz"] = [float(item) for item in data.xquat[body_id]]
            except Exception as exc:
                entry["quaternion_error"] = str(exc)
            body_entries[name] = entry
        result = {
            "sim_type": type(sim).__name__,
            "body_count": body_count,
            "bodies": body_entries,
            "candidate_types": [type(item).__name__ for item in candidates],
        }
        current_obs = getattr(low_level, "_current_obs", None)
        if isinstance(current_obs, dict):
            qpos = current_obs.get("robot0_gripper_qpos")
            if qpos is not None:
                result["robot0_gripper_qpos"] = jsonable(qpos)
        fraction = getattr(low_level, "_gripper_fraction", None)
        if fraction is not None:
            result["commanded_gripper_fraction"] = float(fraction)
        return result

    def privileged_object_poses(low_level: object) -> dict[str, object]:
        """Expose simulator-aligned robot-base poses only for diagnostics."""
        if not args.privileged_api:
            return {}
        getter = getattr(low_level, "_get_all_object_poses", None)
        if not callable(getter):
            return {"error": "low-level environment has no privileged pose helper"}
        try:
            value = getter()
            return {str(name): jsonable(pose) for name, pose in value.items()}
        except Exception as exc:
            return {"error": f"privileged pose read failed: {exc}"}

    def phase_physics(low_level: object) -> dict[str, object]:
        result = physics_diagnostics(low_level)
        privileged = privileged_object_poses(low_level)
        if privileged:
            result["privileged_object_poses_robot_base"] = privileged
        return result

    def call_pose_api(
        function: object,
        object_name: str,
        *,
        use_multiview: bool | None = None,
    ) -> object:
        if not callable(function):
            raise TypeError("pose API is not callable")
        requested_multiview = args.use_multiview if use_multiview is None else use_multiview
        try:
            return function(object_name, use_multiview=requested_multiview)
        except TypeError as exc:
            if "use_multiview" not in str(exc):
                raise
            return function(object_name)

    config = DictLoader.load(args.config_path)
    servers = _start_api_servers(config.get("api_servers"))
    try:
        object_query, target_query = resolve_probe_object_names(
            args.object_query,
            args.target_query,
        )
        bundle = build_capx_runtime_from_yaml(
            args.config_path,
            loader=lambda _: config,
            object_names=(object_query, target_query),
        )
        api = bundle.api.functions()
        verifier = LiberoObservableVerifier()

        if args.capmas_grounded:
            from capmas.contracts.action import ExecutionBudget
            from capmas.contracts.core import SkillRef

            def call_grounded(skill_id: str, name: str) -> object:
                skill = bundle.skill_registry.get(SkillRef(skill_id, "capx-compat-1"))
                result = skill.execute(
                    {"object_name": name, "use_multiview": args.use_multiview},
                    ExecutionBudget(max_duration_ms=30_000, max_sim_steps=500),
                )
                if not result.ok:
                    raise RuntimeError(
                        f"grounded {skill_id} failed: {result.error_type}: {result.error_message}"
                    )
                if not isinstance(result.output, dict) or "result" not in result.output:
                    raise RuntimeError(f"grounded {skill_id} returned no result")
                return result.output["result"]

            def get_pose(name: str) -> object:
                return call_grounded("get_object_pose", name)

            def sample_pose(name: str) -> object:
                return call_grounded("sample_grasp_pose", name)
        else:
            def get_pose(name: str) -> object:
                return call_pose_api(api["get_object_pose"], name)

            def sample_pose(name: str) -> object:
                return call_pose_api(api["sample_grasp_pose"], name)

        bundle.backend.reset(seed=args.seed)
        approaches = tuple(float(item.strip()) for item in args.z_approaches.split(",") if item.strip())
        reports: list[dict[str, object]] = []
        for approach in approaches:
            bundle.backend.reset(seed=args.seed)
            phase_snapshots: list[dict[str, object]] = []
            append_phase_snapshot(
                phase_snapshots,
                "after_reset",
                phase_physics(bundle.low_level_environment),
            )
            observed_position, observed_quaternion = get_pose(object_query)
            (
                position,
                quaternion,
                grasp_source,
                grasp_error,
            ) = select_grasp_pose(
                lambda: sample_pose(object_query),
                lambda: get_pose(object_query),
                allow_pose_fallback=args.pose_fallback,
            )
            sampled_position = [float(x) for x in position]
            sampled_quaternion = [float(x) for x in quaternion]
            observed_position = None if observed_position is None else [float(x) for x in observed_position]
            observed_quaternion = None if observed_quaternion is None else [float(x) for x in observed_quaternion]
            api["goto_pose"](position, quaternion, z_approach=approach)
            append_phase_snapshot(
                phase_snapshots,
                "after_grasp_approach",
                phase_physics(bundle.low_level_environment),
            )
            api["close_gripper"]()
            append_phase_snapshot(
                phase_snapshots,
                "after_close_gripper",
                phase_physics(bundle.low_level_environment),
            )
            api["goto_pose"](
                [float(position[0]), float(position[1]), float(position[2]) + 0.12],
                quaternion,
                z_approach=0.0,
            )
            append_phase_snapshot(
                phase_snapshots,
                "after_lift",
                phase_physics(bundle.low_level_environment),
            )
            target_position = None
            target_quaternion = None
            if args.place_after_grasp:
                target_position, target_quaternion = get_pose(target_query)
                api["goto_pose"](
                    target_position,
                    [0.0, 1.0, 0.0, 0.0],
                    z_approach=0.12,
                )
                append_phase_snapshot(
                    phase_snapshots,
                    "before_release",
                    phase_physics(bundle.low_level_environment),
                )
                api["open_gripper"]()
                append_phase_snapshot(
                    phase_snapshots,
                    "after_release",
                    phase_physics(bundle.low_level_environment),
                )
            scene = bundle.backend.observe()
            post_pose = {}
            for multiview in (True, False):
                try:
                    pose = (
                        call_grounded("get_object_pose", object_query)
                        if args.capmas_grounded and multiview == args.use_multiview
                        else call_pose_api(
                            api["get_object_pose"],
                            object_query,
                            use_multiview=multiview,
                        )
                    )
                    post_pose[str(multiview)] = {
                        "position": None if pose[0] is None else [float(x) for x in pose[0]],
                        "quaternion": None if pose[1] is None else [float(x) for x in pose[1]],
                    }
                except Exception as exc:
                    post_pose[str(multiview)] = {"error": str(exc)}
            sim = getattr(getattr(bundle.low_level_environment, "handle", None), "env", None)
            physics = physics_diagnostics(bundle.low_level_environment)
            predicates = (
                (f"object_at_target({object_query},{target_query})", "gripper_open()")
                if args.place_after_grasp
                else (f"object_in_gripper({object_query})", "gripper_closed()")
            )
            predicate_reports = verifier.evaluate_predicates(predicates, scene)
            report = {
                "z_approach": approach,
                "sample_position": sampled_position,
                "sample_quaternion": sampled_quaternion,
                "grasp_source": grasp_source,
                "grasp_sampler_error": grasp_error,
                "target_position": None if target_position is None else [float(x) for x in target_position],
                "target_quaternion": None if target_quaternion is None else [float(x) for x in target_quaternion],
                "observed_position_before_sample": observed_position,
                "observed_quaternion_before_sample": observed_quaternion,
                "ee_pose": scene.robot.get("ee_pose_wxyz_xyz"),
                "object_pose": next(
                    (
                        track.pose_wxyz_xyz
                        for track in scene.objects
                        if track.track_id == object_query
                    ),
                    None,
                ),
                "physics": physics,
                "phase_snapshots": phase_snapshots,
                "post_get_object_pose": post_pose,
                "task_completed": bool(bundle.low_level_environment.task_completed()),
                "reports": [
                    {"name": item.name, "passed": item.passed, "reason": item.reason}
                    for item in predicate_reports
                ],
            }
            reports.append(report)
            print(json.dumps(jsonable(report), sort_keys=True), flush=True)
        results_path.write_text(json.dumps(jsonable(reports), indent=2, sort_keys=True), encoding="utf-8")
        print(f"probe_results={results_path}")
    finally:
        for process in servers:
            process.terminate()
            process.join(timeout=5)
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_handle.flush()
        log_handle.close()


if __name__ == "__main__":
    main()
