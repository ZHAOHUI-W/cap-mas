"""Build task-specific frozen candidate artifacts from the P5.3 graph template."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from capmas.evaluation.candidate_identity import raw_graph_fingerprint


def _replace(value: object, replacements: tuple[tuple[str, str], ...]) -> object:
    if isinstance(value, str):
        for source, target in replacements:
            value = value.replace(source, target)
            source_slug = source.replace(" ", "_")
            target_slug = target.replace(" ", "_")
            if source_slug != source:
                value = value.replace(source_slug, target_slug)
        return value
    if isinstance(value, list):
        return [_replace(item, replacements) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace(item, replacements)
            for key, item in value.items()
        }
    return value


def _target_pose_for_task(
    *,
    suite_name: str,
    task_id: int,
    target_name: str,
    target_z_offset: float,
) -> tuple[list[float], dict[str, object]]:
    # Importing the low-level wrapper avoids constructing CAP-X perception APIs
    # while still reading the actual task reset state from MuJoCo.
    from capx.envs.simulators.libero import FrankaLiberoTask

    env = FrankaLiberoTask(
        suite_name=suite_name,
        task_id=task_id,
        privileged=False,
        max_steps=32,
        enable_render=False,
    )
    try:
        env.reset(seed=1)
        position, quaternion = env._get_object_pose(target_name)
        if position is None:
            raise RuntimeError(f"target pose unavailable for {target_name!r}")
        pose = np.asarray(position, dtype=float).reshape(3)
        pose[2] += float(target_z_offset)
        sim = env.handle.env.sim
        body_positions: dict[str, list[float]] = {}
        for body_id in range(sim.model.nbody):
            name = sim.model.body_id2name(body_id)
            if name and "_main" in name:
                body_positions[name] = [float(item) for item in sim.data.xpos[body_id]]
        return [float(item) for item in pose], {
            "task_language": env.handle.task_language,
            "target_name": target_name,
            "target_pose_robot_base": [float(item) for item in pose],
            "target_quaternion_robot_base": [float(item) for item in quaternion],
            "body_positions_world": body_positions,
            "reset_seed": 1,
        }
    finally:
        env.close()


def build_candidate_artifact(
    *,
    base_artifact: str | Path,
    output_path: str | Path,
    suite_name: str,
    task_id: int,
    object_name: str,
    target_name: str,
    target_z_offset: float,
    task_id_label: str | None = None,
    target_position: tuple[float, float, float] | None = None,
) -> dict[str, object]:
    raw = json.loads(Path(base_artifact).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("candidates"), list):
        raise ValueError("base candidate artifact must contain candidates")
    if target_position is None:
        target_position, scene = _target_pose_for_task(
            suite_name=suite_name,
            task_id=task_id,
            target_name=target_name,
            target_z_offset=target_z_offset,
        )
    else:
        target_position = [float(item) for item in target_position]
        target_position[2] += float(target_z_offset)
        scene = {
            "target_name": target_name,
            "target_pose_robot_base": list(target_position),
            "target_pose_source": "capx-libero-reset-observation-seed-1",
            "reset_seed": 1,
        }
    replacements = (("akita black bowl", object_name), ("plate", target_name))
    candidates: list[dict[str, object]] = []
    for raw_candidate in raw["candidates"]:
        candidate = deepcopy(raw_candidate)
        graph = _replace(candidate.get("graph"), replacements)
        if not isinstance(graph, dict):
            raise ValueError("base candidate graph must be an object")
        for subgraph in graph.get("subgraphs", []):
            if not isinstance(subgraph, dict):
                continue
            for node in subgraph.get("nodes", []):
                if not isinstance(node, dict):
                    continue
                intent = node.get("motion_intent")
                if isinstance(intent, dict) and intent.get("kind") == "place":
                    intent["target_pose_wxyz_xyz"] = [0.0, 1.0, 0.0, 0.0, *target_position]
                if intent is not None and intent.get("kind") == "place":
                    for call in node.get("skill_calls", []):
                        if not isinstance(call, dict):
                            continue
                        if call.get("skill", {}).get("skill_id") == "goto_pose":
                            args = call.setdefault("args", {})
                            args["position"] = list(target_position)
                            args["quaternion_wxyz"] = [0.0, 1.0, 0.0, 0.0]
        candidate["graph"] = graph
        candidate["candidate_id"] = str(candidate["candidate_id"]).replace(
            "akita_black_bowl", object_name.replace(" ", "_")
        ).replace("plate", target_name.replace(" ", "_"))
        candidate["candidate_fingerprint"] = raw_graph_fingerprint(graph)
        candidate["fingerprint_scope"] = "subgraph"
        candidates.append(candidate)
    artifact = {
        "candidates": candidates,
        "scene_version": int(raw.get("scene_version", 1)),
        "task_id": task_id_label or f"{suite_name}_{task_id}",
        "generator_version": "p55-candidate-template-v1",
        "source_scene": scene,
        "task_spec": {
            "suite_name": suite_name,
            "task_id": task_id,
            "object_name": object_name,
            "target_name": target_name,
            "target_z_offset": target_z_offset,
        },
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--suite-name", required=True)
    parser.add_argument("--task-id", required=True, type=int)
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--target-z-offset", required=True, type=float)
    parser.add_argument("--task-id-label")
    parser.add_argument("--target-position", nargs=3, type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = build_candidate_artifact(
        base_artifact=args.base_artifact,
        output_path=args.output,
        suite_name=args.suite_name,
        task_id=args.task_id,
        object_name=args.object_name,
        target_name=args.target_name,
        target_z_offset=args.target_z_offset,
        task_id_label=args.task_id_label,
        target_position=(
            tuple(args.target_position)
            if args.target_position is not None
            else None
        ),
    )
    print(json.dumps({"output": args.output, "candidate_count": len(artifact["candidates"])}, sort_keys=True))


if __name__ == "__main__":
    main()
