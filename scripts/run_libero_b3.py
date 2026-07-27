from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the CAP-MAS deterministic fixed-graph LIBERO B3 episode."
    )
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--output", default="outputs/capmas_libero_b3/episode.json")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--object-name", default="akita black bowl")
    parser.add_argument("--target-name", default="plate")
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--skip-api-servers", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    capx_root = project_root.parent / "cap-x"
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/capmas-mpl")
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    for path in (
        capx_root / "capx" / "third_party" / "libero_dependencies" / "robosuite",
        capx_root / "capx" / "third_party" / "LIBERO-PRO",
        capx_root,
    ):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from capmas.agents.libero import build_libero_spatial_task0_mission_graph
    from capmas.backends.capx_libero_factory import build_capx_runtime_from_yaml
    from capmas.graph.serialization import mission_graph_to_dict
    from capmas.runtime.action_lease import ActionLeaseManager
    from capmas.runtime.artifact_bus import ArtifactStore, EventBus
    from capmas.runtime.graph_interpreter import FixedGraphInterpreter
    from capmas.runtime.scheduler import FixedGraphScheduler
    from capmas.runtime.orchestrator import RuntimeOrchestrator
    from capmas.runtime.state_store import InMemoryStateStore
    from capmas.verification.libero import LiberoObservableVerifier
    from capmas.runtime.episode_runner import to_jsonable

    from capx.envs.configs.loader import DictLoader

    config = DictLoader.load(args.config_path)
    server_processes = []
    try:
        if not args.skip_api_servers:
            from capx.envs.runner import _start_api_servers

            server_processes = _start_api_servers(config.get("api_servers"))
        bundle = build_capx_runtime_from_yaml(
            args.config_path,
            loader=lambda _: config,
            object_names=(args.object_name, args.target_name),
        )
        verifier = LiberoObservableVerifier()
        runtime = RuntimeOrchestrator(
            backend=bundle.backend,
            state_store=InMemoryStateStore(),
            skill_registry=bundle.skill_registry,
            lease_manager=ActionLeaseManager(),
            verifier=verifier,
        )
        episode = runtime.backend.reset(seed=args.seed)
        runtime.start_episode(episode)
        scene = runtime.state_store.latest()
        graph = build_libero_spatial_task0_mission_graph(
            object_name=args.object_name,
            target_name=args.target_name,
            include_home=False,
            parent_scene_version=scene.scene_version,
        )
        store = ArtifactStore()
        event_bus = EventBus()
        result = FixedGraphInterpreter(
            FixedGraphScheduler(runtime),
            artifact_store=store,
            event_bus=event_bus,
            max_steps=args.max_steps,
        ).run(graph, scene, context=None)
        payload = {
            "baseline": "B3",
            "task_id": bundle.task_id,
            "seed": args.seed,
            "graph": mission_graph_to_dict(graph),
            "result": result,
            "evaluator_success": bundle.backend.evaluator_success(),
            "failure_artifacts": store.snapshot(),
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))
        print(f"CAP-MAS B3 output: {args.output}")
        print(f"CAP-MAS B3 evaluator_success: {payload['evaluator_success']}")
        print(f"CAP-MAS B3 completed: {result.completed}")
        print(f"CAP-MAS B3 terminal_subgraph: {result.terminal_subgraph}")
    finally:
        for process in server_processes:
            process.terminate()
            process.join(timeout=5)


if __name__ == "__main__":
    main()
