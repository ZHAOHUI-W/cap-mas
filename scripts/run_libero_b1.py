from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CAP-MAS P2.5 multi-cycle LIBERO episode.")
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--output", default="outputs/capmas_libero_b1/episode.json")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-cycles", type=int, default=8)
    parser.add_argument("--max-recoveries", type=int, default=2)
    parser.add_argument("--object-name", default="akita black bowl")
    parser.add_argument("--target-name", default="plate")
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

    from capmas.agents.libero import LiberoSpatialTask0MultiStepPolicy
    from capmas.backends.capx_libero_factory import build_capx_runtime_from_yaml
    from capmas.runtime.action_lease import ActionLeaseManager
    from capmas.runtime.episode_runner import MultiCycleEpisodeRunner, write_episode_result
    from capmas.runtime.orchestrator import RuntimeOrchestrator
    from capmas.runtime.state_store import InMemoryStateStore
    from capmas.verification.libero import LiberoObservableVerifier

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
        policy = LiberoSpatialTask0MultiStepPolicy(
            object_name=args.object_name,
            target_name=args.target_name,
            include_home=False,
        )
        object_id = args.object_name.replace(" ", "_")
        target_id = args.target_name.replace(" ", "_")
        result = MultiCycleEpisodeRunner(runtime).run(
            task_id=bundle.task_id,
            seed=args.seed,
            policy_step=policy.propose,
            goal_check=lambda scene: verifier.goal_satisfied(
                (
                    f"object_at_target({object_id},{target_id})",
                    "gripper_open",
                ),
                scene,
            ),
            recovery_step=policy,
            max_cycles=args.max_cycles,
            max_recoveries=args.max_recoveries,
        )
        write_episode_result(result, args.output)
        print(f"CAP-MAS output: {args.output}")
        print(f"CAP-MAS evaluator_success: {result.evaluator_success}")
        print(f"CAP-MAS total_cycles: {result.total_cycles}")
        print(f"CAP-MAS recovery_attempts: {result.recovery_attempts}")
        print(f"CAP-MAS stop_reason: {result.stop_reason}")
    finally:
        for process in server_processes:
            process.terminate()
            process.join(timeout=5)


if __name__ == "__main__":
    main()
