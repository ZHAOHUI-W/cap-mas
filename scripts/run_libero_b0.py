from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CAP-MAS LIBERO V1 smoke episode.")
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--output", default="outputs/capmas_libero_b0/episode.json")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-cycles", type=int, default=1)
    parser.add_argument("--object-name", default="akita black bowl")
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

    from capmas.agents.libero import LiberoSpatialTask0Policy
    from capmas.backends.capx_libero_factory import build_capx_runtime_from_yaml
    from capmas.contracts.core import SkillRef
    from capmas.runtime.action_lease import ActionLeaseManager
    from capmas.runtime.episode_runner import EpisodeRunner, write_episode_result
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
            object_names=(args.object_name, "plate"),
        )
        runtime = RuntimeOrchestrator(
            backend=bundle.backend,
            state_store=InMemoryStateStore(),
            skill_registry=bundle.skill_registry,
            lease_manager=ActionLeaseManager(),
            verifier=LiberoObservableVerifier(),
        )
        policy = LiberoSpatialTask0Policy(
            object_name=args.object_name,
            include_home=bundle.skill_registry.has(
                SkillRef("goto_home_joint_position", "capx-compat-1")
            ),
        )
        result = EpisodeRunner(runtime).run(
            task_id=bundle.task_id,
            seed=args.seed,
            policy_step=policy.propose,
            max_cycles=args.max_cycles,
        )
        write_episode_result(result, args.output)
        print(f"CAP-MAS output: {args.output}")
        print(f"CAP-MAS evaluator_success: {result.evaluator_success}")
    finally:
        for process in server_processes:
            process.terminate()
            process.join(timeout=5)


if __name__ == "__main__":
    main()
