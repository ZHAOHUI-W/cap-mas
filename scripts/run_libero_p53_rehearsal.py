"""Run isolated CAP-X/LIBERO process rehearsal for serialized candidates."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys
from collections.abc import Callable, Mapping, Sequence
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from capmas.evaluation.libero_rehearsal import LiberoRehearsalConfig, LiberoRehearsalWorker
from capmas.evaluation.phase5_artifacts import Phase5RunDirectory
from capmas.evaluation.rehearsal import RehearsalJob, RehearsalResult
from capmas.evaluation.rehearsal_evidence import RehearsalPoolConfig, run_with_respawn


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    graph: Mapping[str, object]
    task_id: str
    scene_version: int
    candidate_fingerprint: str


def load_candidate_artifact(path: str | Path) -> tuple[CandidateSpec, ...]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_candidate_mapping(raw)


def parse_candidate_mapping(raw: object) -> tuple[CandidateSpec, ...]:
    if not isinstance(raw, Mapping):
        raise ValueError("candidate artifact must be an object")
    task_id = raw.get("task_id", "")
    scene_version = raw.get("scene_version", 0)
    candidates = raw.get("candidates")
    if not isinstance(task_id, str):
        raise ValueError("candidate artifact task_id must be a string")
    if not isinstance(scene_version, int) or scene_version < 0:
        raise ValueError("candidate artifact scene_version must be a non-negative integer")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidate artifact candidates must be a non-empty list")
    parsed: list[CandidateSpec] = []
    for index, item in enumerate(candidates):
        if not isinstance(item, Mapping):
            raise ValueError(f"candidate artifact candidates[{index}] must be an object")
        candidate_id = item.get("candidate_id")
        graph = item.get("graph")
        fingerprint = item.get("candidate_fingerprint", candidate_id)
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError(f"candidate artifact candidates[{index}].candidate_id is required")
        if not isinstance(graph, Mapping):
            raise ValueError(f"candidate artifact candidates[{index}].graph must be an object")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError(f"candidate artifact candidates[{index}].candidate_fingerprint is required")
        parsed.append(
            CandidateSpec(
                candidate_id=candidate_id,
                graph=dict(graph),
                task_id=task_id,
                scene_version=scene_version,
                candidate_fingerprint=fingerprint,
            )
        )
    return tuple(parsed)


def build_rehearsal_jobs(
    candidates: Sequence[CandidateSpec],
    *,
    seed: int,
    checkpoint_budget: int = 0,
) -> tuple[RehearsalJob, ...]:
    return tuple(
        RehearsalJob(
            candidate_id=candidate.candidate_id,
            seed=int(seed),
            payload={"graph": dict(candidate.graph)},
            task_id=candidate.task_id,
            scene_version=candidate.scene_version,
            candidate_fingerprint=candidate.candidate_fingerprint,
            checkpoint_budget=checkpoint_budget,
        )
        for candidate in candidates
    )


def run_rehearsal_batches(
    *,
    config_path: str,
    candidates: Sequence[CandidateSpec],
    seeds: Sequence[int],
    output_root: str | Path,
    pool_config: RehearsalPoolConfig,
    max_steps: int = 32,
    object_name: str = "akita black bowl",
    target_name: str = "plate",
    gpu: str = "5",
    run_fn: Callable[..., tuple[RehearsalResult, ...]] = run_with_respawn,
) -> tuple[Phase5RunDirectory, ...]:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    config = LiberoRehearsalConfig(
        config_path=config_path,
        object_name=object_name,
        target_name=target_name,
        max_steps=max_steps,
    )
    run_dirs: list[Phase5RunDirectory] = []
    for seed in seeds:
        run_dir = Phase5RunDirectory.create(
            output_root,
            "P5.3_process_rehearsal",
            f"seed{seed}_{uuid4().hex[:8]}",
        )
        jobs = build_rehearsal_jobs(candidates, seed=seed)
        run_dir.write_json(
            "run_config.json",
            {
                "experiment": "P5.3_process_rehearsal",
                "config_path": str(Path(config_path).resolve()),
                "task_id": candidates[0].task_id if candidates else "",
                "seed": seed,
                "candidate_ids": [job.candidate_id for job in jobs],
                "max_workers": pool_config.max_workers,
                "timeout_s": pool_config.timeout_s,
                "max_restarts": pool_config.max_restarts,
                "gpu": gpu,
                "artifact_dir": str(run_dir.path),
            },
        )
        results = run_fn(
            jobs,
            worker_factory=lambda: LiberoRehearsalWorker(config),
            pool_config=pool_config,
        )
        run_dir.write_json("results/rehearsal.json", [asdict(result) for result in results])
        run_dir.write_text(
            "logs/runner.log",
            "\n".join(
                [
                    f"experiment=P5.3_process_rehearsal seed={seed}",
                    f"results={len(results)}",
                    *(
                        f"candidate={result.candidate_id} success={result.success} "
                        f"failure={result.failure_class} latency_ms={result.latency_ms:.3f}"
                        for result in results
                    ),
                    "",
                ]
            ),
        )
        run_dir.write_json(
            "summary.json",
            {
                "seed": seed,
                "result_count": len(results),
                "success_count": sum(result.success for result in results),
                "failure_classes": sorted(
                    {str(result.failure_class) for result in results if result.failure_class}
                ),
            },
        )
        run_dir.write_text(
            "summary.md",
            f"# CAP-MAS P5.3 rehearsal\n\n- seed: {seed}\n"
            f"- results: {len(results)}\n"
            f"- successes: {sum(result.success for result in results)}\n",
        )
        run_dir.finalize_manifest()
        run_dirs.append(run_dir)
    return tuple(run_dirs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CAP-MAS P5.3 LIBERO process rehearsal")
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--candidate-artifact", required=True)
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--max-restarts", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--gpu", default="5")
    parser.add_argument("--output-root", default="outputs/phase5")
    parser.add_argument("--object-name", default="akita black bowl")
    parser.add_argument("--target-name", default="plate")
    parser.add_argument("--skip-api-servers", action="store_true")
    return parser.parse_args()


def _setup_capx_paths() -> None:
    capx_root = PROJECT_ROOT.parent / "cap-x"
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/capmas-mpl")
    for path in (
        capx_root / "capx" / "third_party" / "libero_dependencies" / "robosuite",
        capx_root / "capx" / "third_party" / "LIBERO-PRO",
        capx_root,
    ):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def main() -> None:
    args = parse_args()
    _setup_capx_paths()
    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    candidates = load_candidate_artifact(args.candidate_artifact)
    servers = []
    try:
        if not args.skip_api_servers:
            from capx.envs.configs.loader import DictLoader
            from capx.envs.runner import _start_api_servers

            config = DictLoader.load(args.config_path)
            servers = _start_api_servers(config.get("api_servers"))
        run_dirs = run_rehearsal_batches(
            config_path=args.config_path,
            candidates=candidates,
            seeds=seeds,
            output_root=args.output_root,
            pool_config=RehearsalPoolConfig(
                max_workers=args.max_workers,
                timeout_s=args.timeout_s,
                max_restarts=args.max_restarts,
            ),
            max_steps=args.max_steps,
            object_name=args.object_name,
            target_name=args.target_name,
            gpu=args.gpu,
        )
        for run_dir in run_dirs:
            print(f"CAP-MAS P5.3 rehearsal: {run_dir.path}")
    finally:
        for process in servers:
            process.terminate()
            process.join(timeout=5)


if __name__ == "__main__":
    main()
