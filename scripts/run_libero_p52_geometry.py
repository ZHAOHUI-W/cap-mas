"""Run the five-seed P5.2 geometry ablation with isolated artifacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from capmas.evaluation.phase5_artifacts import Phase5RunDirectory


P52_MODES = ("geometry_disabled", "geometry_shadow", "geometry_online_bounded")
_B3_MODE = {
    "geometry_disabled": "disabled",
    "geometry_shadow": "shadow",
    "geometry_online_bounded": "online_bounded",
}


@dataclass(frozen=True)
class PilotSpec:
    mode: str
    seed: int
    run_dir: Phase5RunDirectory
    used_privileged_state: bool = False

    def run_config(self) -> dict[str, object]:
        return {
            "experiment": "P5.2_geometry_evidence",
            "mode": self.mode,
            "seed": self.seed,
            "used_privileged_state": self.used_privileged_state,
            "artifact_dir": str(self.run_dir.path),
        }


def pilot_summary_markdown(summary: Mapping[str, object]) -> str:
    """Render the required human-readable summary for one pilot run."""
    return "\n".join(
        (
            "# CAP-MAS P5.2 pilot run",
            "",
            f"- mode: {summary.get('mode')}",
            f"- seed: {summary.get('seed')}",
            f"- return_code: {summary.get('return_code')}",
            f"- evaluator_success: {summary.get('evaluator_success')}",
            f"- output: {summary.get('output')}",
            f"- log: {summary.get('log')}",
            "",
        )
    )


def build_pilot_specs(
    root: str | Path,
    *,
    seeds: Sequence[int] = (1, 2, 3, 4, 5),
    modes: Sequence[str] = P52_MODES,
) -> tuple[PilotSpec, ...]:
    unknown = sorted(set(modes) - set(P52_MODES))
    if unknown:
        raise ValueError(f"unknown P5.2 modes: {', '.join(unknown)}")
    specs = []
    for mode in modes:
        for seed in seeds:
            run_dir = Phase5RunDirectory.create(
                root,
                "P5.2_geometry_evidence",
                f"{mode}_seed{seed}_{uuid4().hex[:8]}",
            )
            spec = PilotSpec(mode, int(seed), run_dir)
            run_dir.write_json("run_config.json", spec.run_config())
            specs.append(spec)
    return tuple(specs)


def run_pilot(
    config_path: str,
    *,
    api_base: str,
    model: str,
    api_key: str | None,
    root: str | Path,
    seeds: Sequence[int] = (1, 2, 3, 4, 5),
    modes: Sequence[str] = P52_MODES,
    gpu: str = "5",
    extra_args: Sequence[str] = (),
) -> tuple[PilotSpec, ...]:
    runner = PROJECT_ROOT / "scripts" / "run_libero_b3_llm.py"
    specs = build_pilot_specs(root, seeds=seeds, modes=modes)
    for spec in specs:
        output = spec.run_dir.path / "results" / "episode.json"
        log_path = spec.run_dir.log_path()
        command = [
            sys.executable,
            str(runner),
            "--config-path",
            config_path,
            "--server-url",
            api_base,
            "--model",
            model,
            "--seed",
            str(spec.seed),
            "--geometry-mode",
            _B3_MODE[spec.mode],
            "--preview-backend",
            "reference_motion_preview",
            "--geometry-deadline-ms",
            "50",
            "--output",
            str(output),
            "--log-file",
            str(log_path),
        ]
        if api_key:
            command.extend(("--api-key", api_key))
        command.extend(extra_args)
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        completed = subprocess.run(command, check=False, env=environment)
        summary = {
            **spec.run_config(),
            "return_code": completed.returncode,
            "output": str(output),
            "log": str(log_path),
        }
        if output.exists():
            try:
                payload = json.loads(output.read_text(encoding="utf-8"))
                summary["evaluator_success"] = payload.get("evaluator_success")
            except (OSError, json.JSONDecodeError):
                summary["output_parse_error"] = True
        failure = output.with_suffix(".failure.json")
        if failure.exists():
            summary["failure_artifact"] = str(failure)
        spec.run_dir.write_json("summary.json", summary)
        spec.run_dir.write_text("summary.md", pilot_summary_markdown(summary))
        spec.run_dir.finalize_manifest()
    return specs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CAP-MAS P5.2 LIBERO pilot")
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--api-key", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output-root", default="outputs/phase5")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--modes", default=",".join(P52_MODES))
    parser.add_argument("--gpu", default="5")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    seeds = tuple(int(item) for item in args.seeds.split(",") if item.strip())
    modes = tuple(item.strip() for item in args.modes.split(",") if item.strip())
    run_pilot(
        args.config_path,
        api_base=args.server_url,
        model=args.model,
        api_key=args.api_key,
        root=args.output_root,
        seeds=seeds,
        modes=modes,
        gpu=args.gpu,
    )


if __name__ == "__main__":
    main()
