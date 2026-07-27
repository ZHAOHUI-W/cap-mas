from __future__ import annotations

import argparse
import json
from pathlib import Path

from capmas.evaluation.parity import compare_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare existing CAP-X and CAP-MAS episode artifacts."
    )
    parser.add_argument("--capx-trial", required=True, type=Path)
    parser.add_argument("--capmas-episode", required=True, type=Path)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = compare_artifacts(
        args.capx_trial,
        args.capmas_episode,
        task_id=args.task_id,
        seed=args.seed,
    )
    encoded = json.dumps(comparison.to_dict(), indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
