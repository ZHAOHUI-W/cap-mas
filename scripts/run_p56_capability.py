"""CLI for read-only P5.6 task-family capability diagnosis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from capmas.evaluation.capability import run_capability_diagnosis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run read-only P5.6 capability diagnosis over retained P5.5 artifacts.",
    )
    parser.add_argument("--suite-dir", required=True, help="Retained P5.5 frozen replay suite")
    parser.add_argument(
        "--family",
        action="append",
        required=True,
        dest="families",
        help="Task family to diagnose; repeat for multiple families",
    )
    parser.add_argument("--split", choices=("id",), default="id")
    parser.add_argument("--output-root", required=True, help="Phase 5 output root")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_capability_diagnosis(
        suite_dir=args.suite_dir,
        families=tuple(args.families),
        split=args.split,
        output_root=args.output_root,
    )
    print(result.run_dir)
    for report in result.reports:
        print(
            f"{report.family_id}: cases={report.case_count} "
            f"physical_execution={report.physical_execution_count}/10 "
            f"evaluator_success={report.evaluator_success_count}/10 "
            f"eligible={report.eligible} gate_failures={','.join(report.gate_failures) or '-'}"
        )
    if result.handoffs:
        print("handoffs=" + ",".join(handoff.family_id for handoff in result.handoffs))
    else:
        print("handoffs=-")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
