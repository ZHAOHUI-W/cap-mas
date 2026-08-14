"""CLI for read-only P5.6 historical compatibility audits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from capmas.evaluation.history_audit import run_history_audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit retained P5.5 family rows for native P5.6 Tier-A history compatibility."
        ),
    )
    parser.add_argument("--suite-dir", required=True, help="Retained P5.5 frozen replay suite")
    parser.add_argument("--family", required=True, help="Task family to audit")
    parser.add_argument("--output-root", required=True, help="Phase 5 output root")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_history_audit(
        suite_dir=args.suite_dir,
        family_id=args.family,
        output_root=args.output_root,
    )
    audit = result.audit
    print(result.run_dir)
    print(
        f"{audit.family_id}: examined={audit.examined_count} "
        f"admissible_tier_a={audit.admissible_tier_a_count} "
        f"rejected={audit.rejected_count}"
    )
    print("rejection_counts=" + repr(dict(audit.rejection_counts)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
