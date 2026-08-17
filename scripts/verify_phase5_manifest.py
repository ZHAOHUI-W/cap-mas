"""Verify and record the integrity of a Phase 5 artifact manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from capmas.evaluation.phase5_artifacts import Phase5RunDirectory

VERIFICATION_SCHEMA_VERSION = "phase5.manifest_verification.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_payload(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"manifest must be a JSON object: {path}")
    return payload


def inspect_manifest(run_dir: str | Path) -> dict[str, object]:
    """Return a deterministic integrity report without changing the run."""

    root = Path(run_dir)
    manifest_path = root / "manifest.json"
    payload = _manifest_payload(manifest_path)
    entries = payload.get("files")
    if not isinstance(entries, list):
        raise TypeError(f"manifest files must be a list: {manifest_path}")

    declared_paths: set[str] = set()
    invalid_entries: list[str] = []
    missing_files: list[str] = []
    size_mismatches: list[str] = []
    digest_mismatches: list[str] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            invalid_entries.append(repr(entry))
            continue
        relative = entry.get("path")
        expected_size = entry.get("size")
        expected_digest = entry.get("sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size < 0
            or not isinstance(expected_digest, str)
            or len(expected_digest) != 64
        ):
            invalid_entries.append(str(relative))
            continue
        if relative in declared_paths:
            invalid_entries.append(relative)
            continue
        declared_paths.add(relative)
        path = root / relative
        if not path.is_file():
            missing_files.append(relative)
            continue
        if path.stat().st_size != expected_size:
            size_mismatches.append(relative)
        if _sha256(path) != expected_digest:
            digest_mismatches.append(relative)

    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    untracked_files = sorted(actual_paths - declared_paths)
    verified = not any(
        (invalid_entries, missing_files, size_mismatches, digest_mismatches, untracked_files)
    )
    return {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "run_dir": str(root),
        "manifest_path": str(manifest_path),
        "entry_count": len(entries),
        "verified": verified,
        "invalid_entries": invalid_entries,
        "missing_files": missing_files,
        "size_mismatches": size_mismatches,
        "digest_mismatches": digest_mismatches,
        "untracked_files": untracked_files,
    }


def verify_and_record_manifest(run_dir: str | Path) -> dict[str, object]:
    """Record an initial audit, regenerate the manifest, then verify it again."""

    root = Path(run_dir)
    initial = inspect_manifest(root)
    artifact_dir = Phase5RunDirectory(root)
    artifact_dir.write_json(
        "results/manifest_verification.json",
        {
            "schema_version": VERIFICATION_SCHEMA_VERSION,
            "initial_manifest": initial,
        },
    )
    artifact_dir.finalize_manifest()
    final = inspect_manifest(root)
    if final["verified"] is not True:
        raise RuntimeError(f"regenerated manifest failed verification: {root}")
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    reports = [verify_and_record_manifest(path) for path in args.run_dir]
    print(json.dumps(reports, indent=2, sort_keys=True))
    if not all(report["verified"] is True for report in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
