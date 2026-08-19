"""Run the P5.6B object-6 offline calibration pipeline without a runtime."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from capmas.contracts.calibration import (
    CalibrationLineage,
    CalibrationOutcome,
)
from capmas.evaluation.dataset import assign_lineage_splits, build_calibration_dataset
from capmas.evaluation.offline import (
    ExactQuotaSplitConfig,
    OfflineCalibrationReport,
    run_offline_calibration,
)
from capmas.evaluation.phase5_artifacts import Phase5RunDirectory
from scripts.verify_phase5_manifest import verify_and_record_manifest

EXPERIMENT_NAME = "P5.6.4_offline_calibration"
SOURCE_AUDIT_SPLIT_SALT = "p56b-source-audit-v1"
_PROVENANCE_FIELDS = (
    "feature_schema_version",
    "memory_skill_version",
    "robot_skill_version",
    "prompt_version",
    "environment_version",
    "code_revision",
)


@dataclass(frozen=True)
class CollectionProvenance:
    manifest_sha256: str
    feature_schema_version: str
    memory_skill_version: str
    robot_skill_version: str
    prompt_version: str
    environment_version: str
    code_revision: str
    source_manifest_paths: tuple[str, ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run P5.6B offline calibration.")
    parser.add_argument("--collection-run", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", default="p56b-object6-offline")
    return parser


def load_collection_rows(
    paths: Sequence[str | Path],
) -> tuple[tuple[CalibrationOutcome, ...], tuple[CalibrationLineage, ...]]:
    """Strictly decode collection rows and reject cross-suite lineage reuse."""

    outcomes: list[CalibrationOutcome] = []
    lineages: list[CalibrationLineage] = []
    seen_episodes: set[str] = set()
    seen_lineage_groups: set[str] = set()
    for raw_path in paths:
        suite = Path(raw_path).resolve(strict=True)
        suite_lineages = tuple(
            CalibrationLineage.from_dict(row)
            for row in _read_json_list(suite / "results" / "lineages.json")
        )
        for lineage in suite_lineages:
            if lineage.episode_id in seen_episodes:
                raise ValueError(f"duplicate episode id across collection runs: {lineage.episode_id}")
            if lineage.lineage_group_id in seen_lineage_groups:
                raise ValueError(
                    f"duplicate lineage group across collection runs: {lineage.lineage_group_id}"
                )
            seen_episodes.add(lineage.episode_id)
            seen_lineage_groups.add(lineage.lineage_group_id)
        suite_outcomes = tuple(
            CalibrationOutcome.from_dict(row)
            for row in _read_json_list(suite / "results" / "outcomes.json")
        )
        known_episodes = {lineage.episode_id for lineage in suite_lineages}
        for outcome in suite_outcomes:
            if outcome.episode_id not in known_episodes:
                raise ValueError(f"outcome lacks a persisted lineage: {outcome.episode_id}")
        lineages.extend(suite_lineages)
        outcomes.extend(suite_outcomes)
    if not outcomes or not lineages:
        raise ValueError("collection runs must contain outcomes and lineages")
    return tuple(outcomes), tuple(lineages)


def _read_json_list(path: Path) -> list[Mapping[str, object]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {path}") from error
    if not isinstance(raw, list):
        raise TypeError(f"expected a JSON list: {path}")
    if any(not isinstance(row, Mapping) for row in raw):
        raise TypeError(f"expected object rows: {path}")
    return [dict(row) for row in raw]


def _read_mapping(path: Path) -> Mapping[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {path}") from error
    if not isinstance(raw, Mapping):
        raise TypeError(f"expected a JSON object: {path}")
    return raw


def _collection_provenance(paths: Sequence[str | Path]) -> CollectionProvenance:
    records: list[tuple[Mapping[str, object], Path]] = []
    for raw_path in paths:
        suite = Path(raw_path).resolve(strict=True)
        run_config = _read_mapping(suite / "run_config.json")
        manifest_sha256 = run_config.get("manifest_sha256")
        if not _is_sha256(manifest_sha256):
            raise ValueError(f"collection run lacks a valid manifest SHA-256: {suite}")
        records.append((_resolve_frozen_manifest(manifest_sha256), suite))

    first, _ = records[0]
    for manifest, suite in records[1:]:
        if any(manifest[field] != first[field] for field in _PROVENANCE_FIELDS):
            raise ValueError(f"collection provenance mismatch: {suite}")
    return CollectionProvenance(
        manifest_sha256=str(first["manifest_sha256"]),
        feature_schema_version=str(first["feature_schema_version"]),
        memory_skill_version=str(first["memory_skill_version"]),
        robot_skill_version=str(first["robot_skill_version"]),
        prompt_version=str(first["prompt_version"]),
        environment_version=str(first["environment_version"]),
        code_revision=str(first["code_revision"]),
        source_manifest_paths=tuple(
            str(_frozen_manifest_path(str(record["manifest_sha256"])))
            for record, _ in records
        ),
    )


def _resolve_frozen_manifest(manifest_sha256: str) -> Mapping[str, object]:
    path = _frozen_manifest_path(manifest_sha256)
    raw = _read_mapping(path)
    if raw.get("manifest_sha256") != manifest_sha256:
        raise ValueError(f"frozen manifest digest mismatch: {path}")
    for field in _PROVENANCE_FIELDS:
        if not isinstance(raw.get(field), str) or not raw[field]:
            raise ValueError(f"frozen manifest lacks {field}: {path}")
    return raw


def _frozen_manifest_path(manifest_sha256: str) -> Path:
    directory = PROJECT_ROOT / "configs" / "phase5"
    matches = [
        path
        for path in sorted(directory.glob("p56_*.json"))
        if _read_mapping(path).get("manifest_sha256") == manifest_sha256
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one frozen P5.6 collection manifest for {manifest_sha256}"
        )
    return matches[0]


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _build_source_dataset(
    paths: Sequence[str | Path],
):
    outcomes, lineages = load_collection_rows(paths)
    provenance = _collection_provenance(paths)
    for outcome in outcomes:
        snapshot = outcome.feature_snapshot
        if (
            snapshot.feature_schema_version != provenance.feature_schema_version
            or snapshot.memory_skill_version != provenance.memory_skill_version
            or snapshot.robot_skill_version != provenance.robot_skill_version
        ):
            raise ValueError(f"outcome provenance mismatch: {outcome.episode_id}")
    assignments = assign_lineage_splits(lineages, salt=SOURCE_AUDIT_SPLIT_SALT)
    manifest = build_calibration_dataset(
        outcomes,
        lineages,
        split_assignments=assignments,
        memory_skill_version=provenance.memory_skill_version,
        robot_skill_version=provenance.robot_skill_version,
        prompt_version=provenance.prompt_version,
        environment_version=provenance.environment_version,
        code_revision=provenance.code_revision,
        split_salt=SOURCE_AUDIT_SPLIT_SALT,
    )
    return manifest, provenance


def _write_report_artifacts(
    run_dir: Phase5RunDirectory,
    report: OfflineCalibrationReport,
) -> None:
    run_dir.write_json("artifacts/exact_quota_split.json", report.split_config.to_dict())
    run_dir.write_json("artifacts/reduced_features.json", [row.to_dict() for row in report.reduced_rows])
    run_dir.write_json(
        "artifacts/constrained_logistic_model.json",
        None if report.model is None else report.model.to_dict(),
    )
    run_dir.write_json(
        "artifacts/isotonic_calibration.json",
        None if report.isotonic is None else report.isotonic.to_dict(),
    )
    run_dir.write_json(
        "results/predictions.json",
        {
            split: [prediction.to_dict() for prediction in predictions]
            for split, predictions in report.predictions.items()
        },
    )
    run_dir.write_json("results/offline_calibration_report.json", report.to_dict())


def _write_log(run_dir: Phase5RunDirectory, lines: Sequence[str]) -> None:
    run_dir.write_text("logs/runner.log", "\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = Phase5RunDirectory.create(args.output_root, EXPERIMENT_NAME, args.run_id)
    collection_runs = tuple(str(Path(path).resolve()) for path in args.collection_run)
    run_dir.write_json(
        "run_config.json",
        {
            "collection_runs": collection_runs,
            "experiment": EXPERIMENT_NAME,
            "run_id": args.run_id,
            "status": "running",
        },
    )
    try:
        manifest, provenance = _build_source_dataset(args.collection_run)
        report = run_offline_calibration(manifest, ExactQuotaSplitConfig.object6_v1())
    except (OSError, TypeError, ValueError) as error:
        _write_log(
            run_dir,
            (
                f"experiment={EXPERIMENT_NAME}",
                "status=failed",
                f"error_type={type(error).__name__}",
                f"error={error}",
            ),
        )
        run_dir.write_json(
            "results/offline_calibration_failure.json",
            {"error": str(error), "error_type": type(error).__name__, "status": "failed"},
        )
        run_dir.finalize_manifest()
        verify_and_record_manifest(run_dir.path)
        print(str(run_dir.path), file=sys.stderr)
        return 1

    run_dir.write_json("artifacts/source_dataset_manifest.json", manifest.to_dict())
    _write_report_artifacts(run_dir, report)
    run_dir.write_json(
        "run_config.json",
        {
            "collection_manifest_paths": provenance.source_manifest_paths,
            "collection_manifest_sha256": provenance.manifest_sha256,
            "collection_runs": collection_runs,
            "experiment": EXPERIMENT_NAME,
            "run_id": args.run_id,
            "status": "completed",
        },
    )
    _write_log(
        run_dir,
        (
            f"experiment={EXPERIMENT_NAME}",
            "status=completed",
            f"source_dataset_id={manifest.dataset_id}",
            f"report_sha256={report.report_sha256}",
            f"fit_reason={report.fit_reason or 'none'}",
            "online_effect=false",
        ),
    )
    run_dir.finalize_manifest()
    verify_and_record_manifest(run_dir.path)
    print(str(run_dir.path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
