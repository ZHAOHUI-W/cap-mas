"""Create and preflight a P5.3.2 object-6 capability manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCHEMA_VERSION = "p532.collection.v1"
REQUIRED_SEED_COUNT = 10
REQUIRED_GPU = "5"


@dataclass(frozen=True)
class Object6Assets:
    task_id: str
    config_path: str
    config_sha256: str
    candidate_artifact: str
    candidate_artifact_sha256: str
    object_name: str
    target_name: str


@dataclass(frozen=True)
class Object6Case:
    case_id: str
    task_id: str
    seed: int
    config_path: str
    config_sha256: str
    candidate_artifact: str
    candidate_artifact_sha256: str
    object_name: str
    target_name: str
    effective_motion_scope: str = "mission_suffix"
    max_restarts: int = 0
    gpu: str = REQUIRED_GPU
    max_physical_executions: int = 1


@dataclass(frozen=True)
class P532Manifest:
    manifest_id: str
    schema_version: str
    cases: tuple[Object6Case, ...]
    manifest_sha256: str = ""


def asset_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_object6_manifest(seeds: Iterable[int], assets: Object6Assets) -> P532Manifest:
    cases = tuple(
        Object6Case(
            case_id=f"object-6-seed{seed}",
            task_id=assets.task_id,
            seed=seed,
            config_path=assets.config_path,
            config_sha256=assets.config_sha256,
            candidate_artifact=assets.candidate_artifact,
            candidate_artifact_sha256=assets.candidate_artifact_sha256,
            object_name=assets.object_name,
            target_name=assets.target_name,
        )
        for seed in seeds
    )
    manifest = P532Manifest("", SCHEMA_VERSION, cases)
    validate_manifest(manifest, check_asset_files=False)
    return _finalize(manifest)


def validate_manifest(
    manifest: P532Manifest,
    *,
    project_root: str | Path | None = None,
    check_asset_files: bool = True,
) -> None:
    if manifest.schema_version != SCHEMA_VERSION:
        raise ValueError("unsupported P5.3.2 manifest schema")
    keys: set[tuple[str, int]] = set()
    case_ids: set[str] = set()
    for case in manifest.cases:
        key = (case.task_id, case.seed)
        if key in keys:
            raise ValueError(f"duplicate collection task/seed entry: {key}")
        keys.add(key)
        if case.case_id in case_ids:
            raise ValueError(f"duplicate collection case id: {case.case_id}")
        case_ids.add(case.case_id)
        if case.seed < 0:
            raise ValueError("collection seed must be non-negative")
        if case.effective_motion_scope != "mission_suffix":
            raise ValueError("P5.3.2 cases require mission_suffix scope")
        if case.max_restarts != 0:
            raise ValueError("P5.3.2 cases require max_restarts=0")
        if case.gpu != REQUIRED_GPU:
            raise ValueError("P5.3.2 cases require GPU 5")
        if case.max_physical_executions != 1:
            raise ValueError("P5.3.2 cases allow one physical execution per case")
        for name, digest in (
            ("config", case.config_sha256),
            ("candidate artifact", case.candidate_artifact_sha256),
        ):
            _validate_sha256(digest, f"{name} digest")
        if check_asset_files:
            root = Path(project_root or PROJECT_ROOT)
            config = _resolve_asset(root, case.config_path)
            candidate_artifact = _resolve_asset(root, case.candidate_artifact)
            if asset_sha256(config) != case.config_sha256:
                raise ValueError(f"config digest mismatch for case {case.case_id}")
            if asset_sha256(candidate_artifact) != case.candidate_artifact_sha256:
                raise ValueError(f"candidate artifact digest mismatch for case {case.case_id}")
    if len(manifest.cases) != REQUIRED_SEED_COUNT:
        raise ValueError("P5.3.2 manifest must contain exactly ten cases")
    expected = manifest_sha256(manifest)
    if manifest.manifest_sha256 and manifest.manifest_sha256 != expected:
        raise ValueError("manifest digest mismatch")
    if manifest.manifest_id and manifest.manifest_id != f"sha256:{expected}":
        raise ValueError("manifest id digest mismatch")


def load_and_preflight(path: str | Path, *, project_root: str | Path | None = None) -> P532Manifest:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    manifest = _from_dict(raw)
    validate_manifest(manifest, project_root=project_root)
    return _finalize(manifest)


def write_manifest(path: str | Path, manifest: P532Manifest) -> Path:
    validate_manifest(manifest, check_asset_files=False)
    finalized = _finalize(manifest)
    destination = Path(path)
    encoded = _manifest_bytes(finalized)
    if destination.exists():
        if destination.read_bytes() != encoded:
            raise ValueError(f"refusing byte-different manifest overwrite: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(encoded)
    return destination


def manifest_sha256(manifest: P532Manifest) -> str:
    payload = _payload(manifest)
    payload["manifest_id"] = ""
    payload["manifest_sha256"] = ""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finalize(manifest: P532Manifest) -> P532Manifest:
    digest = manifest_sha256(manifest)
    return replace(manifest, manifest_id=f"sha256:{digest}", manifest_sha256=digest)


def _payload(manifest: P532Manifest) -> dict[str, object]:
    return {
        "manifest_id": manifest.manifest_id,
        "schema_version": manifest.schema_version,
        "cases": [
            {
                "case_id": case.case_id,
                "task_id": case.task_id,
                "seed": case.seed,
                "config_path": case.config_path,
                "config_sha256": case.config_sha256,
                "candidate_artifact": case.candidate_artifact,
                "candidate_artifact_sha256": case.candidate_artifact_sha256,
                "object_name": case.object_name,
                "target_name": case.target_name,
                "effective_motion_scope": case.effective_motion_scope,
                "max_restarts": case.max_restarts,
                "gpu": case.gpu,
                "max_physical_executions": case.max_physical_executions,
            }
            for case in manifest.cases
        ],
        "manifest_sha256": manifest.manifest_sha256,
    }


def _manifest_bytes(manifest: P532Manifest) -> bytes:
    return (json.dumps(_payload(manifest), indent=2, sort_keys=True) + "\n").encode("utf-8")


def _from_dict(raw: object) -> P532Manifest:
    if not isinstance(raw, dict):
        raise TypeError("P5.3.2 manifest must be an object")
    cases_raw = raw.get("cases")
    if not isinstance(cases_raw, list):
        raise TypeError("P5.3.2 manifest cases must be a list")
    cases = tuple(Object6Case(**case) for case in cases_raw if isinstance(case, dict))
    if len(cases) != len(cases_raw):
        raise ValueError("P5.3.2 manifest cases must contain objects")
    return P532Manifest(
        manifest_id=str(raw.get("manifest_id", "")),
        schema_version=str(raw.get("schema_version", "")),
        cases=cases,
        manifest_sha256=str(raw.get("manifest_sha256", "")),
    )


def _resolve_asset(root: Path, path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve(strict=True)


def _validate_sha256(value: str, name: str) -> None:
    if len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a SHA-256 digest") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--candidate-artifact", required=True)
    parser.add_argument("--task-id", default="libero_object_6")
    parser.add_argument("--object-name", default="butter")
    parser.add_argument("--target-name", default="basket")
    parser.add_argument("--start-seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assets = Object6Assets(
        task_id=args.task_id,
        config_path=args.config_path,
        config_sha256=asset_sha256(args.config_path),
        candidate_artifact=args.candidate_artifact,
        candidate_artifact_sha256=asset_sha256(args.candidate_artifact),
        object_name=args.object_name,
        target_name=args.target_name,
    )
    manifest = build_object6_manifest(range(args.start_seed, args.start_seed + 10), assets)
    path = write_manifest(args.output, manifest)
    print(f"wrote {path} sha256={manifest.manifest_sha256}")


if __name__ == "__main__":
    main()
