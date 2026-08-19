"""Create byte-stable CAP-MAS P5.6 object-6 collection manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from capmas.contracts.calibration import (
    COLLECTION_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION,
    TRANSPORT_SMOKE_COLLECTION_SCHEMA_VERSION,
    CalibrationCollectionCase,
    CalibrationCollectionManifest,
    collection_manifest_sha256,
)

CONFIG_PATH = "configs/phase5/capx_libero_object_6_nonprivileged.yaml"
CANDIDATE_ARTIFACT = "outputs/phase5/P5.5_real_layout_assets_20260803/candidates/object_6.json"
CODE_REVISION = "b32b604358e80b933838ade4d56d58414347ceac"
MEMORY_SKILL_VERSION = "p56-memory-object6-frozen-v1"
ROBOT_SKILL_VERSION = "p56-robot-object6-frozen-v1"
PROMPT_VERSION = "p56-object6-prompt-v1"
ENVIRONMENT_VERSION = "capx-libero-object6-nonprivileged-v1"
FIRST_FILENAME = "p56_object6_id_seeds_11_20.json"
SECOND_FILENAME = "p56_object6_id_seeds_21_30.json"
P56D_SMOKE_FILENAME = "p56d_object6_id_seed_31.json"
P56D_QUALIFICATION_FILENAME = "p56d_object6_id_seeds_32_51.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(project_root: Path, relative_path: str) -> Path:
    return (project_root / relative_path).resolve(strict=True)


def _native_layout_variant() -> dict[str, object]:
    return {
        "generator_version": "capmas-layout-v1",
        "layout_family": "native-object-6",
        "transforms": [
            {
                "body_name": "butter_1_main",
                "translation_delta_xyz": [0.0, 0.0, 0.0],
            },
            {
                "body_name": "basket_1_main",
                "translation_delta_xyz": [0.0, 0.0, 0.0],
            },
        ],
        "variant_id": "native-object-6-zero-delta",
    }


def _case(project_root: Path, seed: int) -> CalibrationCollectionCase:
    case_id = f"object-6-id-seed{seed}"
    return CalibrationCollectionCase(
        case_id=case_id,
        lineage_group_id=case_id,
        family_id="object-6",
        task_id="libero_object_6",
        seed=seed,
        split_identity="id",
        config_path=CONFIG_PATH,
        config_sha256=_sha256(_resolve(project_root, CONFIG_PATH)),
        candidate_artifact=CANDIDATE_ARTIFACT,
        candidate_artifact_sha256=_sha256(_resolve(project_root, CANDIDATE_ARTIFACT)),
        object_name="butter",
        target_name="basket",
        layout_family="native-object-6",
        layout_variant=_native_layout_variant(),
    )


def _manifest(
    project_root: Path,
    *,
    start: int,
    stop: int,
    schema_version: str = COLLECTION_SCHEMA_VERSION,
    collection_purpose: str = "qualification",
) -> CalibrationCollectionManifest:
    manifest = CalibrationCollectionManifest(
        manifest_id="",
        schema_version=schema_version,
        cases=tuple(_case(project_root, seed) for seed in range(start, stop + 1)),
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        memory_skill_version=MEMORY_SKILL_VERSION,
        robot_skill_version=ROBOT_SKILL_VERSION,
        prompt_version=PROMPT_VERSION,
        environment_version=ENVIRONMENT_VERSION,
        code_revision=CODE_REVISION,
        collection_purpose=collection_purpose,
    )
    digest = collection_manifest_sha256(manifest)
    return replace(manifest, manifest_id=f"sha256:{digest}", manifest_sha256=digest)


def create_object6_manifests(
    project_root: str | Path,
) -> tuple[CalibrationCollectionManifest, CalibrationCollectionManifest]:
    """Return the two fixed object-6 ID seed blocks without writing files."""

    root = Path(project_root).resolve()
    return (
        _manifest(root, start=11, stop=20),
        _manifest(root, start=21, stop=30),
    )


def create_p56d_object6_seed31_manifest(
    project_root: str | Path,
) -> CalibrationCollectionManifest:
    """Return the isolated pre-registered P5.6D transport-smoke case."""

    return _manifest(
        Path(project_root).resolve(),
        start=31,
        stop=31,
        schema_version=TRANSPORT_SMOKE_COLLECTION_SCHEMA_VERSION,
        collection_purpose="transport_smoke",
    )


def create_p56d_object6_qualification_manifest(
    project_root: str | Path,
) -> CalibrationCollectionManifest:
    """Return the fixed P5.6D same-runtime qualification block."""

    return _manifest(
        Path(project_root).resolve(),
        start=32,
        stop=51,
        schema_version=TRANSPORT_SMOKE_COLLECTION_SCHEMA_VERSION,
        collection_purpose="qualification",
    )


def _manifest_bytes(manifest: CalibrationCollectionManifest) -> bytes:
    return (
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True, ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def _validate_manifest_bytes(path: Path, expected: bytes) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing committed manifest: {path}")
    actual = path.read_bytes()
    if actual != expected:
        raise ValueError(f"manifest bytes differ from generator output: {path}")
    restored = CalibrationCollectionManifest.from_dict(json.loads(actual.decode("ascii")))
    if collection_manifest_sha256(restored) != restored.manifest_sha256:
        raise ValueError(f"manifest digest check failed: {path}")


def write_object6_manifests(
    project_root: str | Path,
    *,
    output_dir: str | Path | None = None,
    check: bool = False,
) -> tuple[Path, Path]:
    root = Path(project_root).resolve()
    destination = (
        Path(output_dir).resolve()
        if output_dir is not None
        else root / "configs" / "phase5"
    )
    manifests = create_object6_manifests(root)
    paths = (destination / FIRST_FILENAME, destination / SECOND_FILENAME)
    encoded = tuple(_manifest_bytes(manifest) for manifest in manifests)
    if check:
        for path, content in zip(paths, encoded, strict=True):
            _validate_manifest_bytes(path, content)
        return paths
    destination.mkdir(parents=True, exist_ok=True)
    for path, content in zip(paths, encoded, strict=True):
        path.write_bytes(content)
    return paths


def write_p56d_object6_seed31_manifest(
    project_root: str | Path,
    *,
    output_dir: str | Path | None = None,
    check: bool = False,
) -> Path:
    """Write or verify the one-case P5.6D smoke manifest."""

    root = Path(project_root).resolve()
    destination = (
        Path(output_dir).resolve()
        if output_dir is not None
        else root / "configs" / "phase5"
    )
    path = destination / P56D_SMOKE_FILENAME
    encoded = _manifest_bytes(create_p56d_object6_seed31_manifest(root))
    if check:
        _validate_manifest_bytes(path, encoded)
        return path
    destination.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return path


def write_p56d_object6_qualification_manifest(
    project_root: str | Path,
    *,
    output_dir: str | Path | None = None,
    check: bool = False,
) -> Path:
    """Write or verify the fixed P5.6D qualification manifest."""

    root = Path(project_root).resolve()
    destination = (
        Path(output_dir).resolve()
        if output_dir is not None
        else root / "configs" / "phase5"
    )
    path = destination / P56D_QUALIFICATION_FILENAME
    encoded = _manifest_bytes(create_p56d_object6_qualification_manifest(root))
    if check:
        _validate_manifest_bytes(path, encoded)
        return path
    destination.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=PROJECT_ROOT)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--p56d-smoke", action="store_true")
    parser.add_argument("--p56d-qualification", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.p56d_smoke and args.p56d_qualification:
        raise SystemExit("--p56d-smoke and --p56d-qualification are mutually exclusive")
    if args.p56d_smoke:
        paths = (write_p56d_object6_seed31_manifest(
            args.project_root,
            output_dir=args.output_dir,
            check=args.check,
        ),)
        manifests = (create_p56d_object6_seed31_manifest(args.project_root),)
    elif args.p56d_qualification:
        paths = (write_p56d_object6_qualification_manifest(
            args.project_root,
            output_dir=args.output_dir,
            check=args.check,
        ),)
        manifests = (create_p56d_object6_qualification_manifest(args.project_root),)
    else:
        paths = write_object6_manifests(
            args.project_root,
            output_dir=args.output_dir,
            check=args.check,
        )
        manifests = create_object6_manifests(args.project_root)
    action = "validated" if args.check else "wrote"
    for path, manifest in zip(paths, manifests, strict=True):
        print(f"{action} {path} sha256={manifest.manifest_sha256}")


if __name__ == "__main__":
    main()
