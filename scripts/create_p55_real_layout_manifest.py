"""Create a frozen, real-layout P5.5 matched pilot manifest."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
from typing import Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from capmas.evaluation.ood import OODCase, OODSplitManifest, dump_ood_manifest


@dataclass(frozen=True)
class FamilySpec:
    family: str
    task_id: str
    task_goal: str
    config_path: str
    candidate_path: str
    object_name: str
    target_name: str
    native_bodies: tuple[str, ...]
    translation_delta: tuple[float, float, float]


def _layout_variant(
    *,
    variant_id: str,
    layout_family: str,
    bodies: tuple[str, ...],
    delta: tuple[float, float, float],
) -> dict[str, object]:
    return {
        "variant_id": variant_id,
        "layout_family": layout_family,
        "generator_version": "capmas-layout-v1",
        "transforms": [
            {
                "body_name": body,
                "translation_delta_xyz": list(delta),
            }
            for body in bodies
        ],
    }


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_real_layout_manifest(
    *,
    output_path: str | Path,
    assets_root: str | Path,
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> OODSplitManifest:
    assets = Path(assets_root).resolve()
    capx_root = (PROJECT_ROOT.parent / "cap-x").resolve()
    families = (
        FamilySpec(
            family="spatial-0",
            task_id="libero_spatial_0",
            task_goal="place the akita black bowl on the plate",
            config_path=str((capx_root / "env_configs/libero/franka_libero_spatial_0.yaml").resolve()),
            candidate_path=str((assets / "candidates/spatial_0.json").resolve()),
            object_name="akita black bowl",
            target_name="plate",
            native_bodies=(
                "akita_black_bowl_1_main",
                "plate_1_main",
                "glazed_rim_porcelain_ramekin_1_main",
            ),
            translation_delta=(0.08, -0.04, 0.0),
        ),
        FamilySpec(
            family="goal-1",
            task_id="libero_goal_1",
            task_goal="put the bowl on the stove",
            config_path=str((PROJECT_ROOT / "configs/phase5/capx_libero_goal_1_nonprivileged.yaml").resolve()),
            candidate_path=str((assets / "candidates/goal_1.json").resolve()),
            object_name="akita black bowl",
            target_name="flat stove",
            native_bodies=("akita_black_bowl_1_main",),
            translation_delta=(0.10, -0.04, 0.0),
        ),
        FamilySpec(
            family="object-6",
            task_id="libero_object_6",
            task_goal="pick the butter and place it in the basket",
            config_path=str((PROJECT_ROOT / "configs/phase5/capx_libero_object_6_nonprivileged.yaml").resolve()),
            candidate_path=str((assets / "candidates/object_6.json").resolve()),
            object_name="butter",
            target_name="basket",
            native_bodies=("butter_1_main", "basket_1_main"),
            translation_delta=(0.08, 0.04, 0.0),
        ),
    )
    cases: list[OODCase] = []
    for family in families:
        candidate_digest = _sha256(family.candidate_path)
        native_layout_family = f"native-{family.family}"
        ood_layout_family = f"translated-{family.family}-v1"
        for seed in seeds:
            pair_id = f"{family.family}-seed{seed}"
            id_case_id = f"id-{pair_id}"
            native_variant = _layout_variant(
                variant_id=f"{family.family}-native-seed{seed}",
                layout_family=native_layout_family,
                bodies=family.native_bodies,
                delta=(0.0, 0.0, 0.0),
            )
            translated_variant = _layout_variant(
                variant_id=f"{family.family}-translated-v1-seed{seed}",
                layout_family=ood_layout_family,
                bodies=family.native_bodies,
                delta=family.translation_delta,
            )
            common = dict(
                task_id=family.task_id,
                task_goal=family.task_goal,
                task_family=family.family,
                object_name=family.object_name,
                target_name=family.target_name,
                seed=seed,
                pair_id=pair_id,
                config_path=family.config_path,
                candidate_artifact=family.candidate_path,
                candidate_artifact_sha256=candidate_digest,
                environment_version="capx-libero-real-layout-v1",
                generator_version="p55-real-layout-curator-v1",
            )
            cases.append(
                OODCase(
                    case_id=id_case_id,
                    split="id",
                    ood_type="none",
                    layout_family=native_layout_family,
                    layout_variant=native_variant,
                    metadata={
                        "layout_source": "capx-libero-reset-seed1-body-joint-capture",
                        "layout_transform": "zero_delta_native_control",
                    },
                    **common,
                )
            )
            cases.append(
                OODCase(
                    case_id=f"ood-{pair_id}",
                    split="ood",
                    ood_type="layout",
                    layout_family=ood_layout_family,
                    layout_variant=translated_variant,
                    parent_case_id=id_case_id,
                    metadata={
                        "layout_source": "mujoco_free_joint_qpos_after_reset",
                        "layout_transform": "task_objects_translated_in_world_xy",
                        "layout_delta_xyz": ",".join(str(item) for item in family.translation_delta),
                    },
                    **common,
                )
            )
    manifest = OODSplitManifest(
        suite_id=f"p55-real-layout-3family-{len(seeds)}seed",
        manifest_version="2",
        cases=tuple(cases),
        id_task_families=tuple(family.family for family in families),
        ood_task_families=tuple(family.family for family in families),
        id_layout_families=tuple(f"native-{family.family}" for family in families),
        ood_layout_families=tuple(f"translated-{family.family}-v1" for family in families),
        memory_snapshot_version="frozen-memory-v1",
        robot_skill_snapshot_version="frozen-robot-skill-v1",
        prompt_version="frozen-prompt-v1",
        code_revision="cap-mas-p55-real-layout-v1",
        created_at_utc="2026-08-03T00:00:00Z",
    )
    dump_ood_manifest(output_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_real_layout_manifest(
        output_path=args.output,
        assets_root=args.assets_root,
        seeds=tuple(args.seeds),
    )
    print(f"manifest={args.output} cases={len(manifest.cases)}")


if __name__ == "__main__":
    main()
