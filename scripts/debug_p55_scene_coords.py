"""Run one P5.5 case and retain physical scene-coordinate diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from capmas.evaluation.ood import load_ood_manifest
from capmas.evaluation.phase5_artifacts import Phase5RunDirectory
from scripts.run_libero_p55_ood import (
    OODRunConfig,
    _setup_capx_paths,
    _start_capx_api_servers,
    run_ood_case,
)


def main() -> None:
    _setup_capx_paths()
    manifest = load_ood_manifest(
        "outputs/phase5/P5.5_real_layout_assets_20260803/"
        "p55_real_layout_3family_1seed.json"
    )
    case = next(item for item in manifest.cases if item.case_id == "id-object-6-seed1")
    servers = _start_capx_api_servers(case.config_path)
    parent = Phase5RunDirectory.create(
        "outputs/phase5/P5.5_diagnostic_scene_coords_20260806",
        "suite",
        "object6_id_seed1",
    )
    try:
        result = run_ood_case(
            case,
            suite_dir=parent,
            run_config=OODRunConfig(
                selection_repeats=1,
                max_workers=1,
                timeout_s=360.0,
                max_restarts=0,
                max_steps=32,
                gpu="5",
                cache_mode="disabled",
            ),
        )
        print(f"case_dir={result.case_dir}")
        print(f"status={result.status}")
        print(f"error={result.error}")
    finally:
        for process in servers:
            process.terminate()
            process.join(timeout=5)


if __name__ == "__main__":
    main()
