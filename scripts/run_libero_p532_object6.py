"""Preflight or run the separately gated P5.3.2 object-6 capability block."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from capmas.evaluation.phase5_artifacts import Phase5RunDirectory
from capmas.evaluation.rehearsal_evidence import RehearsalPoolConfig
from scripts.create_p532_object6_manifest import Object6Case, P532Manifest, load_and_preflight


@dataclass(frozen=True)
class P532RunResult:
    run_dir: Phase5RunDirectory
    case_count: int
    live_session_count: int
    physical_execution_count: int
    evaluator_success_count: int
    infrastructure_unknown_count: int
    semantic_abstention_count: int
    equal_evidence_tie_count: int
    evidence_selected_count: int
    safety_abstention_count: int
    fingerprint_mismatch_count: int


def _live_session_factory(case: object) -> object:
    from capmas.evaluation.libero_evidence_session import (
        LiveLiberoEvidenceSession,
        LiveLiberoEvidenceSessionConfig,
    )

    return LiveLiberoEvidenceSession(
        LiveLiberoEvidenceSessionConfig(
            config_path=case.config_path,
            object_name=case.object_name,
            target_name=case.target_name,
            seed=case.seed,
            max_steps=32,
        )
    )


def run_capability(
    manifest_path: str | Path,
    *,
    output_root: str | Path,
    dry_run: bool,
) -> P532RunResult:
    manifest = load_and_preflight(manifest_path)
    run_dir = Phase5RunDirectory.create(output_root, "P5.3.2_object6_capability", manifest.manifest_id[7:15])
    preflight = {
        "schema_version": manifest.schema_version,
        "manifest_sha256": manifest.manifest_sha256,
        "case_count": len(manifest.cases),
        "effective_motion_scope": "mission_suffix",
        "gpu": "5",
        "max_restarts": 0,
        "dry_run": dry_run,
    }
    run_dir.write_json("results/preflight.json", preflight)
    run_dir.write_json("run_config.json", {**preflight, "status": "dry_run" if dry_run else "running"})
    if dry_run:
        run_dir.write_text("logs/runner.log", "status=dry_run\ncapx_imported=false\n")
        run_dir.finalize_manifest()
        return P532RunResult(run_dir, len(manifest.cases), 0, 0, 0, 0, 0, 0, 0, 0, 0)
    return _run_live(manifest, run_dir)


def _run_live(manifest: P532Manifest, run_dir: Phase5RunDirectory) -> P532RunResult:
    """Execute serial cases once each behind the explicit CLI execute boundary."""

    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    counts = {
        "live_session_count": 0,
        "physical_execution_count": 0,
        "evaluator_success_count": 0,
        "infrastructure_unknown_count": 0,
        "semantic_abstention_count": 0,
        "equal_evidence_tie_count": 0,
        "evidence_selected_count": 0,
        "safety_abstention_count": 0,
        "fingerprint_mismatch_count": 0,
    }
    for case in manifest.cases:
        case_dir = run_dir.path / "cases" / case.case_id
        (case_dir / "logs").mkdir(parents=True, exist_ok=True)
        stdout_path = case_dir / "logs" / "stdout.log"
        stderr_path = case_dir / "logs" / "stderr.log"
        servers: list[object] = []
        try:
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr, redirect_stdout(stdout), redirect_stderr(stderr):
                servers = _start_capx_api_servers(case.config_path)
                outcome = _run_case(case, case_dir)
            counts["live_session_count"] += 1
            _accumulate_outcome(counts, outcome)
            (case_dir / "results").mkdir(exist_ok=True)
            (case_dir / "results" / "online_outcome.json").write_text(
                json.dumps(_outcome_payload(outcome), indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
            (case_dir / "logs" / "runner.log").write_text(
                "\n".join(
                    (
                        "status=completed",
                        f"physical_candidate_id={outcome.physical_candidate_id}",
                        f"online_run_dir={outcome.run_dir.path}",
                        "",
                    )
                ),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001 - gated runner records all infrastructure faults.
            counts["infrastructure_unknown_count"] += 1
            (case_dir / "logs" / "runner.log").write_text(
                f"status=failed\nerror={type(exc).__name__}: {exc}\n",
                encoding="utf-8",
            )
        finally:
            _terminate_servers(servers)
    run_dir.write_json("results/counts.json", counts)
    run_dir.finalize_manifest()
    return P532RunResult(run_dir, len(manifest.cases), **counts)


def _run_case(case: Object6Case, case_dir: Path) -> object:
    """Run exactly one same-runtime decision; the online runner enforces one submit."""

    from scripts.run_libero_p53_online import (
        _setup_capx_paths,
        load_online_candidates,
        run_online_experiment,
    )

    _setup_capx_paths()
    candidates = load_online_candidates(case.candidate_artifact)
    if not candidates:
        raise ValueError("P5.3.2 candidate artifact must not be empty")
    scene_versions = {candidate.scene_version for candidate in candidates}
    if len(scene_versions) != 1:
        raise ValueError("P5.3.2 candidate artifact must use one scene version")
    session = _live_session_factory(case)
    return run_online_experiment(
        config_path=case.config_path,
        candidates=candidates,
        seed=case.seed,
        scene_version=next(iter(scene_versions)),
        mode="disabled",
        output_root=case_dir / "artifacts",
        pool_config=RehearsalPoolConfig(max_workers=1, timeout_s=360.0, max_restarts=0),
        max_steps=32,
        object_name=case.object_name,
        target_name=case.target_name,
        gpu=case.gpu,
        evidence_session=session,
        effective_motion_scope="mission_suffix",
    )


def _accumulate_outcome(counts: dict[str, int], outcome: object) -> None:
    report = outcome.report
    selection = report.live
    basis = selection.selection_basis
    physical_started = outcome.physical_execution_started_at_ns is not None
    if physical_started:
        counts["physical_execution_count"] += 1
    if basis == "candidate_semantic_equivalence":
        counts["semantic_abstention_count"] += 1
    elif basis == "evidence_tie_break":
        counts["equal_evidence_tie_count"] += 1
    elif basis == "evidence_score":
        counts["evidence_selected_count"] += 1
    elif selection.selected is None:
        counts["safety_abstention_count"] += 1
    physical = outcome.physical_result
    if isinstance(physical, dict) and physical.get("evaluator_success") is True:
        counts["evaluator_success_count"] += 1
    _check_selected_fingerprint(counts, outcome)


def _check_selected_fingerprint(counts: dict[str, int], outcome: object) -> None:
    selected = outcome.physical_candidate_id
    if selected is None:
        return
    selection_path = outcome.run_dir.path / "results" / "selection.json"
    try:
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        fingerprints = selection.get("program_fingerprints")
        if not isinstance(fingerprints, dict):
            raise TypeError("selection is missing candidate program fingerprints")
        expected = fingerprints.get(selected)
        if not isinstance(expected, str) or not expected:
            raise ValueError("selection is missing the selected candidate program fingerprint")
        if selection.get("selected_program_fingerprint") != expected:
            counts["fingerprint_mismatch_count"] += 1
    except (OSError, TypeError, ValueError):
        counts["fingerprint_mismatch_count"] += 1


def _outcome_payload(outcome: object) -> dict[str, object]:
    return {
        "online_run_dir": str(outcome.run_dir.path),
        "physical_candidate_id": outcome.physical_candidate_id,
        "physical_result": outcome.physical_result,
        "selection_basis": outcome.report.live.selection_basis,
        "decision_completed_at_ns": outcome.decision_completed_at_ns,
        "physical_execution_started_at_ns": outcome.physical_execution_started_at_ns,
    }


def _start_capx_api_servers(config_path: str) -> list[object]:
    from capx.envs.configs.loader import DictLoader
    from capx.envs.runner import _start_api_servers

    config = DictLoader.load(config_path)
    return list(_start_api_servers(config.get("api_servers")))


def _terminate_servers(servers: list[object]) -> None:
    for process in servers:
        terminate = getattr(process, "terminate", None)
        if callable(terminate):
            terminate()
        join = getattr(process, "join", None)
        if callable(join):
            join(timeout=5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", default="outputs/phase5")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run the approved manifest; without this flag only preflight is performed",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_capability(args.manifest, output_root=args.output_root, dry_run=not args.execute)
    print(json.dumps(asdict(result), default=str, sort_keys=True))


if __name__ == "__main__":
    main()
