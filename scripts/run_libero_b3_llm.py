"""Run the P3.1 LLM proposal path on the CAP-X LIBERO runtime.

The Manager and local Policy Agents only produce typed graph artifacts. The
selected graph is executed by the same single physical scheduler used by the
deterministic B3 baseline.
"""

from __future__ import annotations

import argparse
import atexit
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlparse
from uuid import uuid4


# Direct script execution starts with ``scripts/`` on sys.path.  Phase 5
# artifact allocation happens before the runtime bootstrap below, so expose
# the project root at module import time rather than relying on PYTHONPATH.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class _TeeStream:
    """Duplicate runner stdout/stderr to the terminal and a per-run log."""

    def __init__(self, *streams: object) -> None:
        self._streams = streams

    def write(self, value: str) -> int:
        for stream in self._streams:
            stream.write(value)  # type: ignore[attr-defined]
        return len(value)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()  # type: ignore[attr-defined]

    def isatty(self) -> bool:
        return any(stream.isatty() for stream in self._streams)  # type: ignore[attr-defined]


def reserve_run_log_path(requested: Path) -> Path:
    """Return a writable per-run path without overwriting an earlier log."""
    if not requested.exists():
        return requested
    stamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = requested.suffix or ".log"
    stem = requested.name[: -len(requested.suffix)] if requested.suffix else requested.name
    candidate = requested.with_name(f"{stem}_{stamp}_{os.getpid()}{suffix}")
    counter = 2
    while candidate.exists():
        candidate = requested.with_name(
            f"{stem}_{stamp}_{os.getpid()}_{counter}{suffix}"
        )
        counter += 1
    return candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CAP-MAS P3.1 LLM graph scheduling on LIBERO."
    )
    parser.add_argument("--config-path", required=True)
    parser.add_argument(
        "--api-base",
        "--server-url",
        dest="api_base",
        default=os.getenv("CAPMAS_LLM_API_BASE"),
    )
    parser.add_argument("--model", default=os.getenv("CAPMAS_LLM_MODEL", "gpt-5.4"))
    parser.add_argument("--api-key-env", default="CAPMAS_LLM_API_KEY")
    parser.add_argument("--api-key", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output", default="outputs/capmas_libero_b3_llm/episode.json")
    parser.add_argument(
        "--phase5-artifact-root",
        default=None,
        help="optional root for a unique Phase 5 run directory",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="per-run stdout/stderr log; defaults to the output path with a .log suffix",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--object-name", default="akita black bowl")
    parser.add_argument("--target-name", default="plate")
    parser.add_argument("--policy-agents", type=int, default=2)
    parser.add_argument(
        "--policy-strategies",
        default="balanced",
        help="comma-separated Policy strategies; one value is repeated for all agents",
    )
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument(
        "--graph-protocol",
        choices=("staged", "legacy"),
        default="staged",
        help="LLM protocol: compact topology->local graph stages or legacy full graph",
    )
    parser.add_argument(
        "--proposal-mode",
        choices=("subgoal_serial", "ready_wave"),
        default="subgoal_serial",
        help="Policy proposal scheduling: serial subgoals or dependency-ready waves",
    )
    parser.add_argument(
        "--execution-mode",
        choices=("fixed_graph", "rolling"),
        default="fixed_graph",
        help="execute the compiled graph once or replan after each verified subgraph",
    )
    parser.add_argument("--llm-max-retries", type=int, default=2)
    parser.add_argument(
        "--llm-proposal-retries",
        type=int,
        default=1,
        help="bounded Manager/Policy repair attempts after schema or graph rejection",
    )
    parser.add_argument("--llm-deadline-ms", type=int, default=60_000)
    parser.add_argument("--llm-max-output-tokens", type=int, default=1536)
    parser.add_argument(
        "--no-provider-structured-output",
        action="store_true",
        help="use strict JSON prompting plus local decoder validation only",
    )
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument(
        "--geometry-mode",
        choices=("disabled", "shadow", "online_bounded"),
        default="disabled",
    )
    parser.add_argument("--geometry-deadline-ms", type=int, default=50)
    parser.add_argument(
        "--geometry-depth-subsample",
        type=int,
        default=16,
        help="RGB-D stride for the live reference local map; lower values increase density and latency",
    )
    parser.add_argument("--preview-backend", default="none")
    parser.add_argument(
        "--privilege-mode",
        choices=("realistic_sensor", "diagnostic_privileged"),
        default="realistic_sensor",
    )
    parser.add_argument("--skip-api-servers", action="store_true")
    parser.add_argument(
        "--allow-manager-plan-fallback",
        action="store_true",
        help="use the validated Manager subgraph when no local Policy proposal is available",
    )
    return parser.parse_args()


def _policy_strategies(raw: str, count: int) -> tuple[str, ...]:
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not values:
        raise SystemExit("--policy-strategies must contain at least one strategy")
    if len(values) == 1:
        values *= count
    if len(values) != count:
        raise SystemExit(
            "--policy-strategies must contain one value or exactly one value per Policy agent"
        )
    allowed = {"balanced", "safety", "robust", "efficient"}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise SystemExit(f"unsupported Policy strategies: {', '.join(unknown)}")
    return values


def _policy_agent_name(index: int, strategy: str) -> str:
    """Keep the homogeneous baseline identity stable across ablations."""
    return f"policy-{index}" if strategy == "balanced" else f"policy-{index}:{strategy}"


def _evaluate_checkpoint(
    subgraph: object,
    node: object,
    context: object,
    verifier: object,
) -> str:
    """Evaluate a graph checkpoint against the current committed scene."""
    validating_checkpoints = tuple(
        item for item in subgraph.checkpoints if item.validate
    )
    checkpoint = next(
        (item for item in validating_checkpoints if item.name == node.node_id),
        None,
    )
    if checkpoint is None and node.postconditions:
        node_predicates = tuple(node.postconditions)
        checkpoint = next(
            (
                item
                for item in validating_checkpoints
                if tuple(item.predicates) == node_predicates
            ),
            None,
        )
    if checkpoint is None and len(validating_checkpoints) == 1:
        checkpoint = validating_checkpoints[0]
    predicates = tuple(checkpoint.predicates) if checkpoint is not None else tuple(
        node.postconditions
    )
    if not predicates:
        return "failure"
    evaluate_predicates = getattr(verifier, "evaluate_predicates", None)
    if callable(evaluate_predicates):
        reports = tuple(evaluate_predicates(predicates, context.scene))
        print(
            "CAP-MAS checkpoint",
            node.node_id,
            "scene_version=",
            context.scene.scene_version,
            "predicates=",
            tuple(
                (report.name, report.passed, report.reason)
                for report in reports
            ),
        )
        return "success" if all(report.passed for report in reports) else "failure"
    return "success" if verifier.goal_satisfied(predicates, context.scene) else "failure"


def _world_model_metrics(enricher: object | None, local_map: object | None) -> dict[str, object]:
    """Expose fail-open live local-map transport diagnostics in artifacts."""
    if enricher is None:
        return {
            "enabled": False,
            "map_version": None,
            "processed_observations": 0,
            "last_error": None,
        }
    map_version = getattr(local_map, "map_version", None)
    return {
        "enabled": True,
        "map_version": map_version() if callable(map_version) else None,
        "processed_observations": getattr(enricher, "processed_observations", 0),
        "last_error": getattr(enricher, "last_error", None),
    }


def main() -> None:
    args = parse_args()
    if not args.api_base:
        raise SystemExit("--api-base or CAPMAS_LLM_API_BASE is required")
    if (
        args.policy_agents <= 0
        or args.max_workers <= 0
        or args.llm_max_retries < 0
        or args.llm_proposal_retries < 0
        or args.llm_deadline_ms <= 0
        or args.llm_max_output_tokens <= 0
        or args.geometry_deadline_ms <= 0
        or args.geometry_depth_subsample <= 0
    ):
        raise SystemExit(
            "--policy-agents/--max-workers must be positive, "
            "--llm-max-retries/--llm-proposal-retries must not be negative, and "
            "--llm-deadline-ms/--llm-max-output-tokens/--geometry-depth-subsample must be positive"
        )
    policy_strategies = _policy_strategies(args.policy_strategies, args.policy_agents)
    if args.graph_protocol == "staged" and args.allow_manager_plan_fallback:
        raise SystemExit(
            "--allow-manager-plan-fallback is only valid with --graph-protocol legacy; "
            "staged topology has no executable Manager fallback"
        )

    phase5_run = None
    if args.phase5_artifact_root:
        from capmas.evaluation.phase5_artifacts import Phase5RunDirectory

        phase5_run = Phase5RunDirectory.create(
            args.phase5_artifact_root,
            "B3-LLM",
            str(uuid4()),
        )
        args.output = str(phase5_run.path / "results" / "episode.json")
        args.log_file = str(phase5_run.log_path())
    requested_log_path = (
        Path(args.log_file) if args.log_file else Path(args.output).with_suffix(".log")
    )
    log_path = reserve_run_log_path(requested_log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = _TeeStream(original_stdout, log_handle)  # type: ignore[assignment]
    sys.stderr = _TeeStream(original_stderr, log_handle)  # type: ignore[assignment]

    def restore_streams() -> None:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_handle.flush()
        log_handle.close()

    atexit.register(restore_streams)
    print(f"CAP-MAS run log: {log_path}")

    project_root = _PROJECT_ROOT
    capx_root = project_root.parent / "cap-x"
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/capmas-mpl")
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    for path in (
        capx_root / "capx" / "third_party" / "libero_dependencies" / "robosuite",
        capx_root / "capx" / "third_party" / "LIBERO-PRO",
        capx_root,
    ):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from capmas.agents.manager import LLMTopologyManager, LLMMissionManager
    from capmas.agents.policy import LLMGraphPolicyAgent, LLMStagedGraphPolicyAgent
    from capmas.backends.capx_libero_factory import (
        build_capx_runtime_from_yaml,
        build_capx_world_model_enricher,
    )
    from capmas.contracts.experiment import ExperimentRunConfig
    from capmas.graph.normalizer import CandidateNormalizer
    from capmas.graph.serialization import mission_graph_to_dict
    from capmas.llm.capx_compatible import CAPXCompatibleLLMClient
    from capmas.llm.protocol import LLMTraceCollector
    from capmas.llm.prompts import (
        build_manager_request,
        build_policy_request,
        build_staged_policy_request,
        build_topology_request,
    )
    from capmas.runtime.action_lease import ActionLeaseManager
    from capmas.runtime.artifact_bus import ArtifactStore, EventBus
    from capmas.runtime.episode_runner import to_jsonable
    from capmas.runtime.graph_interpreter import FixedGraphInterpreter
    from capmas.runtime.llm_scheduler import (
        LLMGraphRunResult,
        LLMGraphScheduleError,
        LLMGraphScheduler,
    )
    from capmas.perception.geometry_evidence import candidate_geometry_evidence
    from capmas.perception.motion_preview import ReferenceMotionPreview
    from capmas.runtime.rolling import RollingGraphRunner
    from capmas.runtime.orchestrator import RuntimeOrchestrator
    from capmas.runtime.scheduler import FixedGraphScheduler
    from capmas.runtime.state_store import InMemoryStateStore
    from capmas.verification.libero import (
        LiberoObservableVerifier,
        compile_time_preconditions,
        ground_libero_grasp_subgraph,
        libero_candidate_evidence,
        repair_libero_grasp_subgraph,
        validate_libero_grasp_subgraph,
        validate_libero_skill_sequence,
    )

    from capx.envs.configs.loader import DictLoader

    config = DictLoader.load(args.config_path)
    server_processes = []
    trace_collector = None
    run_config = None
    runtime = None
    artifact_graph = None
    artifact_arbitrations = {}
    artifact_failures = ()
    artifact_result = None
    scheduler_metrics = {}
    try:
        if not args.skip_api_servers:
            from capx.envs.runner import _start_api_servers

            server_processes = _start_api_servers(config.get("api_servers"))
        bundle = build_capx_runtime_from_yaml(
            args.config_path,
            loader=lambda _: config,
            object_names=(args.object_name, args.target_name),
        )
        world_model_enricher = None
        geometry_local_map = None
        if args.geometry_mode != "disabled":
            world_model_enricher = build_capx_world_model_enricher(
                bundle.observation_provider,
                depth_subsample=args.geometry_depth_subsample,
            )
            bundle.backend.set_scene_enricher(world_model_enricher)
            geometry_local_map = world_model_enricher.local_map
        runtime = RuntimeOrchestrator(
            backend=bundle.backend,
            state_store=InMemoryStateStore(),
            skill_registry=bundle.skill_registry,
            lease_manager=ActionLeaseManager(),
            verifier=LiberoObservableVerifier(),
        )
        episode = runtime.backend.reset(seed=args.seed)
        runtime.start_episode(episode)
        scene = runtime.state_store.latest()
        skill_metadata = tuple(
            item for item in bundle.skill_registry.snapshot_version().split(",") if item
        )
        skill_arg_names = bundle.skill_registry.argument_names()
        skill_arg_schemas = bundle.skill_registry.argument_schemas()
        policy_skill_metadata = tuple(
            item
            for item in skill_metadata
            if not item.startswith(
                ("get_observation@", "get_object_pose@", "lift_after_grasp@")
            )
        )
        task = bundle.task_language or f"Place {args.object_name} on {args.target_name}"
        run_config = ExperimentRunConfig(
            run_id=str(uuid4()),
            task_id=bundle.task_id,
            task=task,
            seed=args.seed,
            protocol=args.graph_protocol,
            proposal_mode=args.proposal_mode,
            execution_mode=args.execution_mode,
            model=args.model,
            endpoint_host=urlparse(args.api_base).hostname or "unknown",
            policy_agents=args.policy_agents,
            max_workers=args.max_workers,
            llm_deadline_ms=args.llm_deadline_ms,
            llm_max_output_tokens=args.llm_max_output_tokens,
            llm_max_retries=args.llm_max_retries,
            llm_proposal_retries=args.llm_proposal_retries,
            schema_mode=(
                "strict_provider_schema"
                if not args.no_provider_structured_output
                else "local_json_validation"
            ),
            manager_plan_fallback=args.allow_manager_plan_fallback,
            policy_strategies=policy_strategies,
            geometry_mode=args.geometry_mode,
            geometry_deadline_ms=args.geometry_deadline_ms,
            geometry_depth_subsample=args.geometry_depth_subsample,
            preview_backend=args.preview_backend,
            privilege_mode=args.privilege_mode,
            artifact_dir=str(phase5_run.path) if phase5_run is not None else "",
        )
        trace_collector = LLMTraceCollector()
        client = CAPXCompatibleLLMClient(
            endpoint=args.api_base,
            model=args.model,
            api_key=args.api_key,
            api_key_env=args.api_key_env,
            structured_outputs=not args.no_provider_structured_output,
            max_retries=args.llm_max_retries,
            trace_sink=trace_collector.record,
        )
        if args.graph_protocol == "staged":
            manager = LLMTopologyManager(
                client,
                lambda task, current_scene: build_topology_request(
                    task,
                    current_scene,
                    include_schema_in_prompt=args.no_provider_structured_output,
                    deadline_ms=args.llm_deadline_ms,
                    max_output_tokens=args.llm_max_output_tokens,
                ),
                proposal_retries=args.llm_proposal_retries,
                repair_request_builder=(
                    lambda task, current_scene, feedback: build_topology_request(
                        task,
                        current_scene,
                        include_schema_in_prompt=args.no_provider_structured_output,
                        deadline_ms=args.llm_deadline_ms,
                        max_output_tokens=args.llm_max_output_tokens,
                        repair_feedback=feedback,
                    )
                ),
            )
            policy_agents = tuple(
                LLMStagedGraphPolicyAgent(
                    client,
                    lambda subgoal, current_scene, context,
                    agent_name=_policy_agent_name(index, policy_strategies[index]),
                    strategy=policy_strategies[index]: (
                        build_staged_policy_request(
                            subgoal,
                            current_scene,
                            context,
                            skill_metadata=policy_skill_metadata,
                            skill_arg_names=skill_arg_names,
                            skill_arg_schemas=skill_arg_schemas,
                            include_schema_in_prompt=args.no_provider_structured_output,
                            policy_strategy=strategy,
                            agent_name=agent_name,
                            deadline_ms=args.llm_deadline_ms,
                            max_output_tokens=args.llm_max_output_tokens,
                        )
                    ),
                    agent_name=_policy_agent_name(index, policy_strategies[index]),
                    proposal_retries=args.llm_proposal_retries,
                    repair_request_builder=(
                        lambda subgoal, current_scene, context, feedback,
                        agent_name=_policy_agent_name(index, policy_strategies[index]),
                        strategy=policy_strategies[index]: (
                            build_staged_policy_request(
                                subgoal,
                                current_scene,
                                context,
                                skill_metadata=policy_skill_metadata,
                                skill_arg_names=skill_arg_names,
                                skill_arg_schemas=skill_arg_schemas,
                                include_schema_in_prompt=args.no_provider_structured_output,
                                policy_strategy=strategy,
                                agent_name=agent_name,
                                deadline_ms=args.llm_deadline_ms,
                                max_output_tokens=args.llm_max_output_tokens,
                                repair_feedback=feedback,
                            )
                        )
                    ),
                )
                for index in range(args.policy_agents)
            )
        else:
            manager = LLMMissionManager(
                client,
                lambda task, current_scene: build_manager_request(
                    task,
                    current_scene,
                    skill_metadata=skill_metadata,
                    skill_arg_names=skill_arg_names,
                    skill_arg_schemas=skill_arg_schemas,
                    include_schema_in_prompt=args.no_provider_structured_output,
                    deadline_ms=args.llm_deadline_ms,
                    max_output_tokens=args.llm_max_output_tokens,
                ),
            )
            policy_agents = tuple(
                LLMGraphPolicyAgent(
                    client,
                    lambda subgoal, current_scene, context,
                    agent_name=_policy_agent_name(index, policy_strategies[index]),
                    strategy=policy_strategies[index]: (
                        build_policy_request(
                            subgoal,
                            current_scene,
                            context,
                            skill_metadata=skill_metadata,
                            skill_arg_names=skill_arg_names,
                            skill_arg_schemas=skill_arg_schemas,
                            include_schema_in_prompt=args.no_provider_structured_output,
                            policy_strategy=strategy,
                            agent_name=agent_name,
                            deadline_ms=args.llm_deadline_ms,
                            max_output_tokens=args.llm_max_output_tokens,
                        )
                    ),
                        agent_name=_policy_agent_name(index, policy_strategies[index]),
                )
                for index in range(args.policy_agents)
            )

        def validate_skills(graph: object, context: object) -> None:
            typed_graph = graph
            typed_context = context
            for subgraph in typed_graph.subgraphs:
                validate_libero_grasp_subgraph(subgraph)
                for node in subgraph.nodes:
                    if node.node_type == "action":
                        contract = subgraph.to_action_contract(node.node_id, typed_context)
                        validate_libero_skill_sequence(contract.skills)
                        bundle.skill_registry.validate_contract(contract)
                        static_preconditions = compile_time_preconditions(
                            contract.preconditions
                        )
                        if static_preconditions:
                            approval = runtime.verifier.approve(
                                replace(contract, preconditions=static_preconditions),
                                typed_context.scene,
                            )
                            if not approval.passed:
                                details = "; ".join(
                                    f"{report.name}: {report.reason or 'failed'}"
                                    for report in approval.predicate_results
                                    if not report.passed
                                ) or "verifier approval failed"
                                raise ValueError(
                                    f"candidate precondition rejected: {details}"
                                )

        geometry_records = []
        preview_backend = ReferenceMotionPreview()

        def candidate_evidence(candidate: object, current_scene: object):
            perception = libero_candidate_evidence(candidate, current_scene)
            if args.geometry_mode == "disabled":
                return perception
            geometry = candidate_geometry_evidence(
                candidate,
                current_scene,
                geometry_local_map,
                preview_backend,
                time.monotonic_ns() + args.geometry_deadline_ms * 1_000_000,
            )
            geometry_records.append(geometry)
            if args.geometry_mode == "shadow":
                return perception
            available = tuple(perception.available_metrics)
            if geometry.measurable:
                available += ("geometry",)
            return replace(
                perception,
                geometry=geometry,
                available_metrics=available,
            )

        scheduler = LLMGraphScheduler(
            manager,
            {"*": policy_agents},
            max_workers=args.max_workers,
            proposal_mode=args.proposal_mode,
            require_policy_proposals=not args.allow_manager_plan_fallback,
            skill_validator=validate_skills,
            candidate_evidence_provider=candidate_evidence,
            candidate_normalizer=CandidateNormalizer(bundle.skill_registry),
            candidate_evidence_timeout_ms=(
                max(1.0, float(args.geometry_deadline_ms) - 10.0)
                if args.geometry_mode != "disabled"
                else None
            ),
            candidate_scene_rewriter=lambda subgraph, current_scene: ground_libero_grasp_subgraph(
                repair_libero_grasp_subgraph(subgraph), current_scene
            ),
        )
        interpreter = FixedGraphInterpreter(
            FixedGraphScheduler(runtime),
            artifact_store=ArtifactStore(),
            event_bus=EventBus(),
            checkpoint_evaluator=(
                lambda subgraph, node, context: _evaluate_checkpoint(
                    subgraph,
                    node,
                    context,
                    runtime.verifier,
                )
            ),
            max_steps=args.max_steps,
        )
        if args.execution_mode == "rolling":
            def refresh_scene(previous_scene: object) -> object:
                current_scene = previous_scene
                refreshed_scene = runtime.backend.observe()
                if not runtime.state_store.compare_and_commit(
                    current_scene.scene_version, refreshed_scene
                ):
                    raise RuntimeError("scene refresh raced with another state commit")
                return refreshed_scene

            result = RollingGraphRunner().run(
                task,
                scene,
                scheduler,
                interpreter,
                protocol=args.graph_protocol,
                max_cycles=args.max_steps,
                scene_refresh=refresh_scene,
            )
            final_compile = result.compilations[-1]
            artifact_graph = final_compile.graph
            artifact_arbitrations = {
                f"cycle-{index}": compile_result.arbitrations
                for index, compile_result in enumerate(result.compilations)
            }
            artifact_failures = tuple(
                failure
                for compile_result in result.compilations
                for failure in compile_result.proposal_failures
            )
            artifact_result = result
            scheduler_metrics = {
                "execution_mode": "rolling",
                "replan_count": result.replan_count,
                "cycles": len(result.compilations),
                "planning_mode": result.planning_mode,
                "frontier_subgraphs": result.frontier_subgraphs,
                "planning_scopes": tuple(
                    compile_result.planning_scope
                    for compile_result in result.compilations
                ),
                "manager_topology_calls": sum(
                    compile_result.manager_topology_calls
                    for compile_result in result.compilations
                ),
                "compile_latency_ms": tuple(
                    compile_result.compile_latency_ms
                    for compile_result in result.compilations
                ),
                "proposal_waves": tuple(
                    compile_result.proposal_waves
                    for compile_result in result.compilations
                ),
                "geometry_evidence": tuple(geometry_records),
                "world_model": _world_model_metrics(
                    world_model_enricher,
                    geometry_local_map,
                ),
            }
        else:
            compiled = scheduler.compile(
                task,
                scene,
                protocol=args.graph_protocol,
            )
            artifact_graph = compiled.graph
            artifact_arbitrations = compiled.arbitrations
            artifact_failures = compiled.proposal_failures
            # LLM compilation can exceed the scene_fresh budget. Refresh and
            # atomically commit the observation, then rerun scene-dependent
            # grounding before the graph reaches the physical interpreter.
            refreshed_scene = runtime.backend.observe()
            if not runtime.state_store.compare_and_commit(
                scene.scene_version, refreshed_scene
            ):
                raise RuntimeError("scene refresh raced with another state commit")
            rebased_graph = scheduler.rebase_graph(
                compiled.graph,
                refreshed_scene,
            )
            artifact_graph = rebased_graph
            rebased_compile = replace(compiled, graph=rebased_graph)
            refreshed_execution = interpreter.run(rebased_graph, refreshed_scene)
            result = LLMGraphRunResult(
                compile_result=rebased_compile,
                execution=refreshed_execution,
            )
            artifact_graph = result.compile_result.graph
            artifact_arbitrations = result.compile_result.arbitrations
            artifact_failures = result.compile_result.proposal_failures
            artifact_result = result.execution
            scheduler_metrics = {
                "execution_mode": "fixed_graph",
                "proposal_mode": result.compile_result.proposal_mode,
                "planning_scope": result.compile_result.planning_scope,
                "proposal_waves": result.compile_result.proposal_waves,
                "compile_latency_ms": result.compile_result.compile_latency_ms,
                "manager_topology_calls": result.compile_result.manager_topology_calls,
                "geometry_evidence": tuple(geometry_records),
                "world_model": _world_model_metrics(
                    world_model_enricher,
                    geometry_local_map,
                ),
            }
        payload = {
            "baseline": f"B3-LLM-{args.graph_protocol}-{args.execution_mode}",
            "log_file": str(log_path),
            "task_id": bundle.task_id,
            "task_language": bundle.task_language,
            "seed": args.seed,
            "model": args.model,
            "graph_protocol": args.graph_protocol,
            "graph": mission_graph_to_dict(artifact_graph),
            "arbitrations": artifact_arbitrations,
            "proposal_failures": artifact_failures,
            "result": artifact_result,
            "evaluator_success": bundle.backend.evaluator_success(),
            "run_config": run_config.to_dict(),
            "llm_calls": trace_collector.snapshot(),
            "scheduler_metrics": scheduler_metrics,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))
        if phase5_run is not None:
            phase5_run.write_json("run_config.json", run_config.to_dict())
            phase5_run.write_json("summary.json", payload)
            phase5_run.write_text(
                "summary.md",
                "# CAP-MAS Phase 5 run\n\n"
                f"- evaluator_success: {payload['evaluator_success']}\n"
                f"- seed: {args.seed}\n"
                f"- geometry_mode: {args.geometry_mode}\n",
            )
            phase5_run.finalize_manifest()
        print(f"CAP-MAS B3-LLM output: {args.output}")
        print(f"CAP-MAS B3-LLM evaluator_success: {payload['evaluator_success']}")
        print(f"CAP-MAS B3-LLM completed: {artifact_result.completed}")
    except Exception as exc:
        if isinstance(exc, LLMGraphScheduleError):
            artifact_failures = exc.proposal_failures
            if exc.arbitrations:
                artifact_arbitrations = exc.arbitrations
            if exc.partial_graph is not None:
                artifact_graph = exc.partial_graph
        failure_path = Path(args.output).with_suffix(".failure.json")
        failure_payload = {
            "baseline": f"B3-LLM-{args.graph_protocol}-{args.execution_mode}",
            "log_file": str(log_path),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "run_config": run_config.to_dict() if run_config is not None else None,
            "llm_calls": trace_collector.snapshot() if trace_collector is not None else (),
            "graph": (
                mission_graph_to_dict(artifact_graph)
                if artifact_graph is not None
                else None
            ),
            "arbitrations": artifact_arbitrations,
            "proposal_failures": artifact_failures,
            "partial_result": artifact_result,
            "scheduler_metrics": scheduler_metrics,
            "scene": (
                runtime.state_store.latest()
                if runtime is not None
                else None
            ),
        }
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(
            json.dumps(to_jsonable(failure_payload), indent=2, sort_keys=True)
        )
        if phase5_run is not None:
            phase5_run.write_json("failure.json", failure_payload)
            phase5_run.finalize_manifest()
        print(f"CAP-MAS B3-LLM failure artifact: {failure_path}")
        raise
    finally:
        for process in server_processes:
            process.terminate()
            process.join(timeout=5)


if __name__ == "__main__":
    main()
