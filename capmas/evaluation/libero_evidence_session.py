"""One-episode CAP-X/LIBERO evidence collection for P5.6D.

The session deliberately owns the physical runtime from reset through a
single graph execution.  Its decision-time evidence is derived from the same
version-one scene that the executor later receives; rehearsal remains an
independent provider layered on by the online runner.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Protocol

from capmas.contracts.candidates import (
    CandidateEvidence,
    EvidenceDimension,
    GeometryEvidence,
    GraphCandidate,
    subgraph_fingerprint,
)
from capmas.contracts.graph import MissionGraph
from capmas.contracts.scene import EpisodeStart, SceneSnapshot
from capmas.perception.effective_motion import (
    CandidateExecutionContext,
    EffectiveMotionProgram,
    bind_effective_motion,
    execution_graph_fingerprint,
    materialize_execution_graph,
)


class PreExecutionEvidenceSession(Protocol):
    """A live episode that supplies only pre-execution candidate evidence."""

    def start(self) -> SceneSnapshot: ...

    def candidate_evidence(self, candidate: GraphCandidate) -> CandidateEvidence: ...

    def execute(self, candidate: GraphCandidate, graph: MissionGraph) -> object: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class PreparedCandidate:
    """One candidate whose program, evidence, and execution graph are bound together."""

    context: CandidateExecutionContext
    program: EffectiveMotionProgram
    materialized_graph: MissionGraph
    evidence: CandidateEvidence


class EffectiveMotionEvidenceSession(PreExecutionEvidenceSession, Protocol):
    """A session that can bind preview evidence to the exact executed graph."""

    def prepare_candidate(
        self,
        candidate: GraphCandidate,
        graph: MissionGraph,
    ) -> PreparedCandidate: ...

    def execute_prepared(self, prepared: PreparedCandidate) -> object: ...


@dataclass(frozen=True)
class LiveLiberoEvidenceSessionConfig:
    """Frozen inputs for one same-runtime CAP-X/LIBERO episode."""

    config_path: str
    object_name: str
    target_name: str
    seed: int
    max_steps: int
    layout_variant: Mapping[str, object] = field(default_factory=dict)
    geometry_deadline_ms: float = 250.0

    def __post_init__(self) -> None:
        if not self.config_path:
            raise ValueError("CAP-X config path must not be empty")
        if not self.object_name or not self.target_name:
            raise ValueError("LIBERO object and target names must not be empty")
        if self.seed < 0:
            raise ValueError("LIBERO seed must not be negative")
        if self.max_steps <= 0:
            raise ValueError("LIBERO max steps must be positive")
        if self.geometry_deadline_ms <= 0.0:
            raise ValueError("geometry deadline must be positive")


@dataclass(frozen=True)
class LiveLiberoEvidenceResources:
    """Runtime products that can be injected for simulator-free testing."""

    runtime: object
    geometry_local_map: object | None = None
    runtime_bundle: object | None = None


ResourcesFactory = Callable[[LiveLiberoEvidenceSessionConfig], LiveLiberoEvidenceResources]
EvidenceCollector = Callable[[GraphCandidate, SceneSnapshot], CandidateEvidence]
GeometryCollector = Callable[[GraphCandidate, SceneSnapshot, object | None, object, int], GeometryEvidence]
GraphExecutor = Callable[[object, MissionGraph, SceneSnapshot, EpisodeStart, int], object]


class LiveLiberoEvidenceSession:
    """Collect pre-execution evidence and execute one winner in one runtime."""

    def __init__(
        self,
        config: LiveLiberoEvidenceSessionConfig,
        *,
        resources_factory: ResourcesFactory | None = None,
        evidence_collector: EvidenceCollector | None = None,
        geometry_collector: GeometryCollector | None = None,
        graph_executor: GraphExecutor | None = None,
    ) -> None:
        self._config = config
        self._resources_factory = resources_factory or _build_live_resources
        self._evidence_collector = evidence_collector or _collect_libero_evidence
        self._geometry_collector = geometry_collector or _collect_geometry_evidence
        self._graph_executor = graph_executor or _execute_live_graph
        self._resources: LiveLiberoEvidenceResources | None = None
        self._episode: EpisodeStart | None = None
        self._decision_scene: SceneSnapshot | None = None
        self._preview_backend: object | None = None
        self._executed = False
        self._closed = False

    @property
    def decision_scene(self) -> SceneSnapshot | None:
        return self._decision_scene

    def start(self) -> SceneSnapshot:
        """Reset once and commit the version-one scene used for arbitration."""

        if self._closed:
            raise RuntimeError("evidence session is closed")
        if self._decision_scene is not None:
            raise RuntimeError("evidence session has already started")
        resources = self._resources_factory(self._config)
        runtime = resources.runtime
        try:
            episode = runtime.backend.reset(seed=self._config.seed)
            runtime.start_episode(episode)
            initial_scene = runtime.state_store.latest()
            decision_scene = runtime.backend.observe()
            if decision_scene.scene_version != initial_scene.scene_version + 1:
                raise ValueError(
                    "same-runtime decision scene must increment initial scene by one"
                )
            if not runtime.state_store.compare_and_commit(
                initial_scene.scene_version, decision_scene
            ):
                raise RuntimeError("failed to commit same-runtime decision scene")
        except Exception:
            self._resources = resources
            self.close()
            raise

        self._resources = resources
        self._episode = episode
        self._decision_scene = decision_scene
        return decision_scene

    def candidate_evidence(self, candidate: GraphCandidate) -> CandidateEvidence:
        """Collect candidate-bound evidence from the committed decision scene."""

        scene = self._require_decision_scene()
        self._require_decision_version(candidate, scene)
        return self._collect_candidate_evidence(candidate, scene)

    def prepare_candidate(
        self,
        candidate: GraphCandidate,
        graph: MissionGraph,
    ) -> PreparedCandidate:
        """Bind the full success path used for both preview and physical execution."""

        scene = self._require_decision_scene()
        self._require_decision_version(candidate, scene)
        context = CandidateExecutionContext(
            candidate=candidate,
            mission_graph=graph,
            selected_subgraph_id=candidate.subgraph.subgraph_id,
            execution_graph_fingerprint=execution_graph_fingerprint(graph),
        )
        program = bind_effective_motion(context, scene)
        evidence = self._collect_candidate_evidence(candidate, scene, program=program)
        materialized_graph = materialize_execution_graph(program, graph)
        return PreparedCandidate(context, program, materialized_graph, evidence)

    def execute_prepared(self, prepared: PreparedCandidate) -> object:
        """Execute exactly the graph whose motion program was previewed."""

        scene = self._require_decision_scene()
        candidate = prepared.context.candidate
        self._require_decision_version(candidate, scene)
        if prepared.program.decision_scene_version != scene.scene_version:
            raise ValueError("prepared program decision scene does not match retained decision scene")
        if (
            prepared.program.execution_graph_fingerprint
            != prepared.context.execution_graph_fingerprint
        ):
            raise ValueError("prepared program execution graph fingerprint does not match context")
        if (
            execution_graph_fingerprint(prepared.context.mission_graph)
            != prepared.context.execution_graph_fingerprint
        ):
            raise ValueError("prepared context execution graph fingerprint does not match graph")
        if prepared.materialized_graph.parent_scene_version != scene.scene_version:
            raise ValueError("prepared materialized graph does not match decision scene")
        expected_graph = materialize_execution_graph(
            prepared.program,
            prepared.context.mission_graph,
        )
        if prepared.materialized_graph != expected_graph:
            raise ValueError("prepared materialized graph does not match execution graph fingerprint")
        geometry = prepared.evidence.geometry
        if geometry is None:
            raise ValueError("prepared candidate requires geometry evidence")
        if geometry.candidate_fingerprint != subgraph_fingerprint(candidate.subgraph):
            raise ValueError("prepared geometry evidence does not match candidate fingerprint")
        if geometry.scene_version != scene.scene_version:
            raise ValueError("prepared geometry evidence does not match decision scene")
        if geometry.program_scope != "mission_suffix":
            raise ValueError("prepared geometry evidence does not cover the mission suffix")
        if geometry.program_fingerprint != prepared.program.program_fingerprint:
            raise ValueError("prepared geometry evidence program fingerprint does not match program")
        if (
            geometry.execution_graph_fingerprint
            != prepared.program.execution_graph_fingerprint
        ):
            raise ValueError("prepared geometry evidence execution graph fingerprint does not match program")
        return self._execute_graph(prepared.materialized_graph)

    def _collect_candidate_evidence(
        self,
        candidate: GraphCandidate,
        scene: SceneSnapshot,
        *,
        program: EffectiveMotionProgram | None = None,
    ) -> CandidateEvidence:
        base = self._evidence_collector(candidate, scene)
        if not isinstance(base, CandidateEvidence):
            raise TypeError("candidate evidence collector must return CandidateEvidence")

        if self._preview_backend is None:
            self._preview_backend = _build_preview_backend()
        deadline_ns = time.monotonic_ns() + int(self._config.geometry_deadline_ms * 1_000_000)
        try:
            if program is None:
                geometry = self._geometry_collector(
                    candidate,
                    scene,
                    self._resources.geometry_local_map if self._resources is not None else None,
                    self._preview_backend,
                    deadline_ns,
                )
            else:
                geometry = _collect_program_geometry_evidence(
                    candidate,
                    scene,
                    self._resources.geometry_local_map if self._resources is not None else None,
                    self._preview_backend,
                    deadline_ns,
                    program,
                )
        except Exception as exc:  # noqa: BLE001 - geometry must fail open as typed unknown evidence.
            geometry = _unknown_geometry(candidate, scene, str(exc), program=program)
        if not isinstance(geometry, GeometryEvidence):
            raise TypeError("geometry collector must return GeometryEvidence")
        expected_fingerprint = subgraph_fingerprint(candidate.subgraph)
        if geometry.candidate_fingerprint != expected_fingerprint:
            raise ValueError("geometry evidence does not match candidate fingerprint")
        if geometry.scene_version != scene.scene_version:
            raise ValueError("geometry evidence does not match decision scene")
        if program is not None:
            if geometry.program_scope != "mission_suffix":
                raise ValueError("program geometry evidence must cover the mission suffix")
            if geometry.program_fingerprint != program.program_fingerprint:
                raise ValueError("geometry evidence does not match effective motion program")
            if geometry.execution_graph_fingerprint != program.execution_graph_fingerprint:
                raise ValueError("geometry evidence does not match execution graph")

        return replace(
            base,
            geometry=geometry,
            available_metrics=tuple(
                dict.fromkeys((*base.available_metrics, "geometry"))
            ),
            scene_version=scene.scene_version,
            captured_at_ns=(
                base.captured_at_ns
                if base.captured_at_ns is not None
                else scene.publish_timestamp_ns
            ),
        )

    def execute(self, candidate: GraphCandidate, graph: MissionGraph) -> object:
        """Execute one selected graph without rebuilding or resetting the backend."""

        scene = self._require_decision_scene()
        self._require_decision_version(candidate, scene)
        return self._execute_graph(graph)

    def _execute_graph(self, graph: MissionGraph) -> object:
        scene = self._require_decision_scene()
        if self._executed:
            raise RuntimeError("same-runtime session has already executed a candidate")
        assert self._resources is not None
        assert self._episode is not None
        self._executed = True
        return self._graph_executor(
            self._resources.runtime,
            graph,
            scene,
            self._episode,
            self._config.max_steps,
        )

    def close(self) -> None:
        """Stop the retained backend once, including after setup failures."""

        if self._closed:
            return
        self._closed = True
        if self._resources is None:
            return
        self._resources.runtime.backend.stop(None)

    def _require_decision_scene(self) -> SceneSnapshot:
        if self._decision_scene is None:
            raise RuntimeError("evidence session has not started")
        if self._closed:
            raise RuntimeError("evidence session is closed")
        return self._decision_scene

    @staticmethod
    def _require_decision_version(candidate: GraphCandidate, scene: SceneSnapshot) -> None:
        if candidate.parent_scene_version != scene.scene_version:
            raise ValueError(
                f"candidate parent scene {candidate.parent_scene_version} does not match "
                f"decision scene {scene.scene_version}"
            )


def _build_live_resources(
    config: LiveLiberoEvidenceSessionConfig,
) -> LiveLiberoEvidenceResources:
    """Construct the real CAP-X boundary lazily so unit tests stay simulator-free."""

    from capmas.backends.capx_libero_factory import (
        build_capx_runtime_from_yaml,
        build_capx_world_model_enricher,
    )
    from capmas.evaluation.layout_variants import LayoutResetHook
    from capmas.runtime.action_lease import ActionLeaseManager
    from capmas.runtime.orchestrator import RuntimeOrchestrator
    from capmas.runtime.state_store import InMemoryStateStore
    from capmas.verification.libero import LiberoObservableVerifier

    bundle = build_capx_runtime_from_yaml(
        config.config_path,
        object_names=(config.object_name, config.target_name),
        reset_hook=(LayoutResetHook(config.layout_variant) if config.layout_variant else None),
    )
    enricher = build_capx_world_model_enricher(bundle.observation_provider)
    bundle.backend.set_scene_enricher(enricher)
    runtime = RuntimeOrchestrator(
        backend=bundle.backend,
        state_store=InMemoryStateStore(),
        skill_registry=bundle.skill_registry,
        lease_manager=ActionLeaseManager(),
        verifier=LiberoObservableVerifier(),
    )
    return LiveLiberoEvidenceResources(
        runtime=runtime,
        geometry_local_map=enricher.local_map,
        runtime_bundle=bundle,
    )


def _collect_libero_evidence(
    candidate: GraphCandidate,
    scene: SceneSnapshot,
) -> CandidateEvidence:
    from capmas.verification.libero import libero_candidate_evidence

    return libero_candidate_evidence(candidate, scene)


def _build_preview_backend() -> object:
    from capmas.perception.motion_preview import ReferenceMotionPreview

    return ReferenceMotionPreview()


def _collect_geometry_evidence(
    candidate: GraphCandidate,
    scene: SceneSnapshot,
    local_map: object | None,
    preview_backend: object,
    deadline_ns: int,
) -> GeometryEvidence:
    from capmas.perception.geometry_evidence import candidate_geometry_evidence

    return candidate_geometry_evidence(
        candidate,
        scene,
        local_map,
        preview_backend,
        deadline_ns,
    )


def _collect_program_geometry_evidence(
    candidate: GraphCandidate,
    scene: SceneSnapshot,
    local_map: object | None,
    preview_backend: object,
    deadline_ns: int,
    program: EffectiveMotionProgram,
) -> GeometryEvidence:
    from capmas.perception.geometry_evidence import candidate_geometry_evidence

    return candidate_geometry_evidence(
        candidate,
        scene,
        local_map,
        preview_backend,
        deadline_ns,
        program=program,
    )


def _unknown_geometry(
    candidate: GraphCandidate,
    scene: SceneSnapshot,
    error: str,
    *,
    program: EffectiveMotionProgram | None = None,
) -> GeometryEvidence:
    reason = f"geometry provider failed: {error}"

    def unknown(name: str) -> EvidenceDimension:
        return EvidenceDimension(name, "unknown", None, None, reason)

    return GeometryEvidence(
        grasp_quality=unknown("grasp_quality"),
        reachability=unknown("reachability"),
        clearance=unknown("clearance"),
        collision_risk=unknown("collision_risk"),
        candidate_fingerprint=subgraph_fingerprint(candidate.subgraph),
        scene_version=scene.scene_version,
        map_version=None,
        map_backend="none",
        provider="motion_preview",
        provider_version="unknown",
        captured_at_ns=time.time_ns(),
        latency_ms=0.0,
        execution_graph_fingerprint=(
            program.execution_graph_fingerprint if program is not None else None
        ),
        program_fingerprint=program.program_fingerprint if program is not None else None,
        program_scope="mission_suffix" if program is not None else "subgraph",
    )


def _execute_live_graph(
    runtime: object,
    graph: MissionGraph,
    scene: SceneSnapshot,
    episode: EpisodeStart,
    max_steps: int,
) -> object:
    """Use the existing interpreter boundary for the retained runtime."""

    from capmas.runtime.artifact_bus import ArtifactStore
    from capmas.runtime.graph_interpreter import FixedGraphInterpreter
    from capmas.runtime.scheduler import FixedGraphScheduler
    from capmas.verification.libero import ground_libero_mission_graph

    grounded = ground_libero_mission_graph(graph, scene)
    result = FixedGraphInterpreter(
        FixedGraphScheduler(runtime),
        artifact_store=ArtifactStore(),
        max_steps=max_steps,
    ).run(
        grounded,
        scene,
        episode_id=episode.handle.episode_id,
        episode_epoch=episode.handle.episode_epoch,
        task_id=getattr(runtime.backend, "task_id", episode.handle.task_id),
    )
    return _execution_payload(
        result,
        grounded,
        evaluator_success=bool(runtime.backend.evaluator_success()),
    )


def _execution_payload(
    result: object,
    graph: MissionGraph,
    *,
    evaluator_success: bool,
) -> dict[str, object]:
    """Project graph telemetry into the physical outcome contract."""

    from capmas.evaluation.labels import extract_horizon

    failure = getattr(result, "failure", None)
    return {
        "completed": bool(getattr(result, "completed", False)),
        "graph_completed": bool(getattr(result, "completed", False)),
        "evaluator_success": evaluator_success,
        "verifier_success": bool(getattr(result, "completed", False)),
        "failure_class": getattr(failure, "failure_class", None),
        "failure_reason": getattr(failure, "message", None),
        "trace_count": len(getattr(result, "traces", ())),
        "horizon": extract_horizon(graph, getattr(result, "events", ())).to_dict(),
    }


__all__ = [
    "EffectiveMotionEvidenceSession",
    "LiveLiberoEvidenceResources",
    "LiveLiberoEvidenceSession",
    "LiveLiberoEvidenceSessionConfig",
    "PreExecutionEvidenceSession",
    "PreparedCandidate",
]
