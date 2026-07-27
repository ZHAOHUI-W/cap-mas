"""LLM proposal coordination above the single physical executor.

This module implements the first executable multi-agent boundary. Manager and
Policy Agents may run concurrently while producing immutable typed artifacts;
only the selected graph reaches ``FixedGraphInterpreter`` and the existing
``ActionLease``/Verifier runtime.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from collections.abc import Mapping, Sequence
import time
from typing import Callable
from uuid import uuid4

from capmas.agents.arbiter import CandidateArbiter
from capmas.contracts.agent import (
    AgentArtifact,
    AgentContext,
    GraphPolicyAgent,
    MissionGraphManager,
    MissionTopologyManager,
)
from capmas.contracts.candidates import (
    ArbitrationResult,
    CandidateEvidence,
    GraphCandidate,
    rewrite_report_for,
)
from capmas.contracts.graph import (
    CheckpointSpec,
    GraphEdge,
    MissionGraph,
    SubgraphNodeSpec,
    SubgraphSpec,
)
from capmas.contracts.scene import SceneSnapshot
from capmas.contracts.staged import MissionTopology, TopologySubgoal
from capmas.graph.staged import TopologyValidator
from capmas.graph.validator import GraphValidator, scene_initial_facts
from capmas.runtime.graph_interpreter import FixedGraphInterpreter, GraphExecutionResult


class LLMGraphScheduleError(RuntimeError):
    """Raised before execution when the LLM proposal phase has no safe result."""

    def __init__(
        self,
        message: str,
        *,
        proposal_failures: Sequence["PolicyProposalFailure"] = (),
        arbitrations: Mapping[str, ArbitrationResult] | None = None,
        partial_graph: MissionGraph | None = None,
    ) -> None:
        super().__init__(message)
        self.proposal_failures = tuple(proposal_failures)
        self.arbitrations = dict(arbitrations or {})
        self.partial_graph = partial_graph


@dataclass(frozen=True)
class PolicyProposalFailure:
    subgraph_id: str
    agent_name: str
    error_type: str
    message: str
    candidate_id: str | None = None
    diagnostics: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMGraphCompileResult:
    graph: MissionGraph
    arbitrations: Mapping[str, ArbitrationResult]
    proposal_failures: tuple[PolicyProposalFailure, ...] = ()
    proposal_mode: str = "subgoal_serial"
    proposal_waves: tuple[tuple[str, ...], ...] = ()
    compile_latency_ms: float = 0.0
    topology: MissionTopology | None = None
    planning_scope: str = "full_graph"
    manager_topology_calls: int = 0


@dataclass(frozen=True)
class LLMGraphRunResult:
    compile_result: LLMGraphCompileResult
    execution: GraphExecutionResult


PolicyMap = Mapping[str, Sequence[GraphPolicyAgent]]
CandidateEvidenceProvider = Callable[[GraphCandidate, SceneSnapshot], CandidateEvidence | None]


class LLMGraphScheduler:
    """Compile a validated graph using Manager and local Policy Agents.

    ``policy_agents`` maps a subgoal id to one or more candidate producers.
    Candidate generation is read-only and uses a bounded thread pool because
    provider calls are I/O-bound. It never shares an executor or robot backend
    with the workers. The caller chooses whether every subgoal requires a
    Policy proposal; the strict multi-agent mode should leave this enabled.
    """

    def __init__(
        self,
        manager: MissionGraphManager | MissionTopologyManager,
        policy_agents: PolicyMap,
        *,
        arbiter: CandidateArbiter | None = None,
        validator: GraphValidator | None = None,
        max_workers: int = 4,
        require_policy_proposals: bool = True,
        candidate_confidence: float | None = None,
        skill_validator: Callable[[MissionGraph, AgentContext], None] | None = None,
        candidate_rewriter: Callable[[SubgraphSpec], SubgraphSpec] | None = None,
        candidate_scene_rewriter: Callable[[SubgraphSpec, SceneSnapshot], SubgraphSpec] | None = None,
        proposal_mode: str = "subgoal_serial",
        candidate_evidence_provider: CandidateEvidenceProvider | None = None,
    ) -> None:
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if candidate_confidence is not None and not 0.0 <= candidate_confidence <= 1.0:
            raise ValueError("candidate_confidence must be in [0, 1]")
        if proposal_mode not in {"subgoal_serial", "ready_wave"}:
            raise ValueError("proposal_mode must be subgoal_serial or ready_wave")
        self.manager = manager
        self.policy_agents = {key: tuple(value) for key, value in policy_agents.items()}
        self.validator = validator or GraphValidator()
        self.arbiter = arbiter or CandidateArbiter(self.validator)
        self.max_workers = max_workers
        self.require_policy_proposals = require_policy_proposals
        self.candidate_confidence = candidate_confidence
        self.skill_validator = skill_validator
        self.candidate_rewriter = candidate_rewriter
        self.candidate_scene_rewriter = candidate_scene_rewriter
        self.proposal_mode = proposal_mode
        self.candidate_evidence_provider = candidate_evidence_provider

    def compile(
        self,
        task: str,
        scene: SceneSnapshot,
        *,
        context: AgentContext | None = None,
        protocol: str = "legacy",
    ) -> LLMGraphCompileResult:
        started = time.monotonic()
        if protocol == "staged":
            return self.compile_staged(task, scene, context=context)
        if protocol != "legacy":
            raise ValueError("graph protocol must be 'legacy' or 'staged'")
        graph = self.manager.propose_graph(task, scene)
        self._validate_manager_graph(graph, scene)
        context = context or AgentContext(
            task_id=task,
            episode_id=scene.episode_id,
            episode_epoch=scene.episode_epoch,
            scene=scene,
        )
        selected: list[SubgraphSpec] = []
        arbitrations: dict[str, ArbitrationResult] = {}
        failures: list[PolicyProposalFailure] = []

        for subgraph in graph.subgraphs:
            agents = self.policy_agents.get(
                subgraph.subgoal_id,
                self.policy_agents.get("*", ()),
            )
            if not agents:
                if self.require_policy_proposals:
                    raise LLMGraphScheduleError(
                        f"no policy agents registered for subgoal {subgraph.subgoal_id!r}"
                    )
                selected.append(subgraph)
                continue

            subgoal = _subgoal_artifact(graph, subgraph, scene)
            candidates, proposal_failures = self._propose_candidates(
                agents,
                expected_subgraph_id=subgraph.subgraph_id,
                expected_subgoal_id=subgraph.subgoal_id,
                subgoal=subgoal,
                scene=scene,
                context=context,
            )
            failures.extend(proposal_failures)
            candidates, skill_failures = self._filter_skill_candidates(
                candidates,
                task=task,
                context=context,
            )
            failures.extend(skill_failures)
            arbitration = self.arbiter.select(
                candidates,
                scene,
                expected_subgoal=subgraph.subgoal_id,
            )
            arbitrations[subgraph.subgraph_id] = arbitration
            if arbitration.selected is None:
                details = "; ".join(
                    f"{failure.agent_name}: {failure.message}"
                    for failure in (*proposal_failures, *skill_failures)
                ) or "all candidates were rejected by GraphValidator or Arbiter"
                raise LLMGraphScheduleError(
                    f"no valid policy candidate for subgoal {subgraph.subgoal_id!r}: {details}",
                    proposal_failures=tuple(failures),
                    arbitrations=arbitrations,
                )
            selected.append(arbitration.selected.subgraph)

        compiled = replace(graph, subgraphs=tuple(selected))
        self._validate_manager_graph(compiled, scene)
        if self.skill_validator is not None:
            self.skill_validator(compiled, context)
        return LLMGraphCompileResult(
            compiled,
            arbitrations,
            tuple(failures),
            proposal_mode=self.proposal_mode,
            proposal_waves=tuple((item.subgraph_id,) for item in graph.subgraphs),
            compile_latency_ms=(time.monotonic() - started) * 1000.0,
        )

    def run(
        self,
        task: str,
        scene: SceneSnapshot,
        interpreter: FixedGraphInterpreter,
        *,
        context: AgentContext | None = None,
        protocol: str = "legacy",
    ) -> LLMGraphRunResult:
        compiled = self.compile(task, scene, context=context, protocol=protocol)
        execution = interpreter.run(compiled.graph, scene, context=context)
        return LLMGraphRunResult(compiled, execution)

    def rebase_graph(
        self,
        graph: MissionGraph,
        scene: SceneSnapshot,
        *,
        context: AgentContext | None = None,
    ) -> MissionGraph:
        """Align a compiled graph with a freshly observed scene.

        LLM compilation can take longer than the runtime freshness budget. A
        fresh observation must therefore be committed before dispatch, while
        retaining the graph's validated topology. Scene-dependent candidate
        rewriting (for example LIBERO target-pose grounding) is rerun against
        the refreshed snapshot before the graph is admitted to the executor.
        """
        context = context or AgentContext(
            task_id=graph.task,
            episode_id=scene.episode_id,
            episode_epoch=scene.episode_epoch,
            scene=scene,
        )
        rebased_subgraphs: list[SubgraphSpec] = []
        for subgraph in graph.subgraphs:
            rewritten = self._rewrite_candidate(subgraph, scene)
            if rewritten.subgraph_id != subgraph.subgraph_id:
                raise LLMGraphScheduleError(
                    f"scene rewriter changed subgraph id from {subgraph.subgraph_id!r} "
                    f"to {rewritten.subgraph_id!r}"
                )
            if rewritten.subgoal_id != subgraph.subgoal_id:
                raise LLMGraphScheduleError(
                    f"scene rewriter changed subgoal id for {subgraph.subgraph_id!r}"
                )
            rebased_subgraphs.append(rewritten)
        rebased = replace(
            graph,
            subgraphs=tuple(rebased_subgraphs),
            parent_scene_version=scene.scene_version,
        )
        self._validate_manager_graph(rebased, scene)
        if self.skill_validator is not None:
            self.skill_validator(rebased, replace(context, scene=scene))
        return rebased

    def compile_staged(
        self,
        task: str,
        scene: SceneSnapshot,
        *,
        context: AgentContext | None = None,
    ) -> LLMGraphCompileResult:
        """Compile topology and local graphs as two independent LLM stages."""
        started = time.monotonic()
        manager = self.manager
        if not hasattr(manager, "propose_topology"):
            raise LLMGraphScheduleError(
                "staged protocol requires a MissionTopologyManager; "
                "use protocol='legacy' for a full MissionGraph manager"
            )
        topology = manager.propose_topology(task, scene)  # type: ignore[attr-defined]
        self._validate_topology(topology, scene)
        context = context or AgentContext(
            task_id=task,
            episode_id=scene.episode_id,
            episode_epoch=scene.episode_epoch,
            scene=scene,
        )
        if self.proposal_mode == "ready_wave":
            return self._compile_staged_ready_waves(
                task,
                topology,
                scene,
                context,
                started=started,
            )
        selected: list[SubgraphSpec] = []
        arbitrations: dict[str, ArbitrationResult] = {}
        failures: list[PolicyProposalFailure] = []
        for topology_subgoal in topology.subgoals:
            if topology_subgoal.execution_kind == "checkpoint_only":
                checkpoint, arbitration = _deterministic_checkpoint_candidate(
                    topology_subgoal,
                    scene,
                )
                selected.append(checkpoint)
                arbitrations[topology_subgoal.subgraph_id] = arbitration
                continue
            agents = self.policy_agents.get(
                topology_subgoal.subgoal_id,
                self.policy_agents.get("*", ()),
            )
            if not agents:
                raise LLMGraphScheduleError(
                    f"staged protocol has no policy agents for subgoal "
                    f"{topology_subgoal.subgoal_id!r}"
                )
            subgoal = _topology_subgoal_artifact(topology, topology_subgoal, scene)
            candidates, proposal_failures = self._propose_candidates(
                agents,
                expected_subgraph_id=topology_subgoal.subgraph_id,
                expected_subgoal_id=topology_subgoal.subgoal_id,
                subgoal=subgoal,
                scene=scene,
                context=context,
            )
            failures.extend(proposal_failures)
            candidates, skill_failures = self._filter_skill_candidates(
                candidates,
                task=task,
                context=context,
            )
            failures.extend(skill_failures)
            eligible_candidates = tuple(
                candidate
                for candidate in candidates
                if self._has_topology_postconditions(
                    topology_subgoal,
                    candidate.subgraph,
                )
            )
            arbitration = self.arbiter.select(
                eligible_candidates,
                scene,
                expected_subgoal=topology_subgoal.subgoal_id,
            )
            arbitrations[topology_subgoal.subgraph_id] = arbitration
            if arbitration.selected is None:
                details = "; ".join(
                    f"{failure.agent_name}: {failure.message}"
                    for failure in (*proposal_failures, *skill_failures)
                )
                if not details and candidates and not eligible_candidates:
                    details = "all staged candidates omitted topology success predicates"
                details = details or "all staged candidates were rejected by GraphValidator or Arbiter"
                raise LLMGraphScheduleError(
                    f"no valid staged policy candidate for subgoal "
                    f"{topology_subgoal.subgoal_id!r}: {details}",
                    proposal_failures=tuple(failures),
                    arbitrations=arbitrations,
                )
            self._require_topology_postconditions(topology_subgoal, arbitration.selected.subgraph)
            selected.append(arbitration.selected.subgraph)

        graph = topology.assemble(tuple(selected))
        self._validate_manager_graph(graph, scene)
        if self.skill_validator is not None:
            self.skill_validator(graph, context)
        return LLMGraphCompileResult(
            graph,
            arbitrations,
            tuple(failures),
            proposal_mode=self.proposal_mode,
            proposal_waves=tuple((item.subgraph_id,) for item in topology.subgoals),
            compile_latency_ms=(time.monotonic() - started) * 1000.0,
            topology=topology,
            manager_topology_calls=1,
        )

    def compile_ready_frontier(
        self,
        task: str,
        scene: SceneSnapshot,
        *,
        topology: MissionTopology | None = None,
        subgraph_id: str | None = None,
        completed_subgraphs: Sequence[str] = (),
        context: AgentContext | None = None,
    ) -> LLMGraphCompileResult:
        """Compile only one dependency-ready subgraph for rolling execution.

        The Manager is called only when ``topology`` is absent. Subsequent
        calls reuse the same topology and update only its scene-version
        envelope, so scene refreshes cannot turn an otherwise fixed topology
        into a stale planning artifact.
        """
        started = time.monotonic()
        manager = self.manager
        manager_called = topology is None
        if topology is None:
            if not hasattr(manager, "propose_topology"):
                raise LLMGraphScheduleError(
                    "ready-frontier compilation requires a MissionTopologyManager"
                )
            topology = manager.propose_topology(task, scene)  # type: ignore[attr-defined]
        topology = replace(topology, parent_scene_version=scene.scene_version)
        self._validate_topology(topology, scene)
        context = context or AgentContext(
            task_id=task,
            episode_id=scene.episode_id,
            episode_epoch=scene.episode_epoch,
            scene=scene,
        )

        completed = set(completed_subgraphs)
        if subgraph_id is None:
            if not completed:
                ready = (topology.entry_subgraph,)
            else:
                ready = tuple(
                    item.subgraph_id
                    for item in topology.subgoals
                    if item.subgraph_id not in completed
                    and set(item.depends_on) <= completed
                )
            if len(ready) != 1:
                raise LLMGraphScheduleError(
                    "ready-frontier compilation requires one explicit subgraph "
                    f"when the frontier has {len(ready)} candidates: {ready}"
                )
            subgraph_id = ready[0]
        if subgraph_id in completed:
            raise LLMGraphScheduleError(
                f"ready-frontier subgraph {subgraph_id!r} was already completed"
            )
        try:
            topology_subgoal = topology.subgoal(subgraph_id)
        except KeyError as exc:
            raise LLMGraphScheduleError(
                f"unknown ready-frontier subgraph {subgraph_id!r}"
            ) from exc
        missing_dependencies = set(topology_subgoal.depends_on) - completed
        if missing_dependencies:
            raise LLMGraphScheduleError(
                f"ready-frontier subgraph {subgraph_id!r} has unresolved "
                f"dependencies: {sorted(missing_dependencies)}"
            )

        failures: tuple[PolicyProposalFailure, ...] = ()
        if topology_subgoal.execution_kind == "checkpoint_only":
            selected, arbitration = _deterministic_checkpoint_candidate(
                topology_subgoal,
                scene,
            )
        else:
            agents = self.policy_agents.get(
                topology_subgoal.subgoal_id,
                self.policy_agents.get("*", ()),
            )
            if not agents:
                raise LLMGraphScheduleError(
                    f"staged protocol has no policy agents for subgoal "
                    f"{topology_subgoal.subgoal_id!r}"
                )
            subgoal = _topology_subgoal_artifact(topology, topology_subgoal, scene)
            candidates, proposal_failures = self._propose_candidates(
                agents,
                expected_subgraph_id=subgraph_id,
                expected_subgoal_id=topology_subgoal.subgoal_id,
                subgoal=subgoal,
                scene=scene,
                context=context,
            )
            selected, arbitration, skill_failures = self._select_staged_candidate(
                topology_subgoal,
                candidates,
                failures=proposal_failures,
                task=task,
                scene=scene,
                context=context,
            )
            failures = (*proposal_failures, *skill_failures)

        graph = MissionGraph(
            mission_id=topology.mission_id,
            task=task,
            subgraphs=(selected,),
            edges=(),
            bindings=(),
            entry_subgraph=subgraph_id,
            success_subgraphs=(subgraph_id,),
            failure_subgraphs=(subgraph_id,),
            parent_scene_version=scene.scene_version,
            graph_version=topology.graph_version,
        )
        self._validate_manager_graph(graph, scene)
        if self.skill_validator is not None:
            self.skill_validator(graph, context)
        return LLMGraphCompileResult(
            graph,
            {subgraph_id: arbitration},
            failures,
            proposal_mode="ready_frontier",
            proposal_waves=((subgraph_id,),),
            compile_latency_ms=(time.monotonic() - started) * 1000.0,
            topology=topology,
            planning_scope="ready_frontier",
            manager_topology_calls=1 if manager_called else 0,
        )

    def _compile_staged_ready_waves(
        self,
        task: str,
        topology: MissionTopology,
        scene: SceneSnapshot,
        context: AgentContext,
        *,
        started: float,
    ) -> LLMGraphCompileResult:
        """Fan out proposals for dependency-ready subgoals in bounded waves.

        This parallelizes only read-only proposal generation. The selected
        graph still reaches the single physical interpreter after compilation.
        """
        pending = {item.subgraph_id for item in topology.subgoals}
        planned: set[str] = set()
        selected_by_id: dict[str, SubgraphSpec] = {}
        arbitrations: dict[str, ArbitrationResult] = {}
        failures: list[PolicyProposalFailure] = []
        waves: list[tuple[str, ...]] = []

        while pending:
            ready = tuple(
                item.subgraph_id
                for item in topology.subgoals
                if item.subgraph_id in pending and set(item.depends_on) <= planned
            )
            if not ready:
                raise LLMGraphScheduleError("ready-wave proposal frontier is cyclic or unresolved")
            waves.append(ready)
            proposals_by_id, proposal_failures = self._propose_candidate_wave(
                topology,
                ready,
                scene,
                context,
            )
            failures.extend(proposal_failures)

            for subgraph_id in ready:
                topology_subgoal = topology.subgoal(subgraph_id)
                if topology_subgoal.execution_kind == "checkpoint_only":
                    selected, arbitration = _deterministic_checkpoint_candidate(
                        topology_subgoal,
                        scene,
                    )
                    selected_by_id[subgraph_id] = selected
                    arbitrations[subgraph_id] = arbitration
                    continue
                selected, arbitration, subgoal_failures = self._select_staged_candidate(
                    topology_subgoal,
                    proposals_by_id.get(subgraph_id, ()),
                    failures=tuple(
                        failure
                        for failure in proposal_failures
                        if failure.subgraph_id == subgraph_id
                    ),
                    task=task,
                    scene=scene,
                    context=context,
                )
                failures.extend(subgoal_failures)
                selected_by_id[subgraph_id] = selected
                arbitrations[subgraph_id] = arbitration

            pending.difference_update(ready)
            planned.update(ready)

        graph = topology.assemble(
            tuple(selected_by_id[item.subgraph_id] for item in topology.subgoals)
        )
        self._validate_manager_graph(graph, scene)
        if self.skill_validator is not None:
            self.skill_validator(graph, context)
        return LLMGraphCompileResult(
            graph,
            arbitrations,
            tuple(failures),
            proposal_mode=self.proposal_mode,
            proposal_waves=tuple(waves),
            compile_latency_ms=(time.monotonic() - started) * 1000.0,
            topology=topology,
            manager_topology_calls=1,
        )

    def _propose_candidate_wave(
        self,
        topology: MissionTopology,
        ready: tuple[str, ...],
        scene: SceneSnapshot,
        context: AgentContext,
    ) -> tuple[dict[str, tuple[GraphCandidate, ...]], tuple[PolicyProposalFailure, ...]]:
        jobs: list[tuple[str, int, GraphPolicyAgent]] = []
        artifacts: dict[str, AgentArtifact] = {}
        for subgraph_id in ready:
            topology_subgoal = topology.subgoal(subgraph_id)
            if topology_subgoal.execution_kind == "checkpoint_only":
                continue
            agents = self.policy_agents.get(
                topology_subgoal.subgoal_id,
                self.policy_agents.get("*", ()),
            )
            if not agents:
                raise LLMGraphScheduleError(
                    f"staged protocol has no policy agents for subgoal "
                    f"{topology_subgoal.subgoal_id!r}"
                )
            artifacts[subgraph_id] = _topology_subgoal_artifact(
                topology, topology_subgoal, scene
            )
            jobs.extend(
                (subgraph_id, index, agent)
                for index, agent in enumerate(agents)
            )

        candidates: dict[str, list[GraphCandidate]] = {subgraph_id: [] for subgraph_id in ready}
        failures: list[PolicyProposalFailure] = []
        workers = min(self.max_workers, len(jobs))
        if not jobs:
            return {subgraph_id: () for subgraph_id in ready}, ()
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="capmas-policy-wave") as pool:
            future_to_job = {
                pool.submit(
                    agent.propose_subgraph,
                    artifacts[subgraph_id],
                    scene,
                    context,
                ): (subgraph_id, index, agent)
                for subgraph_id, index, agent in jobs
            }
            for future in as_completed(future_to_job):
                subgraph_id, index, agent = future_to_job[future]
                try:
                    raw_proposal = future.result()
                    proposal = self._rewrite_candidate(raw_proposal, scene)
                    topology_subgoal = topology.subgoal(subgraph_id)
                    if proposal.subgraph_id != subgraph_id:
                        raise ValueError(
                            f"proposal targets {proposal.subgraph_id!r}, expected {subgraph_id!r}"
                        )
                    if proposal.subgoal_id != topology_subgoal.subgoal_id:
                        raise ValueError(
                            f"proposal targets subgoal {proposal.subgoal_id!r}, "
                            f"expected {topology_subgoal.subgoal_id!r}"
                        )
                    candidate = GraphCandidate(
                        candidate_id=f"{subgraph_id}:{agent.name}:{index}",
                        subgraph=proposal,
                        parent_scene_version=scene.scene_version,
                        producer_agent=agent.name,
                        confidence=self.candidate_confidence,
                        strategy=_agent_strategy(agent),
                        raw_subgraph=raw_proposal,
                        rewrite_report=rewrite_report_for(raw_proposal, proposal),
                    )
                    candidates[subgraph_id].append(
                        self._attach_candidate_evidence(candidate, scene)
                    )
                except Exception as exc:
                    failures.append(
                        PolicyProposalFailure(
                            subgraph_id,
                            agent.name,
                            type(exc).__name__,
                            str(exc),
                        )
                    )
        ordered = {
            subgraph_id: tuple(sorted(items, key=lambda item: item.candidate_id))
            for subgraph_id, items in candidates.items()
        }
        failures.sort(key=lambda item: (item.subgraph_id, item.agent_name))
        return ordered, tuple(failures)

    def _select_staged_candidate(
        self,
        topology_subgoal: TopologySubgoal,
        candidates: tuple[GraphCandidate, ...],
        *,
        failures: tuple[PolicyProposalFailure, ...],
        task: str,
        scene: SceneSnapshot,
        context: AgentContext,
    ) -> tuple[SubgraphSpec, ArbitrationResult, tuple[PolicyProposalFailure, ...]]:
        candidates, skill_failures = self._filter_skill_candidates(
            candidates,
            task=task,
            context=context,
        )
        eligible_candidates = tuple(
            candidate
            for candidate in candidates
            if self._has_topology_postconditions(topology_subgoal, candidate.subgraph)
        )
        arbitration = self.arbiter.select(
            eligible_candidates,
            scene,
            expected_subgoal=topology_subgoal.subgoal_id,
        )
        if arbitration.selected is None:
            details = "; ".join(
                f"{failure.agent_name}: {failure.message}"
                for failure in (*failures, *skill_failures)
            )
            if not details and candidates and not eligible_candidates:
                details = "all staged candidates omitted topology success predicates"
            details = details or "all staged candidates were rejected by GraphValidator or Arbiter"
            raise LLMGraphScheduleError(
                f"no valid staged policy candidate for subgoal "
                f"{topology_subgoal.subgoal_id!r}: {details}",
                proposal_failures=(*failures, *skill_failures),
            )
        self._require_topology_postconditions(topology_subgoal, arbitration.selected.subgraph)
        return arbitration.selected.subgraph, arbitration, tuple(skill_failures)

    def _propose_candidates(
        self,
        agents: Sequence[GraphPolicyAgent],
        *,
        expected_subgraph_id: str,
        expected_subgoal_id: str,
        subgoal: AgentArtifact,
        scene: SceneSnapshot,
        context: AgentContext,
    ) -> tuple[tuple[GraphCandidate, ...], tuple[PolicyProposalFailure, ...]]:
        candidates: list[GraphCandidate] = []
        failures: list[PolicyProposalFailure] = []
        workers = min(self.max_workers, len(agents))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="capmas-policy") as pool:
            future_to_agent = {
                pool.submit(agent.propose_subgraph, subgoal, scene, context): (index, agent)
                for index, agent in enumerate(agents)
            }
            for future in as_completed(future_to_agent):
                index, agent = future_to_agent[future]
                try:
                    raw_proposal = future.result()
                    proposal = self._rewrite_candidate(raw_proposal, scene)
                    if proposal.subgraph_id != expected_subgraph_id:
                        raise ValueError(
                            f"proposal targets {proposal.subgraph_id!r}, "
                            f"expected {expected_subgraph_id!r}"
                        )
                    if proposal.subgoal_id != expected_subgoal_id:
                        raise ValueError(
                            f"proposal targets subgoal {proposal.subgoal_id!r}, "
                            f"expected {expected_subgoal_id!r}"
                        )
                    candidate = GraphCandidate(
                        candidate_id=(
                            f"{expected_subgraph_id}:"
                            f"{agent.name}:{index}"
                        ),
                        subgraph=proposal,
                        parent_scene_version=scene.scene_version,
                        producer_agent=agent.name,
                        confidence=self.candidate_confidence,
                        strategy=_agent_strategy(agent),
                        raw_subgraph=raw_proposal,
                        rewrite_report=rewrite_report_for(raw_proposal, proposal),
                    )
                    candidates.append(self._attach_candidate_evidence(candidate, scene))
                except Exception as exc:
                    failures.append(
                        PolicyProposalFailure(
                            expected_subgraph_id,
                            agent.name,
                            type(exc).__name__,
                            str(exc),
                        )
                    )
        # Candidate id is intentionally unique, but sorting makes the returned
        # evidence deterministic for replay and stable test reports.
        candidates.sort(key=lambda item: item.candidate_id)
        failures.sort(key=lambda item: item.agent_name)
        return tuple(candidates), tuple(failures)

    def _attach_candidate_evidence(
        self,
        candidate: GraphCandidate,
        scene: SceneSnapshot,
    ) -> GraphCandidate:
        if self.candidate_evidence_provider is None:
            return candidate
        evidence = self.candidate_evidence_provider(candidate, scene)
        if evidence is None:
            return candidate
        return replace(candidate, evidence=evidence)

    def _rewrite_candidate(
        self,
        proposal: SubgraphSpec,
        scene: SceneSnapshot,
    ) -> SubgraphSpec:
        if self.candidate_scene_rewriter is not None:
            return self.candidate_scene_rewriter(proposal, scene)
        if self.candidate_rewriter is not None:
            return self.candidate_rewriter(proposal)
        return proposal

    def _filter_skill_candidates(
        self,
        candidates: tuple[GraphCandidate, ...],
        *,
        task: str,
        context: AgentContext,
    ) -> tuple[tuple[GraphCandidate, ...], tuple[PolicyProposalFailure, ...]]:
        if self.skill_validator is None:
            return candidates, ()
        valid: list[GraphCandidate] = []
        failures: list[PolicyProposalFailure] = []
        for candidate in candidates:
            subgraph = candidate.subgraph
            candidate_graph = MissionGraph(
                mission_id=f"candidate:{candidate.candidate_id}",
                task=task,
                subgraphs=(subgraph,),
                edges=(),
                bindings=(),
                entry_subgraph=subgraph.subgraph_id,
                success_subgraphs=(subgraph.subgraph_id,),
                failure_subgraphs=(subgraph.subgraph_id,),
                parent_scene_version=context.scene.scene_version,
            )
            try:
                self.skill_validator(candidate_graph, context)
            except Exception as exc:
                failures.append(
                    PolicyProposalFailure(
                        subgraph.subgraph_id,
                        candidate.producer_agent,
                        type(exc).__name__,
                        f"candidate skill validation failed: {exc}",
                        candidate_id=candidate.candidate_id,
                        diagnostics=_candidate_validation_diagnostics(candidate, exc),
                    )
                )
            else:
                valid.append(candidate)
        return tuple(valid), tuple(failures)

    def _validate_manager_graph(self, graph: MissionGraph, scene: SceneSnapshot) -> None:
        if graph.parent_scene_version != scene.scene_version:
            raise LLMGraphScheduleError(
                f"stale manager graph: targets scene {graph.parent_scene_version}, "
                f"current scene is {scene.scene_version}"
            )
        validation = self.validator.validate(
            graph,
            initial_facts=scene_initial_facts(graph, scene),
        )
        if not validation.valid:
            details = "; ".join(f"{item.code}: {item.message}" for item in validation.errors)
            raise LLMGraphScheduleError(f"manager graph is invalid: {details}")

    def _validate_topology(self, topology: MissionTopology, scene: SceneSnapshot) -> None:
        if topology.parent_scene_version != scene.scene_version:
            raise LLMGraphScheduleError(
                f"stale manager topology: targets scene {topology.parent_scene_version}, "
                f"current scene is {scene.scene_version}"
            )
        validation = TopologyValidator().validate(topology)
        if not validation.valid:
            details = "; ".join(f"{item.code}: {item.message}" for item in validation.errors)
            raise LLMGraphScheduleError(f"manager topology is invalid: {details}")

    @staticmethod
    def _require_topology_postconditions(
        topology_subgoal: TopologySubgoal,
        subgraph: SubgraphSpec,
    ) -> None:
        if not LLMGraphScheduler._has_topology_postconditions(topology_subgoal, subgraph):
            declared = {
                predicate
                for node in subgraph.nodes
                for predicate in node.postconditions
            }
            declared.update(
                predicate
                for checkpoint in subgraph.checkpoints
                if checkpoint.validate
                for predicate in checkpoint.predicates
            )
            missing = sorted(set(topology_subgoal.success_predicates) - declared)
            raise LLMGraphScheduleError(
                f"staged candidate {subgraph.subgraph_id!r} omits topology "
                f"success predicates: {missing}"
            )

    @staticmethod
    def _has_topology_postconditions(
        topology_subgoal: TopologySubgoal,
        subgraph: SubgraphSpec,
    ) -> bool:
        declared = {
            predicate
            for node in subgraph.nodes
            for predicate in node.postconditions
        }
        declared.update(
            predicate
            for checkpoint in subgraph.checkpoints
            if checkpoint.validate
            for predicate in checkpoint.predicates
        )
        return not set(topology_subgoal.success_predicates) - declared


def _candidate_validation_diagnostics(
    candidate: GraphCandidate,
    error: Exception,
) -> dict[str, object]:
    """Attach the failing typed call to a scheduler-level rejection."""
    requested_skill = getattr(error, "skill_id", None)
    requested_index = getattr(error, "call_index", None)
    selected_node: object | None = None
    selected_call: object | None = None
    selected_index: int | None = None

    for node in candidate.subgraph.nodes:
        if node.node_type != "action":
            continue
        for index, call in enumerate(node.skill_calls):
            if requested_skill is not None and call.skill.skill_id != requested_skill:
                continue
            if requested_index is not None and index != requested_index:
                continue
            selected_node = node
            selected_call = call
            selected_index = index
            break
        if selected_call is not None:
            break

    if selected_call is None:
        for node in candidate.subgraph.nodes:
            if node.node_type == "action" and node.skill_calls:
                selected_node = node
                selected_call = node.skill_calls[0]
                selected_index = 0
                break

    diagnostics: dict[str, object] = {
        "candidate_id": candidate.candidate_id,
        "subgraph_id": candidate.subgraph.subgraph_id,
        "error": str(error),
        "raw_fingerprint": candidate.rewrite_report.raw_fingerprint,
        "normalized_fingerprint": candidate.rewrite_report.normalized_fingerprint,
        "rewrite_operations": tuple(candidate.rewrite_report.operations),
    }
    if selected_node is not None and selected_call is not None:
        diagnostics.update(
            {
                "node_id": selected_node.node_id,
                "skill_id": selected_call.skill.skill_id,
                "skill_version": selected_call.skill.version,
                "skill_call_index": selected_index,
                "args": dict(selected_call.args),
            }
        )
        if candidate.raw_subgraph is not None:
            try:
                raw_node = candidate.raw_subgraph.node(selected_node.node_id)
            except KeyError:
                raw_node = None
            if raw_node is not None and selected_index is not None:
                if selected_index < len(raw_node.skill_calls):
                    diagnostics["raw_args"] = dict(
                        raw_node.skill_calls[selected_index].args
                    )
    expected_signature = getattr(error, "expected_signature", None)
    diagnostics["expected_signature"] = expected_signature or "unknown"
    diagnostics["missing_arguments"] = tuple(
        getattr(error, "missing_arguments", ())
    )
    diagnostics["unexpected_arguments"] = tuple(
        getattr(error, "unexpected_arguments", ())
    )
    return diagnostics


def _deterministic_checkpoint_candidate(
    topology_subgoal: TopologySubgoal,
    scene: SceneSnapshot,
) -> tuple[SubgraphSpec, ArbitrationResult]:
    """Build a pure observation barrier without calling a Policy Agent.

    A checkpoint-only subgoal has no physical action to synthesize. Keeping it
    as a typed graph node preserves topology, failure routing, and artifact
    observability while removing an unnecessary provider request.
    """
    predicates = tuple(topology_subgoal.success_predicates)
    if not predicates:
        raise LLMGraphScheduleError(
            f"checkpoint-only subgoal {topology_subgoal.subgoal_id!r} "
            "must declare at least one success predicate"
        )
    node_id = f"{topology_subgoal.subgraph_id}:checkpoint"
    failure_node_id = f"{topology_subgoal.subgraph_id}:checkpoint_failure"
    node = SubgraphNodeSpec(
        node_id=node_id,
        description=topology_subgoal.description,
        postconditions=predicates,
        proposed_by="checkpoint_compiler",
        recovery_policy="replan",
        node_type="checkpoint",
        max_duration_ms=1_000,
        max_sim_steps=1,
    )
    failure_node = SubgraphNodeSpec(
        node_id=failure_node_id,
        description=f"record failed checkpoint for {topology_subgoal.subgoal_id}",
        proposed_by="checkpoint_compiler",
        recovery_policy="fail_terminal",
        node_type="checkpoint",
        max_duration_ms=1_000,
        max_sim_steps=1,
    )
    subgraph = SubgraphSpec(
        subgraph_id=topology_subgoal.subgraph_id,
        subgoal_id=topology_subgoal.subgoal_id,
        description=topology_subgoal.description,
        nodes=(node, failure_node),
        edges=(GraphEdge(node_id, failure_node_id, "failure"),),
        entry_node=node_id,
        success_nodes=(node_id,),
        failure_nodes=(failure_node_id,),
        checkpoints=(CheckpointSpec(node_id, predicates),),
        assigned_agent="checkpoint_compiler",
    )
    candidate = GraphCandidate(
        candidate_id=f"{topology_subgoal.subgraph_id}:checkpoint_compiler:0",
        subgraph=subgraph,
        parent_scene_version=scene.scene_version,
        producer_agent="checkpoint_compiler",
        raw_subgraph=subgraph,
    )
    return subgraph, ArbitrationResult(
        selected=candidate,
        considered=(candidate,),
        selection_basis="deterministic_checkpoint",
    )


def _subgoal_artifact(
    graph: MissionGraph,
    subgraph: SubgraphSpec,
    scene: SceneSnapshot,
) -> AgentArtifact:
    return AgentArtifact(
        artifact_id=str(uuid4()),
        kind="subgoal",
        payload={
            "mission_id": graph.mission_id,
            "task": graph.task,
            "subgraph_id": subgraph.subgraph_id,
            "subgoal_id": subgraph.subgoal_id,
            "description": subgraph.description,
            "parent_scene_version": scene.scene_version,
            "graph_version": graph.graph_version,
        },
        source_agent="mission_manager",
    )


def _topology_subgoal_artifact(
    topology: MissionTopology,
    subgoal: TopologySubgoal,
    scene: SceneSnapshot,
) -> AgentArtifact:
    return AgentArtifact(
        artifact_id=str(uuid4()),
        kind="topology_subgoal",
        payload={
            "mission_id": topology.mission_id,
            "task": topology.task,
            "subgraph_id": subgoal.subgraph_id,
            "subgoal_id": subgoal.subgoal_id,
            "description": subgoal.description,
            "depends_on": list(subgoal.depends_on),
            "success_predicates": list(subgoal.success_predicates),
            "failure_predicates": list(subgoal.failure_predicates),
            "required_agent_role": subgoal.required_agent_role,
            "execution_kind": subgoal.execution_kind,
            "parent_scene_version": scene.scene_version,
            "graph_version": topology.graph_version,
        },
        source_agent="mission_manager",
    )


def _agent_strategy(agent: GraphPolicyAgent) -> str:
    """Resolve explicit strategy metadata or the P3.1 name suffix."""
    declared = getattr(agent, "policy_strategy", None)
    if not isinstance(declared, str) or not declared:
        name = getattr(agent, "name", "")
        declared = name.rsplit(":", 1)[-1] if ":" in name else "balanced"
    from capmas.contracts.strategy import StrategyProfile

    return StrategyProfile.for_name(declared).name


__all__ = [
    "LLMGraphCompileResult",
    "LLMGraphRunResult",
    "LLMGraphScheduleError",
    "LLMGraphScheduler",
    "PolicyProposalFailure",
]
