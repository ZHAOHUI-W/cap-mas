from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from uuid import uuid4

from capmas.contracts.action import SkillCall
from capmas.contracts.agent import AgentContext, CycleHistory
from capmas.contracts.candidates import subgraph_fingerprint
from capmas.contracts.failures import FailureArtifact, FailureClass
from capmas.contracts.graph import (
    MissionGraph,
    SubgraphNodeSpec,
    SubgraphSpec,
)
from capmas.contracts.scene import SceneSnapshot
from capmas.contracts.trace import ExecutionTrace, GraphEventKind, GraphExecutionEvent
from capmas.graph.validator import GraphValidator, scene_initial_facts
from capmas.runtime.artifact_bus import ArtifactEnvelope, ArtifactStore, EventBus, RuntimeEvent
from capmas.runtime.orchestrator import CycleResult
from capmas.runtime.recovery import RecoverySelector
from capmas.runtime.scheduler import Scheduler


class GraphExecutionError(RuntimeError):
    """Raised when a graph cannot produce a safe next transition."""


@dataclass(frozen=True)
class GraphExecutionResult:
    completed: bool
    scene: SceneSnapshot
    context: AgentContext
    terminal_subgraph: str | None = None
    traces: tuple[ExecutionTrace, ...] = ()
    failure: FailureArtifact | None = None
    outputs: dict[str, dict[str, object]] | None = None
    next_subgraph: str | None = None
    events: tuple[GraphExecutionEvent, ...] = ()


@dataclass(frozen=True)
class _SubgraphResult:
    success: bool
    context: AgentContext
    outcome: str
    traces: tuple[ExecutionTrace, ...]
    failure: FailureArtifact | None = None
    outputs: dict[str, object] | None = None


ControlEvaluator = Callable[[SubgraphNodeSpec, AgentContext], str]
CheckpointEvaluator = Callable[[SubgraphSpec, SubgraphNodeSpec, AgentContext], str]


class FixedGraphInterpreter:
    """Execute a validated MissionGraph with one physical dispatch seam.

    The interpreter follows graph control flow, but it never calls skills
    directly.  Every action node is lowered to an ``ActionContract`` and sent
    through ``Scheduler.dispatch``; the scheduler/orchestrator remains the
    sole owner of the actuator lease and observable verification.
    """

    def __init__(
        self,
        scheduler: Scheduler,
        *,
        validator: GraphValidator | None = None,
        control_evaluator: ControlEvaluator | None = None,
        checkpoint_evaluator: CheckpointEvaluator | None = None,
        artifact_store: ArtifactStore | None = None,
        event_bus: EventBus | None = None,
        recovery_selector: RecoverySelector | None = None,
        max_steps: int = 100,
        clock: Callable[[], int] = time.time_ns,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.scheduler = scheduler
        self.validator = validator or GraphValidator()
        self.control_evaluator = control_evaluator
        self.checkpoint_evaluator = checkpoint_evaluator
        self.artifact_store = artifact_store
        self.event_bus = event_bus
        self.recovery_selector = recovery_selector
        self.max_steps = max_steps
        self.clock = clock

    def run(
        self,
        graph: MissionGraph,
        scene: SceneSnapshot,
        *,
        episode_id: str | None = None,
        episode_epoch: int | None = None,
        task_id: str | None = None,
        context: AgentContext | None = None,
        stop_after_subgraph: bool = False,
        initial_outputs: Mapping[str, Mapping[str, object]] | None = None,
    ) -> GraphExecutionResult:
        self.validator.raise_if_invalid(
            graph,
            initial_facts=scene_initial_facts(graph, scene),
        )
        if context is None:
            context = AgentContext(
                task_id=task_id or graph.mission_id,
                episode_id=episode_id or scene.episode_id,
                episode_epoch=scene.episode_epoch if episode_epoch is None else episode_epoch,
                scene=scene,
            )
        elif context.scene != scene:
            context = replace(context, scene=scene)

        current_id = graph.entry_subgraph
        visits: dict[str, int] = {}
        node_event_visits: dict[tuple[str, str], int] = {}
        traces: list[ExecutionTrace] = []
        events: list[GraphExecutionEvent] = []
        mission_outputs: dict[str, dict[str, object]] = {
            subgraph_id: dict(outputs)
            for subgraph_id, outputs in (initial_outputs or {}).items()
        }

        for _ in range(self.max_steps):
            visits[current_id] = visits.get(current_id, 0) + 1
            loop = _loop_for(graph.loops, current_id)
            if loop is not None and visits[current_id] > loop.max_visits:
                failure = self._failure(
                    FailureClass.EXECUTION_ERROR,
                    f"mission loop {current_id!r} exceeded max_visits={loop.max_visits}",
                    context,
                    subgraph_id=current_id,
                )
                self._publish_failure(failure)
                return GraphExecutionResult(
                    False, context.scene, context, current_id, tuple(traces), failure,
                    mission_outputs, events=tuple(events),
                )
            if loop is None and visits[current_id] > 1:
                failure = self._failure(
                    FailureClass.EXECUTION_ERROR,
                    f"subgraph {current_id!r} was re-entered without LoopSpec",
                    context,
                    subgraph_id=current_id,
                )
                self._publish_failure(failure)
                return GraphExecutionResult(
                    False, context.scene, context, current_id, tuple(traces), failure,
                    mission_outputs, events=tuple(events),
                )

            external_inputs = _mission_inputs(graph, current_id, mission_outputs)
            self._emit_event(
                events,
                "subgraph_started",
                current_id,
                attempt=visits[current_id],
            )
            subgraph_result = self._run_subgraph(
                graph.subgraph(current_id),
                context,
                external_inputs,
                events,
                visits[current_id],
                node_event_visits,
            )
            context = subgraph_result.context
            traces.extend(subgraph_result.traces)
            mission_outputs[current_id] = dict(subgraph_result.outputs or {})

            if not subgraph_result.success and subgraph_result.failure is not None:
                self._publish_failure(subgraph_result.failure)
                recovery_target = self._select_recovery_target(
                    graph, current_id, subgraph_result.failure, context
                )
                if recovery_target is not None:
                    current_id = recovery_target
                    continue

            if subgraph_result.success and current_id in graph.success_subgraphs:
                return GraphExecutionResult(
                    True,
                    context.scene,
                    context,
                    current_id,
                    tuple(traces),
                    outputs=mission_outputs,
                    events=tuple(events),
                )
            if not subgraph_result.success and current_id in graph.failure_subgraphs:
                return GraphExecutionResult(
                    False,
                    context.scene,
                    context,
                    current_id,
                    tuple(traces),
                    subgraph_result.failure,
                    mission_outputs,
                    events=tuple(events),
                )

            next_id = _select_transition(
                graph.edges,
                current_id,
                subgraph_result.outcome,
            )
            if next_id is None:
                failure = subgraph_result.failure or self._failure(
                    FailureClass.EXECUTION_ERROR,
                    f"no mission transition from {current_id!r} for outcome "
                    f"{subgraph_result.outcome!r}",
                    context,
                    subgraph_id=current_id,
                )
                return GraphExecutionResult(
                    False, context.scene, context, current_id, tuple(traces), failure,
                    mission_outputs, events=tuple(events),
                )
            if stop_after_subgraph:
                return GraphExecutionResult(
                    False,
                    context.scene,
                    context,
                    current_id,
                    tuple(traces),
                    outputs=mission_outputs,
                    next_subgraph=next_id,
                    events=tuple(events),
                )
            current_id = next_id

        failure = self._failure(
            FailureClass.EXECUTION_ERROR,
            f"mission exceeded max_steps={self.max_steps}",
            context,
        )
        self._publish_failure(failure)
        return GraphExecutionResult(
            False, context.scene, context, current_id, tuple(traces), failure,
            mission_outputs, events=tuple(events),
        )

    def _run_subgraph(
        self,
        subgraph: SubgraphSpec,
        context: AgentContext,
        external_inputs: dict[str, object] | None = None,
        events: list[GraphExecutionEvent] | None = None,
        subgraph_attempt: int = 1,
        node_event_visits: dict[tuple[str, str], int] | None = None,
    ) -> _SubgraphResult:
        current_id = subgraph.entry_node
        visits: dict[str, int] = {}
        loop_started: dict[str, float] = {}
        traces: list[ExecutionTrace] = []
        external_inputs = dict(external_inputs or {})
        node_outputs: dict[str, dict[str, object]] = {}
        pending_failure: tuple[str, dict[str, object]] | None = None
        events = events if events is not None else []
        node_event_visits = node_event_visits if node_event_visits is not None else {}

        def finish(result: _SubgraphResult) -> _SubgraphResult:
            self._emit_event(
                events,
                "subgraph_completed" if result.success else "subgraph_failed",
                subgraph.subgraph_id,
                attempt=subgraph_attempt,
                outcome=result.outcome,
            )
            return result

        while True:
            visits[current_id] = visits.get(current_id, 0) + 1
            loop = _loop_for(subgraph.loops, current_id)
            if loop is not None:
                loop_started.setdefault(current_id, time.monotonic())
                if visits[current_id] > loop.max_visits:
                    failure = self._failure(
                        FailureClass.EXECUTION_ERROR,
                        f"loop {current_id!r} exceeded max_visits={loop.max_visits}",
                        context,
                        node_id=current_id,
                        subgraph_id=subgraph.subgraph_id,
                    )
                    return finish(_SubgraphResult(
                    False, context, "failure", tuple(traces), failure,
                    _bind_subgraph_outputs(subgraph, node_outputs, external_inputs),
                ))
                if loop.max_duration_ms and (
                    (time.monotonic() - loop_started[current_id]) * 1000 > loop.max_duration_ms
                ):
                    failure = self._failure(
                        FailureClass.MOTION_TIMEOUT,
                        f"loop {current_id!r} exceeded max_duration_ms={loop.max_duration_ms}",
                        context,
                        node_id=current_id,
                        subgraph_id=subgraph.subgraph_id,
                    )
                    return finish(_SubgraphResult(False, context, "failure", tuple(traces), failure))
            elif visits[current_id] > 1:
                failure = self._failure(
                    FailureClass.EXECUTION_ERROR,
                    f"node {current_id!r} was re-entered without LoopSpec",
                    context,
                    node_id=current_id,
                    subgraph_id=subgraph.subgraph_id,
                )
                return finish(_SubgraphResult(
                    False, context, "failure", tuple(traces), failure,
                    _bind_subgraph_outputs(subgraph, node_outputs, external_inputs),
                ))

            node = subgraph.node(current_id)
            node_key = (subgraph.subgraph_id, current_id)
            node_event_visits[node_key] = node_event_visits.get(node_key, 0) + 1
            node_attempt = node_event_visits[node_key]
            self._emit_event(
                events,
                "node_started",
                subgraph.subgraph_id,
                node_id=current_id,
                node_type=node.node_type,
                attempt=node_attempt,
            )
            if pending_failure is not None and current_id in subgraph.failure_nodes:
                failure_class, metadata = pending_failure
                failure = self._failure(
                    failure_class,
                    f"failure checkpoint {current_id!r} reached after a failed "
                    "action transition",
                    context,
                    node_id=current_id,
                    subgraph_id=subgraph.subgraph_id,
                    metadata=metadata,
                )
                self._emit_event(
                    events,
                    "node_failed",
                    subgraph.subgraph_id,
                    node_id=current_id,
                    node_type=node.node_type,
                    attempt=node_attempt,
                    outcome=failure_class,
                )
                return finish(_SubgraphResult(
                    False,
                    context,
                    failure_class,
                    tuple(traces),
                    failure,
                    _bind_subgraph_outputs(subgraph, node_outputs, external_inputs),
                ))
            cycle: CycleResult | None = None
            if node.node_type == "action":
                resolved_inputs = _resolve_node_inputs(
                    subgraph,
                    node,
                    node_outputs,
                    external_inputs,
                )
                contract = subgraph.to_action_contract(current_id, context)
                if resolved_inputs:
                    contract = replace(
                        contract,
                        skills=_merge_first_skill_args(contract.skills, resolved_inputs),
                    )
                cycle = self.scheduler.dispatch(contract, context.scene)
                cycle = replace(
                    cycle,
                    trace=replace(
                        cycle.trace,
                        metadata={
                            **cycle.trace.metadata,
                            "subgraph_id": subgraph.subgraph_id,
                            "node_id": current_id,
                            "candidate_fingerprint": subgraph_fingerprint(subgraph),
                        },
                    ),
                )
                traces.append(cycle.trace)
                context = _advance_context(context, subgraph.subgoal_id, cycle)
                node_outputs[current_id] = _cycle_output(cycle)
                outcome = (
                    "success"
                    if cycle.committed
                    else cycle.verification.failure_class or "failure"
                )
            elif node.node_type == "checkpoint":
                outcome = (
                    self.checkpoint_evaluator(subgraph, node, context)
                    if self.checkpoint_evaluator is not None
                    else "success"
                )
                if outcome not in {"success", "failure"}:
                    raise GraphExecutionError(
                        f"checkpoint evaluator returned unsupported outcome {outcome!r}"
                    )
            elif node.node_type == "router" and self.control_evaluator is not None:
                outcome = self.control_evaluator(node, context)
            else:
                failure = self._failure(
                    FailureClass.EXECUTION_ERROR,
                    f"node {current_id!r} requires a control evaluator",
                    context,
                    node_id=current_id,
                    subgraph_id=subgraph.subgraph_id,
                )
                self._emit_event(
                    events,
                    "node_failed",
                    subgraph.subgraph_id,
                    node_id=current_id,
                    node_type=node.node_type,
                    attempt=node_attempt,
                    outcome="failure",
                )
                return finish(_SubgraphResult(
                    False, context, "failure", tuple(traces), failure,
                    _bind_subgraph_outputs(subgraph, node_outputs, external_inputs),
                ))

            self._emit_event(
                events,
                "node_completed" if outcome == "success" else "node_failed",
                subgraph.subgraph_id,
                node_id=current_id,
                node_type=node.node_type,
                attempt=node_attempt,
                outcome=outcome,
            )

            if outcome == "success" and current_id in subgraph.success_nodes:
                return finish(_SubgraphResult(
                    True,
                    context,
                    outcome,
                    tuple(traces),
                    outputs=_bind_subgraph_outputs(subgraph, node_outputs, external_inputs),
                ))
            if outcome != "success" and current_id in subgraph.failure_nodes:
                failure = self._failure(
                    _failure_class_for_outcome(outcome),
                    f"node {current_id!r} returned outcome {outcome!r}",
                    context,
                    node_id=current_id,
                    subgraph_id=subgraph.subgraph_id,
                    metadata=_cycle_failure_metadata(cycle),
                )
                return finish(_SubgraphResult(
                    False,
                    context,
                    outcome,
                    tuple(traces),
                    failure,
                    _bind_subgraph_outputs(subgraph, node_outputs, external_inputs),
                ))

            next_id = _select_transition(subgraph.edges, current_id, outcome)
            if next_id is None:
                failure = self._failure(
                    _failure_class_for_outcome(outcome),
                    f"no local transition from {current_id!r} for outcome {outcome!r}",
                    context,
                    node_id=current_id,
                    subgraph_id=subgraph.subgraph_id,
                    metadata=_cycle_failure_metadata(cycle),
                )
                return finish(_SubgraphResult(
                    False,
                    context,
                    outcome,
                    tuple(traces),
                    failure,
                    _bind_subgraph_outputs(subgraph, node_outputs, external_inputs),
                ))
            if outcome != "success" and next_id in subgraph.failure_nodes:
                pending_failure = (
                    _failure_class_for_outcome(outcome),
                    _cycle_failure_metadata(cycle),
                )
            else:
                pending_failure = None
            current_id = next_id

    @staticmethod
    def _failure(
        failure_class: str,
        message: str,
        context: AgentContext,
        *,
        node_id: str | None = None,
        subgraph_id: str | None = None,
        recovery_policy: str = "replan",
        evidence_refs: tuple[str, ...] = (),
        metadata: Mapping[str, object] | None = None,
        ) -> FailureArtifact:
        return FailureArtifact(
            failure_id=str(uuid4()),
            failure_class=failure_class,
            message=message,
            scene_version=context.scene.scene_version,
            node_id=node_id,
            subgraph_id=subgraph_id,
            recovery_policy=recovery_policy,
            evidence_refs=evidence_refs,
            metadata=dict(metadata or {}),
        )

    def _emit_event(
        self,
        events: list[GraphExecutionEvent],
        kind: GraphEventKind,
        subgraph_id: str,
        *,
        attempt: int,
        node_id: str | None = None,
        node_type: str | None = None,
        outcome: str | None = None,
    ) -> None:
        events.append(
            GraphExecutionEvent(
                sequence=len(events),
                kind=kind,
                subgraph_id=subgraph_id,
                node_id=node_id,
                node_type=node_type,  # type: ignore[arg-type]
                attempt=attempt,
                outcome=outcome,
                occurred_at_ns=self.clock(),
            )
        )

    def _publish_failure(self, failure: FailureArtifact) -> None:
        if self.artifact_store is None and self.event_bus is None:
            return
        envelope = ArtifactEnvelope(
            artifact_id=failure.failure_id,
            kind="failure",
            payload={"failure": failure},
            producer_agent=failure.source_agent,
            parent_scene_version=failure.scene_version,
        )
        if self.artifact_store is not None:
            self.artifact_store.put(envelope)
        if self.event_bus is not None:
            self.event_bus.publish(
                RuntimeEvent(
                    event_id=f"failure-event:{failure.failure_id}",
                    topic="failure",
                    scene_version=failure.scene_version,
                    artifact=envelope,
                )
            )

    def _select_recovery_target(
        self,
        graph: MissionGraph,
        source: str,
        failure: FailureArtifact,
        context: AgentContext,
    ) -> str | None:
        if self.recovery_selector is None:
            return None
        decision = self.recovery_selector.select(failure, context)
        if decision is None or decision.target_subgraph not in {
            subgraph.subgraph_id for subgraph in graph.subgraphs
        }:
            return None
        outgoing = tuple(edge for edge in graph.edges if edge.source == source)
        allowed = {
            failure.failure_class,
            "failure",
            failure.recovery_policy,
        }
        if not any(edge.target == decision.target_subgraph and edge.condition in allowed for edge in outgoing):
            return None
        return decision.target_subgraph


def _advance_context(context: AgentContext, subgoal_id: str, cycle: CycleResult) -> AgentContext:
    history = context.history
    return replace(
        context,
        scene=cycle.after_scene,
        history=CycleHistory(
            traces=history.traces + (cycle.trace,),
            last_verification=cycle.verification,
            current_subgoal=subgoal_id,
            recovery_count=history.recovery_count + (0 if cycle.committed else 1),
        ),
    )


def _loop_for(loops, entry: str):
    return next((loop for loop in loops if loop.entry_node == entry), None)


def _select_transition(edges, source: str, outcome: str) -> str | None:
    outgoing = tuple(edge for edge in edges if edge.source == source)
    if not outgoing:
        return None
    normalized_outcome = _normalize_outcome(outcome)
    matching = tuple(
        edge
        for edge in outgoing
        if _transition_matches(edge.condition, outcome, normalized_outcome)
    )
    if len(matching) == 1:
        return matching[0].target
    if len(matching) > 1:
        raise GraphExecutionError(
            f"multiple transitions from {source!r} match outcome {outcome!r}"
        )
    defaults = tuple(edge for edge in outgoing if edge.condition is None)
    if len(defaults) == 1:
        return defaults[0].target
    if len(defaults) > 1:
        raise GraphExecutionError(f"multiple default transitions from {source!r}")
    return None


def _normalize_transition_condition(condition: str | None) -> str | None:
    """Accept a narrow provider vocabulary while keeping runtime outcomes typed."""
    if condition is None:
        return None
    normalized = condition.strip().lower()
    if normalized in {
        "success",
        "completed",
        "complete",
        "ok",
        "passed",
        "action_complete",
        "action_completed",
        "skill_sequence_complete",
        "skill_sequence_completed",
    }:
        return "success"
    if normalized in {"failure", "failed", "error", "aborted", "timeout"}:
        return "failure"
    if any(
        marker in normalized
        for marker in ("fail", "error", "abort", "timeout", "recover", "unsafe", "not(")
    ):
        return "failure"
    return condition


def _transition_matches(
    condition: str | None,
    outcome: str,
    normalized_outcome: str,
) -> bool:
    normalized_condition = _normalize_transition_condition(condition)
    return normalized_condition == outcome or normalized_condition == normalized_outcome


def _normalize_outcome(outcome: str) -> str:
    if outcome == "success":
        return "success"
    if outcome in {
        FailureClass.STALE_STATE,
        FailureClass.PRECONDITION_FAILED,
        FailureClass.EXECUTION_ERROR,
        FailureClass.MOTION_TIMEOUT,
        FailureClass.POSTCONDITION_FAILED,
        FailureClass.PERCEPTION_UNCERTAIN,
        FailureClass.COLLISION_RISK,
        FailureClass.EPISODE_INVALIDATED,
    }:
        return "failure"
    return outcome


def _failure_class_for_outcome(outcome: str) -> str:
    known = {
        FailureClass.STALE_STATE,
        FailureClass.PRECONDITION_FAILED,
        FailureClass.EXECUTION_ERROR,
        FailureClass.MOTION_TIMEOUT,
        FailureClass.POSTCONDITION_FAILED,
        FailureClass.PERCEPTION_UNCERTAIN,
        FailureClass.COLLISION_RISK,
        FailureClass.EPISODE_INVALIDATED,
    }
    return outcome if outcome in known else FailureClass.POSTCONDITION_FAILED


def _cycle_failure_metadata(cycle: CycleResult | None) -> dict[str, object]:
    if cycle is None:
        return {}
    verification = cycle.verification
    return {
        "rejected": cycle.rejected,
        "reason": cycle.reason,
        "checked_scene_version": verification.checked_scene_version,
        "predicate_results": tuple(
            {
                "name": report.name,
                "passed": report.passed,
                "confidence": report.confidence,
                "evidence": tuple(report.evidence),
                "reason": report.reason,
            }
            for report in verification.predicate_results
        ),
    }


def _cycle_output(cycle: CycleResult) -> dict[str, object]:
    """Return the latest typed output produced by a dispatched action."""
    if not cycle.trace.skill_traces:
        return {}
    return dict(cycle.trace.skill_traces[-1].output)


def _resolve_node_inputs(
    subgraph: SubgraphSpec,
    node: SubgraphNodeSpec,
    node_outputs: dict[str, dict[str, object]],
    external_inputs: dict[str, object],
) -> dict[str, object]:
    """Resolve declared node ports into action arguments.

    A local ``PortBinding`` has precedence over an external input with the
    same name. This keeps bindings explicit while allowing a subgraph input
    to feed its first action without a synthetic identity node.
    """
    bindings = {
        binding.target_port: binding
        for binding in subgraph.bindings
        if binding.target_node == node.node_id
    }
    resolved: dict[str, object] = {}
    for port in node.inputs:
        binding = bindings.get(port.name)
        if binding is not None:
            source = node_outputs.get(binding.source_node, {})
            if binding.source_port not in source:
                raise GraphExecutionError(
                    f"missing output {binding.source_node}.{binding.source_port} "
                    f"for {node.node_id}.{port.name}"
                )
            resolved[port.name] = source[binding.source_port]
            continue
        if port.name in external_inputs:
            resolved[port.name] = external_inputs[port.name]
            continue
        if port.required:
            raise GraphExecutionError(
                f"required input {node.node_id}.{port.name} was not resolved"
            )
    return resolved


def _merge_first_skill_args(
    calls: tuple[SkillCall, ...],
    resolved_inputs: dict[str, object],
) -> tuple[SkillCall, ...]:
    if not calls:
        return calls
    first = calls[0]
    merged = dict(first.args)
    for key, value in resolved_inputs.items():
        if key in merged and merged[key] != value:
            raise GraphExecutionError(
                f"graph input {key!r} conflicts with the action's literal argument"
            )
        merged[key] = value
    return (SkillCall(first.skill, merged), *calls[1:])


def _bind_subgraph_outputs(
    subgraph: SubgraphSpec,
    node_outputs: dict[str, dict[str, object]],
    external_inputs: dict[str, object],
) -> dict[str, object]:
    bound: dict[str, object] = {}
    for binding in subgraph.output_bindings:
        source = node_outputs.get(binding.source_node, {})
        if binding.source_port not in source:
            raise GraphExecutionError(
                f"missing output {binding.source_node}.{binding.source_port} "
                f"for subgraph output {binding.output_port}"
            )
        bound[binding.output_port] = source[binding.source_port]
    return bound


def _mission_inputs(
    graph: MissionGraph,
    target_subgraph: str,
    mission_outputs: dict[str, dict[str, object]],
) -> dict[str, object]:
    bound: dict[str, object] = {}
    for binding in graph.bindings:
        if binding.target_subgraph != target_subgraph:
            continue
        source = mission_outputs.get(binding.source_subgraph, {})
        if binding.source_port not in source:
            raise GraphExecutionError(
                f"missing mission output {binding.source_subgraph}.{binding.source_port} "
                f"for {target_subgraph}.{binding.target_port}"
            )
        if binding.target_port in bound and bound[binding.target_port] != source[binding.source_port]:
            raise GraphExecutionError(
                f"multiple mission bindings conflict on {target_subgraph}.{binding.target_port}"
            )
        bound[binding.target_port] = source[binding.source_port]
    return bound
