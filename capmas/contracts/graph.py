"""Typed graph contracts used by the CAP-MAS agent and runtime planes.

The graph is deliberately a planning artifact, not a second actuator API. A
validated node can be lowered to the existing ``ActionContract`` seam, while
the graph itself remains useful for static validation, candidate arbitration,
rehearsal, and later distributed execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import uuid4

from capmas.contracts.action import ActionContract, SkillCall

if TYPE_CHECKING:
    from capmas.contracts.agent import AgentContext


@dataclass(frozen=True)
class PortSpec:
    """A named, typed input or output of a graph node or subgraph."""

    name: str
    type_name: str
    required: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("port name must not be empty")
        if not self.type_name:
            raise ValueError("port type must not be empty")


@dataclass(frozen=True)
class ResourceRequirement:
    """A resource used by a graph node.

    ``exclusive`` resources cannot be used by parallel branches. This is what
    prevents two candidate execution branches from claiming one robot arm.
    """

    resource_id: str
    mode: str = "exclusive"

    def __post_init__(self) -> None:
        if not self.resource_id:
            raise ValueError("resource id must not be empty")
        if self.mode not in {"exclusive", "shared"}:
            raise ValueError("resource mode must be 'exclusive' or 'shared'")


@dataclass(frozen=True)
class CheckpointSpec:
    """A node/subgraph postcondition checkpoint.

    A validating checkpoint is required for every executable subgraph. Probe
    checkpoints may add diagnostics without blocking execution in later
    versions, mirroring GaP's validation-vs-probe distinction.
    """

    name: str
    predicates: tuple[str, ...]
    validate: bool = True
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("checkpoint name must not be empty")
        if not self.predicates:
            raise ValueError("checkpoint must contain at least one predicate")
        if self.weight <= 0:
            raise ValueError("checkpoint weight must be positive")


@dataclass(frozen=True)
class GraphEdge:
    """A local subgraph control/data-flow edge."""

    source: str
    target: str
    condition: str | None = None


@dataclass(frozen=True)
class LoopSpec:
    """An explicit, bounded control-flow loop.

    Cycles are never accepted merely because an LLM emitted a back-edge.  A
    loop must name its re-entry node and carry a finite visit budget.  The
    interpreter uses ``exit_conditions`` as documentation and as a guard for
    router/action outcomes; ``max_duration_ms`` is an optional wall-clock
    budget (zero means no separate duration budget).
    """

    entry_node: str
    max_visits: int = 1
    max_duration_ms: int = 0
    exit_conditions: tuple[str, ...] = ("success", "failure")

    def __post_init__(self) -> None:
        if not self.entry_node:
            raise ValueError("loop entry node must not be empty")
        if self.max_visits <= 0:
            raise ValueError("loop max_visits must be positive")
        if self.max_duration_ms < 0:
            raise ValueError("loop max_duration_ms must not be negative")
        if not self.exit_conditions:
            raise ValueError("loop must declare at least one exit condition")
        if any(not condition for condition in self.exit_conditions):
            raise ValueError("loop exit conditions must not be empty")


@dataclass(frozen=True)
class PortBinding:
    """Connect an output port to a local node input port."""

    source_node: str
    source_port: str
    target_node: str
    target_port: str


@dataclass(frozen=True)
class SubgraphOutputBinding:
    """Expose a local node output as a subgraph output."""

    source_node: str
    source_port: str
    output_port: str


@dataclass(frozen=True)
class SubgraphNodeSpec:
    """An executable or control node inside a local policy subgraph."""

    node_id: str
    description: str
    skill_calls: tuple[SkillCall, ...] = ()
    inputs: tuple[PortSpec, ...] = ()
    outputs: tuple[PortSpec, ...] = ()
    preconditions: tuple[str, ...] = ()
    postconditions: tuple[str, ...] = ()
    resources: tuple[ResourceRequirement, ...] = ()
    max_duration_ms: int = 60_000
    max_sim_steps: int = 500
    proposed_by: str = "policy_agent"
    recovery_policy: str = "replan"
    node_type: str = "action"

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("node id must not be empty")
        if self.node_type not in {"action", "router", "checkpoint"}:
            raise ValueError("unsupported graph node type")
        if self.max_duration_ms <= 0 or self.max_sim_steps <= 0:
            raise ValueError("node execution budgets must be positive")

    @property
    def exclusive_resources(self) -> frozenset[str]:
        return frozenset(
            resource.resource_id
            for resource in self.resources
            if resource.mode == "exclusive"
        )


@dataclass(frozen=True)
class SubgraphSpec:
    """A bounded local policy graph assigned to one Policy Agent."""

    subgraph_id: str
    subgoal_id: str
    description: str
    nodes: tuple[SubgraphNodeSpec, ...]
    edges: tuple[GraphEdge, ...]
    entry_node: str
    success_nodes: tuple[str, ...]
    failure_nodes: tuple[str, ...]
    inputs: tuple[PortSpec, ...] = ()
    outputs: tuple[PortSpec, ...] = ()
    bindings: tuple[PortBinding, ...] = ()
    output_bindings: tuple[SubgraphOutputBinding, ...] = ()
    checkpoints: tuple[CheckpointSpec, ...] = ()
    assigned_agent: str = "policy_agent"
    loops: tuple[LoopSpec, ...] = ()

    def node(self, node_id: str) -> SubgraphNodeSpec:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(f"unknown node: {node_id}")

    def to_action_contract(
        self,
        node_id: str,
        context: "AgentContext",
        *,
        contract_id: str | None = None,
    ) -> ActionContract:
        """Lower one validated action node to the existing runtime contract.

        The graph interpreter is intentionally not implemented here. This
        compatibility lowering lets P2.5 execute graph-authored nodes while
        the Phase 3 scheduler is developed independently.
        """

        node = self.node(node_id)
        if node.node_type != "action":
            raise ValueError(f"node is not executable: {node_id}")
        if not node.skill_calls:
            raise ValueError(f"action node has no skill calls: {node_id}")
        return ActionContract(
            contract_id=contract_id or str(uuid4()),
            episode_id=context.episode_id,
            episode_epoch=context.episode_epoch,
            parent_scene_version=context.scene.scene_version,
            subgoal_id=self.subgoal_id,
            skills=node.skill_calls,
            expected_postconditions=node.postconditions,
            preconditions=node.preconditions,
            max_duration_ms=node.max_duration_ms,
            max_sim_steps=node.max_sim_steps,
            proposed_by=node.proposed_by or self.assigned_agent,
            recovery_policy=node.recovery_policy,
        )


@dataclass(frozen=True)
class MissionEdge:
    """A dependency or control-flow edge between subgraphs."""

    source: str
    target: str
    condition: str | None = None


@dataclass(frozen=True)
class MissionBinding:
    """Connect an output of one subgraph to another subgraph's input."""

    source_subgraph: str
    source_port: str
    target_subgraph: str
    target_port: str


@dataclass(frozen=True)
class MissionGraph:
    """The shared typed workspace for Manager and local Policy Agents."""

    mission_id: str
    task: str
    subgraphs: tuple[SubgraphSpec, ...]
    edges: tuple[MissionEdge, ...]
    bindings: tuple[MissionBinding, ...]
    entry_subgraph: str
    success_subgraphs: tuple[str, ...]
    failure_subgraphs: tuple[str, ...]
    parent_scene_version: int | None = None
    graph_version: int = 1
    loops: tuple[LoopSpec, ...] = ()

    def subgraph(self, subgraph_id: str) -> SubgraphSpec:
        for subgraph in self.subgraphs:
            if subgraph.subgraph_id == subgraph_id:
                return subgraph
        raise KeyError(f"unknown subgraph: {subgraph_id}")

    def to_dict(self) -> dict[str, object]:
        """Serialize through the strict versioned graph codec."""
        from capmas.graph.serialization import mission_graph_to_dict

        return mission_graph_to_dict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "MissionGraph":
        """Load a graph artifact after strict schema validation."""
        from capmas.graph.serialization import mission_graph_from_dict

        return mission_graph_from_dict(raw)
