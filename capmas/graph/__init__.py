"""Typed mission-graph contracts and static validation."""

from capmas.graph.validator import GraphDiagnostic, GraphValidationResult, GraphValidator
from capmas.graph.serialization import (
    GRAPH_SCHEMA_VERSION,
    GraphSchemaError,
    local_subgraph_from_dict,
    local_subgraph_to_dict,
    mission_graph_from_dict,
    mission_graph_to_dict,
)
from capmas.graph.staged import (
    TopologyDiagnostic,
    TopologySchemaError,
    TopologyValidationResult,
    TopologyValidator,
    topology_from_dict,
    topology_to_dict,
)

__all__ = [
    "GRAPH_SCHEMA_VERSION",
    "GraphDiagnostic",
    "GraphSchemaError",
    "GraphValidationResult",
    "GraphValidator",
    "TopologyDiagnostic",
    "TopologySchemaError",
    "TopologyValidationResult",
    "TopologyValidator",
    "local_subgraph_from_dict",
    "local_subgraph_to_dict",
    "mission_graph_from_dict",
    "mission_graph_to_dict",
    "topology_from_dict",
    "topology_to_dict",
]
