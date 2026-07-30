"""Stable graph/subgraph identity for rehearsal evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json

from capmas.contracts.candidates import subgraph_fingerprint
from capmas.graph.serialization import GraphSchemaError, mission_graph_from_dict


@dataclass(frozen=True)
class CandidateIdentity:
    """Source graph provenance plus the exact local Arbiter identity."""

    graph_fingerprint: str
    subgraph_id: str
    subgraph_fingerprint: str
    scene_version: int

    def __post_init__(self) -> None:
        if not self.graph_fingerprint:
            raise ValueError("graph fingerprint must not be empty")
        if not self.subgraph_id or not self.subgraph_fingerprint:
            raise ValueError("subgraph identity must not be empty")
        if self.scene_version < 0:
            raise ValueError("identity scene version must not be negative")


def raw_graph_fingerprint(raw_graph: Mapping[str, object]) -> str:
    """Hash the source JSON representation without typed normalization."""

    if not isinstance(raw_graph, Mapping):
        raise ValueError("raw graph must be an object")
    try:
        encoded = json.dumps(
            raw_graph,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("raw graph must be JSON serializable") from exc
    return hashlib.sha256(encoded).hexdigest()


def candidate_identity_from_raw_graph(
    raw_graph: Mapping[str, object],
    subgraph_id: str,
    scene_version: int,
) -> CandidateIdentity:
    """Derive dual identity while preserving the raw source graph hash."""

    if not subgraph_id:
        raise ValueError("target subgraph id must not be empty")
    if scene_version < 0:
        raise ValueError("identity scene version must not be negative")
    graph_hash = raw_graph_fingerprint(raw_graph)
    try:
        graph = mission_graph_from_dict(raw_graph)
        subgraph = graph.subgraph(subgraph_id)
    except KeyError as exc:
        raise ValueError(f"unknown target subgraph: {subgraph_id}") from exc
    except GraphSchemaError:
        raise
    return CandidateIdentity(
        graph_fingerprint=graph_hash,
        subgraph_id=subgraph_id,
        subgraph_fingerprint=subgraph_fingerprint(subgraph),
        scene_version=scene_version,
    )


__all__ = [
    "CandidateIdentity",
    "candidate_identity_from_raw_graph",
    "raw_graph_fingerprint",
]
