"""Strict decoders for the staged topology and local graph artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Callable

from capmas.contracts.scene import SceneSnapshot
from capmas.contracts.graph import SubgraphSpec
from capmas.contracts.staged import MissionTopology
from capmas.graph.serialization import (
    GraphSchemaError,
    local_subgraph_from_dict,
)
from capmas.graph.staged import TopologySchemaError, topology_from_dict
from capmas.graph.validator import GraphValidator
from capmas.llm.protocol import LLMRequest, LLMResponse


@dataclass(frozen=True)
class StagedDecodeRejection:
    code: str
    message: str


@dataclass(frozen=True)
class TopologyDecodeResult:
    topology: MissionTopology | None = None
    rejections: tuple[StagedDecodeRejection, ...] = ()

    @property
    def accepted(self) -> bool:
        return self.topology is not None and not self.rejections


@dataclass(frozen=True)
class SubgraphDecodeResult:
    subgraph: SubgraphSpec | None = None
    rejections: tuple[StagedDecodeRejection, ...] = ()

    @property
    def accepted(self) -> bool:
        return self.subgraph is not None and not self.rejections


class StagedProposalError(ValueError):
    """Raised when a staged artifact cannot cross its typed boundary."""

    def __init__(self, result: TopologyDecodeResult | SubgraphDecodeResult) -> None:
        self.result = result
        details = "; ".join(f"{item.code}: {item.message}" for item in result.rejections)
        super().__init__(f"staged proposal rejected: {details}")


class MissionTopologyDecoder:
    """Decode compact Manager output and bind it to the current scene."""

    def decode(
        self,
        response: LLMResponse,
        scene: SceneSnapshot,
        *,
        request: LLMRequest | None = None,
        expected_mission_id: str | None = None,
    ) -> TopologyDecodeResult:
        rejections = _request_rejections(response, request)
        raw, payload_rejection = _payload(response)
        if payload_rejection is not None:
            rejections.append(payload_rejection)
        if rejections:
            return TopologyDecodeResult(rejections=tuple(rejections))
        assert raw is not None
        try:
            topology = topology_from_dict(raw)
        except (TopologySchemaError, TypeError, ValueError) as exc:
            return TopologyDecodeResult(
                rejections=(StagedDecodeRejection("TOPOLOGY_SCHEMA_INVALID", str(exc)),)
            )
        if expected_mission_id is not None and topology.mission_id != expected_mission_id:
            rejections.append(
                StagedDecodeRejection(
                    "MISSION_ID_MISMATCH",
                    f"topology mission id {topology.mission_id!r} does not match {expected_mission_id!r}",
                )
            )
        _scene_rejection(rejections, topology.parent_scene_version, scene)
        return TopologyDecodeResult(topology=topology) if not rejections else TopologyDecodeResult(rejections=tuple(rejections))


class LocalSubgraphDecoder:
    """Decode one direct local Policy graph and validate it in isolation."""

    def __init__(
        self,
        validator: GraphValidator | None = None,
        condition_enricher: Callable[[SubgraphSpec, SceneSnapshot], SubgraphSpec]
        | None = None,
    ) -> None:
        self.validator = validator or GraphValidator()
        self.condition_enricher = condition_enricher

    def decode(
        self,
        response: LLMResponse,
        scene: SceneSnapshot,
        *,
        request: LLMRequest | None = None,
        expected_subgraph_id: str | None = None,
        expected_subgoal_id: str | None = None,
    ) -> SubgraphDecodeResult:
        rejections = _request_rejections(response, request)
        raw, payload_rejection = _payload(response)
        if payload_rejection is not None:
            rejections.append(payload_rejection)
        if rejections:
            return SubgraphDecodeResult(rejections=tuple(rejections))
        assert raw is not None
        try:
            subgraph = local_subgraph_from_dict(raw)
        except (GraphSchemaError, TypeError, ValueError) as exc:
            return SubgraphDecodeResult(
                rejections=(StagedDecodeRejection("SUBGRAPH_SCHEMA_INVALID", str(exc)),)
            )
        if self.condition_enricher is not None:
            try:
                subgraph = self.condition_enricher(subgraph, scene)
            except Exception as exc:
                return SubgraphDecodeResult(
                    rejections=(
                        StagedDecodeRejection(
                            "SUBGRAPH_CONDITION_ENRICHMENT_FAILED",
                            str(exc),
                        ),
                    )
                )
        subgraph = _normalize_terminal_edges(subgraph)
        if expected_subgraph_id is not None and subgraph.subgraph_id != expected_subgraph_id:
            rejections.append(
                StagedDecodeRejection(
                    "SUBGRAPH_ID_MISMATCH",
                    f"subgraph id {subgraph.subgraph_id!r} does not match {expected_subgraph_id!r}",
                )
            )
        if expected_subgoal_id is not None and subgraph.subgoal_id != expected_subgoal_id:
            rejections.append(
                StagedDecodeRejection(
                    "SUBGOAL_ID_MISMATCH",
                    f"subgoal id {subgraph.subgoal_id!r} does not match {expected_subgoal_id!r}",
                )
            )
        validation = self.validator.validate_subgraph(subgraph)
        rejections.extend(
            StagedDecodeRejection(item.code, item.message) for item in validation.errors
        )
        return SubgraphDecodeResult(subgraph=subgraph) if not rejections else SubgraphDecodeResult(rejections=tuple(rejections))


def _request_rejections(
    response: LLMResponse,
    request: LLMRequest | None,
) -> list[StagedDecodeRejection]:
    if request is None or response.request_id == request.request_id:
        return []
    return [
        StagedDecodeRejection(
            "REQUEST_ID_MISMATCH",
            f"response belongs to {response.request_id!r}, expected {request.request_id!r}",
        )
    ]


def _scene_rejection(
    rejections: list[StagedDecodeRejection],
    parent_scene_version: int | None,
    scene: SceneSnapshot,
) -> None:
    if parent_scene_version is None:
        rejections.append(
            StagedDecodeRejection("MISSING_PARENT_SCENE", "topology must declare parent_scene_version")
        )
    elif parent_scene_version != scene.scene_version:
        rejections.append(
            StagedDecodeRejection(
                "STALE_SCENE",
                f"topology targets scene {parent_scene_version!r}, current scene is {scene.scene_version}",
            )
        )


def _normalize_terminal_edges(subgraph: SubgraphSpec) -> SubgraphSpec:
    """Lower provider predicate-labelled terminal edges to typed outcomes.

    Policy models sometimes describe the condition that was verified rather
    than the runtime outcome (for example, ``object_in_gripper(...) &&
    gripper_closed()``). Terminal node membership is the typed source of truth
    for this bounded local graph, so unknown conditions targeting a success or
    failure terminal are lowered accordingly. Explicit success/failure labels
    remain unchanged.
    """
    success_nodes = set(subgraph.success_nodes)
    failure_nodes = set(subgraph.failure_nodes)
    normalized = []
    changed = False
    for edge in subgraph.edges:
        condition = edge.condition
        known_outcome = _terminal_outcome(condition)
        if known_outcome is None:
            if edge.target in success_nodes and edge.target not in failure_nodes:
                condition = "success"
            elif edge.target in failure_nodes and edge.target not in success_nodes:
                condition = "failure"
        if condition != edge.condition:
            changed = True
        normalized.append(replace(edge, condition=condition))
    return replace(subgraph, edges=tuple(normalized)) if changed else subgraph


def _terminal_outcome(condition: str | None) -> str | None:
    if condition is None:
        return None
    value = condition.strip().lower()
    if value in {
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
    if value in {"failure", "failed", "error", "aborted", "timeout"}:
        return "failure"
    if any(
        marker in value
        for marker in ("fail", "error", "abort", "timeout", "recover", "unsafe", "not(")
    ):
        return "failure"
    return None


def _payload(
    response: LLMResponse,
) -> tuple[Mapping[str, object] | None, StagedDecodeRejection | None]:
    if response.structured is not None:
        if not isinstance(response.structured, Mapping):
            return None, StagedDecodeRejection("STRUCTURED_PAYLOAD_INVALID", "structured payload must be an object")
        return response.structured, None
    if not response.content.strip():
        return None, StagedDecodeRejection("EMPTY_RESPONSE", "LLM response contains no JSON artifact")
    try:
        payload = json.loads(response.content)
    except json.JSONDecodeError as exc:
        return None, StagedDecodeRejection("JSON_INVALID", f"invalid JSON response: {exc.msg}")
    if not isinstance(payload, Mapping):
        return None, StagedDecodeRejection("JSON_NOT_OBJECT", "LLM response must be a JSON object")
    return payload, None


__all__ = [
    "LocalSubgraphDecoder",
    "MissionTopologyDecoder",
    "StagedDecodeRejection",
    "StagedProposalError",
    "SubgraphDecodeResult",
    "TopologyDecodeResult",
]
