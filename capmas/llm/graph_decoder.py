"""Strict LLM response decoding for typed mission graphs.

This module is deliberately independent from any model provider.  It turns a
provider response into a validated ``MissionGraph`` or an explicit rejection;
there is no empty-plan or default-action fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from collections.abc import Mapping

from capmas.contracts.graph import MissionGraph
from capmas.contracts.scene import SceneSnapshot
from capmas.graph.serialization import GraphSchemaError, mission_graph_from_dict
from capmas.graph.validator import GraphDiagnostic, GraphValidator, scene_initial_facts
from capmas.llm.protocol import LLMRequest, LLMResponse


@dataclass(frozen=True)
class GraphDecodeRejection:
    code: str
    message: str


@dataclass(frozen=True)
class GraphDecodeResult:
    graph: MissionGraph | None = None
    rejections: tuple[GraphDecodeRejection, ...] = ()

    @property
    def accepted(self) -> bool:
        return self.graph is not None and not self.rejections


class GraphProposalError(ValueError):
    """Raised when an LLM graph proposal cannot cross the typed boundary."""

    def __init__(self, result: GraphDecodeResult) -> None:
        self.result = result
        details = "; ".join(f"{item.code}: {item.message}" for item in result.rejections)
        super().__init__(f"LLM graph proposal rejected: {details}")


class MissionGraphDecoder:
    """Decode and validate one model response before it enters the runtime."""

    def __init__(
        self,
        validator: GraphValidator | None = None,
        *,
        require_current_scene: bool = True,
    ) -> None:
        self.validator = validator or GraphValidator()
        self.require_current_scene = require_current_scene

    def decode(
        self,
        response: LLMResponse,
        scene: SceneSnapshot,
        *,
        request: LLMRequest | None = None,
        expected_mission_id: str | None = None,
    ) -> GraphDecodeResult:
        rejections: list[GraphDecodeRejection] = []
        if request is not None and response.request_id != request.request_id:
            rejections.append(
                GraphDecodeRejection(
                    "REQUEST_ID_MISMATCH",
                    f"response belongs to {response.request_id!r}, "
                    f"expected {request.request_id!r}",
                )
            )
        raw, parse_rejection = self._payload(response)
        if parse_rejection is not None:
            rejections.append(parse_rejection)
        if rejections:
            return GraphDecodeResult(rejections=tuple(rejections))
        assert raw is not None

        try:
            graph = mission_graph_from_dict(raw)
        except (GraphSchemaError, TypeError, ValueError) as exc:
            return GraphDecodeResult(
                rejections=(GraphDecodeRejection("GRAPH_SCHEMA_INVALID", str(exc)),)
            )

        if expected_mission_id is not None and graph.mission_id != expected_mission_id:
            rejections.append(
                GraphDecodeRejection(
                    "MISSION_ID_MISMATCH",
                    f"graph mission id {graph.mission_id!r} does not match "
                    f"{expected_mission_id!r}",
                )
            )
        if self.require_current_scene:
            if graph.parent_scene_version is None:
                rejections.append(
                    GraphDecodeRejection(
                        "MISSING_PARENT_SCENE",
                        "LLM graph must declare parent_scene_version",
                    )
                )
            elif graph.parent_scene_version != scene.scene_version:
                rejections.append(
                    GraphDecodeRejection(
                        "STALE_SCENE",
                        f"graph targets scene {graph.parent_scene_version}, "
                        f"current scene is {scene.scene_version}",
                    )
                )

        validation = self.validator.validate(
            graph,
            initial_facts=scene_initial_facts(graph, scene),
        )
        rejections.extend(_diagnostic_rejections(validation.diagnostics))
        if rejections:
            return GraphDecodeResult(rejections=tuple(rejections))
        return GraphDecodeResult(graph=graph)

    @staticmethod
    def _payload(
        response: LLMResponse,
    ) -> tuple[Mapping[str, object] | None, GraphDecodeRejection | None]:
        if response.structured is not None:
            if not isinstance(response.structured, Mapping):
                return None, GraphDecodeRejection(
                    "STRUCTURED_PAYLOAD_INVALID",
                    "response.structured must be a JSON object",
                )
            return response.structured, None
        if not response.content.strip():
            return None, GraphDecodeRejection(
                "EMPTY_RESPONSE",
                "LLM response contains neither structured data nor JSON content",
            )
        try:
            payload = json.loads(response.content)
        except json.JSONDecodeError as exc:
            return None, GraphDecodeRejection("JSON_INVALID", f"invalid JSON response: {exc.msg}")
        if not isinstance(payload, Mapping):
            return None, GraphDecodeRejection(
                "JSON_NOT_OBJECT",
                "LLM graph response must be a JSON object",
            )
        return payload, None


def _diagnostic_rejections(
    diagnostics: tuple[GraphDiagnostic, ...],
) -> list[GraphDecodeRejection]:
    return [
        GraphDecodeRejection(diagnostic.code, diagnostic.message)
        for diagnostic in diagnostics
        if diagnostic.severity == "error"
    ]


__all__ = [
    "GraphDecodeRejection",
    "GraphDecodeResult",
    "GraphProposalError",
    "MissionGraphDecoder",
]
