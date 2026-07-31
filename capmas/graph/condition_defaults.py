"""Deterministic predicate defaults for typed Policy action nodes."""

from __future__ import annotations

from dataclasses import replace

from capmas.contracts.graph import SubgraphNodeSpec, SubgraphSpec
from capmas.contracts.scene import SceneSnapshot
from capmas.contracts.strategy import StrategyProfile
from capmas.skills.registry import SkillRegistry


_DEFAULT_FRESHNESS = "scene_fresh(2000)"


class SkillConditionEnricher:
    """Add safe typed-skill predicates without inventing task semantics.

    Enrichment is intentionally idempotent. The LLM remains responsible for
    task predicates such as ``object_in_gripper`` and ``object_at_target``;
    this class only supplies predicates that follow from a registered skill or
    an explicitly grounded scene identity.
    """

    def __init__(self, registry: SkillRegistry) -> None:
        self.registry = registry

    def enrich(
        self,
        subgraph: SubgraphSpec,
        scene: SceneSnapshot,
        strategy: str,
    ) -> SubgraphSpec:
        profile = StrategyProfile.for_name(strategy)
        return replace(
            subgraph,
            nodes=tuple(
                self._enrich_node(node, scene, profile) for node in subgraph.nodes
            ),
        )

    def _enrich_node(
        self,
        node: SubgraphNodeSpec,
        scene: SceneSnapshot,
        profile: StrategyProfile,
    ) -> SubgraphNodeSpec:
        if node.node_type != "action" or not node.skill_calls:
            return node

        preconditions = list(node.preconditions)
        postconditions = list(node.postconditions)
        if not _has_family(preconditions, "scene_fresh"):
            preconditions.append(_DEFAULT_FRESHNESS)

        if profile.require_grounding_preconditions:
            for track_id in self._grounded_track_ids(node, scene):
                _append_unique(preconditions, f"track_exists:{track_id}")

        for call in node.skill_calls:
            for predicate in self.registry.default_postconditions(call.skill):
                if predicate.startswith("scene_fresh("):
                    if not _has_family(postconditions, "scene_fresh"):
                        postconditions.append(predicate)
                else:
                    _append_unique(postconditions, predicate)

        return replace(
            node,
            preconditions=tuple(preconditions),
            postconditions=tuple(postconditions),
        )

    @staticmethod
    def _grounded_track_ids(
        node: SubgraphNodeSpec,
        scene: SceneSnapshot,
    ) -> tuple[str, ...]:
        requested: list[str] = []
        if node.motion_intent is not None:
            for value in (
                node.motion_intent.object_track_id,
                node.motion_intent.target_track_id,
            ):
                if value:
                    requested.append(value)
        for call in node.skill_calls:
            value = call.args.get("object_name")
            if isinstance(value, str) and value:
                requested.append(value)

        resolved: list[str] = []
        for value in requested:
            matches = tuple(
                track.track_id
                for track in scene.objects
                if _same_identifier(value, track.track_id)
                or _same_identifier(value, track.label)
            )
            unique_matches = tuple(dict.fromkeys(matches))
            if len(unique_matches) == 1:
                _append_unique(resolved, unique_matches[0])
        return tuple(resolved)


def _has_family(predicates: list[str], family: str) -> bool:
    return any(
        predicate == family or predicate.startswith(f"{family}(")
        for predicate in predicates
    )


def _append_unique(predicates: list[str], predicate: str) -> None:
    if predicate not in predicates:
        predicates.append(predicate)


def _same_identifier(left: str, right: str) -> bool:
    return " ".join(left.replace("_", " ").lower().split()) == " ".join(
        right.replace("_", " ").lower().split()
    )


__all__ = ["SkillConditionEnricher"]
