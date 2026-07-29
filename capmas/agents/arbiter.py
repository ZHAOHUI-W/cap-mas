from __future__ import annotations

from collections.abc import Iterable

from capmas.contracts.candidates import (
    ArbitrationResult,
    CandidateRejection,
    GraphCandidate,
    subgraph_fingerprint,
)
from capmas.contracts.scene import SceneSnapshot
from capmas.contracts.strategy import StrategyProfile
from capmas.graph.validator import GraphValidator


class CandidateArbiter:
    """Deterministically select one valid local graph candidate.

    The arbiter is deliberately not an executor.  It validates all proposals,
    rejects stale or malformed artifacts, and selects one candidate with a
    stable ordering.  Physical execution remains behind the runtime scheduler
    and ``ActionLease``.
    """

    def __init__(
        self,
        validator: GraphValidator | None = None,
        *,
        latency_budget_ms: float = 5_000,
        current_map_version: int | None = None,
    ) -> None:
        if latency_budget_ms <= 0:
            raise ValueError("latency budget must be positive")
        self.validator = validator or GraphValidator()
        self.latency_budget_ms = float(latency_budget_ms)
        if current_map_version is not None and current_map_version < 0:
            raise ValueError("current map version must not be negative")
        self.current_map_version = current_map_version

    def select(
        self,
        candidates: Iterable[GraphCandidate],
        scene: SceneSnapshot,
        *,
        expected_subgoal: str | None = None,
        current_map_version: int | None = None,
    ) -> ArbitrationResult:
        proposals = tuple(candidates)
        rejections: list[CandidateRejection] = []
        valid: list[GraphCandidate] = []
        seen_ids: set[str] = set()

        score_breakdowns: dict[str, dict[str, float]] = {}
        for candidate in proposals:
            if candidate.candidate_id in seen_ids:
                rejections.append(
                    CandidateRejection(
                        candidate.candidate_id,
                        "DUPLICATE_CANDIDATE",
                        "candidate id was already submitted",
                    )
                )
                continue
            seen_ids.add(candidate.candidate_id)

            if candidate.parent_scene_version != scene.scene_version:
                rejections.append(
                    CandidateRejection(
                        candidate.candidate_id,
                        "STALE_SCENE",
                        f"candidate targets scene {candidate.parent_scene_version}, "
                        f"current scene is {scene.scene_version}",
                    )
                )
                continue

            validation = self.validator.validate_subgraph(candidate.subgraph)
            if not validation.valid:
                first = validation.errors[0]
                rejections.append(
                    CandidateRejection(
                        candidate.candidate_id,
                        first.code,
                        "; ".join(error.message for error in validation.errors),
                    )
                )
                continue
            subgoal_id = candidate.subgraph.subgoal_id
            if expected_subgoal is None:
                expected_subgoal = subgoal_id
            elif subgoal_id != expected_subgoal:
                rejections.append(
                    CandidateRejection(
                        candidate.candidate_id,
                        "SUBGOAL_MISMATCH",
                        f"candidate targets {subgoal_id!r}, expected {expected_subgoal!r}",
                    )
                )
                continue
            evidence_rejection = self._evidence_gate(
                candidate,
                scene,
                self.current_map_version if current_map_version is None else current_map_version,
            )
            if evidence_rejection is not None:
                rejections.append(
                    CandidateRejection(
                        candidate.candidate_id,
                        "STALE_EVIDENCE",
                        evidence_rejection,
                    )
                )
                continue
            geometry_rejection = self._geometry_gate(candidate)
            if geometry_rejection is not None:
                rejections.append(
                    CandidateRejection(
                        candidate.candidate_id,
                        "GEOMETRY_GATE",
                        geometry_rejection,
                    )
                )
                continue
            perception_rejection = self._perception_gate(candidate)
            if perception_rejection is not None:
                rejections.append(
                    CandidateRejection(
                        candidate.candidate_id,
                        "PERCEPTION_GATE",
                        perception_rejection,
                    )
                )
                continue
            valid.append(candidate)

        evidence_mode = any(candidate.evidence is not None for candidate in valid)
        if evidence_mode:
            evidence_valid: list[GraphCandidate] = []
            for candidate in valid:
                if candidate.evidence is None:
                    rejections.append(
                        CandidateRejection(
                            candidate.candidate_id,
                            "MISSING_EVIDENCE",
                            "candidate evidence is required when evidence mode is active",
                        )
                    )
                    continue
                evidence_valid.append(candidate)
            valid = evidence_valid

        selected = max(valid, key=self._ordering_key) if valid else None
        if not valid:
            selection_basis = (
                "hard_gate"
                if any(
                    rejection.code in {"PERCEPTION_GATE", "STALE_EVIDENCE", "MISSING_EVIDENCE"}
                    for rejection in rejections
                )
                else "none"
            )
            tie_broken = False
        else:
            scores = []
            for candidate in valid:
                score, breakdown = self._score_breakdown(candidate)
                scores.append(score)
                score_breakdowns[candidate.candidate_id] = breakdown
            top_score = max(scores)
            tie_broken = sum(score == top_score for score in scores) > 1
            selection_basis = (
                "evidence_tie_break"
                if tie_broken and evidence_mode
                else "evidence_score"
                if evidence_mode
                else "confidence_fallback"
            )
        return ArbitrationResult(
            selected,
            proposals,
            tuple(rejections),
            selection_basis=selection_basis,
            tie_broken=tie_broken,
            score_breakdowns=score_breakdowns,
        )

    def score(self, candidate: GraphCandidate) -> float:
        return self._score_breakdown(candidate)[0]

    def _score_breakdown(self, candidate: GraphCandidate) -> tuple[float, dict[str, float]]:
        evidence = candidate.evidence
        if evidence is None:
            confidence = candidate.confidence if candidate.confidence is not None else 0.0
            return confidence, {"confidence_fallback": confidence}
        profile = StrategyProfile.for_name(candidate.strategy)
        # Evidence mode intentionally excludes the legacy self-reported/default
        # confidence. Providers declare available dimensions so missing
        # rehearsal/OOD data is not silently treated as a zero-quality result.
        if not evidence.available_metrics:
            latency = min(evidence.expected_latency_ms / self.latency_budget_ms, 2.0)
            breakdown = {
                "verifier": 0.20 * evidence.verifier_pass_rate,
                "rehearsal": 0.25 * evidence.rehearsal_success_rate,
                "ood": 0.25 * evidence.ood_success_rate,
                "latency": -0.03 * latency,
                "recovery": -0.02 * min(evidence.recovery_cost, 2.0),
            }
            return sum(breakdown.values()), breakdown

        available = set(evidence.available_metrics)
        breakdown: dict[str, float] = {}
        if "verifier" in available:
            breakdown["verifier"] = profile.verifier_weight * evidence.verifier_pass_rate
        if "rehearsal" in available:
            breakdown["rehearsal"] = profile.rehearsal_weight * evidence.rehearsal_success_rate
        if "ood" in available:
            breakdown["ood"] = profile.ood_weight * evidence.ood_success_rate
        if "perception" in available and evidence.perception is not None:
            breakdown["perception"] = profile.perception_weight * evidence.perception.score()
        if "geometry" in available and evidence.geometry is not None:
            geometry_score = _geometry_score(evidence.geometry)
            if geometry_score is not None:
                breakdown["geometry"] = profile.geometry_weight * geometry_score
        if "latency" in available:
            latency = min(evidence.expected_latency_ms / self.latency_budget_ms, 2.0)
            breakdown["latency"] = -profile.latency_penalty * latency
        if "recovery" in available:
            breakdown["recovery"] = -profile.recovery_penalty * min(evidence.recovery_cost, 2.0)
        return sum(breakdown.values()), breakdown

    @staticmethod
    def _evidence_gate(
        candidate: GraphCandidate,
        scene: SceneSnapshot,
        current_map_version: int | None,
    ) -> str | None:
        evidence = candidate.evidence
        if evidence is None or evidence.scene_version is None:
            return None
        if evidence.scene_version != scene.scene_version:
            return (
                f"evidence targets scene {evidence.scene_version}, "
                f"current scene is {scene.scene_version}"
            )
        if evidence.geometry is not None:
            if evidence.geometry.candidate_fingerprint != subgraph_fingerprint(candidate.subgraph):
                return "geometry evidence fingerprint does not match effective candidate"
            if (
                current_map_version is not None
                and evidence.geometry.map_version is not None
                and evidence.geometry.map_version != current_map_version
            ):
                return (
                    f"geometry evidence targets map {evidence.geometry.map_version}, "
                    f"current map is {current_map_version}"
                )
        return None

    @staticmethod
    def _geometry_gate(candidate: GraphCandidate) -> str | None:
        evidence = candidate.evidence
        if evidence is None or evidence.geometry is None:
            return None
        profile = StrategyProfile.for_name(candidate.strategy)
        geometry = evidence.geometry
        reachability = geometry.reachability
        if (
            reachability.status == "fail"
            and reachability.score is not None
            and reachability.score < profile.min_reachability
        ):
            return (
                f"reachability={reachability.score:.3f} is below "
                f"{profile.name} threshold {profile.min_reachability:.3f}"
            )
        collision = geometry.collision_risk
        if (
            collision.status == "fail"
            and collision.score is not None
            and collision.score > profile.max_collision_risk
        ):
            return (
                f"collision_risk={collision.score:.3f} exceeds "
                f"{profile.name} limit {profile.max_collision_risk:.3f}"
            )
        return None

    def _perception_gate(self, candidate: GraphCandidate) -> str | None:
        evidence = candidate.evidence
        if evidence is None or evidence.perception is None:
            return None
        profile = StrategyProfile.for_name(candidate.strategy)
        perception = evidence.perception
        thresholds = (
            ("scene_freshness", profile.min_scene_freshness),
            ("scene_confidence", profile.min_scene_confidence),
            ("target_visibility", profile.min_target_visibility),
            ("track_confidence", profile.min_track_confidence),
            ("identity_confidence", profile.min_identity_confidence),
            ("pose_reliability", profile.min_pose_reliability),
        )
        for name, threshold in thresholds:
            value = getattr(perception, name)
            if value is not None and value < threshold:
                return f"{name}={value:.3f} is below {profile.name} threshold {threshold:.3f}"
        return None

    def _ordering_key(self, candidate: GraphCandidate) -> tuple[float, int, int, str]:
        subgraph = candidate.subgraph
        duration = sum(node.max_duration_ms for node in subgraph.nodes)
        resources = sum(len(node.exclusive_resources) for node in subgraph.nodes)
        return (self.score(candidate), -duration, -resources, candidate.candidate_id)


def _geometry_score(geometry: object) -> float | None:
    dimensions = (
        (getattr(geometry, "grasp_quality"), 0.30, False),
        (getattr(geometry, "reachability"), 0.30, False),
        (getattr(geometry, "clearance"), 0.25, False),
        (getattr(geometry, "collision_risk"), 0.15, True),
    )
    weighted = 0.0
    total_weight = 0.0
    for dimension, weight, invert in dimensions:
        if dimension.status not in {"pass", "fail"} or dimension.score is None:
            continue
        weighted += weight * (1.0 - dimension.score if invert else dimension.score)
        total_weight += weight
    return weighted / total_weight if total_weight else None
