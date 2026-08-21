from __future__ import annotations

import time

import pytest

from capmas.contracts.candidates import GraphCandidate, subgraph_fingerprint
from capmas.contracts.graph import CheckpointSpec, MotionIntent, SubgraphNodeSpec, SubgraphSpec
from capmas.contracts.scene import ObjectTrack, SceneSnapshot
from capmas.perception.effective_motion import EffectiveMotionProgram, EffectiveMotionSegment
from capmas.perception.geometry import GeometryUpdate
from capmas.perception.geometry_evidence import candidate_geometry_evidence
from capmas.perception.local_map import SparseVoxelMap
from capmas.perception.motion_preview import ReferenceMotionPreview


def _scene() -> SceneSnapshot:
    return SceneSnapshot(
        "episode",
        1,
        7,
        100,
        101,
        {},
        objects=(
            ObjectTrack(
                "bowl",
                "bowl",
                (1.0, 0.0, 0.0, 0.0, 0.40, 0.00, 0.30),
                0.95,
                100,
            ),
        ),
        freshness_ms=1.0,
    )


def _candidate(approach: tuple[float, float, float]) -> GraphCandidate:
    node = SubgraphNodeSpec(
        "pick-action",
        "pick bowl",
        postconditions=("object_in_gripper(bowl)",),
        motion_intent=MotionIntent("grasp", "bowl", approach_vector_xyz=approach),
    )
    subgraph = SubgraphSpec(
        "pick",
        "pick",
        "pick bowl",
        (node,),
        (),
        "pick-action",
        ("pick-action",),
        ("pick-action",),
        checkpoints=(CheckpointSpec("pick-check", node.postconditions),),
    )
    return GraphCandidate(
        candidate_id="candidate-" + "-".join(str(value) for value in approach),
        subgraph=subgraph,
        parent_scene_version=7,
        producer_agent="policy",
    )


def _map() -> SparseVoxelMap:
    local_map = SparseVoxelMap(voxel_size_m=0.02, local_radius_m=2.0)
    local_map.integrate(
        GeometryUpdate(
            timestamp_ns=100,
            camera_poses={},
            points_world=((0.40, 0.00, 0.50),),
        ),
        100,
    )
    return local_map


def _map_with_place_obstacle() -> SparseVoxelMap:
    local_map = SparseVoxelMap(voxel_size_m=0.02, local_radius_m=2.0)
    local_map.integrate(
        GeometryUpdate(
            timestamp_ns=100,
            camera_poses={},
            points_world=((0.60, 0.25, 0.13),),
        ),
        100,
    )
    return local_map


def _program(candidate: GraphCandidate, *, place_approach_m: float) -> EffectiveMotionProgram:
    segments = (
        EffectiveMotionSegment(
            "grasp_approach",
            "grasp_approach",
            "pick",
            "pick-action",
            (1.0, 0.0, 0.0, 0.0, 0.40, 0.00, 0.35),
            (1.0, 0.0, 0.0, 0.0, 0.40, 0.00, 0.30),
            (0.0, 0.0, -1.0),
            0.05,
            "bowl",
        ),
        EffectiveMotionSegment(
            "lift",
            "lift",
            "pick",
            "pick-action",
            (1.0, 0.0, 0.0, 0.0, 0.40, 0.00, 0.30),
            (1.0, 0.0, 0.0, 0.0, 0.40, 0.00, 0.42),
            (0.0, 0.0, 1.0),
            0.12,
            "bowl",
        ),
        EffectiveMotionSegment(
            "transfer",
            "transfer",
            "place",
            "place-action",
            (1.0, 0.0, 0.0, 0.0, 0.40, 0.00, 0.42),
            (0.0, 1.0, 0.0, 0.0, 0.60, 0.25, 0.04 + place_approach_m),
            None,
            0.31,
            "bowl",
        ),
        EffectiveMotionSegment(
            "place_approach",
            "place_approach",
            "place",
            "place-action",
            (0.0, 1.0, 0.0, 0.0, 0.60, 0.25, 0.04 + place_approach_m),
            (0.0, 1.0, 0.0, 0.0, 0.60, 0.25, 0.04),
            (0.0, 0.0, -1.0),
            place_approach_m,
            "bowl",
        ),
        EffectiveMotionSegment(
            "release",
            "release",
            "place",
            "place-action",
            (0.0, 1.0, 0.0, 0.0, 0.60, 0.25, 0.04),
            (0.0, 1.0, 0.0, 0.0, 0.60, 0.25, 0.04),
            None,
            0.0,
            "bowl",
        ),
    )
    return EffectiveMotionProgram(
        candidate_fingerprint=subgraph_fingerprint(candidate.subgraph),
        execution_graph_fingerprint="graph-fingerprint",
        program_fingerprint=f"program-{place_approach_m}",
        decision_scene_version=7,
        selected_subgraph_id="pick",
        segments=segments,
        semantic_signature=f"semantic-{place_approach_m}",
    )


def test_reference_preview_distinguishes_approach_clearance_without_execution() -> None:
    scene = _scene()
    local_map = _map()
    preview = ReferenceMotionPreview()

    top = candidate_geometry_evidence(
        _candidate((0.0, 0.0, -1.0)),
        scene,
        local_map,
        preview,
        deadline_ns=time.monotonic_ns() + 50_000_000,
    )
    side = candidate_geometry_evidence(
        _candidate((1.0, 0.0, 0.0)),
        scene,
        local_map,
        preview,
        deadline_ns=time.monotonic_ns() + 50_000_000,
    )

    assert top.candidate_fingerprint != side.candidate_fingerprint
    assert top.clearance.score != side.clearance.score
    assert top.used_privileged_state is False


def test_geometry_deadline_expiration_returns_unknown_instead_of_zero() -> None:
    evidence = candidate_geometry_evidence(
        _candidate((0.0, 0.0, -1.0)),
        _scene(),
        _map(),
        ReferenceMotionPreview(),
        deadline_ns=time.monotonic_ns() - 1,
    )

    assert evidence.grasp_quality.status == "unknown"
    assert evidence.reachability.status == "unknown"
    assert evidence.clearance.status == "unknown"
    assert evidence.collision_risk.status == "unknown"
    assert evidence.clearance.score is None


def test_reference_preview_does_not_need_or_call_an_executor() -> None:
    result = ReferenceMotionPreview().preview(
        _candidate((0.0, 0.0, -1.0)).subgraph.nodes[0].motion_intent,
        _scene(),
        _map(),
    )

    assert result.backend == "reference_motion_preview"


def test_program_preview_distinguishes_place_approach_lengths() -> None:
    candidate = _candidate((0.0, 0.0, -1.0))
    preview = ReferenceMotionPreview(corridor_samples=5)

    short = preview.preview_program(
        _program(candidate, place_approach_m=0.05),
        _scene(),
        _map_with_place_obstacle(),
    )
    long = preview.preview_program(
        _program(candidate, place_approach_m=0.20),
        _scene(),
        _map_with_place_obstacle(),
    )

    assert short.by_segment("place_approach").collision_free is True
    assert long.by_segment("place_approach").collision_free is False
    queries = long.by_segment("place_approach").map_queries
    assert queries[-1].point_xyz == pytest.approx((0.6, 0.25, 0.04))
    occupied = next(query for query in queries if query.occupied)
    assert occupied.clearance_m == 0.0
    assert occupied.map_version == 1
    assert occupied.snapshot_timestamp_ns == 100


def test_program_geometry_is_conservative_and_has_program_lineage() -> None:
    candidate = _candidate((0.0, 0.0, -1.0))
    program = _program(candidate, place_approach_m=0.20)

    evidence = candidate_geometry_evidence(
        candidate,
        _scene(),
        _map_with_place_obstacle(),
        ReferenceMotionPreview(),
        deadline_ns=time.monotonic_ns() + 50_000_000,
        program=program,
    )

    assert evidence.program_scope == "mission_suffix"
    assert evidence.clearance.score == 0.0
    assert evidence.collision_risk.score == 1.0
    assert evidence.program_fingerprint == program.program_fingerprint


def test_legacy_geometry_keeps_default_program_provenance() -> None:
    evidence = candidate_geometry_evidence(
        _candidate((0.0, 0.0, -1.0)),
        _scene(),
        _map(),
        ReferenceMotionPreview(),
        deadline_ns=time.monotonic_ns() + 50_000_000,
    )

    assert evidence.program_scope == "subgraph"
    assert evidence.execution_graph_fingerprint is None
