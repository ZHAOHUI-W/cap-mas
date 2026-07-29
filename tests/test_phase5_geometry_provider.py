from __future__ import annotations

import time

from capmas.contracts.candidates import GraphCandidate
from capmas.contracts.graph import CheckpointSpec, MotionIntent, SubgraphNodeSpec, SubgraphSpec
from capmas.contracts.scene import ObjectTrack, SceneSnapshot
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
