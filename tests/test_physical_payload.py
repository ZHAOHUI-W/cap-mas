from __future__ import annotations

from types import SimpleNamespace

from capmas.contracts.core import ArtifactRef
from capmas.contracts.failures import FailureArtifact
from capmas.contracts.graph import MissionGraph, SubgraphNodeSpec, SubgraphSpec
from capmas.contracts.scene import ObjectTrack, SceneSnapshot
from capmas.contracts.trace import GraphExecutionEvent, SkillTrace
from capmas.contracts.verification import PredicateReport, VerificationResult


def _scene(version: int, *, gripper_opening: float) -> SceneSnapshot:
    return SceneSnapshot(
        episode_id="episode",
        episode_epoch=1,
        scene_version=version,
        sensor_timestamp_ns=1_000 + version,
        publish_timestamp_ns=1_100 + version,
        robot={
            "ee_pose_wxyz_xyz": (1.0, 0.0, 0.0, 0.0, 0.4, 0.2, 0.3),
            "gripper_opening": gripper_opening,
        },
        objects=(
            ObjectTrack(
                track_id="butter",
                label="butter",
                pose_wxyz_xyz=(1.0, 0.0, 0.0, 0.0, 0.5, 0.2, 0.1),
                confidence=0.9,
                last_seen_ns=1_000,
            ),
        ),
        freshness_ms=12.0,
    )


def _graph() -> MissionGraph:
    subgraph = SubgraphSpec(
        subgraph_id="sg_pick",
        subgoal_id="pick_butter",
        description="Pick butter.",
        nodes=(SubgraphNodeSpec(node_id="pick", description="Pick butter."),),
        edges=(),
        entry_node="pick",
        success_nodes=("pick",),
        failure_nodes=("pick",),
    )
    return MissionGraph(
        mission_id="mission",
        task="Pick butter.",
        subgraphs=(subgraph,),
        edges=(),
        bindings=(),
        entry_subgraph="sg_pick",
        success_subgraphs=("sg_pick",),
        failure_subgraphs=(),
        parent_scene_version=1,
    )


def test_physical_payload_preserves_predicates_traces_events_and_scene_transition() -> None:
    from capmas.evaluation.physical_payload import physical_result_payload

    before = _scene(1, gripper_opening=1.0)
    after = _scene(2, gripper_opening=1.0)
    postcondition = VerificationResult(
        contract_id="contract-1",
        decision="recover",
        checked_scene_version=2,
        predicate_results=(
            PredicateReport(
                "gripper_closed()",
                False,
                confidence=0.8,
                evidence=("artifact://gripper",),
                reason="gripper remained open",
            ),
        ),
        failure_class="POSTCONDITION_FAILED",
    )
    trace = SimpleNamespace(
        trace_id="trace-1",
        episode_id="episode",
        episode_epoch=1,
        contract_id="contract-1",
        lease_id="lease-1",
        parent_scene_version=1,
        start_scene_version=1,
        end_scene_version=2,
        started_at_ns=10,
        finished_at_ns=20,
        status="failed",
        skill_traces=(
            SkillTrace(
                invocation_id="invoke-1",
                skill_id="close_gripper",
                skill_version="capx-compat-1",
                args={},
                started_at_ns=11,
                finished_at_ns=12,
                status="completed",
                output={"gripper_fraction": 0.0},
            ),
        ),
        precondition_result=VerificationResult("contract-1", "approve", 1),
        postcondition_result=postcondition,
        failure_class="POSTCONDITION_FAILED",
        observation_before=ArtifactRef("artifact://before", "application/json"),
        observation_after=ArtifactRef("artifact://after", "application/json"),
        metadata={"attempt": 1},
    )
    failure = FailureArtifact(
        failure_id="failure-1",
        failure_class="POSTCONDITION_FAILED",
        message="gripper did not close",
        scene_version=2,
        node_id="pick",
        subgraph_id="sg_pick",
        metadata={"predicate": "gripper_closed()"},
    )
    result = SimpleNamespace(
        completed=False,
        traces=(trace,),
        failure=failure,
        terminal_subgraph="sg_pick",
        next_subgraph=None,
        events=(
            GraphExecutionEvent(
                sequence=0,
                kind="node_failed",
                subgraph_id="sg_pick",
                node_id="pick",
                node_type="action",
                attempt=1,
                outcome="POSTCONDITION_FAILED",
                occurred_at_ns=20,
            ),
        ),
    )

    payload = physical_result_payload(
        result,
        evaluator_success=False,
        graph=_graph(),
        scene_before=before,
        scene_after=after,
        object_ids=("butter",),
    )

    assert payload["failure"]["metadata"] == {"predicate": "gripper_closed()"}
    assert payload["traces"][0]["postcondition_result"]["predicate_results"] == [
        {
            "name": "gripper_closed()",
            "passed": False,
            "confidence": 0.8,
            "evidence": ["artifact://gripper"],
            "reason": "gripper remained open",
        }
    ]
    assert payload["traces"][0]["skill_traces"][0]["output"] == {"gripper_fraction": 0.0}
    assert payload["traces"][0]["observation_before"]["uri"] == "artifact://before"
    assert payload["graph_events"][0]["outcome"] == "POSTCONDITION_FAILED"
    assert payload["scene_diagnostics"]["before"]["scene_version"] == 1
    assert payload["scene_diagnostics"]["after"]["scene_version"] == 2
    assert payload["scene_diagnostics"]["after"]["robot"]["gripper_opening"] == 1.0
    assert payload["scene_diagnostics"]["after"]["objects"][0]["track_id"] == "butter"
