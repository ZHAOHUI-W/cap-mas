from capmas.contracts.agent import AgentContext, AgentArtifact, PolicyDecision
from capmas.contracts.core import ArtifactRef
from capmas.contracts.scene import ObjectTrack, SceneSnapshot, SpatialRelation, VisualEvidence
from capmas.perception.fusion import tracks_from_result
from capmas.perception.protocol import ObjectPoseEstimate, PerceptionRequest, PerceptionResult
from capmas.agents.policy import CallableGroundedPolicyAgent


def make_scene() -> SceneSnapshot:
    rgb = ArtifactRef("artifact://rgb/frame-1", "image/rgb")
    return SceneSnapshot(
        episode_id="episode-1",
        episode_epoch=1,
        scene_version=3,
        sensor_timestamp_ns=100,
        publish_timestamp_ns=110,
        robot={},
        objects=(
            ObjectTrack(
                track_id="obj-7",
                label="red cube",
                pose_wxyz_xyz=(1.0, 0.0, 0.0, 0.0, 0.2, 0.3, 0.4),
                confidence=0.94,
                last_seen_ns=100,
                visual_evidence=(
                    VisualEvidence(
                        artifact=rgb,
                        evidence_type="rgb_crop",
                        camera_id="agentview",
                        captured_at_ns=100,
                    ),
                ),
            ),
        ),
        spatial_relations=(
            SpatialRelation("obj-7", "obj-8", "left_of", 0.88),
        ),
    )


def test_policy_context_contains_grounded_visual_evidence_without_raw_frames() -> None:
    context = AgentContext(
        task_id="task-1",
        episode_id="episode-1",
        episode_epoch=1,
        scene=make_scene(),
    )

    track = context.scene.objects[0]

    assert track.visual_evidence[0].artifact.uri == "artifact://rgb/frame-1"
    assert context.scene.spatial_relations[0].relation == "left_of"
    assert context.scene.uncertainty.scene_confidence == 1.0
    assert not hasattr(context, "observation")


def test_policy_can_request_targeted_visual_evidence() -> None:
    request = PerceptionRequest(
        request_id="request-1",
        query="confirm the identity of obj-7",
        target_track_ids=("obj-7",),
        evidence_types=("rgb_crop", "depth_crop", "mask"),
        purpose="identity_disambiguation",
        max_latency_ms=120,
    )

    assert request.target_track_ids == ("obj-7",)
    assert request.evidence_types == ("rgb_crop", "depth_crop", "mask")
    assert request.purpose == "identity_disambiguation"


def test_grounded_policy_adapter_preserves_perception_request_boundary() -> None:
    subgoal = AgentArtifact("subgoal-1", "subgoal", {}, "manager")
    context = AgentContext("task-1", "episode-1", 1, make_scene())
    request = PerceptionRequest(
        request_id="request-2",
        target_track_ids=("obj-7",),
        evidence_types=("mask",),
        purpose="occlusion_check",
    )
    policy = CallableGroundedPolicyAgent(
        lambda received_subgoal, received_scene, received_context: PolicyDecision(
            perception_request=request,
            grounded_track_ids=received_scene.uncertainty.ambiguous_track_ids,
        )
    )

    decision = policy.decide(subgoal, context.scene, context)

    assert decision.action is None
    assert decision.perception_request == request
    assert decision.grounded_track_ids == ()


def test_perception_evidence_is_attached_to_the_grounded_object_track() -> None:
    evidence = VisualEvidence(
        artifact=ArtifactRef("artifact://mask/7", "image/mask"),
        evidence_type="mask",
        captured_at_ns=200,
        camera_id="agentview",
        track_id="obj-7",
    )
    result = PerceptionResult(
        request_id="request-3",
        timestamp_ns=200,
        poses_3d=(
            ObjectPoseEstimate(
                label="red cube",
                pose_wxyz_xyz=(1.0, 0.0, 0.0, 0.0, 0.2, 0.3, 0.4),
                confidence=0.9,
                covariance=None,
                frame="world",
                track_id="obj-7",
            ),
        ),
        visual_evidence=(evidence,),
    )

    tracks = tracks_from_result(result, 4, "episode-1", 1)

    assert tracks[0].track_id == "obj-7"
    assert tracks[0].visual_evidence == (evidence,)
