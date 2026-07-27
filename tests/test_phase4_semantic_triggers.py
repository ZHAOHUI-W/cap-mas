from capmas.contracts.scene import ObjectTrack, SceneSnapshot
from capmas.perception.protocol import PerceptionRequest
from capmas.perception.semantic_triggers import (
    DeterministicSemanticRequestQueue,
    DeterministicSemanticTrigger,
)


def track(track_id: str, *, confidence: float) -> ObjectTrack:
    return ObjectTrack(
        track_id=track_id,
        label="cube",
        pose_wxyz_xyz=(1.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0),
        confidence=confidence,
        last_seen_ns=100,
    )


def make_scene() -> SceneSnapshot:
    return SceneSnapshot(
        "ep",
        1,
        3,
        100,
        100,
        {},
        objects=(track("cube-1", confidence=0.2),),
    )


def make_request(request_id: str, *, priority: int) -> PerceptionRequest:
    return PerceptionRequest(
        request_id=request_id,
        target_track_ids=(request_id,),
        evidence_types=("rgb_crop",),
        priority=priority,
        episode_id="ep",
        episode_epoch=1,
        scene_version=3,
    )


class Clock:
    def __init__(self) -> int:
        self.value = 0

    def __call__(self) -> int:
        return self.value


def test_low_confidence_track_emits_deduplicated_request():
    queue = DeterministicSemanticRequestQueue()
    trigger = DeterministicSemanticTrigger(queue, confidence_threshold=0.5)
    scene = make_scene()

    trigger.inspect(scene)
    trigger.inspect(scene)

    requests = queue.poll(max_items=10)
    assert len(requests) == 1
    assert requests[0].target_track_ids == ("cube-1",)
    assert queue.metrics().deduplicated == 1


def test_queue_returns_high_priority_request_first():
    queue = DeterministicSemanticRequestQueue()
    queue.submit(make_request("low", priority=1))
    queue.submit(make_request("high", priority=10))

    assert [request.request_id for request in queue.poll(max_items=2)] == ["high", "low"]


def test_queue_timeout_does_not_change_fast_snapshot_processing():
    clock = Clock()
    queue = DeterministicSemanticRequestQueue(max_latency_ms=1, clock=clock)
    queue.submit(make_request("timeout", priority=1))
    clock.value = 2_000_000

    assert queue.poll(max_items=1) == ()
    assert queue.metrics().timed_out == 1
