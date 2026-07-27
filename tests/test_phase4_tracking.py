from capmas.perception.tracking import KnownObjectTracker, ObjectMeasurement


def measurement(
    label: str,
    position: tuple[float, float, float],
    *,
    track_id: str | None = None,
    confidence: float = 0.9,
    timestamp_ns: int,
) -> ObjectMeasurement:
    return ObjectMeasurement(
        track_id=track_id,
        label=label,
        pose_wxyz_xyz=(1.0, 0.0, 0.0, 0.0, *position),
        confidence=confidence,
        timestamp_ns=timestamp_ns,
    )


def test_explicit_track_id_is_preserved_across_updates():
    tracker = KnownObjectTracker(max_match_distance_m=0.2)
    first = tracker.update((measurement("cube", (0, 0, 0), track_id="cube-7", timestamp_ns=100),))
    second = tracker.update((measurement("cube", (0.1, 0, 0), track_id="cube-7", timestamp_ns=200),))

    assert first[0].track_id == "cube-7"
    assert second[0].track_id == "cube-7"
    assert second[0].last_seen_ns == 200


def test_label_and_distance_gate_associates_measurement_without_id():
    tracker = KnownObjectTracker(max_match_distance_m=0.2)
    tracker.update((measurement("cube", (0, 0, 0), timestamp_ns=100),))

    tracks = tracker.update((measurement("cube", (0.1, 0, 0), timestamp_ns=200),))

    assert len(tracks) == 1
    assert tracks[0].track_id == "cube-0"


def test_missing_measurement_uses_constant_velocity_prediction():
    tracker = KnownObjectTracker(max_match_distance_m=0.2, prediction_timeout_ms=500)
    tracker.update((measurement("cube", (0, 0, 0), timestamp_ns=100),))
    tracker.update((measurement("cube", (0.1, 0, 0), timestamp_ns=200),))

    predicted = tracker.predict(300)

    assert predicted[0].track_status == "predicted"
    assert predicted[0].pose_wxyz_xyz[4] == 0.2


def test_prediction_confidence_decays_then_enters_stale_and_lost_states():
    tracker = KnownObjectTracker(
        max_match_distance_m=0.2,
        prediction_timeout_ms=100,
        stale_timeout_ms=200,
        confidence_decay=0.5,
    )
    tracker.update((measurement("cube", (0, 0, 0), timestamp_ns=100_000_000),))

    stale = tracker.predict(250_000_000)[0]
    lost = tracker.predict(400_000_001)[0]

    assert stale.track_status == "stale"
    assert stale.confidence < 0.9
    assert lost.track_status == "lost"
