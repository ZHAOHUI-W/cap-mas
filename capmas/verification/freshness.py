from __future__ import annotations

from capmas.contracts.scene import SceneSnapshot


def is_fresh(snapshot: SceneSnapshot, now_ns: int, max_age_ms: float) -> bool:
    return (now_ns - snapshot.publish_timestamp_ns) <= max_age_ms * 1_000_000
