from capmas.contracts.core import SkillRef
from capmas.contracts.memory import MemoryItem, MemoryLayer, MemoryOperation, MemoryUpdate
from capmas.memory.store import InMemoryMemoryStore


def make_update(base_version: str = "0", key: str = "update-1") -> MemoryUpdate:
    item = MemoryItem(
        memory_id="memory-1",
        memory_version="1.0.0",
        kind="failure_rule",
        content={"failure": "grasp_failed"},
        applicability={"task_family": "pick_place"},
        confidence=0.9,
        evidence_count=1,
        source_episode_ids=("episode-1",),
        source_trace_ids=("trace-1",),
        status="active",
    )
    return MemoryUpdate(
        update_id=key,
        episode_id="episode-1",
        task_id="task-1",
        base_memory_version=base_version,
        target_layer=MemoryLayer.EXPERIENCE,
        operation=MemoryOperation.ADD,
        items=(item,),
        source_trace_ids=("trace-1",),
        produced_by_skill=SkillRef("extract_failure_cause", "1.0.0"),
        confidence=0.9,
        idempotency_key=key,
    )


def test_memory_update_is_idempotent_and_advances_version() -> None:
    store = InMemoryMemoryStore()

    first = store.commit(make_update())
    repeated = store.commit(make_update())

    assert first.version == "1"
    assert repeated.version == "1"
    assert len(repeated.items) == 1


def test_memory_update_requires_current_base_version() -> None:
    store = InMemoryMemoryStore()
    store.commit(make_update())

    try:
        store.commit(make_update(key="update-2"))
    except ValueError as exc:
        assert str(exc) == "memory base version conflict"
    else:
        raise AssertionError("stale memory update was accepted")
