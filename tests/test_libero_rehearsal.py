from dataclasses import dataclass

from capmas.contracts.core import EpisodeHandle
from capmas.contracts.graph import CheckpointSpec, MissionGraph, SubgraphNodeSpec, SubgraphSpec
from capmas.contracts.scene import EpisodeStart, SceneSnapshot
from capmas.evaluation.libero_rehearsal import (
    LiberoRehearsalConfig,
    run_libero_rehearsal_job,
)
from capmas.evaluation.rehearsal import RehearsalFailureClass, RehearsalJob
from capmas.graph.serialization import mission_graph_to_dict
from capmas.skills.registry import SkillRegistry


def _checkpoint_graph() -> dict[str, object]:
    node = SubgraphNodeSpec(
        node_id="checkpoint",
        description="checkpoint",
        node_type="checkpoint",
        proposed_by="test",
    )
    subgraph = SubgraphSpec(
        subgraph_id="verify",
        subgoal_id="verify",
        description="verify",
        nodes=(node,),
        edges=(),
        entry_node="checkpoint",
        success_nodes=("checkpoint",),
        failure_nodes=("checkpoint",),
        checkpoints=(CheckpointSpec("checkpoint-spec", ("scene_advanced",)),),
    )
    graph = MissionGraph(
        mission_id="test-mission",
        task="test",
        subgraphs=(subgraph,),
        edges=(),
        bindings=(),
        entry_subgraph="verify",
        success_subgraphs=("verify",),
        failure_subgraphs=("verify",),
    )
    return mission_graph_to_dict(graph)


@dataclass
class _FakeBackend:
    def reset(self, seed=None, options=None):
        del options
        scene = SceneSnapshot("episode", 1, 0, 1, 2, {})
        handle = EpisodeHandle("episode", "task", "libero", "fake", seed, 1, 0)
        return EpisodeStart(handle, scene)

    def observe(self):
        return SceneSnapshot("episode", 1, 1, 3, 4, {})

    def execute_skill(self, skill, args, budget):
        del skill, args, budget
        raise AssertionError("checkpoint-only test must not execute a skill")

    def stop(self, lease):
        del lease

    def evaluator_success(self):
        return True


@dataclass
class _FakeBundle:
    backend: _FakeBackend
    skill_registry: SkillRegistry
    task_id: str = "fake-task"


def test_libero_rehearsal_executes_serialized_graph_and_records_identity(monkeypatch):
    import capmas.evaluation.libero_rehearsal as module

    monkeypatch.setattr(
        module,
        "_default_build_runtime",
        lambda config: _FakeBundle(_FakeBackend(), SkillRegistry()),
    )
    job = RehearsalJob(
        "candidate-a",
        3,
        {"graph": _checkpoint_graph()},
        task_id="fake-task",
        scene_version=0,
        candidate_fingerprint="fp-a",
    )

    result = run_libero_rehearsal_job(job, LiberoRehearsalConfig("fake.yaml"))

    assert result.success is True
    assert result.worker_pid is not None
    assert result.scene_version == 0
    assert result.candidate_fingerprint == "fp-a"
    assert result.checkpoint_results[-1]["mission_completed"] is True


def test_invalid_serialized_graph_returns_auditable_failure():
    job = RehearsalJob("candidate-a", 3, {"graph": {"schema_version": 1}})

    result = run_libero_rehearsal_job(job, LiberoRehearsalConfig("fake.yaml"))

    assert result.success is False
    assert result.failure_class == RehearsalFailureClass.INVALID_GRAPH
    assert result.failure_reason
