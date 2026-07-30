import json

import pytest

from capmas.contracts.action import SkillCall, SkillOutputRef
from capmas.contracts.core import SkillRef
from capmas.contracts.graph import CheckpointSpec, MissionGraph, SubgraphNodeSpec, SubgraphSpec
from capmas.evaluation.candidate_identity import raw_graph_fingerprint
from capmas.graph.serialization import mission_graph_to_dict

from capmas.evaluation.rehearsal import RehearsalResult
from capmas.evaluation.rehearsal_evidence import RehearsalPoolConfig
from scripts.run_libero_p53_rehearsal import (
    build_rehearsal_jobs,
    load_candidate_artifact,
    run_rehearsal_batches,
)


def _graph_payload():
    node = SubgraphNodeSpec(
        node_id="act",
        description="act",
        skill_calls=(
            SkillCall(
                SkillRef("goto_pose", "1.0.0"),
                {"position": SkillOutputRef(0, ("result", 0))},
            ),
        ),
        postconditions=("step_done",),
        proposed_by="policy",
    )
    subgraph = SubgraphSpec(
        subgraph_id="sg_pick",
        subgoal_id="pick",
        description="pick",
        nodes=(node,),
        edges=(),
        entry_node="act",
        success_nodes=("act",),
        failure_nodes=("act",),
        checkpoints=(CheckpointSpec("check", ("step_done",)),),
    )
    return mission_graph_to_dict(
        MissionGraph(
            mission_id="mission",
            task="pick",
            subgraphs=(subgraph,),
            edges=(),
            bindings=(),
            entry_subgraph="sg_pick",
            success_subgraphs=("sg_pick",),
            failure_subgraphs=("sg_pick",),
        )
    )


def test_candidate_artifact_builds_seeded_serializable_jobs(tmp_path):
    artifact = tmp_path / "candidates.json"
    artifact.write_text(
        json.dumps(
            {
                "task_id": "libero_spatial_0",
                "scene_version": 8,
                "candidates": [
                    {
                        "candidate_id": "candidate-a",
                        "candidate_fingerprint": "fp-a",
                        "graph": {"schema_version": 1, "mission_id": "m"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    candidates = load_candidate_artifact(artifact)
    jobs = build_rehearsal_jobs(candidates, seed=4)

    assert len(jobs) == 1
    assert jobs[0].candidate_id == "candidate-a"
    assert jobs[0].seed == 4
    assert jobs[0].scene_version == 8
    assert jobs[0].candidate_fingerprint == "fp-a"
    assert jobs[0].payload["graph"]["mission_id"] == "m"


def test_graph_scoped_candidate_derives_explicit_arbiter_identity():
    from scripts.run_libero_p53_rehearsal import parse_candidate_mapping

    graph = _graph_payload()
    candidates = parse_candidate_mapping(
        {
            "task_id": "task",
            "scene_version": 4,
            "candidates": [
                {
                    "candidate_id": "candidate-a",
                    "candidate_fingerprint": raw_graph_fingerprint(graph),
                    "fingerprint_scope": "graph",
                    "arbiter_subgraph_id": "sg_pick",
                    "graph": graph,
                }
            ],
        }
    )

    assert candidates[0].identity is not None
    assert candidates[0].identity.subgraph_id == "sg_pick"
    assert candidates[0].identity.graph_fingerprint == candidates[0].candidate_fingerprint


def test_graph_scoped_candidate_rejects_missing_explicit_arbiter_subgraph():
    from scripts.run_libero_p53_rehearsal import parse_candidate_mapping

    graph = _graph_payload()
    with pytest.raises(ValueError, match="unknown target subgraph"):
        parse_candidate_mapping(
            {
                "task_id": "task",
                "scene_version": 4,
                "candidates": [
                    {
                        "candidate_id": "candidate-a",
                        "candidate_fingerprint": raw_graph_fingerprint(graph),
                        "fingerprint_scope": "graph",
                        "arbiter_subgraph_id": "sg_missing",
                        "graph": graph,
                    }
                ],
            }
        )


def test_run_batches_allocates_independent_secret_free_artifacts(tmp_path):
    candidates = load_candidate_artifact_from_mapping(
        {
            "task_id": "task",
            "scene_version": 0,
            "candidates": [
                {
                    "candidate_id": "candidate-a",
                    "graph": {"schema_version": 1, "mission_id": "m"},
                }
            ],
        }
    )

    def fake_run(jobs, worker_factory, pool_config):
        del worker_factory, pool_config
        return tuple(
            RehearsalResult(
                candidate_id=job.candidate_id,
                seed=job.seed,
                success=True,
                latency_ms=1.0,
                scene_version=job.scene_version,
                candidate_fingerprint=job.candidate_fingerprint,
            )
            for job in jobs
        )

    first, second = run_rehearsal_batches(
        config_path="libero.yaml",
        candidates=candidates,
        seeds=(1, 2),
        output_root=tmp_path,
        pool_config=RehearsalPoolConfig(max_workers=1, timeout_s=1.0),
        run_fn=fake_run,
    )

    assert first.path != second.path
    assert (first.path / "logs").is_dir()
    assert (first.path / "results" / "rehearsal.json").exists()
    assert (first.path / "manifest.json").exists()
    config_text = (first.path / "run_config.json").read_text(encoding="utf-8")
    assert "api_key" not in config_text.lower()


def load_candidate_artifact_from_mapping(value):
    from scripts.run_libero_p53_rehearsal import parse_candidate_mapping

    return parse_candidate_mapping(value)
