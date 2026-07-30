from __future__ import annotations

import hashlib
import json

import pytest

from capmas.contracts.action import SkillCall, SkillOutputRef
from capmas.contracts.core import SkillRef
from capmas.contracts.graph import (
    CheckpointSpec,
    MissionGraph,
    SubgraphNodeSpec,
    SubgraphSpec,
)
from capmas.graph.serialization import mission_graph_to_dict


def _raw_graph() -> dict[str, object]:
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
    graph = MissionGraph(
        mission_id="mission",
        task="pick",
        subgraphs=(subgraph,),
        edges=(),
        bindings=(),
        entry_subgraph="sg_pick",
        success_subgraphs=("sg_pick",),
        failure_subgraphs=("sg_pick",),
        parent_scene_version=4,
    )
    return mission_graph_to_dict(graph)


def test_raw_graph_fingerprint_is_stable_over_json_key_order():
    from capmas.evaluation.candidate_identity import raw_graph_fingerprint

    raw = _raw_graph()
    reordered = {key: raw[key] for key in reversed(tuple(raw))}

    expected = hashlib.sha256(
        json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    assert raw_graph_fingerprint(raw) == expected
    assert raw_graph_fingerprint(reordered) == expected


def test_identity_derives_source_graph_and_explicit_arbiter_subgraph_fingerprints():
    from capmas.evaluation.candidate_identity import candidate_identity_from_raw_graph

    identity = candidate_identity_from_raw_graph(_raw_graph(), "sg_pick", 4)

    assert identity.graph_fingerprint
    assert identity.subgraph_id == "sg_pick"
    assert identity.subgraph_fingerprint
    assert identity.scene_version == 4
    assert identity.graph_fingerprint != identity.subgraph_fingerprint


def test_identity_rejects_unknown_explicit_target_subgraph():
    from capmas.evaluation.candidate_identity import candidate_identity_from_raw_graph

    with pytest.raises(ValueError, match="unknown target subgraph"):
        candidate_identity_from_raw_graph(_raw_graph(), "sg_missing", 4)


def test_identity_preserves_legacy_call_index_source_graph():
    from capmas.evaluation.candidate_identity import candidate_identity_from_raw_graph

    raw = _raw_graph()
    raw["subgraphs"][0]["nodes"][0]["skill_calls"][0]["args"]["position"] = {
        "call_index": 0,
        "path": ["result", 0],
    }

    identity = candidate_identity_from_raw_graph(raw, "sg_pick", 4)

    assert identity.graph_fingerprint == hashlib.sha256(
        json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert identity.subgraph_fingerprint


def test_evaluation_public_exports_include_identity_and_shadow_api():
    import capmas.evaluation as evaluation

    assert "CandidateIdentity" in evaluation.__all__
    assert "raw_graph_fingerprint" in evaluation.__all__
    assert "ShadowArbitrationReport" in evaluation.__all__
    assert "run_shadow_arbitration" in evaluation.__all__
