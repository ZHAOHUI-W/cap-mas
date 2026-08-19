from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from capmas.contracts.candidates import (
    CandidateEvidence,
    EvidenceDimension,
    GeometryEvidence,
    GraphCandidate,
    PerceptionEvidence,
    rewrite_report_for,
)
from capmas.contracts.core import EpisodeHandle
from capmas.contracts.graph import MissionGraph, SubgraphNodeSpec, SubgraphSpec
from capmas.contracts.scene import EpisodeStart, ObjectTrack, SceneSnapshot
from capmas.contracts.trace import GraphExecutionEvent
from capmas.evaluation.libero_evidence_session import (
    LiveLiberoEvidenceResources,
    LiveLiberoEvidenceSession,
    LiveLiberoEvidenceSessionConfig,
    _execution_payload,
)
from capmas.verification.evidence import VerifierEvidence, VerifierPredicateEvidence


def _scene(version: int) -> SceneSnapshot:
    return SceneSnapshot(
        episode_id="episode",
        episode_epoch=1,
        scene_version=version,
        sensor_timestamp_ns=1_000,
        publish_timestamp_ns=1_100,
        robot={},
        objects=(
            ObjectTrack(
                track_id="butter",
                label="butter",
                pose_wxyz_xyz=(1.0, 0.0, 0.0, 0.0, 0.5, 0.2, 0.1),
                confidence=0.9,
                last_seen_ns=1_000,
            ),
        ),
    )


def _subgraph() -> SubgraphSpec:
    node = SubgraphNodeSpec(
        node_id="pick",
        description="pick",
        postconditions=("object_in_gripper(butter)",),
    )
    return SubgraphSpec(
        subgraph_id="sg_pick",
        subgoal_id="pick",
        description="pick butter",
        nodes=(node,),
        edges=(),
        entry_node="pick",
        success_nodes=("pick",),
        failure_nodes=("pick",),
    )


def _candidate(version: int = 1) -> GraphCandidate:
    subgraph = _subgraph()
    return GraphCandidate(
        candidate_id="candidate",
        subgraph=subgraph,
        parent_scene_version=version,
        producer_agent="test",
        raw_subgraph=subgraph,
        rewrite_report=rewrite_report_for(subgraph, subgraph),
    )


def _graph() -> MissionGraph:
    subgraph = _subgraph()
    return MissionGraph(
        mission_id="mission",
        task="pick",
        subgraphs=(subgraph,),
        edges=(),
        bindings=(),
        entry_subgraph="sg_pick",
        success_subgraphs=("sg_pick",),
        failure_subgraphs=("sg_pick",),
        parent_scene_version=1,
    )


@dataclass
class _FakeStore:
    snapshot: SceneSnapshot | None = None

    def latest(self) -> SceneSnapshot:
        assert self.snapshot is not None
        return self.snapshot

    def compare_and_commit(self, parent_version: int, snapshot: SceneSnapshot) -> bool:
        assert self.snapshot is not None
        assert self.snapshot.scene_version == parent_version
        self.snapshot = snapshot
        return True


@dataclass
class _FakeBackend:
    initial: SceneSnapshot
    decision: SceneSnapshot
    stopped: bool = False

    def reset(self, *, seed: int) -> EpisodeStart:
        return EpisodeStart(
            EpisodeHandle("episode", "task", "libero", "fake", seed, 1, 0),
            self.initial,
        )

    def observe(self) -> SceneSnapshot:
        return self.decision

    def stop(self, _lease: object) -> None:
        self.stopped = True


@dataclass
class _FakeRuntime:
    backend: _FakeBackend
    state_store: _FakeStore
    started: bool = False

    def start_episode(self, episode: EpisodeStart) -> None:
        self.started = True
        self.state_store.snapshot = episode.initial_scene


def _base_evidence(_candidate: GraphCandidate, scene: SceneSnapshot) -> CandidateEvidence:
    result = VerifierEvidence(
        candidate_fingerprint="placeholder",
        scene_version=scene.scene_version,
        pass_rate=1.0,
        coverage=1.0,
        provider="static",
        captured_at_ns=scene.publish_timestamp_ns,
        static_results=(
            VerifierPredicateEvidence(
                predicate="scene_fresh(1000)",
                phase="static",
                status="pass",
                confidence=1.0,
                reason=None,
            ),
        ),
    )
    return CandidateEvidence(
        verifier_pass_rate=1.0,
        perception=PerceptionEvidence(
            scene_freshness=0.9,
            scene_confidence=0.8,
            target_visibility=1.0,
            track_confidence=0.9,
            identity_confidence=1.0,
            pose_reliability=0.9,
        ),
        available_metrics=("perception", "verifier"),
        scene_version=scene.scene_version,
        provider="perception",
        captured_at_ns=scene.publish_timestamp_ns,
        verifier=result,
    )


def _geometry(candidate: GraphCandidate, scene: SceneSnapshot, _map: object, _preview: object, _deadline: int) -> GeometryEvidence:
    fingerprint = candidate.rewrite_report.normalized_fingerprint
    known = lambda name, score: EvidenceDimension(name, "pass", score, 0.5, "test")
    return GeometryEvidence(
        grasp_quality=known("grasp_quality", 0.7),
        reachability=known("reachability", 1.0),
        clearance=known("clearance", 0.8),
        collision_risk=known("collision_risk", 0.0),
        candidate_fingerprint=fingerprint,
        scene_version=scene.scene_version,
        map_version=1,
        map_backend="fake",
        provider="geometry",
        provider_version="1",
        captured_at_ns=scene.publish_timestamp_ns,
        latency_ms=1.0,
    )


def _session(scene_version: int = 1) -> tuple[LiveLiberoEvidenceSession, _FakeBackend]:
    backend = _FakeBackend(_scene(0), _scene(scene_version))
    runtime = _FakeRuntime(backend, _FakeStore())
    config = LiveLiberoEvidenceSessionConfig(
        config_path="libero.yaml",
        object_name="butter",
        target_name="basket",
        seed=7,
        max_steps=32,
    )
    session = LiveLiberoEvidenceSession(
        config,
        resources_factory=lambda _config: LiveLiberoEvidenceResources(
            runtime=runtime,
            geometry_local_map=object(),
        ),
        evidence_collector=_base_evidence,
        geometry_collector=_geometry,
        graph_executor=lambda _runtime, graph, _scene, _episode, _steps: {"graph": graph},
    )
    return session, backend


def test_session_rejects_candidate_from_another_decision_scene() -> None:
    session, _backend = _session()

    assert session.start().scene_version == 1
    with pytest.raises(ValueError, match="decision scene"):
        session.candidate_evidence(_candidate(version=2))


def test_session_merges_perception_verifier_and_geometry_evidence() -> None:
    session, _backend = _session()
    candidate = _candidate()

    session.start()
    evidence = session.candidate_evidence(candidate)

    assert {"perception", "verifier", "geometry"} <= set(evidence.available_metrics)
    assert evidence.geometry is not None
    assert evidence.geometry.candidate_fingerprint == candidate.rewrite_report.normalized_fingerprint


def test_session_executes_once_and_closes_idempotently() -> None:
    session, backend = _session()

    session.start()
    assert session.execute(_candidate(), _graph()) == {"graph": _graph()}
    with pytest.raises(RuntimeError, match="already executed"):
        session.execute(_candidate(), _graph())
    session.close()
    session.close()

    assert backend.stopped is True


def test_execution_payload_uses_graph_events_for_realized_horizon() -> None:
    result = SimpleNamespace(
        completed=True,
        failure=None,
        traces=(),
        events=(
            GraphExecutionEvent(
                sequence=0,
                kind="subgraph_started",
                subgraph_id="sg_pick",
                node_id=None,
                node_type=None,
                attempt=1,
                outcome=None,
                occurred_at_ns=1,
            ),
        ),
    )

    payload = _execution_payload(result, _graph(), evaluator_success=True)

    assert payload["horizon"]["planned_valid"] is True
    assert payload["horizon"]["realized_valid"] is True
