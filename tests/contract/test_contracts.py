"""Contract-level tests for CAP-MAS contracts system.

Tests verify public behavior at the contract boundary (dataclass integrity,
state machine transitions, serialization compatibility).
"""
from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from capmas.contracts.action import ActionContract, ExecutionBudget, SkillCall
from capmas.contracts.core import ArtifactRef, EpisodeHandle, SkillRef
from capmas.contracts.failures import FailureClass
from capmas.contracts.scene import EpisodeStart, SceneSnapshot
from capmas.contracts.trace import ExecutionTrace, SkillTrace
from capmas.contracts.verification import PredicateReport, VerificationResult
from capmas.runtime.action_lease import ActionLease, ActionLeaseManager


# ── SceneSnapshot version monotonicity ──────────────────────────

def test_scene_snapshot_is_immutable():
    snapshot = SceneSnapshot(
        episode_id="ep-1",
        episode_epoch=1,
        scene_version=5,
        sensor_timestamp_ns=100,
        publish_timestamp_ns=200,
        robot={},
    )
    with pytest.raises(Exception):
        snapshot.scene_version = 6  # type: ignore[misc]


# ── ActionContract epoch validation ─────────────────────────────

def test_action_contract_rejects_wrong_epoch():
    contract = ActionContract(
        contract_id="c-1",
        episode_id="ep-1",
        episode_epoch=2,
        parent_scene_version=0,
        subgoal_id="sg-1",
        skills=(SkillCall(SkillRef("move", "1.0.0"), {"target": "box"}),),
        expected_postconditions=("moved",),
        max_duration_ms=1000,
        max_sim_steps=10,
        proposed_by="policy_agent",
    )
    # contract has epoch 2, but episode handle has epoch 1 → should be rejected
    handle = EpisodeHandle("ep-1", "task-1", "suite", "mock", 7, 1, 100)
    assert contract.episode_epoch != handle.episode_epoch


# ── ActionLease mutual exclusion ─────────────────────────────────

def test_lease_cannot_be_acquired_twice():
    mgr = ActionLeaseManager(clock=lambda: 1000)
    mgr.acquire("agent-1", "contract-1", 5000)
    with pytest.raises(RuntimeError, match="already held"):
        mgr.acquire("agent-2", "contract-2", 5000)


def test_lease_cannot_release_wrong_holder():
    mgr = ActionLeaseManager(clock=lambda: 1000)
    mgr.acquire("agent-1", "contract-1", 5000)
    # Try to release with wrong lease_id — should raise ValueError
    with pytest.raises(ValueError, match="cannot release another holder"):
        mgr.release("wrong-lease-id")


def test_lease_expires_after_duration():
    mgr = ActionLeaseManager(clock=lambda: 1000)
    mgr.acquire("agent-1", "contract-1", 5000)
    assert mgr.active() is not None
    # After duration, lease should be expired
    expired = mgr.expire_if_needed(now_ns=1000 + 5000 * 1_000_000 + 1)
    assert expired is True
    assert mgr.active() is None


# ── FailureClass exhaustiveness ──────────────────────────────────

def test_failure_class_enum_is_exhaustive():
    expected = {
        "STALE_STATE",
        "PRECONDITION_FAILED",
        "EXECUTION_ERROR",
        "MOTION_TIMEOUT",
        "POSTCONDITION_FAILED",
        "PERCEPTION_UNCERTAIN",
        "COLLISION_RISK",
        "EPISODE_INVALIDATED",
    }
    actual = {v for v in dir(FailureClass) if not v.startswith("_") and v.isupper()}
    assert actual == expected


# ── VerificationResult.passed ────────────────────────────────────

def test_verification_result_passed():
    approved = VerificationResult(
        "c-1", "approve", 0, (PredicateReport("ok", True),)
    )
    assert approved.passed is True

    rejected = VerificationResult(
        "c-1", "reject", 0, ()
    )
    assert rejected.passed is False

    failed_predicate = VerificationResult(
        "c-1", "approve", 0, (PredicateReport("post", False),)
    )
    assert failed_predicate.passed is False


# ── JSON serialization round-trip ────────────────────────────────

def test_scene_snapshot_serializes_to_json():
    snapshot = SceneSnapshot(
        episode_id="ep-1",
        episode_epoch=1,
        scene_version=3,
        sensor_timestamp_ns=100,
        publish_timestamp_ns=200,
        robot={"joint_position": ArtifactRef("artifact://a/1", "array/joint-position")},
    )
    data = asdict(snapshot)
    text = json.dumps(data, default=str)
    assert "ep-1" in text
    assert "artifact://a/1" in text


def test_action_contract_serializes_to_json():
    contract = ActionContract(
        contract_id="c-1",
        episode_id="ep-1",
        episode_epoch=1,
        parent_scene_version=0,
        subgoal_id="sg-1",
        skills=(SkillCall(SkillRef("move", "1.0.0"), {"target": "box"}),),
        expected_postconditions=("moved",),
        preconditions=("gripper.open",),
        safety_invariants=("joint_limits_valid",),
        max_duration_ms=1000,
        max_sim_steps=10,
        proposed_by="policy",
        recovery_policy="replan",
    )
    data = asdict(contract)
    text = json.dumps(data, default=str)
    assert "move" in text
    assert "1.0.0" in text


# ── ExecutionTrace integrity ─────────────────────────────────────

def test_execution_trace_links_skill_and_contract():
    skill_trace = SkillTrace(
        invocation_id="inv-1",
        skill_id="move",
        skill_version="1.0.0",
        args={"target": "box"},
        started_at_ns=100,
        finished_at_ns=200,
        status="completed",
    )
    trace = ExecutionTrace(
        trace_id="t-1",
        episode_id="ep-1",
        episode_epoch=1,
        contract_id="c-1",
        lease_id="lease-1",
        parent_scene_version=0,
        start_scene_version=0,
        end_scene_version=1,
        started_at_ns=100,
        finished_at_ns=200,
        status="completed",
        skill_traces=(skill_trace,),
    )
    assert trace.contract_id == "c-1"
    assert len(trace.skill_traces) == 1
    assert trace.skill_traces[0].skill_id == "move"