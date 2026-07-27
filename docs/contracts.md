# Runtime Contracts

The initial wire format is JSON-compatible. Binary images, point clouds, and map deltas are referenced by content-addressed artifact IDs rather than embedded in agent messages.

## 1. Common envelope

~~~json
{
  "message_id": "uuid",
  "task_id": "string",
  "episode_id": "uuid",
  "episode_epoch": 3,
  "sender": "policy_agent",
  "recipient": "verifier",
  "kind": "action_contract",
  "created_at_ns": 0,
  "parent_message_id": "uuid|null",
  "scene_version": 12,
  "schema_version": "capmas.v0"
}
~~~

## 2. Episode start and scene snapshot

`EpisodeHandle` identifies an episode. It does not contain the backend
environment object. `EpisodeStart` pairs the handle with the first observable
scene snapshot.

```json
{
  "handle": {
    "episode_id": "uuid",
    "task_id": "string",
    "suite_name": "libero_spatial",
    "backend_id": "capx_libero",
    "seed": 7,
    "episode_epoch": 3,
    "started_at_ns": 0,
    "status": "active"
  },
  "initial_scene": {"scene_version": 0}
}
```

## 3. Scene snapshot

~~~json
{
  "kind": "scene_snapshot",
  "scene_version": 12,
  "sensor_timestamp_ns": 0,
  "publish_timestamp_ns": 0,
  "robot": {
    "joint_position": "artifact://array/...",
    "ee_pose": "artifact://array/...",
    "gripper_opening": 1.0
  },
  "objects": [
    {
      "track_id": "obj-7",
      "label": "red cube",
      "pose": [0, 0, 0, 1, 0, 0, 0],
      "covariance": "artifact://array/...",
      "confidence": 0.96,
      "last_seen_ns": 0
    }
  ],
  "local_map_delta": "artifact://voxel-delta/...",
  "map_frame": "world",
  "freshness_ms": 24
}
~~~

### Multimodal grounding

`track_id` and `label` are not sufficient visual grounding. A track may carry
typed `visual_evidence` references, and a snapshot may carry scene-level
evidence, spatial relations, and explicit uncertainty. The references point to
artifacts in the artifact store; they do not embed mutable RGB-D arrays in an
agent message.

```json
{
  "objects": [{
    "track_id": "obj-7",
    "label": "red cube",
    "visual_evidence": [{
      "artifact": "artifact://rgb/frame-42/crop-7",
      "evidence_type": "rgb_crop",
      "camera_id": "agentview",
      "region_xyxy": [120, 80, 220, 180],
      "captured_at_ns": 42000000000
    }]
  }],
  "spatial_relations": [{
    "subject_track_id": "obj-7",
    "object_track_id": "obj-8",
    "relation": "left_of",
    "confidence": 0.88
  }],
  "uncertainty": {
    "scene_confidence": 0.91,
    "ambiguous_track_ids": ["obj-7"]
  }
}
```

The Policy Agent receives this compact representation by default. For an
ambiguity it emits a targeted request rather than receiving the complete
observation bundle:

```json
{
  "kind": "perception_request",
  "target_track_ids": ["obj-7"],
  "evidence_types": ["rgb_crop", "depth_crop", "mask"],
  "purpose": "identity_disambiguation",
  "max_latency_ms": 120
}
```

Only the Perception Agent reads `ObservationBundle` and raw frame artifacts.
Semantic inference is asynchronous and outside the high-frequency control
loop.

## 4. Action contract

~~~json
{
  "kind": "action_contract",
  "contract_id": "uuid",
  "parent_scene_version": 12,
  "subgoal_id": "sg-3",
  "skill_graph": [
    {"skill_id": "goto_pose", "skill_version": "1.0.0", "args": {"target": "obj-7"}},
    {"skill_id": "close_gripper", "skill_version": "1.0.0", "args": {}}
  ],
  "preconditions": ["track(obj-7).confidence >= 0.85", "gripper.open == true"],
  "expected_postconditions": [
    "object_at_target(akita_black_bowl,plate)",
    "gripper_open"
  ],
  "safety_invariants": ["distance_to_obstacle >= 0.02", "joint_limits_valid == true"],
  "max_duration_ms": 5000,
  "max_sim_steps": 120,
  "recovery_policy": "reacquire_and_retry",
  "proposed_by": "policy_agent"
}
~~~

## 5. Verification result

~~~json
{
  "kind": "verification_result",
  "contract_id": "uuid",
  "decision": "approve|reject|commit|recover",
  "checked_scene_version": 12,
  "predicate_results": [
    {"predicate": "object_at_target(akita_black_bowl,plate)", "value": true, "confidence": 1.0},
    {"predicate": "gripper_open", "value": true, "confidence": 1.0}
  ],
  "violated_invariants": [],
  "evidence_artifacts": ["artifact://rgbd/...", "artifact://trace/..."],
  "failure_class": null
}
~~~

## 6. Failure classes

~~~text
STALE_STATE          parent scene version no longer current
PRECONDITION_FAILED  contract cannot safely start
EXECUTION_ERROR      skill or code raised an error
MOTION_TIMEOUT       bounded motion did not converge
POSTCONDITION_FAILED action ran but intended state was not achieved
PERCEPTION_UNCERTAIN evidence is insufficient for a safe decision
COLLISION_RISK       invariant or safety monitor triggered
EPISODE_INVALIDATED  reset, timeout, or external cancellation occurred
~~~

## 7. Action lease

The lease is issued by the runtime, not an LLM. It has lease_id, holder, contract_id, issued_at, expires_at, and cancellation_token. Lease expiry causes a controlled stop or safe hold; it does not authorize another agent to assume the robot silently.
