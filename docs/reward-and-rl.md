# Reward, Credit Assignment, and RL Boundary

## 1. Current RL inventory

The CAP-MAS runtime specification does not currently contain an RL update loop.
The following components are deterministic, rule-based, or LLM-inference
components rather than RL policies:

| Component | Current mechanism | RL status |
| --- | --- | --- |
| Mission Manager | Subgoal graph and budget scheduling | No online RL |
| Perception Agent | Event-triggered semantic inference | No RL update |
| Policy Agent | LLM proposes typed ActionContracts | No RL update |
| Verifier | Preconditions, invariants, postconditions | Deterministic/checker |
| Recovery Agent | Failure taxonomy and bounded recovery contracts | No RL update |
| Runtime scheduler and lease manager | Fixed graph and authority arbitration | Deterministic |
| Fast scene estimator | Tracking, geometry, and map update | Estimation/control, not RL |
| Critic and Skill Evolver | Trace analysis and quarantined candidates | Search/evolution, not RL |
| Reward/evaluator | Reports task and subgoal outcomes | Signal only; no update |

CAP-MAS still reserves the CAP-X compatibility path. CAP-X has a separate
CaP-RL/GRPO post-training path and RL-related simulator reward implementations.
Those are training and baseline facilities, not evidence that the CAP-MAS
episode runtime already performs RL. A CAP-X RL model may be plugged in as the
Policy Agent backend, but it must be reported as a separate condition.

## 2. Two reward channels

CAP-X's final benchmark definition remains unchanged:

```text
R_task = 1 if the evaluator verifies final task completion, otherwise 0
```

This score is the primary comparison metric and must be reported for every
system. CAP-MAS must not redefine a partially completed task as a successful
task merely because it received intermediate credit.

For training and diagnosis, CAP-MAS additionally emits a structured learning
return from observable, signed verification events. The two channels are kept
separate:

```text
benchmark return: R_task
learning return:  R_learn = terminal + progress + reliability - cost
```

The learning return is never used as the final success label.

## 3. Verified intermediate reward

Let `phi_i(s)` be the progress of subgoal `i`, measured only by a typed
verifier from the current SceneSnapshot. It is in `[0, 1]` and is tied to a
declared predicate, for example `object_in_gripper`, `object_at_target`, or
`gripper_open`. The task graph defines dependency order and prevents the same
subgoal from being credited repeatedly.

Define the normalized progress potential:

```text
Phi(s) = sum_i weight_i * phi_i(s) / sum_i weight_i
```

For an action chunk from `s_t` to `s_(t+1)`, the default shaped signal is:

```text
r_learn(t) =
    w_p * (Phi(s_(t+1)) - Phi(s_t))
  + w_v * verified_transition(t)
  - w_d * normalized_duration(t)
  - w_q * repeated_retry(t)
  - w_h * human_intervention(t)
  - w_n * unnecessary_replan(t)
```

The terminal term is added once per episode:

```text
r_terminal = w_f * R_task
```

Safety violations, invalid leases, stale contracts, or collision-risk stops
are hard constraint events. They terminate or block the candidate action and
are logged separately; the learner cannot compensate for a safety violation
with task progress. This is a constrained-learning interface, not a single
scalar trade-off that permits unsafe behavior.

The progress potential should be potential-based where possible. This gives
intermediate credit for long-horizon progress while preserving the terminal
task objective under the usual shaping assumptions. Distance or detector
signals that are not stable enough to serve as predicates may be used as
diagnostic features, but not as promotion evidence.

## 4. Agent-level credit assignment

All rewards carry artifact lineage: the subgoal, ActionContract, selected
Memory Skills, Robot Skills, verifier evidence, and responsible agent IDs.
This makes it possible to assign delayed outcomes without rewarding every
agent equally:

| Learner or role | Useful credit signal |
| --- | --- |
| Memory Controller | Downstream progress after selected Memory Skills, memory validation quality, and cost |
| Policy Agent | Valid contract rate and verified progress after its contract |
| Perception Agent | Correctness and freshness of facts later confirmed by the verifier |
| Verifier | Correct approvals/rejections and false-completion avoidance |
| Recovery Agent | Recovery success, time-to-recovery, and repeated-failure avoidance |

Only the Memory Controller is in the initial RL scope. The other roles remain
frozen LLM or deterministic modules until separate experiments justify their
optimization. This prevents an increase in trainable agents from being
mistaken for a coordination improvement.

## 5. Three-stage Memory Controller RL

The controller selects a bounded Top-K set of Memory Skills for the current
trace span. It does not write memory directly and does not execute robot
actions.

### Stage 1: Offline skill-selection policy

Train a lightweight ranker or contextual bandit from archived traces and
verified memory updates. The context contains task family, scene summary,
failure class, current subgoal, retrieved memories, and budget state. The
action is `select Top-K`, `skip`, or `request a specific memory operation`.

The initial reward emphasizes memory quality: validated extraction,
non-contradiction, retrieval precision, useful update coverage, and low token
or latency cost. Existing successful and failed traces provide the dataset;
no new robot rollouts are required. A rules-based selector remains the
fallback and ablation baseline.

### Stage 2: Short-horizon online adaptation

Freeze the Robot Skill Registry and run the controller in simulation on short
subtask episodes. Use a contextual bandit or conservative PPO over memory
selection decisions. The reward adds verified subgoal progress and recovery
success, while keeping memory quality and cost terms.

The controller may explore only among active Memory Skills. Newly proposed
Memory Skills stay quarantined and are evaluated in a separate shadow pass.
No gradient update occurs in the robot control or servo process.

### Stage 3: Long-horizon constrained optimization

Freeze both the active Robot Skill Registry and the active Memory Skill
semantics for each training snapshot. Optimize the controller over complete
multi-subgoal episodes using PPO/GRPO or an equivalent constrained policy
optimizer. The terminal CAP-X-compatible `R_task` is combined with verified
potential shaping, recovery outcome, budget, and intervention cost.

Use a sliding hard-case buffer and evaluate every new controller snapshot on a
locked regression suite, targeted failures, and OOD compositions. Keep the
best validated snapshot and roll back on regression. At inference time the
controller is a cached ranker or bounded selector; training is asynchronous
and never adds latency to the high-frequency loop.

The staged progression is intentional: it reduces variance and credit
assignment difficulty before exposing the controller to long-horizon sparse
outcomes. It also makes the contribution auditable through stage-wise
ablations.

## 6. Human-intervention reduction

The system should reduce human involvement through automatic evidence and
gates, not by removing safety checks:

1. Typed postconditions automatically label subgoal progress.
2. Failure taxonomy automatically creates hard cases from traces.
3. Memory Skill candidates are validated for schema, contradiction, replay,
   regression, and OOD behavior.
4. Snapshot promotion and rollback are automatic at episode boundaries.
5. Human approval is restricted to initial safety allowlists, policy thresholds,
   emergency intervention, and periodic audit.

The number of interventions, intervention causes, and intervention-free task
success must be reported as first-class metrics.

## 7. Required reward ablations

- CAP-X binary reward only.
- CAP-MAS binary terminal reward only.
- Verified subgoal potential shaping.
- Shaping without reliability/cost terms.
- Shaping with unverified detector signals.
- Human-intervention cost enabled or disabled.
- Stage 1 only, Stage 1+2, and Stage 1+2+3.

The primary success-rate and horizon-stability comparisons must use matched
model calls, rollout counts, action budgets, and wall-clock budgets.

The executable foundation exposes `CAPXBinaryReward.benchmark(episode,
evaluator_success)`. The evaluator result is passed only by the evaluation
plane; it is deliberately not a field in `EpisodeTrace` or `SceneSnapshot`.
