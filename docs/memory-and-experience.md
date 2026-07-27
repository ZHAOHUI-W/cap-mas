# Memory and Experience Architecture

Memory and Skill are related but different objects. A Robot Skill changes the
physical world. A Memory Skill changes how an execution trace becomes reusable
information. They therefore have separate namespaces, registries, contracts,
tests, and promotion histories.

## 1. Memory layers

The storage layers describe lifecycle, while the content types describe what
is stored:

| Layer | Lifetime | Content |
| --- | --- | --- |
| Episode Working Memory | One episode | observations, contracts, traces, failures, recovery attempts |
| Experience Memory | Cross-episode | successful/failed cases, skill statistics, recovery outcomes, hard cases |
| Semantic/Procedural Memory | Long-lived | generalized task patterns, physical priors, verified failure rules, skill boundaries |
| Memory Skill Bank | Versioned capability bank | procedures for extracting, consolidating, validating, revising, or forgetting memory |

Episode Working Memory is append-only during execution. Experience and
semantic memory are updated asynchronously after evidence validation. The
Memory Skill Bank is immutable within an active snapshot.

## 2. Memory item schema

Every persistent item must include:

```text
memory_id, memory_version, source_episode_ids, source_trace_ids,
task_family, scene_context, subgoal_context, skill_versions,
outcome, failure_class, applicability_conditions, evidence_count,
confidence, contradiction_set, freshness_or_ttl, status,
created_at, last_validated_at
```

`status` is one of `candidate`, `active`, `stale`, `contradicted`, or
`retired`. A memory item without provenance cannot be used as promotion
evidence.

## 3. Read/write authority

| Component | Read | Write |
| --- | --- | --- |
| Mission/Policy/Recovery Agents | active relevant memories | episode annotations only |
| Typed Executor | applicable skill metadata | execution trace |
| Critic | episode and experience memory | candidate diagnoses |
| Memory Controller | retrieved candidates and context | selection record only |
| Memory Executor | selected Memory Skills and trace span | proposed MemoryUpdate |
| Memory Designer | hard-case buffer and bank snapshot | quarantined candidates |
| Promotion Manager | validation reports and snapshots | atomic active snapshot |

The Memory Controller is not allowed to silently mutate persistent memory.
All writes pass through a Memory Executor, provenance validator, and conflict
resolver.

## 4. Trace-to-memory loop

```text
Execution Trace
  -> retrieve related memories
  -> Memory Controller selects Top-K Memory Skills
  -> Memory Executor emits structured MemoryUpdate
  -> provenance/confidence/conflict/TTL checks
  -> commit to episode or persistent memory
  -> downstream outcome evaluation
  -> hard-case buffer and controller feedback
```

A failed action produces both a failure record and an attempted recovery
record. A recovery that succeeds is not automatically generalized: it becomes
promotion evidence only after replay and applicability checks.

## 5. Failure experience accumulation

For each failure, the system stores the smallest reproducible span: parent
scene version, contract, skill versions, perception evidence, failure class,
recovery attempts, postcondition results, and resource costs. Similar cases
are clustered by task, scene, contract, and failure signature. Representative
hard cases are retained in a bounded buffer so that frequent easy successes do
not erase rare catastrophic failures.

Repeated contradictory evidence lowers confidence and can mark a memory stale
or contradicted. No agent is allowed to delete the source trace; forgetting
only removes an item from active retrieval after its provenance remains
archived.

## 6. Separation from the robot execution path

Memory updates are asynchronous and take effect only at an episode boundary
or declared subgoal checkpoint. A memory update cannot alter the meaning of an
active Robot Skill, invalidate an already leased action, or bypass the
verifier. This is required both for physical stability and for causal
evaluation of the memory contribution.
