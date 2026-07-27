# Memory Skill Contracts

## 1. Definition

A Memory Skill is a typed, versioned transformation over a trace span and
retrieved memory context. It does not control the robot and cannot access the
simulator directly.

```text
MemorySkill = identity + typed input + applicability conditions
              + structured output schema + conflict policy
              + evidence requirements + tests + version
```

## 2. Initial Memory Skill set

| Memory Skill | Input | Output | Typical trigger |
| --- | --- | --- | --- |
| `extract_object_state` | scene/trace span | object-state fact with confidence | state ambiguity |
| `extract_failure_cause` | contract, trace, verifier report | typed failure diagnosis | failed action |
| `derive_recovery_rule` | failure plus recovery outcome | conditional recovery rule | successful recovery |
| `consolidate_skill_outcome` | repeated skill cases | updated statistics/boundary | enough evidence |
| `invalidate_contradictory_memory` | conflicting evidence | invalidation record | contradiction |
| `forget_stale_memory` | item plus freshness policy | retirement proposal | TTL or drift |
| `skip` | trace span | no update with reason | low evidence/confidence |

## 3. Contract example

```json
{
  "skill_id": "derive_recovery_rule",
  "version": "0.1.0",
  "input_schema": ["failure_case", "recovery_trace"],
  "preconditions": ["failure_case.has_verifier_evidence == true"],
  "output_schema": {
    "condition": "typed_predicate",
    "action_hint": "recovery_contract_reference",
    "confidence": "float[0,1]",
    "evidence_ids": "list[artifact_id]",
    "ttl": "duration"
  },
  "conflict_policy": "downgrade_or_invalidate",
  "side_effects": "persistent_memory_proposal_only"
}
```

The output is a proposal, not an active fact. It is committed only after
schema, provenance, contradiction, replay, and confidence checks.

## 4. Validation gates

Every candidate Memory Skill must pass:

1. schema and type checks;
2. deterministic replay on its source trace;
3. no-provenance and contradiction tests;
4. idempotence or bounded-duplication tests;
5. hard-case and regression tests;
6. OOD applicability tests;
7. latency and token-budget checks.

The active Memory Skill Bank and the active Robot Skill Registry are separate
snapshots. A Memory Skill cannot call a Robot Skill; it may only emit a typed
reference used by a later planner or recovery process.

## 5. Controller interface

```python
class MemoryController(Protocol):
    def select(
        self,
        context: MemoryContext,
        candidates: list[MemorySkillRef],
        budget: MemoryBudget,
    ) -> MemorySelection: ...
```

```python
class MemoryExecutor(Protocol):
    def apply(
        self,
        selection: MemorySelection,
        trace_span: TraceSpan,
    ) -> MemoryUpdate: ...
```

Neither interface receives an actuator lease or an environment handle.
