# Project Structure

The implementation should preserve the separation between real-time execution, asynchronous state estimation, agent reasoning, and research evaluation.

~~~text
cap-mas/
├── capmas/
│   ├── contracts/
│   │   ├── core.py
│   │   ├── scene.py
│   │   ├── action.py
│   │   ├── verification.py
│   │   ├── failures.py
│   │   ├── trace.py
│   │   ├── memory.py
│   │   └── agent.py
│   ├── runtime/
│   │   ├── scheduler.py
│   │   ├── event_bus.py
│   │   ├── state_store.py
│   │   ├── action_lease.py
│   │   └── checkpoints.py
│   ├── agents/
│   │   ├── base.py
│   │   ├── manager.py
│   │   ├── perception.py
│   │   ├── policy.py
│   │   ├── verifier.py
│   │   ├── recovery.py
│   │   └── critic.py
│   ├── llm/
│   │   ├── protocol.py
│   │   ├── capx_compatible.py
│   │   ├── openai_compatible.py
│   │   ├── local.py
│   │   ├── mock.py
│   │   └── replay.py
│   ├── perception/
│   │   ├── sensor_sync.py
│   │   ├── geometry.py
│   │   ├── tracking.py
│   │   ├── local_map.py
│   │   └── semantic_triggers.py
│   ├── skills/
│   │   ├── schema.py
│   │   ├── registry.py
│   │   ├── validator.py
│   │   ├── quarantine.py
│   │   ├── evolution.py
│   │   └── capx_adapter.py
│   ├── execution/
│   │   ├── typed_executor.py
│   │   ├── sandbox.py
│   │   ├── safety_monitor.py
│   │   └── trace.py
│   ├── verification/
│   │   ├── predicates.py
│   │   ├── preconditions.py
│   │   ├── postconditions.py
│   │   └── freshness.py
│   ├── backends/
│   │   ├── protocol.py
│   │   ├── capx_legacy.py
│   │   ├── capx_typed.py
│   │   └── libero.py
│   └── evaluation/
│       ├── runner.py
│       ├── budgets.py
│       ├── metrics.py
│       └── failure_analysis.py
├── configs/
├── tests/
│   ├── unit/
│   ├── contract/
│   ├── integration/
│   ├── realtime/
│   └── regression/
├── scripts/
├── docs/
└── pyproject.toml
~~~

The foundation prototype currently implements contracts, the in-memory state
store, action leases, one-cycle orchestration, CAP-X typed bindings,
artifactized observations, Memory Controller protocols, an in-memory memory
store, the LLM protocol, and the reward boundary. The remaining tree entries
are reserved extension points and are not claimed as implemented.

## Dependency direction

~~~text
contracts
  <- runtime, perception, skills, execution, verification, agents

backends -> skills -> execution
perception -> contracts
agents -> contracts + runtime + perception + skills + verification
evaluation -> all public interfaces, never private simulator state
~~~

The control process may depend on contracts, execution, and safety-critical geometry, but must not import LLM clients, agent prompts, or experiment reporting code.

## Module ownership

| Area | Owns | Must not own |
| --- | --- | --- |
| contracts | Schemas and validation primitives | Scheduling or model calls |
| llm | Model/API calls, parsing, usage, and deadlines | Robot execution or state mutation |
| runtime | Ordering, leases, epochs, checkpoints | Robot-specific perception |
| perception | Sensor fusion, map, tracks, confidence | Task decomposition |
| agents | Proposals and decisions as artifacts | Direct actuator access |
| skills | Typed robot capabilities and versions | Global task policy |
| execution | Bounded invocation and safety | LLM generation |
| verification | Predicates and evidence | Rewriting history |
| backends | CAP-X/LIBERO integration | Experiment conclusions |
| evaluation | Metrics and comparisons | Runtime authority |
