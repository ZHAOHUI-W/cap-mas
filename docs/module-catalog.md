# Module Catalog

Every module has a stable interface, a CAP-X replacement, and an ablation flag. The implementation should make these substitutions configuration-driven.

| Module | CAP-MAS implementation | CAP-X comparison | Ablation |
| --- | --- | --- | --- |
| Agent scheduler | Event-driven Manager and sparse graph | CAP-X single-agent loop | `scheduler=capx_single` |
| Graph policy layer | Typed `MissionGraph`/`SubgraphSpec` plus static `GraphValidator` | CAP-X free-form policy code | `graph=disabled` |
| State store | Versioned blackboard and snapshot store | Prompt/history state | `state_store=unversioned` |
| Contract validator | Typed schema, limits, preconditions | Markdown code extraction | `validator=syntax_only` |
| Policy generator | Code/skill graph proposal | CAP-X code generation | `policy=capx_generator` |
| LLM backend | CAP-X-compatible, local, mock, or replay adapter | CAP-X LLM client | `llm_backend=capx_compatible/local/mock` |
| Verifier | Precondition, invariant, postcondition checks | Reward/task signal after execution | `verifier=none` |
| Actuator arbitration | Single action lease | Implicit sequential execution | `lease=disabled` |
| Executor | Bounded typed skill execution | CAP-X `exec` path | `executor=capx_exec` |
| Scene estimator | Fast geometric incremental map | CAP-X observation callback | `scene_estimator=capx_observation` |
| Semantic perception | Event-triggered Perception Agent | CAP-X VLM visual differencing | `semantic_perception=capx_vdm` |
| Recovery | Failure taxonomy and compensating actions | Regenerate future code | `recovery=regenerate` |
| Skill registry | Versioned active/quarantine registries | CAP-X evolving skill library | `skill_registry=capx_library` |
| Topology controller | Event-triggered sparse graph edits | Fixed single loop or fixed MAS graph | `topology=fixed` |
| Memory | Versioned episode, experience, semantic, and procedural memory | CAP-X prompt/history/library | `memory=none/episodic` |
| Memory Controller | Top-K Memory Skill selection with bounded budget | CAP-X fixed prompt/history retrieval | `memory_controller=rules` |
| Memory Executor/Designer | Typed MemoryUpdates, hard-case design, validation | CAP-X no persistent memory evolution | `memory_evolution=disabled` |
| Reward engine | CAP-X binary benchmark score plus verified shaping for learning | CAP-X evaluator reward | `reward=binary_only` |

## Module contract

Each module must document:

- input and output schemas;
- timing budget and blocking behavior;
- authority and denied capabilities;
- failure modes;
- deterministic test fixture;
- CAP-X adapter;
- ablation configuration;
- metrics emitted.

## Recommended implementation order

1. `capx_compat` and baseline runner.
2. `llm_backend`, `contracts`, and `state_store`.
3. `executor` and action leases.
4. `verifier` and recovery.
5. `scene_estimator` and asynchronous map.
6. agent scheduler and communication bus.
7. separate Robot Skill and Memory Skill registries.
8. memory controller and cross-task experience.
9. sequential evolution and adaptive topology.
