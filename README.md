# CAP-MAS

CAP-MAS is a modular multi-agent Code-as-Policy runtime for long-horizon robot manipulation. It is designed as a separate project built on the CAP-X execution and API ecosystem, while preserving enough compatibility to compare every proposed component against CAP-X.

The research contributions are **contract-driven multi-agent coordination** and
an auditable self-evolution system: Memory Skills and Robot Skills are separate,
intermediate progress is verifier-derived for learning, and evolution proceeds
sequentially with hard-case validation and rollback.

## Status

The repository contains the architecture specification and a dependency-light
runtime foundation. It also includes the first CAP-X/LIBERO integration slice:
CAP-X YAML loading, direct low-level environment construction, reuse of CAP-X's
registered API factories, typed-skill binding, an observable LIBERO verifier,
and CAP-MAS-only episode output runners. The P2.5 single-agent multi-cycle
runner now supports staged replanning, observable task goals, bounded recovery,
and cycle history. The initial GaP-inspired graph foundation now provides typed
`MissionGraph`/`SubgraphSpec` contracts, strict graph serialization, bounded-loop
validation, a deterministic fixed-graph interpreter, candidate arbitration, and a
lowering seam into the P2.5 runtime. It now also has a strict LLM graph decoder and
provider-independent Manager/Policy proposal adapters. Real LLM-driven multi-agent scheduling,
asynchronous world modeling, and self-evolution follow the milestones in
[docs/implementation-roadmap.md](docs/implementation-roadmap.md).

## V1 LIBERO smoke run

Run from the CAP-X environment that contains LIBERO and its vision/motion
dependencies:

```bash
cd /data/MLLM/wzh/agent/paper/infiAgent/workspace/cap-mas
PYTHONPATH=. ../cap-x/.venv-libero/bin/python scripts/run_libero_b0.py \
  --config-path ../cap-x/env_configs/libero/franka_libero_spatial_0.yaml \
  --output outputs/capmas_libero_b0/episode.json
```

The runner reads the CAP-X YAML, constructs only its `low_level` environment,
reuses the API name configured by the YAML through CAP-X's registry, and emits
CAP-MAS's episode JSON. It does not execute CAP-X's Python code executor. The
current V1 smoke policy targets the LIBERO Spatial task 0 pick-and-lift flow;
the policy is deterministic while the runtime and adapter seams are being
validated.

For the P2.5 multi-cycle loop:

```bash
PYTHONPATH=. ../cap-x/.venv-libero/bin/python scripts/run_libero_b1.py \
  --config-path ../cap-x/env_configs/libero/franka_libero_spatial_0_privileged.yaml \
  --output outputs/capmas_libero_b1/episode.json \
  --max-cycles 8 --max-recoveries 2 --skip-api-servers
```

This staged runner emits one bounded contract per subgoal, re-plans from each
committed scene, and stops on observable placement predicates. The existing
`run_libero_b0.py` remains the one-contract baseline.

For the deterministic B3 fixed-graph scheduler (same YAML, API factory, seed,
and observable verifier):

```bash
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. ../cap-x/.venv-libero/bin/python \
  scripts/run_libero_b3.py \
  --config-path ../cap-x/env_configs/libero/franka_libero_spatial_0_privileged.yaml \
  --output outputs/capmas_libero_b3/episode.json \
  --skip-api-servers
```

B3 is the fixed deterministic graph baseline, not yet the LLM-driven or
process-distributed system. Its episode artifact includes the graph, all
execution traces, failure artifacts, and evaluator-only success for parity
analysis.

For the P3.1 endpoint-backed LLM graph path:

```bash
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. ../cap-x/.venv-libero/bin/python \
  scripts/run_libero_b3_llm.py \
  --config-path ../cap-x/env_configs/libero/franka_libero_spatial_0_privileged.yaml \
  --api-base "$CAPMAS_LLM_API_BASE" \
  --model gpt-5.4 \
  --graph-protocol staged \
  --output outputs/capmas_libero_b3_llm/episode.json \
  --skip-api-servers
```

The API key is read from `CAPMAS_LLM_API_KEY` (or the variable named by
`--api-key-env`). Policy calls are read-only and bounded; only the selected
graph reaches the same single physical executor used by B3. The endpoint-backed
run is a separate experiment and must not overwrite the deterministic B3
artifact. Endpoint-backed artifacts record a secret-free `run_config` and one
`llm_calls` entry per Manager/Policy request, including latency, tokens, schema
mode, fallback status, and provider outcome.

Every endpoint-backed invocation also tees stdout/stderr to a per-run log. By
default the runner requests `<output>.log`; if that path already exists, it
automatically reserves a timestamp/PID-suffixed sibling instead of overwriting
the earlier run. Use `--log-file` to choose the base path. Failed invocations
additionally write `<output>.failure.json` with the secret-free configuration,
LLM call trace, and exception details.

The proposal and execution ablations are explicit:

```bash
# serial proposal compilation, fixed graph execution
... scripts/run_libero_b3_llm.py ... \
  --proposal-mode subgoal_serial --execution-mode fixed_graph

# dependency-ready proposal waves, rolling verified execution
... scripts/run_libero_b3_llm.py ... \
  --proposal-mode ready_wave --execution-mode rolling
```

`rolling` recompiles from the latest committed scene after each subgraph and
keeps the single physical executor unchanged.

If the provider rejects the large graph schema while accepting ordinary JSON,
add `--no-provider-structured-output`; local decoding and graph validation
remain enforced.

For the Phase 4 live World Model gate, run observation-only CAP-X/LIBERO
capture. This does not execute robot skills or claim task completion:

```bash
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. ../cap-x/.venv-libero/bin/python \
  scripts/run_libero_b5.py \
  --config-path ../cap-x/env_configs/libero/franka_libero_spatial_0_privileged.yaml \
  --runtime thread --skip-api-servers --max-observations 20 --target-hz 20 \
  --server-url "$CAPMAS_LLM_API_BASE" --model gpt-5.5 \
  --api-key-env CAPMAS_LLM_API_KEY \
  --output outputs/capmas_libero_b5/live.json
```

`run_libero_b5.py` probes the optional endpoint once, keeps the key out of
artifacts, and writes a unique per-run log. The live CAP-X provider currently
uses in-process artifacts, so thread mode is the live path; process mode is
covered by replay and IPC tests.

The staged protocol is the default and separates the compact Manager topology
request from direct local `SubgraphSpec` requests. Use `--graph-protocol legacy`
for the full-MissionGraph P3.1 ablation. See
[docs/staged-graph-protocol.md](docs/staged-graph-protocol.md) for the wire
contracts and matched comparison guidance.

To compare already-produced artifacts without starting either environment:

```bash
PYTHONPATH=. python scripts/compare_artifacts.py \
  --capx-trial /path/to/trial_01_sandboxrc_0_reward_0.000_taskcompleted_0 \
  --capmas-episode outputs/capmas_libero_b3/episode.json \
  --task-id libero_spatial_0 --seed 1 \
  --output outputs/parity/libero_spatial_0_seed_1.json
```

The comparison requires task and seed to be supplied explicitly; it does not
infer matching conditions from output directory names.

## Scope

- First environment: LIBERO-PRO.
- First embodiment: one Franka robot with multiple software agents.
- Primary objective: improve success-rate stability as task horizon grows.
- Secondary objectives: reduce human intervention and improve OOD generalization.
- Primary baseline: CAP-X single-agent loop with the same API backend and matched budgets.

## Design rules

1. Generated code can call only registered typed skills; it cannot access `env`, the simulator, arbitrary imports, or the skill registry.
2. The high-frequency controller never waits for an LLM or heavyweight semantic perception call.
3. Agents consume observable postconditions. Privileged completion signals are evaluator-only.
4. Skill candidates are isolated, versioned, and activated only at safe boundaries.
5. Every module is replaceable and independently ablatable.

## Documentation map

| Document | Purpose |
| --- | --- |
| [docs/research-charter.md](docs/research-charter.md) | Research question, hypotheses, scope, and contribution boundary |
| [docs/architecture.md](docs/architecture.md) | Runtime planes, agent roles, data flow, and invariants |
| [docs/graph-as-policy.md](docs/graph-as-policy.md) | GaP-inspired typed graph layer, validation, parallel candidate boundary, and status |
| [docs/gap-code-review.md](docs/gap-code-review.md) | Local GaP implementation review and CAP-MAS adoption boundaries |
| [docs/whole-system-structure.md](docs/whole-system-structure.md) | End-to-end structure, timing boundaries, state ownership, and ablation map |
| [docs/module-catalog.md](docs/module-catalog.md) | Module responsibilities, replacements, and ablations |
| [docs/contracts.md](docs/contracts.md) | State, action, verification, recovery, and communication contracts |
| [docs/llm-backend.md](docs/llm-backend.md) | LLM API adapter, structured outputs, budgets, and latency policy |
| [docs/real-time-perception.md](docs/real-time-perception.md) | Multi-rate perception and incremental 3D scene-map design |
| [docs/skill-system.md](docs/skill-system.md) | Primitive skills, candidate evolution, quarantine, and promotion |
| [docs/reward-and-rl.md](docs/reward-and-rl.md) | RL inventory, verified shaping, and three-stage Memory Controller training |
| [docs/memory-and-experience.md](docs/memory-and-experience.md) | Memory layers, provenance, hard cases, and failure accumulation |
| [docs/memory-skill-contracts.md](docs/memory-skill-contracts.md) | Memory Skill interfaces and validation rules |
| [docs/memory-skill-evolution.md](docs/memory-skill-evolution.md) | Sequential Memory Skill then Robot Skill evolution |
| [docs/capx-compatibility.md](docs/capx-compatibility.md) | CAP-X API adapter and parity requirements |
| [docs/experiments.md](docs/experiments.md) | Baselines, ablations, metrics, and evaluation protocol |
| [docs/staged-graph-protocol.md](docs/staged-graph-protocol.md) | Compact topology-to-local-graph protocol and legacy ablation |
| [docs/implementation-roadmap.md](docs/implementation-roadmap.md) | Staged implementation plan |
| [docs/adr/](docs/adr/) | Architecture decision records |
| [docs/glossary.md](docs/glossary.md) | Stable terminology |
| [docs/design-audit.md](docs/design-audit.md) | Defined decisions, deferred decisions, and implementation risks |

## Reference materials

The design is informed by CAP-X, Playful Agentic Robot Learning, MetaGen, ENPIRE, ReVeal, SEEA-R1, MemSkill, SkillRL, and CoEvoSkills. Local paper copies remain in the CAP-X workspace under `docs/papers`; claims must be checked against the primary paper before publication.
