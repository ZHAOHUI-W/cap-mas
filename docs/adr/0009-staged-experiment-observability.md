# ADR 0009: Observable and Phase-Separated Multi-Policy Experiments

## Status

Accepted

## Context

The staged graph path now supports real Manager and Policy proposals, strict
provider schemas, deterministic arbitration, and one physical executor. A
single successful episode is not enough to compare latency, fallback behavior,
candidate quality, or parallelism. Without run-level and request-level
provenance, later ablations would mix configuration changes with algorithmic
changes.

## Decision

Separate the research progression into observable phases. First add immutable
run and LLM-call telemetry without changing scheduling. Then measure bounded
candidate fan-out, add evidence-aware arbitration, introduce rolling replanning,
and only later add process-level rehearsal or resource-disjoint execution.

Every LLM request emits a sanitized `LLMCallTrace`; every experiment artifact
contains an `ExperimentRunConfig` and the collected traces. Strict provider
structured output and local validation fallback are separate, explicitly
recorded modes. The single physical executor and `ActionLease` remain the only
actuator authority in every phase.

Rolling execution is allowed to trim a completed graph prefix only when the
next subgraph ID is still present in the newly compiled graph. A missing or
changed next ID is a fail-closed planning error. Candidate grounding may use a
scene-aware rewriter so that a replan cannot accidentally reuse initial-scene
geometry.

## Consequences

Positive:

- Latency and token budgets can be compared at Manager, Policy, compile, and
  execution boundaries.
- Strict-schema and fallback runs cannot be silently conflated.
- Concurrency remains a controlled, independently ablatable change.
- API keys, prompts, and raw model responses are not persisted.

Negative:

- Episode artifacts become larger.
- All future runners must populate the experiment configuration boundary.
- A successful artifact without request traces is incomplete for performance
  claims, even if its evaluator score is successful.
