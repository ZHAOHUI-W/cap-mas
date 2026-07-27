# CAP-MAS Staged Multi-Policy Experiment Phases

## Problem Statement

The current staged Multi-Policy LIBERO path has passed a strict structured-output
smoke run, but the artifact does not contain enough information to establish
which configuration produced it, how much LLM time it consumed, whether a
provider fallback was used, or whether the Arbiter selected a candidate using
actual evidence. The scheduler also fans out candidates only within one
subgoal, while subgoals are compiled serially. Consequently, latency,
parallelism, and multi-agent quality cannot yet be measured fairly.

## Solution

Implement the staged experiments in controlled vertical phases:

1. **P3.1a Observability** — record run configuration and one normalized
   `LLMCallTrace` per Manager/Policy request, then include them in the episode
   artifact.
2. **P3.1b Candidate fan-out A/B** — expose and measure bounded Policy fan-out
   and a dependency-aware proposal wave without changing physical execution.
3. **P3.1c Evidence-aware multi-policy arbitration** — make candidate evidence,
   disagreement, and agent strategy identity explicit; never present a
   deterministic tie-break as a quality decision.
4. **P3.2 Rolling staged execution** — compile only ready subgraphs, execute one
   verified subgraph, refresh the `SceneSnapshot`, and replan stale suffixes.
5. **P3.3 Distributed rehearsal and optional parallel execution** — add process
   workers for simulator rehearsal and introduce parallel execution only for
   resource-disjoint graph branches with explicit joins.

The first implementation slice is P3.1a. Later slices must preserve the B3
deterministic baseline, CAP-X API compatibility, strict structured-output mode,
and the single physical executor invariant.

## User Stories

1. As an experimenter, I want the artifact to record the exact protocol and
   model, so that results are reproducible.
2. As an experimenter, I want the artifact to record Policy-agent count and
   worker count, so that parallelism claims are auditable.
3. As a researcher, I want every LLM request to expose latency and token usage,
   so that compile cost can be separated from physical execution cost.
4. As a researcher, I want schema mode and fallback status recorded, so that
   strict and compatibility experiments are not mixed.
5. As a maintainer, I want failed LLM calls recorded without exposing API keys,
   so that provider and retry failures remain diagnosable.
6. As an Arbiter, I want candidate evidence to be explicit, so that a tie-break
   is not mistaken for quality ranking.
7. As a scheduler, I want proposal fan-out to be bounded, so that provider rate
   limits and robot execution authority remain safe.
8. As a long-horizon controller, I want stale suffixes to be replanned after a
   verified state transition, so that early plans do not silently control later
   actions.
9. As a robot runtime, I want physical execution to remain single-owner, so
   that planning parallelism never becomes actuator contention.
10. As an evaluator, I want phase-specific artifacts and matched baselines, so
    that each contribution can be ablated independently.

## Implementation Decisions

- The public observability seams are the LLM client, the episode artifact, and
  the scheduler's compile result. Internal provider or thread-pool details are
  not test seams.
- `LLMCallTrace` is an immutable, JSON-safe record containing request identity,
  agent role, model, status, latency, token usage, provider status, request
  attempts, schema mode, schema hash, fallback status, and sanitized error
  information. It never contains an API key or authorization header.
- `LLMTraceCollector` is a thread-safe sink used by the client and snapshots
  traces in deterministic start-time/request-id order.
- `ExperimentRunConfig` records protocol, seed, task, model, Policy count,
  worker count, deadlines, output budget, retry budgets, schema mode, and
  fallback policy.
- A successful provider schema response is distinct from local JSON validation
  fallback. The artifact records both explicitly.
- The first phase does not change candidate selection or execute Policy calls
  across independent subgoals. Those changes belong to later A/B phases.
- The deterministic B3 and CAP-X artifacts remain separate and are never
  overwritten by LLM experiment output.

## Testing Decisions

- Test only the public LLM client and collector behavior: successful calls,
  schema fallback, retry/failure recording, and deterministic trace ordering.
- Test artifact assembly through the runner-facing serialization boundary rather
  than private client state.
- Reuse the existing transport-injection tests in `test_llm_backend.py` and
  scheduler seam tests in `test_llm_scheduler.py`.
- Each new behavior follows red → minimal implementation → green before the
  next behavior is added.

## Out of Scope

- Changing the physical executor or robot control loop.
- Introducing true parallel robot action execution.
- Adding LLM Recovery/Monitor roles.
- Claiming improved success rate from a single smoke run.
- Persisting endpoint URLs, API keys, raw prompts, or raw model responses in
  episode artifacts.

## Further Notes

The phase exit condition is not merely that a trace exists. The artifact must
contain enough information to reproduce the run and compare strict-schema,
fallback, worker-count, and Policy-count ablations under matched task seeds.
