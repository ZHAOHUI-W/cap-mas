# Research Charter

## 1. Problem

CAP-X demonstrates that a coding agent can control robot manipulation through executable Python and test-time feedback. Its core execution loop remains single-agent and its physical state transitions are not represented by a typed, independently verifiable contract. In long-horizon tasks, this creates failure modes including stale plans, accumulated state error, false completion, and expensive regeneration after irreversible actions.

CAP-MAS studies whether explicit multi-agent coordination can make long-horizon Code-as-Policy execution more stable without relying on uncontrolled agent-to-agent chat or privileged simulator state.

## 2. Primary research question

> Under matched model-call, action, and interaction budgets, does contract-driven multi-agent coordination improve CAP-X task success-rate stability as the number of sequential subgoals increases, compared with CAP-X's single-agent loop?

## 3. Secondary questions

1. Which contract mechanisms matter most: versioned state, pre/postconditions, verification, action leases, or recovery policies?
2. Can asynchronous scene estimation preserve control-loop deadlines while providing sufficiently fresh 3D state to the multi-agent planner?
3. Does verified intermediate feedback improve long-horizon credit assignment
   while preserving CAP-X's binary benchmark definition?
4. Does Memory Skill evolution improve diagnosis, recovery, and OOD performance
   before Robot Skill evolution is applied?
5. Does sequential Memory Skill then Robot Skill evolution improve performance
   without destabilizing active behavior or hiding regressions?
6. Does adaptive sparse communication improve success-cost trade-offs over
   fixed multi-agent topologies?

## 4. Hypotheses

### H1 — Horizon stability

CAP-MAS has a smaller decline in success rate as sequential subgoal count grows than CAP-X under matched budgets.

### H2 — State consistency

Version checks and postcondition verification reduce stale-action and false-completion failures.

### H3 — Recovery

Bounded action chunks and explicit recovery policies increase recovery success after induced execution failures.

### H4 — Real-time compatibility

Separating fast geometric estimation from event-triggered semantic perception keeps control deadline misses near the non-agent baseline.

### H5 — Controlled memory evolution

Memory Skill candidates improve failure diagnosis, recovery, and OOD performance
after promotion while avoiding regression on the in-distribution suite.

### H6 — Sequential skill evolution

Evolving Memory Skills first and Robot Skills second gives better attribution
and lower regression than a joint-evolution condition at matched evaluation
budgets.

## 5. Contribution boundary

The contributions are evaluated as two connected but separable mechanisms:

1. a contract-driven multi-agent coordination runtime; and
2. a separated Memory Skill/Robot Skill system with verified intermediate
   feedback and sequential, hard-case-driven evolution.

Adaptive topology and model distillation remain supporting mechanisms. The
experiments must still isolate coordination, memory, reward shaping, and skill
evolution rather than attributing every gain to the whole stack.

## 6. Scope

### In scope

- CAP-X-compatible Code-as-Policy execution.
- LIBERO-PRO first, with later Robosuite and BEHAVIOR adapters.
- One Franka robot controlled by multiple software agents.
- Observable postcondition verification.
- Long-horizon task composition, failure recovery, OOD layouts and object/task variants.
- Module-level ablations against CAP-X.

### Out of scope for v0

- Multiple physical robots.
- End-to-end VLA replacement.
- Fully autonomous skill promotion during an irreversible action.
- Claiming real-robot safety from simulation-only evidence.
- Using privileged simulator state as the agent's normal observation.

## 7. Fair-comparison rules

- Keep the CAP-X API backend available in both systems.
- Report the same model family, temperature, number of trials, and seed set where possible.
- Match total model calls, token budget, action budget, and wall-clock budget in primary comparisons.
- Report both evaluator-only privileged success and agent-visible verification outcomes.
- Run CAP-X, fixed-graph MAS, and CAP-MAS under identical task splits and initial states.
