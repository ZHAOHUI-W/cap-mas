Status: ready-for-agent
Type: task
Labels: ready-for-agent

# Implement CAP-MAS P2.5 Single-Agent Multi-Cycle Contract Loop

Implement the P2.5 specification in `spec.md`: a reusable single-agent
multi-cycle runner, bounded recovery, a deterministic LIBERO task-0 staged
policy, and a CAP-MAS-only multi-cycle episode artifact. Preserve the existing
one-cycle B0 runner and CAP-X compatibility boundary.

## Acceptance Criteria

- The public multi-cycle runner can execute at least two ActionContracts in one
  episode and pass the latest committed SceneSnapshot to the next policy step.
- Each cycle retains its ExecutionTrace and VerificationResult in the final
  EpisodeTrace.
- A postcondition or execution failure invokes a Recovery Agent when the retry
  budget allows it, and a successful recovery continues the episode.
- Failed physical history is never removed or rewritten.
- Maximum cycles, maximum recovery attempts, and policy exhaustion terminate
  deterministically with structured stop reasons.
- Task-level termination uses observable predicates and does not expose
  evaluator-only completion to the policy or recovery collaborators.
- LIBERO task 0 can run through multiple bounded contracts using the existing
  CAP-X YAML/API factory and produce a CAP-MAS-only JSON artifact.
- Existing tests continue to pass and new public-seam tests cover success,
  recovery, budget exhaustion, and policy completion.
