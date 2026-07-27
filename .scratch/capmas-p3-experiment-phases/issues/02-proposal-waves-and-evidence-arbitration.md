Status: ready-for-agent
Type: task
Labels: ready-for-agent

# Implement Policy proposal waves and evidence-aware arbitration

Implement P3.1b/P3.1c from `spec.md`: dependency-ready proposal waves,
scene-aware candidate rewriting, explicit Policy strategy profiles, injected
candidate evidence, and auditable Arbiter selection basis.

## Acceptance Criteria

- Serial and `ready_wave` proposal modes are selectable without changing the
  physical executor.
- Only dependency-ready subgoals share a bounded proposal worker pool.
- Scene-aware candidate rewriting receives the current `SceneSnapshot`.
- Policy strategy profiles are visible in the request and agent identity.
- Evidence-backed selection is labeled `evidence_score`.
- Evidence-free selection is labeled `confidence_fallback`; equal candidates
  with usable evidence are labeled `evidence_tie_break`.
- Matching scheduler tests cover concurrency, evidence ranking, and scene-aware
  grounding.
