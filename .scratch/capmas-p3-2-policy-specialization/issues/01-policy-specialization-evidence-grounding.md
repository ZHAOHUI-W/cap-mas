Status: completed
Labels: ready-for-agent
Title: P3.2 typed Policy specialization, perception evidence, and auditable grounding

## Scope

Implement the P3.2 spec in the adjacent `spec.md`. Preserve CAP-X parity and
the single physical executor while making Policy strategy differences,
perception-aware arbitration, and LIBERO normalization observable.

## Acceptance Criteria

- Candidate artifacts retain raw and normalized fingerprints plus rewrite data.
- Balanced/safety/robust/efficient map to typed strategy profiles.
- A real SceneSnapshot-backed evidence provider populates perception evidence.
- Arbiter hard-gates unsafe perception and uses strategy-aware evidence scores.
- Evidence-free arbitration remains explicitly labeled as fallback.
- Existing deterministic and P3.1 tests remain green.
- Documentation describes the new contracts, scoring, and ablation seams.
