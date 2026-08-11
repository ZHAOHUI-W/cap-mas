# ADR-0014: Qualified Evidence Calibration and Immutable Snapshot Activation

## Status

Proposed. The design is approved for repository review; implementation starts
only after the P5.6 spec is reviewed and an implementation plan is approved.

## Context

P5.5 provides candidate-conditioned geometry, verifier, rehearsal, and frozen
OOD evidence, but most formal decisions still use `evidence_tie_break`, and
the OOD split is evaluation metadata. Raw evidence dimensions are correlated,
and an unselected candidate has no physical task outcome. A direct learned
weight would therefore risk double-counting, leakage, false labels, and
unsafe promotion.

The two target families `spatial-0` and `goal-1` also have zero evaluator
success in the corrected ten-seed P5.5 suite. A calibration model cannot repair
that capability gap. P5.6 needs a family-level readiness gate and an explicit
abstention path.

## Decision

1. The online primary output is `rank_score`; `success_probability` is emitted
   only for a qualified, non-abstained family and immutable snapshot.
2. Physical `task_success` is the only primary supervised label. Graph
   completion, observable verifier success, and horizon are separate
   diagnostics. Unselected and rejected candidates are not physical failures.
3. Data is separated into Tier A physical outcomes, Tier B isolated rehearsal
   outcomes, and Tier C unlabeled evidence. Tier B never masquerades as Tier A.
4. A deterministic correlation-group reducer runs before a constrained logistic
   model and isotonic transform. Support/evidence coefficients are
   non-negative; latency, recovery, collision, and safety-risk coefficients are
   non-positive. Unknown remains explicit.
5. The initial active model has `ood_weight=0`. OOD evidence remains shadow
   metadata until independent counterfactual multi-layout support and a new
   snapshot qualify it.
6. Calibration is family-scoped. A family requires a fixed ten-seed capability
   gate, then at least 20 Tier A outcomes with at least five positive and five
   negative labels before it can publish a qualified probability.
7. Calibration abstention falls back through safety hard gates, qualified
   calibration, fixed-weight evidence scoring, deterministic evidence tie-break,
   and finally confidence fallback. `evidence_tie_break` and
   `confidence_fallback` remain distinct observables.
8. Calibration snapshots are immutable and content-addressed. Activation is
   atomic, each episode pins one snapshot, and rollback is explicit and
   auditable. Background training cannot mutate active or pinned snapshots.
9. No global online calibrated Arbiter is enabled until target families pass
   capability gates. A bounded canary is safety-only and cannot support a
   downstream improvement claim.
10. P5.6.0 is diagnostic-only. Task mapping, prompt, skill-argument, and
    physical-parameter repair is isolated in P5.3.2 and blocks only the
    affected family's promotion.
11. Task horizon is bucketed by action-bearing subgraphs on the planned
    critical path. Checkpoint-only subgraphs and typed skill-call counts remain
    diagnostics and do not inflate horizon.

## Consequences

Positive:

- The Arbiter can use evidence without treating correlated signals as
  independent votes.
- Probability claims have explicit data, family, schema, and uncertainty
  qualifications.
- Ineligible families remain operational through a known fixed-weight fallback.
- Snapshot pinning makes an episode reproducible across background training and
  later activations.
- OOD evidence cannot silently turn an easier OOD layout into an online weight.

Negative:

- Existing P5.5 physical results need a compatibility audit and potentially
  pre-registered fresh seed blocks before they become P5.6 Tier A data.
- Zero-success families require capability debugging before calibration can
  be promoted for them.
- Group policy and snapshot lifecycle add contracts and artifact overhead.
- A calibrated probability may be unavailable even when fixed-weight ranking
  can still select a candidate.

## Alternatives rejected

- Fit one global model over all task families.
- Use P5.5 OOD outcomes as positive online features immediately.
- Treat every unselected candidate as `task_success=false`.
- Sum verifier and rehearsal success as independent evidence.
- Use `max_steps` as the task horizon.
- Mutate a live calibration model in place or change a pinned episode's model.
- Let calibration bypass existing safety, freshness, schema, or lease gates.
- Repair zero-success task families inside the calibration work package.
- Mark every semantically single benchmark instruction H1 without inspecting
  its action-bearing Mission Graph structure.
