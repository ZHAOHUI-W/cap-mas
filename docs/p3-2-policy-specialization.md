# P3.2 Policy Specialization, Perception Evidence, and Grounding

## Closure status

The small P3.2 closure is complete. Scheduler-created candidates no longer
receive a synthetic `confidence=0.5` prior. Confidence remains an optional
legacy value for evidence-free ablations only. When typed evidence is present,
the Arbiter ranks candidates using evidence dimensions and excludes that
legacy value from the quality score.

Evidence is bound to the `SceneSnapshot.scene_version` from which it was
computed. A mismatched evidence version is rejected before arbitration. The
artifact now distinguishes:

- `evidence_score`: evidence exists and the highest score is unique;
- `evidence_tie_break`: evidence exists but scores are equal, so a stable
  structural ordering chooses the winner;
- `confidence_fallback`: no candidate evidence exists and legacy confidence is
  the only ranking signal;
- `hard_gate`: all otherwise valid candidates were rejected by an evidence or
  perception gate.

This closure makes the arbitration claim precise. It does not claim that the
current SceneSnapshot provider can distinguish two candidates that reference
the same tracks and have the same observable geometry.

## Motivation

P3.1c established that multiple Policy Agents can run concurrently and carry
different strategy names. It did not yet establish that the strategies affect
candidate quality: all candidates used the same model/schema, candidate
confidence defaulted to `0.5`, and the LIBERO runner did not attach a real
evidence provider. Candidate normalization also happened before arbitration,
so approximate model geometry could disappear without an artifact explaining
why.

P3.2 separates three questions:

```text
Did Policy intent differ?
        ↓ raw candidate fingerprint
Did safety normalization change it?
        ↓ rewrite report and normalized fingerprint
Which feasible candidate is better for this strategy and scene?
        ↓ perception-aware Arbiter score
```

## Typed strategy profile

`StrategyProfile` is a runtime contract shared by the local Policy request
builder and the Arbiter. The existing names remain stable:

| Profile | Main priority | Typical gate/weight |
| --- | --- | --- |
| balanced | feasibility and moderate cost | balanced perception and declared evidence |
| safety | fresh, unambiguous, high-confidence geometry | strong perception weight and gates |
| robust | verification and OOD tolerance | stronger rehearsal/OOD terms |
| efficient | short and low-latency execution | stronger latency term; confidence only in fallback |

The profile is serialized into the Policy user payload as structured data.
Prompt language remains useful for explanation, but it is no longer the only
semantic difference: Arbiter thresholds and score weights consume the same
typed profile.

## Candidate lifecycle

Every scheduler-created candidate has two views:

```text
raw_subgraph       = Policy output before scene/safety rewriting
subgraph           = normalized graph used for static and skill validation
rewrite_report     = fingerprints, changed flag, bounded edit metadata
strategy           = typed profile name
```

The normalized graph remains the only graph that can reach the executor. The
raw view is retained for disagreement measurement and artifact analysis. A
candidate that becomes identical to another after grounding is therefore
reported as normalization convergence rather than silently appearing to have
been generated identically.

## Perception evidence

The LIBERO evidence provider is read-only and consumes the committed
`SceneSnapshot`. It reports scene freshness, scene confidence, target
visibility, target track confidence, identity confidence from ambiguity state,
pose reliability, and evidence references.

Rehearsal and OOD fields are not fabricated. `available_metrics` records which
dimensions are present, and the Arbiter excludes unavailable dimensions from
the score. This prevents an online perception-only run from being penalized as
if it had a zero rehearsal success rate.

The Arbiter first applies evidence-version and perception gates when evidence
is available. It then computes a profile-aware score over the available
dimensions; legacy candidate confidence is excluded in this mode. If no
evidence is attached, the result remains explicitly labeled as a confidence
fallback.

## Grounding and repair boundary

LIBERO grasp repair and target-pose grounding remain required safety
canonicalization. They are not allowed to become an unobservable second Policy
Agent. The scheduler records the raw candidate before invoking the existing
rewriter and records the normalized candidate after it. The selected normalized
graph is grounded/repaired and validated again before physical execution.

This preserves CAP-X compatibility while making raw candidate diversity,
normalized candidate diversity, rewrite count, strategy-specific score
breakdown, and evidence-backed selection rate measurable.

## Deferred evidence work

The following work is intentionally deferred to Phase 5, after the world-model
and experience boundaries are stable:

- candidate-specific static verifier coverage and dynamic postcondition score;
- action-conditioned geometric reachability, grasp quality, clearance, and
  collision evidence;
- CAP-X/MuJoCo process rehearsal with matched initial states and multiple
  seeds;
- frozen OOD replay suites, confidence intervals, and test-leakage controls;
- asynchronous evidence caching and invalidation after scene refresh;
- calibrated evidence aggregation that avoids double-counting correlated
  perception, verifier, rehearsal, and OOD signals.

These are not represented as synthetic zero values in P3.2. The current
provider exposes only SceneSnapshot-backed perception dimensions and records
the source scene version.

The implementation handoff for these deferred items is maintained in
[`phase5-evidence-evolution.md`](phase5-evidence-evolution.md). The existing
`ProcessRehearsalPool` is only a serializable offline boundary; it is not yet
connected to the online LIBERO candidate provider or the live executor.

## Ablations and exit condition

The minimum ablations are homogeneous balanced Policies, heterogeneous
Prompt-only profiles, typed profiles with perception evidence, and a diagnosis
run that reports grounding/repair metadata. P3.2 is not considered a quality
result merely because LIBERO succeeds. The closure artifact must show
optional/null scheduler confidence, explicit selection basis, scene-bound
evidence, and raw/normalized fingerprints. Candidate-specific dynamic
evidence, simulator rehearsal, OOD replay, separate model checkpoints, and
physical parallel execution remain later phases.
