# CAP-MAS P3.2 Policy Specialization, Perception Evidence, and Non-Collapsing Grounding

## Problem Statement

The staged multi-policy LIBERO path can invoke multiple Policy Agents with
different named strategies, but the current distinction is mostly a soft
prompt instruction. All policies share the same model, scene, schema, and
deterministic decoding defaults. The scheduler assigns the same default
confidence to every candidate, and the runner does not attach a live candidate
evidence provider. As a result, the Arbiter often performs a stable confidence
tie-break rather than a perception- and execution-aware decision.

Candidate normalization also happens before arbitration. LIBERO grasp repair
inserts missing motion steps and scene grounding replaces approximate placement
poses with the same target-track pose. These transformations are required for
safe execution, but the raw Policy intent and the normalization cost are not
currently preserved. The system therefore cannot distinguish deliberate Policy
convergence from diversity that was erased by post-processing.

## Solution

Add a typed Policy specialization and evidence pipeline while preserving the
single physical executor, CAP-X skill/API compatibility, staged graph protocol,
and observable verifier boundary.

Each candidate will retain both its raw Policy proposal and its executable
normalized subgraph. A rewrite report will record whether normalization changed
the proposal, the before/after fingerprints, and the bounded repair/grounding
edits. The Arbiter will validate the normalized graph but can inspect the raw
intent and rewrite metadata when ranking candidates.

Policy strategies will be represented by typed profiles rather than only prose.
Profiles define explicit risk, perception, verification, latency, and recovery
requirements/weights. Existing balanced, safety, robust, and efficient names
remain compatible and are mapped to stable defaults.

Candidate evidence will include perception quality derived from the immutable
SceneSnapshot, including scene confidence/freshness, target visibility and
track confidence, identity ambiguity, and pose/grounding reliability. Evidence
is produced by CAP-MAS infrastructure, not self-reported by the LLM. Arbiter
selection will use hard rejection gates for unsafe or stale perception and a
strategy-aware soft score for feasible candidates. Missing evidence remains
explicit and must not be silently interpreted as a quality score.

LIBERO grounding and grasp repair remain safety boundaries. They will be
observable, bounded, and applied through a shadow normalized view for candidate
evaluation. The selected candidate is grounded/repaired again before physical
execution and is revalidated. Policy-specific symbolic intent such as target,
approach margin, checkpoint coverage, and recovery policy is preserved instead
of being replaced by a single concrete pose representation.

## User Stories

1. As a Policy Agent, I want a typed strategy profile, so that safety and
   robustness requirements are executable constraints rather than suggestions.
2. As a Policy Agent, I want to preserve symbolic action intent, so that scene
   grounding resolves geometry without deleting my approach and verification
   decisions.
3. As an Arbiter, I want to distinguish raw intent from normalized execution
   graphs, so that candidate diversity can be measured before safety rewriting.
4. As an Arbiter, I want perception evidence for every candidate, so that a
   stale or ambiguous scene cannot win because of a confidence tie-break.
5. As an Arbiter, I want strategy-specific score weights, so that a safety
   Policy is not evaluated with the same latency/robustness trade-off as an
   efficient Policy.
6. As a scheduler, I want missing evidence to be explicit, so that an
   evidence-free result is reported as incomplete rather than as a successful
   perception assessment.
7. As a LIBERO adapter, I want repair edits to be bounded and auditable, so that
   automatic grasp completion cannot silently change a candidate's intent.
8. As a LIBERO adapter, I want placement grounding to bind symbolic targets to
   current tracks, so that stale or approximate LLM coordinates do not reach
   the robot.
9. As a researcher, I want candidate fingerprints before and after
   normalization, so that policy diversity and normalization collapse can be
   reported quantitatively.
10. As a researcher, I want perception score breakdowns in artifacts, so that
    Arbiter decisions are reproducible and diagnosable.
11. As an experimenter, I want the existing homogeneous balanced baseline to
    remain unchanged, so that P3.2 can be ablated against P3.1c.
12. As an experimenter, I want all existing candidate constructors to remain
    valid, so that deterministic graph and scheduler tests do not require
    unrelated migration.
13. As a runtime, I want the final selected graph to be revalidated after
    grounding and repair, so that evidence computation cannot weaken execution
    safety.
14. As a runtime operator, I want perception scoring to use cached scene facts
    and bounded computation, so that it does not enter the high-frequency
    control loop.
15. As a maintainer, I want a provider seam for candidate evidence, so that
    verifier-only, perception-only, rehearsal, and OOD scoring can be ablated
    independently.
16. As an evaluator, I want artifacts to show whether a decision used evidence,
    strategy weights, or a fallback tie-break, so that multi-policy claims are
    not overstated.
17. As a recovery agent, I want the selected candidate's recovery cost and
    failure assumptions to be available, so that recovery can prefer a bounded
    alternative after failure.
18. As a CAP-X adapter author, I want the new evidence and normalization layer
    to consume the existing SceneSnapshot and typed skill registry, so that CAP-X
    API parity is preserved.

## Implementation Decisions

- Add a frozen `StrategyProfile` contract with stable names for balanced,
  safety, robust, and efficient. The profile carries hard perception gates and
  score weights for perception quality, verifier confidence, latency, and
  recovery cost.
- Extend `GraphCandidate` with strategy identity, raw proposal metadata, raw
  subgraph retention, normalized/grounded fingerprints, and a rewrite report.
  Existing callers may omit the new fields and receive the balanced default.
- Extend `CandidateEvidence` with a structured perception evidence record and
  an explicit set of available evidence dimensions. Unknown dimensions are
  excluded from normalization rather than treated as zero-quality evidence.
- Keep `CandidateEvidenceProvider` as the highest public scheduler seam. The
  provider receives a candidate and immutable `SceneSnapshot`, returns typed
  evidence, and must not execute the live robot.
- Add a deterministic LIBERO evidence provider that scores scene freshness,
  scene confidence, target/object track existence, track confidence, identity
  ambiguity, and pose availability. Simulator rehearsal remains a separate
  provider and is not faked by this slice.
- Update `CandidateArbiter` to apply hard perception gates first, then compute a
  strategy-aware score over available evidence. The arbitration result records
  whether the result came from evidence, a confidence fallback, or a hard-gate
  decision.
- Normalize candidates through an observable rewrite boundary. The current
  callback shape remains accepted for compatibility; the scheduler derives a
  stable before/after fingerprint and edit report around it.
- Preserve raw candidate intent for arbitration and artifacts. Use the
  normalized view for graph validation and skill validation. The selected graph
  is grounded/repaired and validated again immediately before execution.
- Grounding may fill missing scene-dependent geometry, but it must preserve
  candidate-provided approach margins, orientation policy, checkpoints, and
  recovery policy. Any automatic repair is counted in rewrite metadata and
  contributes a bounded penalty to ranking.
- Candidate fingerprints are canonical, deterministic hashes of the typed
  subgraph representation. Reports include raw unique ratio, normalized unique
  ratio, rewrite count, and strategy compliance/coverage at the artifact seam.
- Do not put LLM inference, raw RGB-D capture, or expensive rehearsal in the
  servo loop. Perception evidence uses the latest committed snapshot and
  bounded/cached calculations.
- Preserve strict structured-output schemas for Manager and Policy responses;
  strategy and evidence metadata are runtime-side contracts, not open-ended
  LLM response fields.

## Testing Decisions

- Test the public `LLMGraphScheduler.compile()` seam and the serialized
  arbitration artifact, not private rewrite helpers or thread-pool internals.
- Add a red test proving two candidates with the same normalized graph still
  retain distinct raw fingerprints and rewrite metadata.
- Add a red test proving a candidate with strong perception evidence outranks a
  nominally higher-confidence candidate when the selected strategy prioritizes
  perception safety.
- Add a red test proving stale/ambiguous scene evidence is hard-rejected under
  the safety profile.
- Add a red test proving unavailable rehearsal evidence is reported as
  unavailable and does not become an implicit zero-quality claim.
- Add a red test proving the LIBERO provider emits evidence from SceneSnapshot
  object tracks and freshness without invoking the robot backend.
- Add prompt/profile compatibility tests showing existing strategy names map to
  typed profiles while the wire response schema remains unchanged.
- Reuse the existing scheduler arbitration tests, LIBERO verifier tests,
  serialization tests, and multimodal SceneSnapshot contract tests as prior
  art. All new behavior follows one vertical red → minimal implementation →
  green cycle at a public seam.
- Run the full pytest suite, compileall, and deterministic LIBERO B3 regression
  smoke after implementation. A real endpoint-backed run is optional and must
  use an API key supplied at runtime, never persisted in artifacts.

## Out of Scope

- Training a new Policy model or introducing separate model checkpoints.
- Claiming that prompt-only temperature/seed changes constitute Policy
  specialization.
- Process-level simulator rehearsal, online OOD rollouts, candidate-specific
  dynamic geometry scoring, and distributed robot execution; these are deferred
  to Phase 5 or later.
- Changing CAP-X skill implementations, robot control timing, or ActionLease
  ownership.
- Adding new raw RGB-D transport through the LLM prompt path.
- Changing the benchmark's binary success definition.

## Further Notes

P3.2 is complete only when an artifact can distinguish intentional Policy
convergence from normalization-induced candidate collapse, when at least one
real arbitration decision uses non-null scene-bound perception evidence, and
when evidence ties are distinct from confidence fallback. A successful LIBERO
episode with `confidence_fallback` alone is a regression success, not evidence
that the new multi-policy mechanism improves selection quality.
