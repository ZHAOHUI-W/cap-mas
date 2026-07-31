# P5.4 Evidence Cache Evaluation Design

## Status

Approved experiment direction. This document defines the first real,
run-scoped evaluation of the P5.4 cache contract. It does not claim a
downstream robot success-rate improvement and does not modify physical
execution.

## Goal

Measure whether `VersionedEvidenceCache` reuses candidate-specific evidence
only when the candidate fingerprint and source scene version are exact, and
whether that reuse reduces evidence-provider calls under a deterministic
request trace.

The evaluation must make the following behavior observable:

1. A first request is a miss, invokes the provider, and stores evidence.
2. A repeated request for the same candidate and scene version is an exact
   hit and does not invoke the provider.
3. Publishing a newer scene version invalidates older entries.
4. A request for an old scene version is rejected as stale and never attaches
   stale evidence.
5. A different candidate fingerprint is independently evaluated on the same
   scene version.

## Non-goals

- No LLM request, CAP-X environment startup, LIBERO simulation, or physical
  robot execution.
- No persistent or cross-process cache.
- No integration of the cache into `LLMGraphScheduler`, `CandidateArbiter`, or
  the P5.3.1 live executor.
- No multi-task success-rate or statistical generalization claim.
- No change to the cache implementation contract established by
  `docs/superpowers/specs/2026-07-30-p5-4-evidence-cache-design.md`.

The experiment is intentionally isolated so cache correctness and provider
call suppression can be evaluated without conflating them with rehearsal
quality, scene grounding, or robot execution.

## Alternatives considered

### A. Deterministic protocol replay (selected)

Run the same fixed request sequence against a cache-disabled control and a
cache-enabled implementation. Use the public candidate, scene, evidence, and
`VersionedEvidenceCache` interfaces. This is deterministic, fast, repeatable,
and directly tests the P5.4 contract.

### B. Replay existing P5.3.1 rehearsal artifacts

Load candidates and evidence from a retained CAP-X/LIBERO rehearsal artifact.
This would add realism, but also introduces artifact-version compatibility,
candidate identity mapping, and environment-specific provenance into a cache
correctness gate. It remains a later integration test.

### C. Enable the cache in online physical selection

Wire cache reads and writes into the P5.3.1 live path. This would test a
larger end-to-end behavior, but it would mix cache correctness with online
Arbiter selection and physical execution. It is explicitly deferred until
the isolated P5.4 gate is closed.

Option A is selected for this increment. Options B and C are follow-up
evaluations, not hidden requirements of P5.4.

## Experiment driver

Add `scripts/run_p54_evidence_cache.py`. The driver must:

- build a small deterministic fixture containing at least two distinct
  `GraphCandidate` values and `SceneSnapshot` values for scene versions 1 and
  2;
- use a local deterministic evidence provider whose call count and returned
  scene version are observable;
- execute the identical request sequence in `cache_disabled` and
  `cache_enabled` modes;
- create a separate `Phase5RunDirectory` for each mode under
  `outputs/phase5/P5.4_cache_evaluation/`;
- write artifacts before finalizing a SHA-256 manifest;
- exit non-zero if any acceptance assertion fails.

The CLI must support at least:

```text
python scripts/run_p54_evidence_cache.py \
  --output-root outputs/phase5 \
  --seed 1
```

The seed is recorded for reproducibility. It may affect fixture labels, but
must not make the expected cache semantics probabilistic.

## Request trace

Both modes use one identical ordered trace. The logical operations are:

1. Publish scene version 1.
2. Query candidate A at scene version 1.
3. Query candidate B at scene version 1.
4. Repeat candidate A at scene version 1.
5. Repeat candidate B at scene version 1.
6. Publish scene version 2.
7. Probe candidate A using its old scene-version-1 cache key.
8. Query candidate A at scene version 2.
9. Query candidate B at scene version 2.
10. Repeat candidate A at scene version 2.
11. Query a distinct candidate C at scene version 2.

The cache-enabled mode must use `get`/`put` or the corresponding candidate
helpers at every eligible request. The cache-disabled mode must invoke the
provider for every logical query and must not use `VersionedEvidenceCache`.
For the stale probe, the disabled lane calls the provider as an ordinary
uncached request and records `disabled`; the enabled lane records
`stale_rejection` and does not call the provider. In both lanes, the stale
probe must not attach old evidence to a current candidate. This keeps the
request sequence identical while making provider-call suppression measurable.

The driver records a trace entry for every operation with:

- operation index and operation kind;
- candidate fingerprint (full value, not only a shortened display label);
- requested scene version;
- cache result: `hit`, `miss`, `stale_rejection`, or `disabled`;
- whether the provider was called;
- returned evidence scene version, if any;
- whether evidence was attached to the candidate.

## Metrics

Each mode records:

- logical request count;
- provider call count;
- exact cache hits;
- cache misses;
- stale rejections;
- stores;
- invalidations;
- evictions;
- final cache size and current scene version;
- disabled-request count (control lane only);
- exact hit rate, defined as
  `hits / (hits + misses + stale_rejections)`;
- provider-call reduction for the enabled mode relative to the disabled
  control;
- stale-evidence attachment count.

The expected enabled-mode properties are more important than a particular
latency value. The provider is local and deterministic, so wall-clock latency
is diagnostic only and must not be used as a performance claim.

## Artifact layout

Each mode gets its own run directory. A completed directory contains:

```text
outputs/phase5/P5.4_cache_evaluation/<timestamp>_<run_id>/
  run_config.json
  logs/runner.log
  results/cache_trace.json
  results/summary.json
  summary.md
  manifest.json
```

`run_config.json` records the mode, seed, request-trace version, cache bounds,
fixture candidate IDs, scene versions, and execution scope. It must not contain
API keys or provider secrets.

`results/summary.json` records per-mode metrics, expected-vs-observed
assertions, and the paired control/enabled comparison. `summary.md` is a
short human-readable rendering of the same result. `manifest.json` hashes all
published files except itself.

If either mode raises, the driver must still retain that mode's run directory,
write `failure.json` with the stage and exception, write the partial trace,
and finalize the manifest before re-raising.

## Acceptance gates

The P5.4 real evaluation is successful only when all of the following hold:

1. Control and enabled modes execute the same trace length and candidate/
   scene request sequence.
2. Enabled mode has at least one exact hit and at least one store.
3. Advancing from scene version 1 to 2 produces at least one invalidation.
4. The stale probe produces a stale rejection.
5. No stale evidence is attached to a candidate.
6. Enabled mode invokes the provider fewer times than the disabled control.
7. The enabled final cache scene version is 2 and no entry has scene version 1.
8. Both run directories contain all required artifacts and a valid manifest.
9. A focused test suite, full test suite, compile check, and diff check pass.

This closes only the P5.4 isolated cache evaluation gate. It does not close
P5.3.1's matched physical baseline, multi-seed or multi-task evaluation, and
does not establish that cache reuse improves downstream task success.

## Testing seams

Tests target public behavior at these seams:

- the driver-level paired control/enabled result;
- the serialized trace and summary artifacts;
- failure artifact retention;
- the existing `VersionedEvidenceCache` public API for exact hit, stale
  rejection, invalidation, and candidate attachment semantics.

The tests do not mock private cache methods or assert implementation-specific
container state.
