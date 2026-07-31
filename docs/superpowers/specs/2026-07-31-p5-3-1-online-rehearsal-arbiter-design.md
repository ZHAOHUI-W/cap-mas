# P5.3.1 Online Rehearsal Arbiter Design

## Goal

Expose one controlled scheduler seam that can attach candidate-specific
process-rehearsal evidence before arbitration, while preserving a shadow mode
that cannot change physical execution. This increment closes the code-level
online-selection boundary; it does not claim a downstream success-rate gain.

## Scope

The scheduler receives a batch provider:

```python
RehearsalEvidenceProvider = Callable[
    [Sequence[GraphCandidate], SceneSnapshot],
    Mapping[str, RehearsalEvidence],
]
```

The mapping is keyed by the exact `GraphCandidate.candidate_id`. Every
returned `RehearsalEvidence` must pass the existing candidate fingerprint,
optional graph-to-subgraph identity, and parent scene-version checks in
`merge_rehearsal_evidence()`.

The scheduler supports three modes:

| Mode | Provider | Live selection |
| --- | --- | --- |
| `disabled` | not called | baseline Arbiter result |
| `shadow` | called and validated | baseline result; evidence result is diagnostic |
| `online_bounded` | called and validated | evidence-aware result when available, otherwise baseline fallback |

`online_bounded` is still bounded and fail-closed. A provider exception,
invalid evidence, stale evidence, or no evidence-aware winner never creates a
second physical action. The baseline result is retained as an explicit
fallback and the report records the reason.

## Data flow

```text
Policy candidates
      |
      v
static skill/geometry/verifier filtering
      |
      +--> baseline Arbiter --------------------+
      |                                         |
      +--> batch rehearsal provider             |
              |                                 |
              +--> identity/version gate       |
                              |                 |
                              v                 v
                    evidence-aware Arbiter --> live result
```

The baseline and evidence-aware results are retained in a typed
`RehearsalArbitrationReport`. The `LLMGraphCompileResult` exposes reports by
subgraph id, while its existing `arbitrations` field always describes the
result that the caller is allowed to execute. In shadow mode that is the
baseline; in online mode it is the evidence-aware result or an explicit
baseline fallback.

## Contracts

```python
@dataclass(frozen=True)
class RehearsalArbitrationReport:
    mode: Literal["disabled", "shadow", "online_bounded"]
    baseline: ArbitrationResult
    evidence_aware: ArbitrationResult | None
    live: ArbitrationResult
    attached_candidate_ids: tuple[str, ...]
    evidence_rejections: tuple[str, ...]
    provider_latency_ms: float
    fallback_reason: str | None = None

    @property
    def would_change_selection(self) -> bool: ...
```

Reports are diagnostic artifacts and must be serialized with the compile
result. They must not carry API keys, authorization headers, or raw provider
payloads.

## Integration boundary

`LLMGraphScheduler` accepts:

```python
rehearsal_mode: str = "disabled"
rehearsal_evidence_provider: RehearsalEvidenceProvider | None = None
```

The batch provider is called once per candidate-selection decision, outside
the physical executor and before `ActionLease`. The same helper is used by
legacy, staged serial, staged ready-wave, and rolling-frontier selection.
The existing `candidate_evidence_provider` remains per-candidate and is not
replaced by rehearsal.

## Safety and failure semantics

- Rehearsal workers never receive or acquire the live `ActionLease`.
- A candidate identity mismatch is unavailable and is listed in
  `evidence_rejections`.
- A scene-version mismatch is unavailable and is listed in
  `evidence_rejections`.
- Missing rehearsal evidence remains unknown; it is never converted to a zero
  success score.
- `shadow` never mutates the candidates passed by the caller.
- `online_bounded` has at most one live Arbiter result and never executes the
  baseline and evidence-aware winners both.
- Provider latency is measured separately from LLM compile latency.

## Verification gate

The code gate requires focused tests proving disabled/shadow/online semantics,
identity and scene-version rejection, provider failure fallback, report
serialization, and no candidate mutation. A real smoke must then show a
run-scoped artifact containing baseline/live winners, selection basis, evidence
provenance, and exactly one physical execution path. A smoke is not evidence
of downstream improvement; the later matched gate requires at least ten seeds
and multiple tasks.

## Non-goals

- No TSDF or semantic adapter implementation.
- No learned Arbiter weights or calibration.
- No adaptive topology edits.
- No persistent or cross-process evidence cache; that remains P5.4.
- No automatic promotion of a shadow winner.
