# P5.3 Graph/Subgraph Identity and Shadow Arbiter Design

## Status

Approved design for the next P5.3 closure increment.

## Problem

The real LIBERO rehearsal driver executes complete `MissionGraph` candidates,
so its source artifact fingerprint identifies the complete graph. The live
CAP-MAS Arbiter ranks `GraphCandidate` objects, whose identity is the effective
local `SubgraphSpec`. Rehearsal results therefore cannot currently pass the
Arbiter fingerprint gate without weakening identity checks.

The next increment must also prove that rehearsal evidence can change a
hypothetical Arbiter decision while preserving the live selection and the
single-owner physical executor invariant.

## Goals

- Preserve both graph-level provenance and subgraph-level Arbiter identity.
- Require an explicit target `subgraph_id`; never infer it from a candidate ID.
- Reject mismatched graph, subgraph, and scene identities.
- Convert a real full-graph rehearsal result into evidence attachable to the
  corresponding local `GraphCandidate`.
- Run a shadow Arbiter in parallel with the baseline Arbiter without executing
  a second physical action.
- Record baseline/shadow winners, selection bases, score breakdowns, and
  whether the hypothetical winner changes.

## Non-goals

- No 10+ seed statistical expansion in this increment.
- No multi-task rehearsal expansion in this increment.
- No evidence cache implementation; that remains P5.4.
- No automatic online use of shadow evidence.
- No change to CAP-X skill APIs or physical execution ownership.

## Identity model

```python
@dataclass(frozen=True)
class CandidateIdentity:
    graph_fingerprint: str
    subgraph_id: str
    subgraph_fingerprint: str
    scene_version: int
```

`graph_fingerprint` is computed from the source JSON mapping with sorted keys
and compact separators. This preserves the identity of legacy artifacts whose
skill-output references use `call_index`. `subgraph_fingerprint` is computed
from the typed, normalized `SubgraphSpec` using the existing canonical local
graph serializer.

Graph candidate artifacts declare the mapping explicitly:

```json
{
  "candidate_id": "sg_pick:policy-1",
  "candidate_fingerprint": "<source graph fingerprint>",
  "fingerprint_scope": "graph",
  "arbiter_subgraph_id": "sg_pick",
  "graph": {}
}
```

Legacy small test artifacts may continue to use the existing subgraph-scoped
`candidate_fingerprint` format. Real graph-scoped artifacts must provide
`arbiter_subgraph_id`; candidate-id string parsing is not allowed.

`RehearsalJob`, `RehearsalResult`, and `RehearsalEvidence` carry:

- source `candidate_fingerprint` and `fingerprint_scope`;
- optional `arbiter_subgraph_id`;
- optional `arbiter_fingerprint`;
- the existing `scene_version`.

The Arbiter attachment gate compares `arbiter_fingerprint` with
`subgraph_fingerprint(candidate.subgraph)`. If the evidence is legacy
subgraph-scoped, `candidate_fingerprint` is used as the fallback. If the
evidence is graph-scoped but has no mapped Arbiter fingerprint, attachment is
rejected.

## Shadow arbitration

The shadow API accepts the same local candidate tuple and scene as the live
Arbiter:

```python
run_shadow_arbitration(
    candidates: Sequence[GraphCandidate],
    rehearsals: Mapping[str, RehearsalEvidence],
    scene: SceneSnapshot,
    arbiter: CandidateArbiter | None = None,
) -> ShadowArbitrationReport
```

It runs the baseline selection on the original candidates, attaches matched
rehearsal evidence only to a copied shadow candidate tuple, then runs a second
Arbiter selection over that tuple. It returns:

```python
@dataclass(frozen=True)
class ShadowArbitrationReport:
    baseline: ArbitrationResult
    shadow: ArbitrationResult
    baseline_winner: str | None
    shadow_winner: str | None
    would_change_selection: bool
    physical_execution_required: bool = False
```

The function has no backend, lease, or executor dependency. A changed shadow
winner is a diagnostic result only; the baseline candidate remains the live
selection until a later explicitly enabled online mode is implemented.

## Failure handling

- Missing target subgraph: reject the artifact before spawning a worker.
- Source graph fingerprint mismatch: reject the artifact before execution.
- Subgraph fingerprint mismatch at attachment: reject evidence and preserve
  the baseline candidate unchanged.
- Scene version mismatch: reject evidence using the existing compatibility
  contract.
- Missing rehearsal for one candidate: leave that candidate's baseline
  evidence unchanged; do not interpret missing evidence as zero success.

## Acceptance criteria

- Unit tests cover graph-scoped identity derivation, legacy reference
  preservation, explicit target-subgraph validation, and mismatch rejection.
- A shadow test shows `selection_basis="evidence_score"` can appear in the
  shadow result while the baseline candidate remains unchanged.
- The shadow path reports `physical_execution_required=False` and has no
  physical backend dependency.
- Full local suite, compile check, and diff check pass.
- Existing P5.3 real artifacts remain readable and are documented as source
  graph evidence until regenerated with explicit mapping fields.
