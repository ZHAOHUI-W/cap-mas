Status: completed
Labels: ready-for-agent
Title: P3.2 closure for evidence precedence and arbitration semantics

## Scope

Implement the adjacent P3.2 closure spec while preserving CAP-X compatibility,
the staged protocol, and the single physical Executor.

## Acceptance Criteria

- Scheduler-created candidates default to no confidence value.
- Evidence-mode score breakdowns do not contain the legacy confidence term.
- Evidence records include provider and source scene version.
- Scene-version-mismatched evidence is rejected before arbitration.
- Evidence ties are labeled `evidence_tie_break`.
- Evidence-free selection is labeled `confidence_fallback`.
- Existing deterministic, scheduler, staged, and LIBERO tests remain green.
- Deferred dynamic evidence work is documented under Phase 5.
