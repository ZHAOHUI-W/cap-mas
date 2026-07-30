# P5.1 Verifier Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Add candidate-specific, scene-versioned static and dynamic verifier evidence while preserving the existing scalar Arbiter contract and correcting object_in_gripper semantics.

**Architecture:** A new capmas.verification.evidence module owns immutable per-predicate evidence, summary calculation, static collection, dynamic conversion, and scalar compatibility attachment. CandidateEvidence receives an optional typed verifier projection; the existing Arbiter consumes the legacy scalar but rejects typed evidence with a stale scene, wrong effective-candidate fingerprint, or inconsistent projection. Static evidence is created before arbitration from selected preconditions. Dynamic evidence is created after execution from VerificationResult and is passed only to later cycles.

**Tech Stack:** Python 3.10+, frozen dataclasses, existing PredicateBasedVerifier, GraphCandidate, SceneSnapshot, VerificationResult, pytest, no new runtime dependency.

## Global Constraints

- Use the effective normalized GraphCandidate.subgraph for fingerprints.
- Bind every typed evidence record to exactly one source SceneSnapshot.scene_version.
- Never use post-execution evidence to select the action that already executed.
- Keep old scalar CandidateEvidence(verifier_pass_rate=..., available_metrics=("verifier",)) construction valid.
- Keep the reference verifier deterministic and model-free.
- Preserve unrelated user changes in the dirty worktree.
- Do not put API keys, simulator handles, or raw image bytes into evidence objects or logs.
- Run focused tests after each vertical slice and the full suite before empirical validation.

---

### Task 1: Add the typed verifier evidence contract

**Files:**
- Create: capmas/verification/evidence.py
- Create: tests/test_verifier_evidence.py
- Modify: capmas/verification/__init__.py

**Interfaces:**
- Produces VerifierPredicateEvidence, VerifierEvidence, summarize_verifier_results(), and predicate_report_to_evidence().
- Later tasks consume the exact constructor and to_dict() shape below.
- Uses only PredicateReport; it does not import the scheduler or executor.

- [ ] Step 1: Write failing contract tests

Create tests/test_verifier_evidence.py with tests equivalent to:

~~~
def test_typed_evidence_is_frozen_and_serializable():
    item = VerifierPredicateEvidence(
        "gripper_closed()", "dynamic", "pass", 0.9, None,
        ("artifact://scene/4",),
    )
    evidence = VerifierEvidence(
        "fp", 4, 1.0, 1.0, "test", 10,
        dynamic_results=(item,), source_verification="contract-1",
    )
    assert evidence.to_dict()["dynamic_results"][0]["status"] == "pass"
    with pytest.raises(FrozenInstanceError):
        item.status = "fail"


def test_summary_excludes_unknown_from_pass_rate_but_counts_it_in_coverage():
    results = (
        VerifierPredicateEvidence("a", "static", "pass", 1.0, None),
        VerifierPredicateEvidence("b", "static", "fail", 1.0, "measured failure"),
        VerifierPredicateEvidence("c", "static", "unknown", None, "track not found"),
    )
    assert summarize_verifier_results(results) == (0.5, 2 / 3)


def test_invalid_phase_confidence_and_duplicate_are_rejected():
    with pytest.raises(ValueError, match="phase"):
        VerifierEvidence(
            "fp", 1, 1.0, 1.0, "test", 1,
            static_results=(
                VerifierPredicateEvidence("a", "dynamic", "pass", 1.0, None),
            ),
        )
    with pytest.raises(ValueError, match="confidence"):
        VerifierPredicateEvidence("a", "static", "pass", None, "missing")
    with pytest.raises(ValueError, match="duplicate"):
        VerifierEvidence(
            "fp", 1, 1.0, 1.0, "test", 1,
            static_results=(
                VerifierPredicateEvidence("a", "static", "pass", 1.0, None),
                VerifierPredicateEvidence("a", "static", "fail", 1.0, "failure"),
            ),
        )


def test_unavailable_false_report_maps_to_unknown():
    item = predicate_report_to_evidence(
        PredicateReport(
            "object_at_target(bowl,plate)", False, 0.0, ("bowl",),
            "track not found",
        ),
        phase="dynamic",
    )
    assert item.status == "unknown"
    assert item.confidence is None
~~~

- [ ] Step 2: Run pytest -q tests/test_verifier_evidence.py and verify collection fails because the module and symbols do not exist.
- [ ] Step 3: Implement frozen dataclasses with fields:

~~~
VerifierPredicateEvidence(
    predicate, phase, status, confidence, reason, evidence_refs=()
)
VerifierEvidence(
    candidate_fingerprint, scene_version, pass_rate, coverage,
    provider, captured_at_ns, static_results=(), dynamic_results=(),
    source_verification=None
)
~~~

Validate non-empty identifiers, non-negative versions/timestamps, ranges in [0, 1], phase/tuple agreement, required confidence for pass/fail, duplicate names within each phase, and that supplied summaries equal recomputed summaries. Implement deterministic JSON-compatible to_dict(). Implement summarize_verifier_results() as passed/determined and determined/total, with zeroes for an empty tuple. predicate_report_to_evidence() maps passed reports to pass; reasons containing unknown, not found, unavailable, not observed, or not available map false reports to unknown, otherwise fail.

- [ ] Step 4: Export the public types and run pytest -q tests/test_verifier_evidence.py. Expected: all focused tests pass.
- [ ] Step 5: Commit only capmas/verification/evidence.py, capmas/verification/__init__.py, and tests/test_verifier_evidence.py with message feat: add typed verifier evidence contract.

### Task 2: Attach typed evidence to CandidateEvidence without breaking legacy callers

**Files:**
- Modify: capmas/contracts/candidates.py
- Modify: capmas/contracts/__init__.py
- Modify: capmas/verification/evidence.py
- Create: tests/test_candidate_verifier_evidence.py

**Interfaces:**
- Adds CandidateEvidence.verifier: VerifierEvidence | None = None.
- Produces attach_verifier_evidence(base, verifier) -> CandidateEvidence.
- Legacy scalar evidence remains accepted unchanged.

- [ ] Step 1: Write failing tests for a legacy scalar constructor, immutable attachment, scalar projection, metric propagation, and inconsistent typed scalar/version rejection.
- [ ] Step 2: Run pytest -q tests/test_candidate_verifier_evidence.py and verify failure because CandidateEvidence.verifier and attach_verifier_evidence are absent.
- [ ] Step 3: Add verifier after existing CandidateEvidence defaults. When typed evidence exists, enforce matching verifier_pass_rate, scene_version, provider, and captured_at_ns. If typed verifier is declared in available_metrics, require positive coverage. The helper uses dataclasses.replace, propagates provenance, and adds the verifier metric only for positive coverage.
- [ ] Step 4: Export the type from capmas.contracts and the helper from capmas.verification. Run pytest -q tests/test_candidate_verifier_evidence.py tests/test_phase5_geometry_contracts.py tests/test_phase5_geometry_arbiter.py. Expected: all pass, including legacy evidence tests.
- [ ] Step 5: Commit only the four task paths with message feat: project typed verifier evidence into candidates.

### Task 3: Implement static candidate evidence and dynamic result conversion

**Files:**
- Modify: capmas/verification/evidence.py
- Create: tests/test_verifier_evidence_collection.py

**Interfaces:**
- Produces collect_static_verifier_evidence(candidate, scene, verifier, predicate_selector=None, provider=..., clock=...).
- Produces verifier_evidence_from_result(candidate_fingerprint, result, provider=..., clock=...).
- Uses EvidenceCompatibilityError for candidate/scene mismatches.

- [ ] Step 1: Write a fixture with a version-4 SceneSnapshot and version-4 GraphCandidate. The action preconditions are track_exists:bowl and object_in_gripper(bowl); its postcondition is object_at_target(bowl,plate). Test that a selector for track_exists returns only that precondition, never the postcondition, and binds effective fingerprint and scene version.
- [ ] Step 2: Test a parent scene version mismatch raises EvidenceCompatibilityError.
- [ ] Step 3: Test dynamic conversion of a measured gripper failure and unavailable track report. Assert source_verification, checked scene version, statuses fail/unknown, pass_rate 0.0, and coverage 0.5.
- [ ] Step 4: Implement static collection by collecting action-node preconditions in stable order, deduplicating exact strings, applying the selector, calling verifier.evaluate_predicates(), converting with phase static, and binding subgraph_fingerprint(candidate.subgraph) and scene.scene_version.
- [ ] Step 5: Implement dynamic conversion from every result.predicate_results with phase dynamic, checked_scene_version, contract_id, injected clock, and immutable evidence refs.
- [ ] Step 6: Run pytest -q tests/test_verifier_evidence_collection.py tests/test_verifier_evidence.py. Expected: all pass. Commit the two task paths.

### Task 4: Integrate static verifier evidence into the LIBERO provider

**Files:**
- Modify: capmas/verification/libero.py
- Modify: capmas/verification/__init__.py
- Create: tests/test_libero_verifier_evidence.py

**Interfaces:**
- libero_candidate_evidence(candidate, scene) returns current perception evidence plus typed static verifier evidence.
- compile_time_preconditions remains the LIBERO selector for predicates safe before dispatch.

- [ ] Step 1: Write a provider test with track_exists:bowl and object_in_gripper(bowl) preconditions and a visible bowl. Assert typed verifier exists, only track_exists is static, verifier is available, and scene version matches.
- [ ] Step 2: Run pytest -q tests/test_libero_verifier_evidence.py and verify the typed verifier is absent.
- [ ] Step 3: Compose current perception evidence with collect_static_verifier_evidence using LiberoObservableVerifier and a selector applying compile_time_preconditions. Use attach_verifier_evidence. Do not evaluate postconditions or checkpoints.
- [ ] Step 4: Run pytest -q tests/test_libero_verifier_evidence.py tests/test_phase5_geometry_provider.py tests/test_phase5_geometry_scheduler.py. Expected: all pass.
- [ ] Step 5: Commit only task paths with message feat: add LIBERO static verifier evidence.

### Task 5: Add Arbiter typed-evidence freshness and identity gates

**Files:**
- Modify: capmas/agents/arbiter.py
- Create: tests/test_verifier_arbiter_gate.py

**Interfaces:**
- CandidateArbiter.select() remains unchanged.
- _evidence_gate() validates optional typed verifier evidence; scalar-only legacy evidence is unaffected.

- [ ] Step 1: Write tests for wrong typed scene version, wrong typed effective-candidate fingerprint, and correctly bound typed evidence. Wrong evidence must produce STALE_EVIDENCE; correct evidence must select and score via the existing verifier scalar.
- [ ] Step 2: Run pytest -q tests/test_verifier_arbiter_gate.py and verify stale typed evidence is accepted or cannot be constructed.
- [ ] Step 3: After the existing scene check, reject typed evidence when its scene_version differs, candidate_fingerprint differs from subgraph_fingerprint(candidate.subgraph), or pass_rate differs from CandidateEvidence.verifier_pass_rate. Keep rejection reason verifier-specific.
- [ ] Step 4: Run pytest -q tests/test_verifier_arbiter_gate.py tests/test_phase5_geometry_arbiter.py tests/test_shadow_arbiter.py. Expected: all pass. Commit only task paths.

### Task 6: Correct object/gripper predicate semantics and update regressions

**Files:**
- Modify: capmas/verification/predicates.py
- Modify: tests/test_runtime_cycle.py
- Modify: tests/test_graph_runtime.py
- Create: tests/test_predicate_semantics.py

**Interfaces:**
- object_in_gripper(obj_id) checks pose and distance only.
- gripper_closed() remains an independent opening threshold.
- object_held(obj_id) checks distance and closure.

- [ ] Step 1: Write a scene with the object within the gripper distance and gripper_opening=0.9. Assert object_in_gripper and gripper_open pass while object_held fails.
- [ ] Step 2: Run pytest -q tests/test_predicate_semantics.py and verify current object_in_gripper fails because it requires closure.
- [ ] Step 3: Change only the closure branch to apply to object_held; preserve thresholds, track matching, pose conversion, and object_near_gripper behavior.
- [ ] Step 4: Update tests whose stated purpose is strict holding to object_held or an explicit pair. Run pytest -q tests/test_predicate_semantics.py tests/test_runtime_cycle.py tests/test_graph_runtime.py. Expected: all pass.
- [ ] Step 5: Commit only predicate and regression test paths with message fix: separate gripper region and closure predicates.

### Task 7: Update P5.1 documentation and repository checks

**Files:**
- Modify: docs/phase5-evidence-evolution.md
- Modify: docs/implementation-roadmap.md
- Modify: docs/experiments.md
- Create: tests/test_phase5_p51_docs.py

**Interfaces:**
- Documents static/dynamic timing, typed evidence, legacy projection, and remaining empirical LIBERO gate.
- No success rate is claimed from unit tests.

- [ ] Step 1: Write failing documentation checks for VerifierEvidence, dynamic post-execution evidence, candidate fingerprint, and the LIBERO gate.
- [ ] Step 2: Run pytest -q tests/test_phase5_p51_docs.py and verify failure before documentation changes.
- [ ] Step 3: Add the P5.1 implementation subsection to phase5-evidence-evolution.md and update roadmap/experiments. State compile-time-safe preconditions, dynamic VerificationResult conversion, candidate/scene gates, object_in_gripper versus object_held, and independent run directories with logs, results, summaries, and manifest.
- [ ] Step 4: Run pytest -q tests/test_phase5_p51_docs.py tests/test_phase5_docs.py; python -m compileall -q capmas scripts; git diff --check. Expected: all exit 0.
- [ ] Step 5: Commit only task documentation and test paths with message docs: record P5.1 verifier evidence progress.

### Task 8: Run the complete software gate

**Files:**
- No source changes expected.
- New artifacts only under fresh outputs/phase5/P5.1_* directories.

- [ ] Step 1: Run the focused P5.1 suite: pytest -q tests/test_verifier_evidence.py tests/test_candidate_verifier_evidence.py tests/test_verifier_evidence_collection.py tests/test_libero_verifier_evidence.py tests/test_verifier_arbiter_gate.py tests/test_predicate_semantics.py.
- [ ] Step 2: Run pytest -q. Expected: zero failures and errors. Record the exact count in the final manifest.
- [ ] Step 3: Run python -m compileall -q capmas scripts; git diff --check; git status --short. Do not revert unrelated changes.

### Task 9: Run the empirical LIBERO P5.1 gate

**Files:**
- Use the existing CAP-X/LIBERO runner and environment.
- Create only fresh run-scoped directories under outputs/phase5/P5.1_*.

**Environment and safety:**

- Use CUDA_VISIBLE_DEVICES=5.
- Use /data/MLLM/wzh/agent/paper/infiAgent/workspace/cap-x/.venv-libero/bin/python when required by the runner.
- Pass API endpoint/model/key through the runner's existing command-line or environment handling; never write the key to files or logs.
- Preserve stdout/stderr in each run's logs directory.

- [ ] Step 1: Run one fixed-seed smoke episode and verify logs/, results/, summary.json, summary.md, and manifest.json exist.
- [ ] Step 2: Verify at least one candidate artifact has typed static evidence and one post-execution dynamic result. If runner integration cannot emit dynamic evidence, report the exact missing boundary and leave the empirical gate open.
- [ ] Step 3: Run the matched seed set used by the current Phase 5 baseline, one fresh directory per seed. Record seed, task, scene version, candidate fingerprint, static/dynamic results, selection basis, success, failure class, and latency.
- [ ] Step 4: Validate required files, absence of the literal API key in logs, evidence identity/version fields, and dynamic evidence ordering after its execution trace.
- [ ] Step 5: Update phase documents only with measured results after artifact validation; otherwise document the remaining integration gap without claiming empirical success.

## Plan Self-Review

- Spec coverage: typed data model is Task 1; CandidateEvidence compatibility is Task 2; static/dynamic collection is Task 3; LIBERO integration is Task 4; Arbiter identity/freshness is Task 5; predicate semantics is Task 6; documentation is Task 7; software and empirical gates are Tasks 8-9.
- Placeholder scan: no task relies on TBD, TODO, or an unspecified schema.
- Type consistency: VerifierEvidence is defined in Task 1, attached in Task 2, collected in Task 3, consumed by Tasks 4-5, and exported from the verification package.
- Scope: no TSDF, geometry, rehearsal, OOD, calibration, learned weights, or unrelated runtime refactor is included.
