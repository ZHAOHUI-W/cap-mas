# Phase 5.2 Candidate-Conditioned Geometry Evidence Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

Goal: Add candidate-conditioned geometry and grasp evidence so the Arbiter can distinguish typed motion proposals without changing CAP-X execution authority or the high-frequency World Model path.

Architecture: Extend the typed graph with optional MotionIntent metadata and CandidateEvidence with a separate GeometryEvidence object. A read-only MotionPreviewBackend evaluates one effective candidate against one immutable SceneSnapshot and frozen local map, returning three-state evidence under a 50 ms proposal-wave deadline. The Arbiter applies fixed transparent weights and hard gates; realistic sensor-derived evidence is primary, while privileged LIBERO pose is diagnostic-only.

Tech Stack: Python 3.12, frozen dataclasses, typing Protocols, pytest, existing CAP-MAS graph serialization, SceneSnapshot, sparse voxel map, CAP-X typed skill registry, and CandidateArbiter.

## Global Constraints

- Geometry remains separate from PerceptionEvidence.
- Evidence dimensions use pass, fail, and unknown; unknown is never converted to zero.
- MotionIntent is planning metadata, never an executable CAP-X skill argument.
- Preview is side-effect free and cannot receive an environment handle or ActionLease.
- The primary mode uses sensor-derived SceneSnapshot; privileged LIBERO state is diagnostic-only.
- The proposal-wave geometry deadline is 50 ms; candidate timeout returns unknown.
- Fixed geometry weights are used until P5.6 calibration.
- Every experiment writes a new run-scoped directory below outputs/phase5 and never persists secrets.
- Do not modify CAP-X Robot Skill signatures or introduce physical action parallelism.

---

### Task 1: Add typed motion intent and evidence contracts

Files:
- Modify: capmas/contracts/graph.py
- Modify: capmas/contracts/candidates.py
- Modify: capmas/graph/serialization.py
- Test: tests/test_phase5_geometry_contracts.py

Interfaces:
- MotionIntent(kind, object_track_id, target_track_id, approach_vector_xyz, standoff_m, target_pose_wxyz_xyz) is optional on SubgraphNodeSpec.
- EvidenceDimension(name, status, score, threshold, reason) validates pass, fail, unknown and scores in [0, 1].
- GeometryEvidence stores four dimensions, candidate fingerprint, scene/map versions, map backend, provider metadata, latency, privilege flag, and artifact refs.
- CandidateEvidence.geometry is optional and available_metrics accepts geometry.

- [x] Step 1: Write failing tests for MotionIntent round-trip, three-state EvidenceDimension validation, GeometryEvidence provenance, and CandidateEvidence geometry serialization.
- [x] Step 2: Run pytest tests/test_phase5_geometry_contracts.py -q and verify failure because the types and fields do not exist.
- [x] Step 3: Add frozen dataclasses and explicit validation. Add motion_intent: MotionIntent | None = None to SubgraphNodeSpec. Add geometry: GeometryEvidence | None = None to CandidateEvidence. Add optional motion_intent to the strict node allowlist and serializer.
- [x] Step 4: Run pytest tests/test_phase5_geometry_contracts.py -q and verify pass.
- [ ] Step 5: Commit with git commit -m "feat: add typed candidate geometry evidence contracts".

### Task 2: Canonicalize MotionIntent

Files:
- Create: capmas/graph/normalizer.py
- Modify: capmas/contracts/graph.py
- Modify: capmas/skills/registry.py
- Test: tests/test_phase5_geometry_normalizer.py

Interfaces:
- normalize_motion_intent(node: SubgraphNodeSpec) -> SubgraphNodeSpec
- CandidateNormalizer.normalize(candidate: GraphCandidate) -> GraphCandidate

Rules:

- Derive intent only from typed fields and registered calls.
- Recognize current CAP-X-compatible calls: sample_grasp_pose(object_name, use_multiview) and goto_pose(position, quaternion_wxyz, z_approach).
- Reject unregistered geometry keys such as approach or standoff inside skill_calls.args.
- Never parse description or rationale.
- Preserve raw and normalized fingerprints using rewrite_report_for.

- [x] Step 1: Write tests for deriving a grasp target, rejecting unregistered arguments, and changing the normalized fingerprint when effective intent changes.
- [x] Step 2: Run pytest tests/test_phase5_geometry_normalizer.py -q and verify failure.
- [x] Step 3: Implement CandidateNormalizer and the typed-call checks.
- [x] Step 4: Run pytest tests/test_phase5_geometry_normalizer.py -q and verify pass.
- [ ] Step 5: Commit with git commit -m "feat: canonicalize typed motion intents".

### Task 3: Add read-only MotionPreview and reference geometry provider

Files:
- Create: capmas/perception/motion_preview.py
- Create: capmas/perception/geometry_evidence.py
- Modify: capmas/perception/local_map.py
- Test: tests/test_phase5_geometry_provider.py

Interfaces:
- MotionPreviewBackend.preview(intent, scene, local_map) -> MotionPreview
- ReferenceMotionPreview.preview(intent, scene, local_map) -> MotionPreview
- candidate_geometry_evidence(candidate, scene, local_map, preview_backend, deadline_ns) -> GeometryEvidence
- LocalMapBackend.query(region: MapRegion) -> MapQueryResult and map_version() -> int

Rules:

- Resolve objects from immutable SceneSnapshot.objects, not mutable tracker state.
- Use MapQueryResult.clearance_m as a continuous value.
- Return unknown for missing surface normals, unavailable IK, stale map, unresolved target, or expired deadline.
- Never call goto_pose, acquire ActionLease, mutate CAP-X state, or read evaluator success.
- Mark used_privileged_state and reject privileged state in realistic mode.
- Reference preview may implement workspace, pose validity, target freshness, corridor occupancy, and conservative clearance.

- [x] Step 1: Write tests where two approach vectors on one scene produce different fingerprints and clearance scores, deadline expiration returns unknown, and preview performs zero executor calls.
- [x] Step 2: Run pytest tests/test_phase5_geometry_provider.py -q and verify failure.
- [x] Step 3: Implement MotionPreviewBackend, ReferenceMotionPreview, LocalMapBackend read-only seam, and candidate_geometry_evidence.
- [x] Step 4: Run pytest tests/test_phase5_geometry_provider.py -q and verify pass.
- [ ] Step 5: Commit with git commit -m "feat: add read-only candidate geometry preview".

### Task 4: Integrate geometry scoring and hard gates

Files:
- Modify: capmas/contracts/strategy.py
- Modify: capmas/agents/arbiter.py
- Test: tests/test_phase5_geometry_arbiter.py

Interfaces:
- StrategyProfile gains geometry_weight, min_reachability, and max_collision_risk.
- CandidateArbiter.score(candidate) includes a geometry component when geometry evidence is available.
- CandidateArbiter rejects declared reachability and collision failures before soft scoring.

Rules:

- Use fixed weights: grasp quality 0.30, reachability 0.30, clearance 0.25, inverted collision risk 0.15.
- Renormalize only over measurable pass/fail dimensions.
- Do not use truthiness fallbacks such as value or 0.0.
- Reject stale candidate fingerprint, scene version, and map version.
- Keep legacy confidence out of evidence mode.

- [x] Step 1: Write tests for geometry changing the winner, collision hard-gate rejection, unknown not becoming zero, and stale geometry rejection.
- [x] Step 2: Run pytest tests/test_phase5_geometry_arbiter.py -q and verify failure.
- [x] Step 3: Implement the fixed geometry score and strategy-profile hard gates.
- [x] Step 4: Run pytest tests/test_phase5_geometry_arbiter.py -q and verify pass.
- [ ] Step 5: Commit with git commit -m "feat: arbitrate candidates with geometry evidence".

### Task 5: Wire normalization and evidence into the LLM scheduler

Files:
- Modify: capmas/runtime/llm_scheduler.py
- Modify: scripts/run_libero_b3_llm.py
- Modify: capmas/contracts/experiment.py
- Test: tests/test_phase5_geometry_scheduler.py

Interfaces:
- LLMGraphScheduler normalizes the effective candidate before candidate_evidence_provider.
- candidate_evidence_provider(candidate, scene) -> CandidateEvidence | None receives the normalized candidate.
- ExperimentRunConfig records geometry mode, deadline, preview backend, privilege mode, and artifact directory.

Rules:

- Normalize after scene grounding/repair and before evidence construction.
- If repair changes effective intent, discard and recompute previous geometry evidence.
- Preserve the single physical Executor and ActionLease path.
- Record geometry deadline misses and candidate-level timeouts.

- [x] Step 1: Write tests proving provider sees the normalized fingerprint and timeout does not block scheduler completion.
- [x] Step 2: Run pytest tests/test_phase5_geometry_scheduler.py -q and verify failure.
- [x] Step 3: Implement scheduler wiring and run-config fields.
- [x] Step 4: Run pytest tests/test_phase5_geometry_scheduler.py -q and verify pass.
- [ ] Step 5: Commit with git commit -m "feat: bind normalized geometry evidence to scheduler runs".

### Task 6: Add immutable Phase 5 experiment artifacts

Files:
- Create: capmas/evaluation/phase5_artifacts.py
- Create: tests/test_phase5_artifacts.py
- Modify: scripts/run_libero_b3_llm.py
- Modify: docs/experiments.md

Interfaces:
- Phase5RunDirectory.create(root, experiment_name, run_id) -> Phase5RunDirectory
- Phase5RunDirectory.write_json(name, payload) -> Path
- Phase5RunDirectory.finalize_manifest() -> Path
- Phase5RunDirectory.log_path(name="runner.log") -> Path

Rules:

- Create logs, results, traces, evidence, and artifacts subdirectories.
- Redact api_key, authorization, and provider headers before writing.
- Publish files atomically with a temporary file and os.replace.
- manifest.json records file size and SHA-256.
- Failed runs retain failure JSON and complete logs.

- [x] Step 1: Write tests for unique non-overwriting run directories, checksum manifests, and secret-free config.
- [x] Step 2: Run pytest tests/test_phase5_artifacts.py -q and verify failure.
- [x] Step 3: Implement the artifact manager and integrate the LLM runner's output path.
- [x] Step 4: Run pytest tests/test_phase5_artifacts.py -q and verify pass.
- [ ] Step 5: Commit with git commit -m "feat: isolate and checksum Phase 5 experiment artifacts".

### Task 7: Run the realistic five-seed P5.2 pilot

Files:
- Create: scripts/run_libero_p52_geometry.py
- Create: tests/test_libero_p52_geometry.py
- Modify: docs/phase5-evidence-evolution.md
- Output: outputs/phase5/P5.2_geometry_evidence/<timestamp>_<run_id>/

Interfaces:
- The runner reuses build_capx_runtime_from_yaml, the staged LLM runner, SceneSnapshot, reference preview, and the single physical Executor seam.
- Modes are geometry_disabled, geometry_shadow, and geometry_online_bounded.
- Primary pilot seeds are (1, 2, 3, 4, 5) with realistic sensor-derived state.

- [x] Step 1: Write a runner contract test for mode, seeds, artifact directory, and used_privileged_state=False.
- [x] Step 2: Run pytest tests/test_libero_p52_geometry.py -q and verify failure.
- [x] Step 3: Implement one independent artifact directory per mode and seed, recording candidate fingerprints, geometry dimensions, Arbiter basis, gates, latency, action execution, verifier result, interventions, and evaluator success.
- [x] Step 4: Run pytest tests/test_libero_p52_geometry.py -q and verify pass.
- [x] Step 5: Execute the pilot with CUDA_VISIBLE_DEVICES=5 and the existing CAP-X LIBERO YAML. Preserve every success and failure log.
- [ ] Step 6: Commit with git commit -m "test: add realistic P5.2 geometry evidence pilot".

## Verification checklist

- [x] Focused P5.2 tests pass.
- [ ] Full existing pytest suite passes.
- [ ] python -m compileall capmas scripts passes.
- [ ] git diff --check passes.
- [x] Realistic pilot contains five seeds for all three geometry modes.
- [x] At least one controlled intent pair produces distinct geometry evidence.
- [x] No primary artifact has used_privileged_state=True.
- [x] P95 geometry wave latency is within 50 ms or the run is explicitly failed.
- [x] No physical execution occurs during preview.
- [x] Every run has run_config.json, manifest.json, summary.json, summary.md, and complete logs.
- [x] Every arbitration records evidence basis and fallback labels.

### Task 8: Close the CAP-X live local-map transport seam

Files:

- Create: `capmas/perception/capx_world_model.py`
- Modify: `capmas/backends/capx.py`
- Modify: `capmas/backends/capx_libero_factory.py`
- Modify: `scripts/run_libero_b3_llm.py`
- Test: `tests/test_capx_world_model_bridge.py`

Rules:

- Reuse the existing CAP-X RGB-D artifact boundary and reference
  `WorldModelService`; do not add a second execution or API registry path.
- Attach the map only when geometry mode is enabled.
- Preserve CAP-X scene-version ownership and fail open to the base snapshot if
  reference geometry cannot process a frame.
- Keep `freshness_ms` compatibility semantics separate from
  `processing_latency_ms`; do not convert a slow CAP-X grounding call into a
  false World Model failure.

- [x] Add the optional `SceneEnricher` backend seam and bridge tests.
- [x] Wire `SparseVoxelMap` into the live B3-LLM candidate evidence provider.
- [x] Record map version, processed observations, and bridge errors in artifacts.
- [x] Run a real endpoint-backed LIBERO execution with measurable geometry.

Transport-closure evidence (2026-07-29): the fresh run at
`outputs/phase5/P5.2_live_map_fix_20260729/B3-LLM/20260729_085300_5d0d420c-4302-41b3-82e1-2f0e7d59f055/`
completed physical execution with evaluator success, map version 4, four
candidate records, candidate-specific clearance scores, and maximum geometry
latency 23.83 ms. The five-seed post-transport pilot and grasp-quality
surface/contact adapter remain open; this is sufficient to unblock P5.3
engineering, not to claim a statistically significant downstream gain.
