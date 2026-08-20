# P5.3.2 Object-6 Effective Motion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Bind, preview, and execute the same decision-time full pick-and-place motion program for object-6, while recording and rejecting semantically duplicate candidates.

**Architecture:** capmas.perception.effective_motion is a pure graph/scene transformation: it creates a full successful-path program and materializes symbolic pose references to its bound literals. The reference preview emits conservative segment aggregates. The live evidence session and online runner opt into this path, preserve legacy subgraph behavior, and record identifiability before the sole physical execution.

**Tech Stack:** Python dataclasses, pytest, CAP-MAS graph contracts, CAP-X/LIBERO compatibility runner, JSON experiment artifacts, Ruff.

## Global Constraints

- Preserve CAP-X skill IDs and the existing FixedGraphInterpreter execution API.
- Never mutate, filter, relabel, or use P5.6D seed-32--51 rows as new data.
- Do not change calibration, Shadow Arbiter, canary, learned grasp selection, TSDF, semantic adapters, or P5.6 features.
- Unresolved binding, stale scene/map, deadline expiry, or missing segment geometry is typed unknown, never zero.
- Keep legacy candidate_geometry_evidence() and PreExecutionEvidenceSession.candidate_evidence() behavior intact by default.
- New physical work uses GPU 5 only, one reset and at most one submitted candidate per seed; do not commit outputs/.
- Each code change begins with a focused failing pytest test and ends with focused green verification.

---

## File Structure

- capmas/perception/effective_motion.py: pure execution context, segment binding, fingerprints, and materialization.
- capmas/perception/motion_preview.py: segment/program preview values and reference implementation.
- capmas/perception/geometry_evidence.py: optional mission-suffix evidence and conservative aggregation.
- capmas/contracts/candidates.py: backward-compatible geometry provenance and identifiability values.
- capmas/agents/candidate_diversity.py: semantic/evidence comparison and regeneration decision.
- capmas/evaluation/libero_evidence_session.py: opt-in prepared candidate evidence/execution.
- scripts/run_libero_p53_online.py: opt-in mission-suffix orchestration and audit artifact output.
- scripts/create_p532_object6_manifest.py and scripts/run_libero_p532_object6.py: deterministic, independently pre-registered capability experiment.
- tests/test_effective_motion.py, tests/test_candidate_diversity.py, and tests/test_p532_manifest.py: new unit/dry-run coverage.

## Task 1: Bind and Materialize an Effective Motion Program

**Files:**
- Create: capmas/perception/effective_motion.py
- Create: tests/test_effective_motion.py
- Modify: capmas/contracts/__init__.py

**Interfaces:**

~~~python
@dataclass(frozen=True)
class CandidateExecutionContext:
    candidate: GraphCandidate
    mission_graph: MissionGraph
    selected_subgraph_id: str
    execution_graph_fingerprint: str

@dataclass(frozen=True)
class EffectiveMotionSegment:
    segment_id: str
    kind: Literal["grasp_approach", "lift", "transfer", "place_approach", "release"]
    source_subgraph_id: str
    source_node_id: str
    start_pose_wxyz_xyz: tuple[float, ...] | None
    end_pose_wxyz_xyz: tuple[float, ...] | None
    approach_vector_xyz: tuple[float, float, float] | None
    approach_distance_m: float | None
    payload_track_id: str | None

@dataclass(frozen=True)
class EffectiveMotionProgram:
    candidate_fingerprint: str
    execution_graph_fingerprint: str
    program_fingerprint: str
    decision_scene_version: int
    selected_subgraph_id: str
    segments: tuple[EffectiveMotionSegment, ...]
    semantic_signature: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

def execution_graph_fingerprint(graph: MissionGraph) -> str:
    raise NotImplementedError

def bind_effective_motion(
    context: CandidateExecutionContext, scene: SceneSnapshot
) -> EffectiveMotionProgram:
    raise NotImplementedError

def materialize_execution_graph(
    program: EffectiveMotionProgram, graph: MissionGraph
) -> MissionGraph:
    raise NotImplementedError
~~~

- [ ] **Step 1: Write a failing full-path extraction test**

~~~python
def test_bind_effective_motion_extracts_pick_lift_transfer_place_release() -> None:
    context, scene = _pick_place_context()
    program = bind_effective_motion(context, scene)
    assert [item.kind for item in program.segments] == [
        "grasp_approach", "lift", "transfer", "place_approach", "release",
    ]
    assert program.segments[0].start_pose_wxyz_xyz[-1] == pytest.approx(0.15)
    assert program.segments[1].end_pose_wxyz_xyz[-1] == pytest.approx(0.22)
    assert program.segments[3].end_pose_wxyz_xyz[-3:] == pytest.approx((0.60, 0.25, 0.04))

def test_materialized_graph_uses_bound_literal_poses() -> None:
    context, scene = _pick_place_context()
    graph = materialize_execution_graph(bind_effective_motion(context, scene), context.mission_graph)
    assert graph.subgraph("sg_pick").node("pick").skill_calls[1].args["position"] == [0.5, 0.2, 0.1]
    assert graph.subgraph("sg_place").node("place").skill_calls[0].args["position"] == [0.6, 0.25, 0.04]

def test_binding_rejects_wrong_scene_or_selected_subgraph() -> None:
    context, scene = _pick_place_context()
    with pytest.raises(ValueError, match="decision scene"):
        bind_effective_motion(context, replace(scene, scene_version=2))
    with pytest.raises(ValueError, match="selected subgraph"):
        bind_effective_motion(replace(context, selected_subgraph_id="sg_place"), scene)
~~~

The fixture uses typed sample_grasp_pose, symbolic pick goto_pose,
close_gripper, symbolic lift_after_grasp, literal place goto_pose, and
open_gripper; use z_approach=0.05, z_lift=0.12, and place z_approach=0.08.

- [ ] **Step 2: Verify RED**

Run: pytest tests/test_effective_motion.py -q

Expected: ModuleNotFoundError for capmas.perception.effective_motion.

- [ ] **Step 3: Implement the pure builder**

~~~python
def bind_effective_motion(context: CandidateExecutionContext, scene: SceneSnapshot) -> EffectiveMotionProgram:
    if scene.scene_version != context.candidate.parent_scene_version:
        raise ValueError("decision scene version does not match candidate")
    segments = _segments_for_success_suffix(context.mission_graph, context.selected_subgraph_id)
    return EffectiveMotionProgram(
        candidate_fingerprint=subgraph_fingerprint(context.candidate.subgraph),
        execution_graph_fingerprint=context.execution_graph_fingerprint,
        program_fingerprint=_sha256(_program_json(context, scene.scene_version, segments)),
        decision_scene_version=scene.scene_version,
        selected_subgraph_id=context.selected_subgraph_id,
        segments=tuple(segments),
        semantic_signature=_semantic_signature(segments),
    )
~~~

Hash mission_graph_to_dict() with sorted compact JSON. Follow one and only one
MissionEdge(condition="success") from selected subgraph to success subgraph;
raise ValueError("unsupported successful control-flow branch") for zero or
multiple nonterminal successors. Resolve symbolic grasp outputs only from the
normalized MotionIntent target pose, never by calling CAP-X. Emit five explicit
segments: approach-offset to grasp, grasp to lifted pose, lift to place
pre-approach, pre-approach to release pose, and zero-length release.

- [ ] **Step 4: Implement materialization**

~~~python
def materialize_execution_graph(program: EffectiveMotionProgram, graph: MissionGraph) -> MissionGraph:
    if program.execution_graph_fingerprint != execution_graph_fingerprint(graph):
        raise ValueError("program execution graph fingerprint does not match graph")
    return replace(graph, subgraphs=tuple(_materialize_subgraph(program, sg) for sg in graph.subgraphs))
~~~

Replace only goto_pose.position, goto_pose.quaternion_wxyz,
lift_after_grasp.position, and lift_after_grasp.quaternion_wxyz when they are
SkillOutputRef values or serialized output-reference mappings. Preserve skill
IDs, z_approach, z_lift, all other arguments, and topology.

- [ ] **Step 5: Verify GREEN and commit**

Run: pytest tests/test_effective_motion.py -q

Expected: 3 passed.

~~~sh
git add capmas/perception/effective_motion.py capmas/contracts/__init__.py tests/test_effective_motion.py
git commit -m "feat: bind object6 effective motion programs"
~~~

## Task 2: Preview the Full Program and Preserve Geometry Provenance

**Files:**
- Modify: capmas/contracts/candidates.py
- Modify: capmas/perception/motion_preview.py
- Modify: capmas/perception/geometry_evidence.py
- Modify: tests/test_phase5_geometry_contracts.py
- Modify: tests/test_phase5_geometry_provider.py

**Interfaces:**

~~~python
@dataclass(frozen=True)
class SegmentMotionPreview:
    segment_id: str
    ik_valid: bool | None
    collision_free: bool | None
    clearance_m: float | None
    path_length_m: float | None
    reason: str
    start_pose_wxyz_xyz: tuple[float, ...] | None
    end_pose_wxyz_xyz: tuple[float, ...] | None
    sampled_points_xyz: tuple[tuple[float, float, float], ...] = ()
    occupied_points_xyz: tuple[tuple[float, float, float], ...] = ()

@dataclass(frozen=True)
class ProgramMotionPreview:
    segments: tuple[SegmentMotionPreview, ...]
    aggregate_status: Literal["feasible", "infeasible", "unknown"]
    scene_version: int
    map_version: int | None
    corridor_radius_m: float

def candidate_geometry_evidence(
    candidate: GraphCandidate,
    scene: SceneSnapshot,
    local_map: LocalMapBackend | None,
    preview_backend: MotionPreviewBackend,
    deadline_ns: int,
    *,
    program: EffectiveMotionProgram | None = None,
) -> GeometryEvidence:
    raise NotImplementedError
~~~

- [ ] **Step 1: Write failing program-preview tests**

~~~python
def test_program_preview_distinguishes_place_approach_lengths() -> None:
    preview = ReferenceMotionPreview(corridor_samples=5)
    short = preview.preview_program(_program(place_approach_m=0.05), _scene(), _map_with_place_obstacle())
    long = preview.preview_program(_program(place_approach_m=0.20), _scene(), _map_with_place_obstacle())
    assert short.by_segment("place_approach").collision_free is True
    assert long.by_segment("place_approach").collision_free is False

def test_program_geometry_is_conservative_and_has_program_lineage() -> None:
    program = _program(place_approach_m=0.20)
    evidence = candidate_geometry_evidence(_candidate(), _scene(), _map_with_place_obstacle(), ReferenceMotionPreview(), _deadline(), program=program)
    assert evidence.program_scope == "mission_suffix"
    assert evidence.clearance.score == 0.0
    assert evidence.collision_risk.score == 1.0
    assert evidence.program_fingerprint == program.program_fingerprint

def test_legacy_geometry_keeps_default_program_provenance() -> None:
    evidence = candidate_geometry_evidence(_candidate(), _scene(), _map(), ReferenceMotionPreview(), _deadline())
    assert evidence.program_scope == "subgraph"
    assert evidence.execution_graph_fingerprint is None
~~~

- [ ] **Step 2: Verify RED**

Run: pytest tests/test_phase5_geometry_provider.py tests/test_phase5_geometry_contracts.py -q

Expected: missing preview_program and provenance attributes.

- [ ] **Step 3: Add compatible values and implement segment sampling**

Append to GeometryEvidence, with defaults, execution_graph_fingerprint,
program_fingerprint, program_scope: Literal["subgraph", "mission_suffix"] =
"subgraph", and segment_artifact_refs. Validate non-empty present fingerprints
and valid scope. Implement preview_program() by sampling every explicit
start-to-end segment, including endpoints, with max(2, corridor_samples)
points. Query with existing corridor radius, retain sampled/occupied points,
and return unknown for absent endpoint/map/clearance rather than using fixed
approach distance.

- [ ] **Step 4: Add evidence aggregation**

For supplied program, verify candidate fingerprint and scene version. Set
reachability fail if any measurable segment has invalid IK, clearance to minimum
measurable clearance, collision risk to maximum measurable risk, and grasp
quality only when grasp pose plus approach are bound. Carry program/graph
fingerprints. Keep current first-intent behavior when no program is supplied.

- [ ] **Step 5: Verify GREEN and commit**

Run: pytest tests/test_phase5_geometry_provider.py tests/test_phase5_geometry_contracts.py tests/test_phase5_geometry_arbiter.py -q

Expected: all selected tests pass.

~~~sh
git add capmas/contracts/candidates.py capmas/perception/motion_preview.py capmas/perception/geometry_evidence.py tests/test_phase5_geometry_contracts.py tests/test_phase5_geometry_provider.py
git commit -m "feat: preview full effective motion programs"
~~~

## Task 3: Detect Equivalent Candidates and Provide Typed Abstention

**Files:**
- Create: capmas/agents/candidate_diversity.py
- Modify: capmas/contracts/candidates.py
- Modify: capmas/agents/arbiter.py
- Create: tests/test_candidate_diversity.py
- Modify: tests/test_phase5_geometry_arbiter.py

**Interfaces:**

~~~python
@dataclass(frozen=True)
class CandidateIdentifiability:
    candidate_id: str
    semantic_signature: str
    program_fingerprint: str | None
    candidate_semantic_equivalent: bool
    candidate_evidence_identical: bool
    selection_identifiable: bool
    abstention_reason: str | None

class CandidateDiversityValidator:
    def inspect(
        self,
        programs: Sequence[EffectiveMotionProgram],
        candidates: Sequence[GraphCandidate],
    ) -> CandidateDiversityDecision:
        raise NotImplementedError

@dataclass(frozen=True)
class CandidateDiversityDecision:
    identifiability: tuple[CandidateIdentifiability, ...]
    requires_regeneration: bool
    required_difference_fields: tuple[str, ...] = ()
    reason: str | None = None

class CandidateArbiter:
    def abstain(
        self,
        candidates: Iterable[GraphCandidate],
        code: str,
        reason: str,
    ) -> ArbitrationResult:
        raise NotImplementedError
~~~

- [ ] **Step 1: Write failing diversity tests**

~~~python
def test_policy_names_do_not_make_duplicate_programs_diverse() -> None:
    decision = CandidateDiversityValidator().inspect(_same_programs("policy-0", "safety"), _candidates())
    assert decision.requires_regeneration is True
    assert all(item.candidate_semantic_equivalent for item in decision.identifiability)

def test_different_programs_with_equal_evidence_are_not_identifiable() -> None:
    decision = CandidateDiversityValidator().inspect(_different_programs(), _equal_evidence_candidates())
    assert decision.requires_regeneration is False
    assert not any(item.selection_identifiable for item in decision.identifiability)

def test_typed_semantic_abstention_selects_nothing() -> None:
    result = CandidateArbiter().abstain(_candidates(), "CANDIDATE_SEMANTIC_EQUIVALENCE", "duplicate motion program")
    assert result.selected is None
    assert result.selection_basis == "candidate_semantic_equivalence"
    assert result.rejections[0].code == "CANDIDATE_SEMANTIC_EQUIVALENCE"
~~~

- [ ] **Step 2: Verify RED**

Run: pytest tests/test_candidate_diversity.py -q

Expected: import error for capmas.agents.candidate_diversity.

- [ ] **Step 3: Implement semantic/evidence signatures and abstention**

Semantic signature hashes only segment kind, payload, start/end pose, approach
vector, and approach distance. Evidence signature uses available metrics plus
geometry status/score, excluding candidate ID, policy/agent name, timestamps,
fingerprints, and refs. If all semantics match, require exactly
("grasp_pose_or_offset", "approach", "lift", "transfer", "place_or_release").
If programs differ but evidence is equal, mark evidence identical and not
selection-identifiable.

CandidateArbiter.abstain() returns directly with selected=None, original
considered candidates, one CandidateRejection("candidate-wave", code, reason),
and candidate_semantic_equivalence basis only for that code. It does not call
select().

- [ ] **Step 4: Verify GREEN and commit**

Run: pytest tests/test_candidate_diversity.py tests/test_phase5_geometry_arbiter.py -q

Expected: all selected tests pass; ordinary evidence selection remains green.

~~~sh
git add capmas/agents/candidate_diversity.py capmas/contracts/candidates.py capmas/agents/arbiter.py tests/test_candidate_diversity.py tests/test_phase5_geometry_arbiter.py
git commit -m "feat: audit candidate identifiability"
~~~

## Task 4: Prepare and Execute in the Same Live Session

**Files:**
- Modify: capmas/evaluation/libero_evidence_session.py
- Modify: capmas/evaluation/__init__.py
- Modify: tests/test_libero_evidence_session.py

**Interfaces:**

~~~python
class EffectiveMotionEvidenceSession(Protocol):
    def prepare_candidate(
        self, candidate: GraphCandidate, graph: MissionGraph
    ) -> PreparedCandidate:
        raise NotImplementedError

    def execute_prepared(self, prepared: PreparedCandidate) -> object:
        raise NotImplementedError

@dataclass(frozen=True)
class PreparedCandidate:
    context: CandidateExecutionContext
    program: EffectiveMotionProgram
    materialized_graph: MissionGraph
    evidence: CandidateEvidence
~~~

- [ ] **Step 1: Write a failing session test**

~~~python
def test_session_prepares_geometry_and_executes_the_same_materialized_graph() -> None:
    session, _backend = _session()
    scene = session.start()
    candidate, graph = _pick_place_candidate_and_graph(scene.scene_version)
    prepared = session.prepare_candidate(candidate, graph)
    result = session.execute_prepared(prepared)
    assert prepared.evidence.geometry.program_fingerprint == prepared.program.program_fingerprint
    assert result["graph"] == prepared.materialized_graph

def test_session_rejects_tampered_prepared_graph() -> None:
    session, _backend = _session()
    scene = session.start()
    candidate, graph = _pick_place_candidate_and_graph(scene.scene_version)
    prepared = session.prepare_candidate(candidate, graph)
    with pytest.raises(ValueError, match="execution graph fingerprint"):
        session.execute_prepared(replace(prepared, materialized_graph=_different_graph()))
~~~

- [ ] **Step 2: Verify RED**

Run: pytest tests/test_libero_evidence_session.py -q

Expected: missing prepare_candidate.

- [ ] **Step 3: Implement session preparation without breaking legacy injection**

Refactor current candidate_evidence(candidate) body into private
_collect_candidate_evidence(candidate, scene, program=None). The legacy public
method calls it with no program and keeps custom geometry_collector injection.
prepare_candidate() creates CandidateExecutionContext, binds the program,
collects program geometry through the Task 2
`candidate_geometry_evidence(candidate, scene, local_map, preview_backend, deadline_ns, program=program)`
interface,
materializes graph, and returns PreparedCandidate.

- [ ] **Step 4: Implement prepared execution and validate provenance**

execute_prepared() requires program scene equals retained decision scene,
program graph fingerprint equals original graph, materialized graph parent scene
equals retained scene, and no earlier execution. It invokes existing graph
executor once with materialized graph. Existing execute(candidate, graph)
remains unchanged. Export the new protocol.

- [ ] **Step 5: Verify GREEN and commit**

Run: pytest tests/test_libero_evidence_session.py tests/test_phase5_geometry_provider.py -q

Expected: all selected tests pass.

~~~sh
git add capmas/evaluation/libero_evidence_session.py capmas/evaluation/__init__.py tests/test_libero_evidence_session.py
git commit -m "feat: bind live evidence to effective motion"
~~~

## Task 5: Add Online Orchestration and Decision-Bound Artifacts

**Files:**
- Modify: scripts/run_libero_p53_online.py
- Modify: capmas/evaluation/feature_snapshots.py
- Modify: tests/test_libero_p53_online.py
- Modify: tests/test_p56_feature_snapshots.py

**Interfaces:**

~~~python
CandidateRegenerator = Callable[[Sequence[CandidateSpec], CandidateDiversityDecision], Sequence[CandidateSpec] | None]

def run_online_experiment(
    *,
    config_path: str,
    candidates: Sequence[CandidateSpec],
    seed: int,
    scene_version: int,
    mode: RehearsalMode,
    output_root: str | Path,
    pool_config: RehearsalPoolConfig,
    evidence_session: PreExecutionEvidenceSession | None = None,
    effective_motion_scope: Literal["subgraph", "mission_suffix"] = "subgraph",
    candidate_regenerator: CandidateRegenerator | None = None,
) -> OnlineSelectionOutcome:
    raise NotImplementedError
~~~

- [ ] **Step 1: Write failing online tests**

~~~python
def test_duplicate_programs_abstain_without_physical_execution(tmp_path) -> None:
    outcome = run_online_experiment(
        config_path="libero.yaml", candidates=_duplicate_specs(), seed=7,
        scene_version=1, mode="disabled", output_root=tmp_path / "runs",
        pool_config=RehearsalPoolConfig(max_workers=1, timeout_s=1.0),
        evidence_session=_EffectiveSession(_scene()), effective_motion_scope="mission_suffix",
    )
    assert outcome.physical_candidate_id is None
    assert outcome.report.live.selection_basis == "candidate_semantic_equivalence"
    assert json.loads((outcome.run_dir.path / "results/candidate_identifiability.json").read_text())[0]["candidate_semantic_equivalent"]

def test_one_regenerated_wave_executes_prepared_winner(tmp_path) -> None:
    calls = []
    outcome = run_online_experiment(
        config_path="libero.yaml", candidates=_duplicate_specs(), seed=7,
        scene_version=1, mode="disabled", output_root=tmp_path / "runs",
        pool_config=RehearsalPoolConfig(max_workers=1, timeout_s=1.0),
        evidence_session=_EffectiveSession(_scene()), effective_motion_scope="mission_suffix",
        candidate_regenerator=lambda specs, decision: (_record(calls), _diverse_specs())[1],
    )
    assert len(calls) == 1
    selection = json.loads((outcome.run_dir.path / "results/selection.json").read_text())
    assert selection["selected_program_fingerprint"]
~~~

- [ ] **Step 2: Verify RED**

Run: pytest tests/test_libero_p53_online.py -q

Expected: TypeError for unsupported effective_motion_scope.

- [ ] **Step 3: Implement one bounded regeneration and typed abstention**

In mission_suffix scope, require callable prepare_candidate and execute_prepared
or raise ValueError("mission_suffix scope requires an effective-motion evidence
session"). Prepare all candidates, replace graph mapping with materialized
graphs, inspect programs, then invoke candidate_regenerator at most once only
for semantic equivalence. Reject None, empty, or unchanged regenerated specs.
If no diversity remains after one try, use CandidateArbiter.abstain() and skip
rehearsal/physical execution. Never use regeneration to retry geometry,
simulator, or physical failures.

- [ ] **Step 4: Persist and validate provenance**

Before decision_completed_at_ns, write
results/candidate_identifiability.json and results/effective_motion_programs.json.
Add scope, regeneration count, identifiability counts, and program fingerprints
to run_config, selection, summary, and logs. Add graph/program provenance to
CandidateFeatureSnapshot.rewrite_metadata; mission-suffix evidence with absent
program fingerprint or mismatched graph fingerprint raises ValueError. Keep the
p56.feature.v1 vector unchanged.

- [ ] **Step 5: Verify GREEN and commit**

Run: pytest tests/test_libero_p53_online.py tests/test_p56_feature_snapshots.py tests/test_online_rehearsal.py -q

Expected: all selected tests pass; default subgraph-mode has one execution.

~~~sh
git add scripts/run_libero_p53_online.py capmas/evaluation/feature_snapshots.py tests/test_libero_p53_online.py tests/test_p56_feature_snapshots.py
git commit -m "feat: record effective motion identifiability"
~~~

## Task 6: Build a Preflightable Object-6 Capability Harness

**Files:**
- Create: scripts/create_p532_object6_manifest.py
- Create: scripts/run_libero_p532_object6.py
- Create: tests/test_p532_manifest.py

**Interfaces:**

~~~python
SCHEMA_VERSION = "p532.collection.v1"
def build_object6_manifest(seeds: Iterable[int], assets: Object6Assets) -> P532Manifest:
    raise NotImplementedError

def load_and_preflight(path: str | Path) -> P532Manifest:
    raise NotImplementedError

def run_capability(
    manifest_path: str | Path, *, output_root: str | Path, dry_run: bool
) -> P532RunResult:
    raise NotImplementedError
~~~

- [ ] **Step 1: Write failing manifest/dry-run tests**

~~~python
def test_manifest_is_deterministic_and_rejects_duplicate_seed_or_digest_mismatch(tmp_path) -> None:
    manifest = build_object6_manifest(range(52, 62), _assets())
    path = write_manifest(tmp_path / "manifest.json", manifest)
    assert load_and_preflight(path).manifest_sha256 == manifest_sha256(manifest)
    with pytest.raises(ValueError, match="duplicate collection task/seed"):
        validate_manifest(replace(manifest, cases=(manifest.cases[0], manifest.cases[0])))

def test_dry_run_constructs_no_live_session(tmp_path, monkeypatch) -> None:
    path = write_manifest(tmp_path / "manifest.json", build_object6_manifest(range(52, 62), _assets()))
    monkeypatch.setattr(module, "_live_session_factory", pytest.fail)
    assert run_capability(path, output_root=tmp_path / "outputs", dry_run=True).live_session_count == 0
~~~

- [ ] **Step 2: Verify RED**

Run: pytest tests/test_p532_manifest.py -q

Expected: import error for scripts.create_p532_object6_manifest.

- [ ] **Step 3: Implement manifest preflight and runner boundary**

Require ten unique non-negative seeds, SHA-256 asset hashes,
effective_motion_scope="mission_suffix", max_restarts=0, GPU "5", and one
physical execution maximum per case. write_manifest() refuses byte-different
overwrite. Dry-run validates and creates a unique P5.3.2 output folder without
CAP-X import or GPU allocation. The live branch runs cases serially, logs
stdout/stderr under each case logs/ directory, and counts physical reach,
evaluator success, infrastructure unknowns, semantic abstentions,
equal-evidence ties, evidence-selected decisions, safety abstentions, and
fingerprint mismatches.

- [ ] **Step 4: Verify GREEN and commit**

Run: pytest tests/test_p532_manifest.py -q

Expected: 2 passed with no GPU allocation.

~~~sh
git add scripts/create_p532_object6_manifest.py scripts/run_libero_p532_object6.py tests/test_p532_manifest.py
git commit -m "feat: add object6 capability experiment harness"
~~~

## Task 7: Document, Verify, and Stop Before Physical Seed Selection

**Files:**
- Modify: docs/phase5-evidence-evolution.md
- Modify: docs/implementation-roadmap.md
- Modify: docs/experiments.md
- Modify: docs/superpowers/specs/2026-08-20-p5-3-2-object6-effective-motion-design.md
- Modify: tests/test_phase5_docs.py

- [ ] **Step 1: Write a failing documentation test**

~~~python
def test_phase5_docs_record_effective_motion_capability_boundary() -> None:
    documents = [(ROOT / name).read_text(encoding="utf-8") for name in (
        "docs/phase5-evidence-evolution.md", "docs/implementation-roadmap.md", "docs/experiments.md",
    )]
    for document in documents:
        assert "EffectiveMotionProgram" in document
        assert "candidate_semantic_equivalence" in document
        assert "P5.6D" in document and "immutable" in document
~~~

- [ ] **Step 2: Verify RED**

Run: pytest tests/test_phase5_docs.py::test_phase5_docs_record_effective_motion_capability_boundary -q

Expected: failure until all documents state implemented boundary.

- [ ] **Step 3: Record the implementation boundary**

State that full-program binding, preview, materialized execution, and
identifiability are implemented; physical ten-seed gate is unrun; P5.6D is
immutable; and no calibration/shadow/canary effect exists. Mark the design
implemented only after Tasks 1-6 pass. Do not write a concrete manifest until
the user approves its seed range.

- [ ] **Step 4: Verify repository state**

~~~sh
pytest -q
python -m compileall -q capmas scripts
ruff check capmas/perception/effective_motion.py capmas/perception/motion_preview.py capmas/perception/geometry_evidence.py capmas/contracts/candidates.py capmas/agents/candidate_diversity.py capmas/agents/arbiter.py capmas/evaluation/libero_evidence_session.py capmas/evaluation/feature_snapshots.py scripts/run_libero_p53_online.py scripts/create_p532_object6_manifest.py scripts/run_libero_p532_object6.py tests/test_effective_motion.py tests/test_candidate_diversity.py tests/test_p532_manifest.py
git diff --check
~~~

Expected: pytest and compileall exit zero, Ruff has no touched-file violations,
and git diff --check prints nothing.

- [ ] **Step 5: Commit and stop for pre-registration approval**

~~~sh
git add docs/phase5-evidence-evolution.md docs/implementation-roadmap.md docs/experiments.md docs/superpowers/specs/2026-08-20-p5-3-2-object6-effective-motion-design.md tests/test_phase5_docs.py
git commit -m "docs: record effective motion capability gate"
~~~

Report verification results, then request one explicit seed range before
creating a manifest, allocating GPU 5, or running CAP-X/LIBERO.

## Plan Self-Review

| Requirement | Task |
| --- | --- |
| Full graph-level label/evidence alignment | 1, 4, 5 |
| Same bound program for preview and execution | 1, 2, 4 |
| Candidate semantic/evidence diagnostics | 3, 5 |
| Bounded regeneration or typed abstention | 3, 5 |
| Segment collision provenance | 2, 5 |
| Independent ten-seed capability gate | 6, 7 |
| P5.6D/calibration remain fail-closed | Global constraints, 7 |

All later-task types are introduced earlier; no production implementation is
specified without a preceding red test. The design has no unresolved
placeholder, no synchronous rehearsal dependency, and no physical experiment
before manifest approval.
