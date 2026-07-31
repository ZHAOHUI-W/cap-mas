"""Run the isolated P5.4 versioned evidence-cache evaluation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Literal, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from capmas.contracts.action import SkillCall
from capmas.contracts.candidates import (
    CandidateEvidence,
    GraphCandidate,
    subgraph_fingerprint,
)
from capmas.contracts.core import SkillRef
from capmas.contracts.graph import CheckpointSpec, SubgraphNodeSpec, SubgraphSpec
from capmas.contracts.scene import SceneSnapshot
from capmas.evaluation.evidence_cache import (
    EvidenceCacheKey,
    VersionedEvidenceCache,
)
from capmas.evaluation.phase5_artifacts import Phase5RunDirectory


CacheMode = Literal["cache_disabled", "cache_enabled"]
CacheResult = Literal["publish", "hit", "miss", "stale_rejection", "disabled"]


@dataclass(frozen=True)
class CacheTraceEntry:
    operation_index: int
    operation: str
    candidate_id: str | None
    candidate_fingerprint: str | None
    scene_version: int
    cache_result: CacheResult
    provider_called: bool
    evidence_scene_version: int | None
    attached: bool


@dataclass(frozen=True)
class CacheModeMetrics:
    request_count: int
    provider_calls: int
    hits: int
    misses: int
    stale_rejections: int
    stores: int
    invalidations: int
    evictions: int
    current_scene_version: int | None
    size: int
    disabled_requests: int
    stale_attachments: int


@dataclass(frozen=True)
class CacheModeResult:
    mode: CacheMode
    metrics: CacheModeMetrics
    trace: tuple[CacheTraceEntry, ...]
    run_dir: Path


@dataclass(frozen=True)
class CacheEvaluationReport:
    control: CacheModeResult
    enabled: CacheModeResult
    provider_call_reduction: float
    same_trace: bool


@dataclass(frozen=True)
class _TraceSpec:
    operation: Literal["publish", "query", "stale_probe"]
    candidate_id: str | None
    scene_version: int


@dataclass(frozen=True)
class _Fixture:
    scenes: Mapping[int, SceneSnapshot]
    candidates: Mapping[tuple[str, int], GraphCandidate]


class _DeterministicProvider:
    def __init__(self) -> None:
        self.calls = 0
        self._scores = {
            "candidate-a": 0.80,
            "candidate-b": 0.60,
            "candidate-c": 0.70,
        }

    def call(self, candidate: GraphCandidate, scene: SceneSnapshot) -> CandidateEvidence:
        self.calls += 1
        return CandidateEvidence(
            rehearsal_success_rate=self._scores.get(candidate.candidate_id, 0.5),
            available_metrics=("rehearsal",),
            scene_version=scene.scene_version,
            provider="p5.4-deterministic",
            captured_at_ns=scene.publish_timestamp_ns,
        )


def _build_subgraph(description: str) -> SubgraphSpec:
    node = SubgraphNodeSpec(
        node_id="act",
        description=description,
        skill_calls=(SkillCall(SkillRef("noop", "1.0.0"), {}),),
        postconditions=("scene_advanced",),
        proposed_by="p5.4-fixture",
    )
    return SubgraphSpec(
        subgraph_id="pick",
        subgoal_id="pick",
        description="deterministic cache fixture",
        nodes=(node,),
        edges=(),
        entry_node="act",
        success_nodes=("act",),
        failure_nodes=("act",),
        checkpoints=(CheckpointSpec("check", ("scene_advanced",)),),
    )


def _candidate(
    candidate_id: str,
    scene_version: int,
    subgraph: SubgraphSpec,
) -> GraphCandidate:
    return GraphCandidate(
        candidate_id=candidate_id,
        subgraph=subgraph,
        parent_scene_version=scene_version,
        producer_agent="p5.4-fixture",
        raw_subgraph=subgraph,
    )


def _build_fixture(seed: int) -> _Fixture:
    del seed
    scene_v1 = SceneSnapshot(
        episode_id="p5.4-cache-evaluation",
        episode_epoch=1,
        scene_version=1,
        sensor_timestamp_ns=1_000_000_000,
        publish_timestamp_ns=1_000_000_100,
        robot={"fixture": True},
    )
    scene_v2 = SceneSnapshot(
        episode_id="p5.4-cache-evaluation",
        episode_epoch=1,
        scene_version=2,
        sensor_timestamp_ns=2_000_000_000,
        publish_timestamp_ns=2_000_000_100,
        robot={"fixture": True},
    )

    subgraph_a = _build_subgraph("candidate A")
    subgraph_b = _build_subgraph("candidate B")
    subgraph_c = _build_subgraph("candidate C")
    candidates = {
        ("candidate-a", 1): _candidate("candidate-a", 1, subgraph_a),
        ("candidate-a", 2): _candidate("candidate-a", 2, subgraph_a),
        ("candidate-b", 1): _candidate("candidate-b", 1, subgraph_b),
        ("candidate-b", 2): _candidate("candidate-b", 2, subgraph_b),
        ("candidate-c", 2): _candidate("candidate-c", 2, subgraph_c),
    }
    return _Fixture(scenes={1: scene_v1, 2: scene_v2}, candidates=candidates)


def _trace_spec() -> tuple[_TraceSpec, ...]:
    return (
        _TraceSpec("publish", None, 1),
        _TraceSpec("query", "candidate-a", 1),
        _TraceSpec("query", "candidate-b", 1),
        _TraceSpec("query", "candidate-a", 1),
        _TraceSpec("query", "candidate-b", 1),
        _TraceSpec("publish", None, 2),
        _TraceSpec("stale_probe", "candidate-a", 1),
        _TraceSpec("query", "candidate-a", 2),
        _TraceSpec("query", "candidate-b", 2),
        _TraceSpec("query", "candidate-a", 2),
        _TraceSpec("query", "candidate-c", 2),
    )


def _metrics_from_cache(
    *,
    provider_calls: int,
    request_count: int,
    stale_attachments: int,
    disabled_requests: int,
    cache: VersionedEvidenceCache | None,
) -> CacheModeMetrics:
    if cache is None:
        return CacheModeMetrics(
            request_count=request_count,
            provider_calls=provider_calls,
            hits=0,
            misses=0,
            stale_rejections=0,
            stores=0,
            invalidations=0,
            evictions=0,
            current_scene_version=None,
            size=0,
            disabled_requests=disabled_requests,
            stale_attachments=stale_attachments,
        )
    stats = cache.stats()
    return CacheModeMetrics(
        request_count=request_count,
        provider_calls=provider_calls,
        hits=stats.hits,
        misses=stats.misses,
        stale_rejections=stats.stale_rejections,
        stores=stats.stores,
        invalidations=stats.invalidations,
        evictions=stats.evictions,
        current_scene_version=stats.current_scene_version,
        size=stats.size,
        disabled_requests=disabled_requests,
        stale_attachments=stale_attachments,
    )


def _write_mode_artifacts(
    run_dir: Phase5RunDirectory,
    *,
    mode: CacheMode,
    seed: int,
    trace: Sequence[CacheTraceEntry],
    metrics: CacheModeMetrics,
    cache: VersionedEvidenceCache | None,
    max_entries: int,
    event_limit: int,
) -> None:
    run_dir.write_json(
        "run_config.json",
        {
            "experiment": "P5.4_evidence_cache_evaluation",
            "mode": mode,
            "seed": seed,
            "trace_version": "p5.4-v1",
            "cache_max_entries": max_entries,
            "cache_event_limit": event_limit,
            "candidate_ids": ["candidate-a", "candidate-b", "candidate-c"],
            "scene_versions": [1, 2],
            "execution_scope": "local_deterministic_no_llm_no_robot",
        },
    )
    run_dir.write_json(
        "results/cache_trace.json",
        [asdict(entry) for entry in trace],
    )
    events = [] if cache is None else [asdict(event) for event in cache.events()]
    run_dir.write_json(
        "results/summary.json",
        {
            "mode": mode,
            "metrics": asdict(metrics),
            "cache_events": events,
            "acceptance_assertions": {
                "has_exact_hit": metrics.hits > 0 if mode == "cache_enabled" else True,
                "has_store": metrics.stores > 0 if mode == "cache_enabled" else True,
                "has_invalidation": (
                    metrics.invalidations > 0 if mode == "cache_enabled" else True
                ),
                "has_stale_rejection": (
                    metrics.stale_rejections > 0 if mode == "cache_enabled" else True
                ),
                "no_stale_attachments": metrics.stale_attachments == 0,
            },
        },
    )
    run_dir.write_text(
        "summary.md",
        "# CAP-MAS P5.4 evidence cache evaluation\n\n"
        f"- mode: {mode}\n"
        f"- provider_calls: {metrics.provider_calls}\n"
        f"- hits: {metrics.hits}\n"
        f"- stale_rejections: {metrics.stale_rejections}\n"
        f"- invalidations: {metrics.invalidations}\n"
        f"- stale_attachments: {metrics.stale_attachments}\n",
    )
    run_dir.write_text(
        "logs/runner.log",
        "\n".join(
            [
                "experiment=P5.4_evidence_cache_evaluation",
                f"mode={mode}",
                f"seed={seed}",
                f"trace_entries={len(trace)}",
                f"provider_calls={metrics.provider_calls}",
                f"hits={metrics.hits}",
                f"stale_rejections={metrics.stale_rejections}",
                f"invalidations={metrics.invalidations}",
                "",
            ]
        ),
    )
    run_dir.finalize_manifest()


def _write_failure_artifacts(
    run_dir: Phase5RunDirectory,
    *,
    mode: CacheMode,
    seed: int,
    trace: Sequence[CacheTraceEntry],
    error: BaseException,
) -> None:
    run_dir.write_json(
        "failure.json",
        {
            "status": "failed",
            "stage": "cache_trace",
            "mode": mode,
            "seed": seed,
            "error_type": type(error).__name__,
            "error": str(error),
        },
    )
    run_dir.write_json("results/cache_trace.json", [asdict(entry) for entry in trace])
    run_dir.write_text(
        "logs/runner.log",
        "\n".join(
            [
                "experiment=P5.4_evidence_cache_evaluation",
                f"mode={mode}",
                f"seed={seed}",
                "status=failed",
                f"error_type={type(error).__name__}",
                f"error={error}",
                f"trace_entries={len(trace)}",
                "",
            ]
        ),
    )
    run_dir.finalize_manifest()


def run_cache_mode(
    *,
    output_root: str | Path,
    mode: CacheMode,
    seed: int = 1,
    provider: object | None = None,
    max_entries: int = 256,
    event_limit: int = 512,
) -> CacheModeResult:
    if mode not in ("cache_disabled", "cache_enabled"):
        raise ValueError(f"unsupported cache mode: {mode}")
    fixture = _build_fixture(seed)
    provider_instance = provider or _DeterministicProvider()
    run_dir = Phase5RunDirectory.create(
        output_root,
        "P5.4_cache_evaluation",
        f"{mode}_seed{seed}",
    )
    run_dir.write_json(
        "run_config.json",
        {
            "experiment": "P5.4_evidence_cache_evaluation",
            "mode": mode,
            "seed": seed,
            "trace_version": "p5.4-v1",
            "cache_max_entries": max_entries,
            "cache_event_limit": event_limit,
            "candidate_ids": ["candidate-a", "candidate-b", "candidate-c"],
            "scene_versions": [1, 2],
            "execution_scope": "local_deterministic_no_llm_no_robot",
            "status": "running",
        },
    )

    cache = (
        VersionedEvidenceCache(max_entries=max_entries, event_limit=event_limit)
        if mode == "cache_enabled"
        else None
    )
    trace: list[CacheTraceEntry] = []
    request_count = 0
    stale_attachments = 0
    disabled_requests = 0

    try:
        for operation_index, spec in enumerate(_trace_spec()):
            if spec.operation == "publish":
                if cache is not None:
                    cache.advance_scene(spec.scene_version)
                trace.append(
                    CacheTraceEntry(
                        operation_index=operation_index,
                        operation=spec.operation,
                        candidate_id=None,
                        candidate_fingerprint=None,
                        scene_version=spec.scene_version,
                        cache_result="publish",
                        provider_called=False,
                        evidence_scene_version=None,
                        attached=False,
                    )
                )
                continue

            request_count += 1
            candidate = fixture.candidates[(spec.candidate_id, spec.scene_version)]
            scene = fixture.scenes[spec.scene_version]
            fingerprint = subgraph_fingerprint(candidate.subgraph)
            key = EvidenceCacheKey(fingerprint, spec.scene_version)
            evidence: CandidateEvidence | None = None
            provider_called = False
            attached = False

            if cache is not None:
                evidence = cache.get(key)
                cache_result: CacheResult = "hit" if evidence is not None else "miss"
                if evidence is None and spec.operation != "stale_probe":
                    evidence = provider_instance.call(candidate, scene)
                    provider_called = True
                    cache.put(key, evidence)
                if spec.operation != "stale_probe" and evidence is not None:
                    attached = True
                if spec.operation == "stale_probe":
                    cache_result = "stale_rejection"
            else:
                evidence = provider_instance.call(candidate, scene)
                provider_called = True
                disabled_requests += 1
                cache_result = "disabled"
                attached = spec.operation != "stale_probe"

            if spec.operation == "stale_probe" and attached:
                stale_attachments += 1

            trace.append(
                CacheTraceEntry(
                    operation_index=operation_index,
                    operation=spec.operation,
                    candidate_id=spec.candidate_id,
                    candidate_fingerprint=fingerprint,
                    scene_version=spec.scene_version,
                    cache_result=cache_result,
                    provider_called=provider_called,
                    evidence_scene_version=(
                        evidence.scene_version if evidence is not None else None
                    ),
                    attached=attached,
                )
            )

        metrics = _metrics_from_cache(
            provider_calls=int(getattr(provider_instance, "calls", 0)),
            request_count=request_count,
            stale_attachments=stale_attachments,
            disabled_requests=disabled_requests,
            cache=cache,
        )
        _write_mode_artifacts(
            run_dir,
            mode=mode,
            seed=seed,
            trace=trace,
            metrics=metrics,
            cache=cache,
            max_entries=max_entries,
            event_limit=event_limit,
        )
        return CacheModeResult(mode, metrics, tuple(trace), run_dir.path)
    except BaseException as error:
        _write_failure_artifacts(
            run_dir,
            mode=mode,
            seed=seed,
            trace=trace,
            error=error,
        )
        raise


def _trace_projection(result: CacheModeResult) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (entry.operation, entry.candidate_id, entry.scene_version)
        for entry in result.trace
    )


def _write_pair_comparison(
    result: CacheModeResult,
    *,
    control: CacheModeResult,
    enabled: CacheModeResult,
    provider_call_reduction: float,
    same_trace: bool,
) -> None:
    result.run_dir.joinpath("results", "paired_comparison.json").write_text(
        __import__("json").dumps(
            {
                "same_trace": same_trace,
                "provider_call_reduction": provider_call_reduction,
                "control_provider_calls": control.metrics.provider_calls,
                "enabled_provider_calls": enabled.metrics.provider_calls,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    result.run_dir.joinpath("manifest.json").unlink(missing_ok=True)
    Phase5RunDirectory(result.run_dir).finalize_manifest()


def run_cache_evaluation(
    *,
    output_root: str | Path,
    seed: int = 1,
    max_entries: int = 256,
    event_limit: int = 512,
) -> CacheEvaluationReport:
    control = run_cache_mode(
        output_root=output_root,
        mode="cache_disabled",
        seed=seed,
        max_entries=max_entries,
        event_limit=event_limit,
    )
    enabled = run_cache_mode(
        output_root=output_root,
        mode="cache_enabled",
        seed=seed,
        max_entries=max_entries,
        event_limit=event_limit,
    )
    if control.metrics.provider_calls <= 0:
        raise AssertionError("control provider must receive at least one call")
    provider_call_reduction = (
        control.metrics.provider_calls - enabled.metrics.provider_calls
    ) / control.metrics.provider_calls
    same_trace = _trace_projection(control) == _trace_projection(enabled)
    if not same_trace:
        raise AssertionError("cache modes did not execute the same request trace")
    if enabled.metrics.hits <= 0:
        raise AssertionError("enabled cache produced no exact hits")
    if enabled.metrics.invalidations <= 0:
        raise AssertionError("enabled cache produced no invalidations")
    if enabled.metrics.stale_rejections <= 0:
        raise AssertionError("enabled cache produced no stale rejection")
    if enabled.metrics.stale_attachments != 0:
        raise AssertionError("enabled cache attached stale evidence")
    if enabled.metrics.provider_calls >= control.metrics.provider_calls:
        raise AssertionError("cache did not reduce provider calls")

    for result in (control, enabled):
        _write_pair_comparison(
            result,
            control=control,
            enabled=enabled,
            provider_call_reduction=provider_call_reduction,
            same_trace=same_trace,
        )
    return CacheEvaluationReport(
        control=control,
        enabled=enabled,
        provider_call_reduction=provider_call_reduction,
        same_trace=same_trace,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs/phase5")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-entries", type=int, default=256)
    parser.add_argument("--event-limit", type=int, default=512)
    args = parser.parse_args(argv)
    report = run_cache_evaluation(
        output_root=args.output_root,
        seed=args.seed,
        max_entries=args.max_entries,
        event_limit=args.event_limit,
    )
    print(f"control_run_dir={report.control.run_dir}")
    print(f"enabled_run_dir={report.enabled.run_dir}")
    print(f"control_provider_calls={report.control.metrics.provider_calls}")
    print(f"enabled_provider_calls={report.enabled.metrics.provider_calls}")
    print(f"provider_call_reduction={report.provider_call_reduction:.6f}")
    print(f"enabled_hits={report.enabled.metrics.hits}")
    print(f"enabled_invalidations={report.enabled.metrics.invalidations}")
    print(f"enabled_stale_rejections={report.enabled.metrics.stale_rejections}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
