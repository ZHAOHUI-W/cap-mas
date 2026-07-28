from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlparse
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from capmas.perception.capx_depth import CAPXDepthDecoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the CAP-MAS Phase 4 replay or live CAP-X B5 World Model benchmark."
    )
    parser.add_argument("--recording", help="JSONL ObservationBundle recording")
    parser.add_argument("--config-path", help="CAP-X YAML for a live LIBERO observation run")
    parser.add_argument("--runtime", choices=("thread", "process"), default="thread")
    parser.add_argument("--output", default="outputs/capmas_libero_b5/episode.json")
    parser.add_argument("--log", default=None)
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--target-hz", type=float, default=20.0)
    parser.add_argument("--queue-capacity", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--object-name", default="akita black bowl")
    parser.add_argument("--target-name", default="plate")
    parser.add_argument("--max-observations", type=int, default=20)
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument(
        "--depth-subsample",
        type=int,
        default=16,
        help="Depth stride; 16 is the process-mode latency-safe default, 8 increases map density",
    )
    parser.add_argument(
        "--server-url",
        "--api-base",
        dest="api_base",
        default=os.getenv("CAPMAS_LLM_API_BASE"),
        help="Optional OpenAI-compatible endpoint to probe once; B5 itself is LLM-free",
    )
    parser.add_argument("--model", default=os.getenv("CAPMAS_LLM_MODEL", "gpt-5.4"))
    parser.add_argument("--api-key-env", default="CAPMAS_LLM_API_KEY")
    parser.add_argument("--api-key", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--llm-deadline-ms", type=int, default=30_000)
    parser.add_argument(
        "--require-llm-probe",
        action="store_true",
        help="fail the B5 run if the optional endpoint probe fails",
    )
    parser.add_argument("--skip-api-servers", action="store_true")
    args = parser.parse_args()
    if (args.recording is None) == (args.config_path is None):
        parser.error("provide exactly one of --recording or --config-path")
    return args


class EmptyDepthDecoder:
    def decode(self, frame, depth, artifact_store):
        del frame, depth, artifact_store
        return ()


def build_live_world_model(provider, *, depth_subsample: int = 8):
    """Build a World Model using the exact CAP-X provider artifact boundary."""
    from capmas.perception.geometry import ReferenceGeometryEstimator
    from capmas.perception.local_map import SparseVoxelMap
    from capmas.perception.tracking import KnownObjectTracker, ObjectMeasurement
    from capmas.perception.world_model import WorldModelService

    def measurements(observation):
        if observation.object_measurements:
            return observation.object_measurements
        tracks = provider.capture_object_tracks(
            timestamp_ns=observation.timestamp_ns,
            episode_id=observation.episode_id or "unknown",
            episode_epoch=observation.episode_epoch or 0,
        )
        return tuple(
            ObjectMeasurement(
                track_id=track.track_id,
                label=track.label,
                pose_wxyz_xyz=track.pose_wxyz_xyz,
                confidence=track.confidence,
                timestamp_ns=observation.timestamp_ns,
                covariance=track.covariance,
                evidence=track.evidence,
            )
            for track in tracks
        )

    return WorldModelService(
        geometry=ReferenceGeometryEstimator(
            artifact_store=provider.artifacts,
            depth_decoder=CAPXDepthDecoder(subsample=depth_subsample),
        ),
        local_map=SparseVoxelMap(voxel_size_m=0.01, local_radius_m=1.0),
        tracker=KnownObjectTracker(max_match_distance_m=0.08),
        artifact_store=provider.artifacts,
        measurement_provider=measurements,
    )


def _endpoint_host(endpoint: str | None) -> str | None:
    if not endpoint:
        return None
    return urlparse(endpoint).hostname or "unknown"


def probe_llm_endpoint(
    endpoint: str,
    model: str,
    *,
    api_key: str | None,
    api_key_env: str,
    deadline_ms: int,
) -> dict[str, object]:
    """Probe an endpoint without putting the key or response content in artifacts."""
    from capmas.llm.capx_compatible import CAPXCompatibleLLMClient
    from capmas.llm.protocol import LLMRequest

    started = time.monotonic()
    client = CAPXCompatibleLLMClient(
        endpoint=endpoint,
        model=model,
        api_key=api_key,
        api_key_env=api_key_env,
        structured_outputs=False,
        max_retries=0,
    )
    try:
        response = client.complete(
            LLMRequest(
                request_id=f"b5-probe-{uuid4()}",
                agent_name="b5-endpoint-probe",
                messages=(
                    {"role": "system", "content": "Reply with the single word OK."},
                    {"role": "user", "content": "Health check."},
                ),
                deadline_ms=deadline_ms,
                max_output_tokens=8,
            )
        )
        return {
            "status": "completed",
            "host": _endpoint_host(endpoint),
            "model": model,
            "latency_ms": response.latency_ms,
            "elapsed_ms": (time.monotonic() - started) * 1000.0,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "host": _endpoint_host(endpoint),
            "model": model,
            "elapsed_ms": (time.monotonic() - started) * 1000.0,
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }


def build_reference_world_model():
    from capmas.perception.artifact_bridge import FileArtifactStore
    from capmas.perception.artifacts import InMemoryArtifactStore
    from capmas.perception.geometry import ReferenceGeometryEstimator
    from capmas.perception.local_map import SparseVoxelMap
    from capmas.perception.tracking import KnownObjectTracker
    from capmas.perception.world_model import WorldModelService

    artifact_root = os.environ.get("CAPMAS_ARTIFACT_ROOT")
    artifact_store = FileArtifactStore(artifact_root) if artifact_root else InMemoryArtifactStore()
    geometry = ReferenceGeometryEstimator(
        artifact_store=artifact_store,
        depth_decoder=EmptyDepthDecoder(),
    )
    return WorldModelService(
        geometry=geometry,
        local_map=SparseVoxelMap(voxel_size_m=0.01, local_radius_m=1.0),
        tracker=KnownObjectTracker(max_match_distance_m=0.08),
        artifact_store=artifact_store,
    )


def reserve_log_path(requested: Path) -> Path:
    """Keep every B5 invocation's log instead of overwriting an earlier run."""
    if not requested.exists():
        return requested
    stamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = requested.suffix or ".log"
    stem = requested.name[: -len(requested.suffix)] if requested.suffix else requested.name
    candidate = requested.with_name(f"{stem}_{stamp}_{os.getpid()}{suffix}")
    counter = 2
    while candidate.exists():
        candidate = requested.with_name(f"{stem}_{stamp}_{os.getpid()}_{counter}{suffix}")
        counter += 1
    return candidate


def snapshot_summary(snapshot) -> dict[str, object] | None:
    if snapshot is None:
        return None
    return {
        "scene_version": snapshot.scene_version,
        "sensor_timestamp_ns": snapshot.sensor_timestamp_ns,
        "publish_timestamp_ns": snapshot.publish_timestamp_ns,
        "processing_latency_ms": snapshot.processing_latency_ms,
        "freshness_ms": snapshot.freshness_ms,
        "object_tracks": [
            {
                "track_id": track.track_id,
                "label": track.label,
                "confidence": track.confidence,
                "track_status": track.track_status,
            }
            for track in snapshot.objects
        ],
        "source_artifact_count": len(snapshot.source_artifacts),
        "source_artifact_bytes": sum(
            artifact.byte_size or 0 for artifact in snapshot.source_artifacts
        ),
        "source_artifact_media_types": sorted(
            {artifact.media_type for artifact in snapshot.source_artifacts}
        ),
        "local_map_uri": snapshot.local_map.uri if snapshot.local_map is not None else None,
        "scene_confidence": snapshot.uncertainty.scene_confidence,
    }


def _prepare_capx_imports() -> tuple[Path, object]:
    project_root = PROJECT_ROOT
    capx_root = project_root.parent / "cap-x"
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/capmas-mpl")
    for path in (
        capx_root / "capx" / "third_party" / "libero_dependencies" / "robosuite",
        capx_root / "capx" / "third_party" / "LIBERO-PRO",
        capx_root,
    ):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from capx.envs.configs.loader import DictLoader

    return capx_root, DictLoader


def main() -> None:
    args = parse_args()
    if args.max_observations <= 0:
        raise SystemExit("--max-observations must be positive")
    if args.duration_s is not None and args.duration_s <= 0.0:
        raise SystemExit("--duration-s must be positive")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    requested_log_path = Path(args.log) if args.log else output_path.with_suffix(".log")
    log_path = reserve_log_path(requested_log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=(logging.FileHandler(log_path), logging.StreamHandler()),
        force=True,
    )
    logger = logging.getLogger("capmas.b5")

    from capmas.perception.artifact_bridge import FileArtifactStore
    from capmas.perception.metrics import RealTimeMetrics
    from capmas.backends.capx import CAPXStreamingObservationSource
    from capmas.perception.sensor_sync import (
        BoundedSensorSynchronizer,
        JsonlReplaySource,
    )
    from capmas.perception.world_model import (
        ProcessWorldModelRuntime,
        SynchronizerConfig,
        ThreadWorldModelRuntime,
        WorldModelRuntimeConfig,
    )

    artifact_root = Path(args.artifact_root or output_path.parent / "artifacts")
    artifact_root.mkdir(parents=True, exist_ok=True)
    endpoint_probe = None
    if args.api_base:
        endpoint_probe = probe_llm_endpoint(
            args.api_base,
            args.model,
            api_key=args.api_key,
            api_key_env=args.api_key_env,
            deadline_ms=args.llm_deadline_ms,
        )
        logger.info(
            "LLM endpoint probe status=%s host=%s model=%s latency_ms=%s",
            endpoint_probe["status"],
            endpoint_probe["host"],
            endpoint_probe["model"],
            endpoint_probe.get("latency_ms", endpoint_probe.get("elapsed_ms")),
        )
        if args.require_llm_probe and endpoint_probe["status"] != "completed":
            raise RuntimeError("required LLM endpoint probe failed")

    server_processes = []
    capx_bundle = None
    initial_scene = None
    live_mode = args.config_path is not None and args.recording is None
    if live_mode:
        _, loader_type = _prepare_capx_imports()
        config = loader_type.load(args.config_path)
        if not args.skip_api_servers:
            from capx.envs.runner import _start_api_servers

            server_processes = _start_api_servers(config.get("api_servers"))
        from capmas.backends.capx_libero_factory import build_capx_runtime_from_yaml

        capture_artifact_store = None
        if args.runtime == "process":
            from capmas.perception.artifact_bridge import EncodedArtifactStore, NumpyArtifactCodec

            capture_artifact_store = EncodedArtifactStore(
                FileArtifactStore(artifact_root, fsync=False),
                NumpyArtifactCodec(),
            )

        capx_bundle = build_capx_runtime_from_yaml(
            args.config_path,
            loader=lambda _: config,
            object_names=(args.object_name, args.target_name),
            artifact_store=capture_artifact_store,
        )
        episode = capx_bundle.backend.reset(seed=args.seed)
        initial_scene = episode.initial_scene
        source = CAPXStreamingObservationSource(
            capx_bundle.observation_provider,
            source="capx-libero-live",
            episode_id=episode.handle.episode_id,
            episode_epoch=episode.handle.episode_epoch,
        )
        if args.runtime == "thread":
            service = build_live_world_model(
                capx_bundle.observation_provider,
                depth_subsample=args.depth_subsample,
            )
            runtime = ThreadWorldModelRuntime(
                service=service,
                synchronizer=BoundedSensorSynchronizer(
                    capacity=args.queue_capacity,
                    episode_id=episode.handle.episode_id,
                    episode_epoch=episode.handle.episode_epoch,
                ),
                config=WorldModelRuntimeConfig(queue_capacity=args.queue_capacity),
                clock=time.time_ns,
            )
        else:
            from capmas.backends.capx_libero_factory import CAPXProcessWorldModelFactory

            runtime = ProcessWorldModelRuntime(
                service_factory=CAPXProcessWorldModelFactory(
                    str(artifact_root),
                    depth_subsample=args.depth_subsample,
                ),
                synchronizer_config=SynchronizerConfig(
                    queue_capacity=args.queue_capacity,
                    max_age_ms=150.0,
                ),
                config=WorldModelRuntimeConfig(queue_capacity=args.queue_capacity),
                artifact_store=FileArtifactStore(artifact_root, fsync=False),
            )
    elif args.runtime == "thread":
        from capmas.perception.artifacts import InMemoryArtifactStore

        store = InMemoryArtifactStore()
        from capmas.perception.geometry import ReferenceGeometryEstimator
        from capmas.perception.local_map import SparseVoxelMap
        from capmas.perception.tracking import KnownObjectTracker
        from capmas.perception.world_model import WorldModelService

        service = WorldModelService(
            geometry=ReferenceGeometryEstimator(
                artifact_store=store,
                depth_decoder=EmptyDepthDecoder(),
            ),
            local_map=SparseVoxelMap(voxel_size_m=0.01, local_radius_m=1.0),
            tracker=KnownObjectTracker(max_match_distance_m=0.08),
            artifact_store=store,
        )
        runtime = ThreadWorldModelRuntime(
            service=service,
            synchronizer=BoundedSensorSynchronizer(capacity=args.queue_capacity),
            config=WorldModelRuntimeConfig(queue_capacity=args.queue_capacity),
            clock=time.time_ns,
        )
        source = JsonlReplaySource(args.recording)
    else:
        os.environ["CAPMAS_ARTIFACT_ROOT"] = str(artifact_root)
        runtime = ProcessWorldModelRuntime(
            service_factory=build_reference_world_model,
            synchronizer_config=SynchronizerConfig(queue_capacity=args.queue_capacity),
            config=WorldModelRuntimeConfig(queue_capacity=args.queue_capacity),
            artifact_store=FileArtifactStore(artifact_root),
        )
        source = JsonlReplaySource(args.recording)

    metrics = RealTimeMetrics(target_hz=args.target_hz)
    snapshots: list[int] = []
    submitted = 0
    processed = 0
    evaluator_success = None
    evaluator_source = "not_connected_in_replay_mode"
    final_snapshot = initial_scene
    started = time.monotonic()
    next_capture_at = started
    runtime.start(initial_scene=initial_scene)
    try:
        while live_mode or not source.exhausted():
            if live_mode and submitted >= args.max_observations:
                break
            if live_mode and args.duration_s is not None and time.monotonic() - started >= args.duration_s:
                break
            if live_mode:
                next_capture_at = max(next_capture_at, time.monotonic())
                time.sleep(max(0.0, next_capture_at - time.monotonic()))
            observation = source.capture()
            next_capture_at += 1.0 / args.target_hz
            metrics.record_observation(observation.timestamp_ns)
            if not runtime.submit(observation):
                metrics.record_drop()
                logger.warning("dropped observation sequence=%s", observation.sequence)
                continue
            submitted += 1
            if args.runtime == "process":
                acknowledgement = runtime.wait_for_sequence(
                    observation.sequence,
                    timeout_s=5.0,
                    episode_id=observation.episode_id,
                    episode_epoch=observation.episode_epoch,
                )
                if acknowledgement is None:
                    metrics.record_drop()
                    logger.error("world model processing timeout sequence=%s", observation.sequence)
                    continue
                if acknowledgement.kind == "dropped":
                    metrics.record_drop()
                    logger.warning(
                        "world model dropped sequence=%s reason=%s",
                        observation.sequence,
                        acknowledgement.error,
                    )
                    continue
                if acknowledgement.kind == "failed":
                    logger.error(
                        "world model failed sequence=%s error=%s",
                        observation.sequence,
                        acknowledgement.error,
                    )
                    continue
            elif not runtime.wait_until_processed(timeout_s=5.0):
                metrics.record_drop()
                logger.error("world model processing timeout sequence=%s", observation.sequence)
                continue
            snapshot = runtime.latest_observation()
            processed += 1
            snapshots.append(snapshot.scene_version)
            final_snapshot = snapshot
            metrics.record_processing_latency(snapshot.processing_latency_ms)
            metrics.record_snapshot_age(max(0.0, (time.time_ns() - snapshot.publish_timestamp_ns) / 1_000_000))
            logger.info(
                "snapshot version=%s sequence=%s processing_latency_ms=%.3f",
                snapshot.scene_version,
                observation.sequence,
                snapshot.processing_latency_ms,
            )
    finally:
        runtime.stop()
        if capx_bundle is not None:
            try:
                evaluator_success = capx_bundle.backend.evaluator_success()
                evaluator_source = "capx_backend_task_completed"
            except Exception:
                logger.exception("CAP-X evaluator check failed")
            for process in server_processes:
                process.terminate()
                process.join(timeout=5)
            try:
                capx_bundle.backend.stop(None)
            except Exception:
                logger.exception("CAP-X backend stop failed")

    summary = metrics.summary(now_ns=time.time_ns())
    capture_artifact_io = None
    if capx_bundle is not None:
        artifact_metrics = getattr(capx_bundle.observation_provider.artifacts, "metrics", None)
        if callable(artifact_metrics):
            capture_artifact_io = asdict(artifact_metrics())
    payload = {
        "phase": "4-b5-live" if live_mode else "4-b5-reference",
        "mode": "live_capx" if live_mode else "replay",
        "recording": str(Path(args.recording).resolve()) if args.recording else None,
        "config_path": str(Path(args.config_path).resolve()) if args.config_path else None,
        "runtime": args.runtime,
        "depth_subsample": args.depth_subsample,
        "perception_mode": (
            "privileged_object_pose_plus_real_rgbd" if live_mode else "replay_reference"
        ),
        "semantic_models_enabled": False,
        "artifact_codec": "numpy-npy" if args.runtime == "process" else "in_memory",
        "artifact_fsync": False if args.runtime == "process" else None,
        "submitted": submitted,
        "processed": processed,
        "snapshot_versions": snapshots,
        "final_snapshot": snapshot_summary(final_snapshot),
        "capture_artifact_io": capture_artifact_io,
        "metrics": asdict(summary),
        "health": asdict(runtime.health()),
        "evaluator_success": evaluator_success,
        "evaluator_success_source": evaluator_source,
        "endpoint_probe": endpoint_probe,
        "log": str(log_path),
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"CAP-MAS B5 output: {output_path}")
    print(f"CAP-MAS B5 log: {log_path}")
    print(f"CAP-MAS B5 evaluator_success: {evaluator_success}")


if __name__ == "__main__":
    main()
