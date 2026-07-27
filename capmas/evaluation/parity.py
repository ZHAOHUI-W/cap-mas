"""Read-only normalization of CAP-X and CAP-MAS episode artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping


_CAPX_TRIAL_PATTERN = re.compile(
    r"trial_(?P<trial>\d+)_sandboxrc_(?P<sandbox_rc>-?\d+)"
    r"_reward_(?P<reward>[-+]?\d+(?:\.\d+)?)"
    r"_taskcompleted_(?P<completed>[01])$"
)


@dataclass(frozen=True)
class NormalizedEpisode:
    """Comparable episode-level fields shared by both systems."""

    system: str
    artifact: str
    task_id: str
    seed: int | None
    success: bool
    reward: float
    action_count: int | None = None
    failure_reason: str | None = None
    sandbox_rc: int | None = None


@dataclass(frozen=True)
class ParityComparison:
    """One explicitly matched CAP-X/CAP-MAS comparison."""

    task_id: str
    seed: int | None
    capx: NormalizedEpisode
    capmas: NormalizedEpisode

    @property
    def success_delta(self) -> int:
        return int(self.capmas.success) - int(self.capx.success)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "success_delta_capmas_minus_capx": self.success_delta,
            "capx": asdict(self.capx),
            "capmas": asdict(self.capmas),
        }


def load_capmas_episode(
    path: str | Path,
    *,
    task_id: str,
    seed: int | None = None,
) -> NormalizedEpisode:
    """Load a CAP-MAS JSON artifact without consulting evaluator state."""
    artifact = Path(path)
    payload = _load_json_mapping(artifact)
    success = _as_bool(payload.get("evaluator_success", payload.get("completed", False)))
    return NormalizedEpisode(
        system="capmas",
        artifact=str(artifact),
        task_id=task_id,
        seed=seed,
        success=success,
        reward=1.0 if success else 0.0,
        action_count=_trace_count(payload),
        failure_reason=_capmas_failure_reason(payload),
    )


def load_capx_trial(
    path: str | Path,
    *,
    task_id: str,
    seed: int | None = None,
) -> NormalizedEpisode:
    """Load a CAP-X trial directory using its stable directory-name schema."""
    artifact = Path(path)
    if not artifact.is_dir():
        raise ValueError(f"CAP-X trial must be a directory: {artifact}")
    match = _CAPX_TRIAL_PATTERN.fullmatch(artifact.name)
    if match is None:
        raise ValueError(
            "CAP-X trial directory must match "
            "trial_<n>_sandboxrc_<rc>_reward_<r>_taskcompleted_<0|1>"
        )
    summary = artifact / "summary.txt"
    if not summary.is_file():
        raise ValueError(f"CAP-X trial is missing summary.txt: {artifact}")
    completed = bool(int(match.group("completed")))
    summary_text = summary.read_text()
    return NormalizedEpisode(
        system="capx",
        artifact=str(artifact),
        task_id=task_id,
        seed=seed,
        success=completed,
        reward=float(match.group("reward")),
        action_count=_capx_action_count(summary_text),
        failure_reason=None if completed else _capx_failure_reason(summary_text),
        sandbox_rc=int(match.group("sandbox_rc")),
    )


def compare_artifacts(
    capx_trial: str | Path,
    capmas_episode: str | Path,
    *,
    task_id: str,
    seed: int | None = None,
) -> ParityComparison:
    """Build a matched comparison; no external state is accessed."""
    return ParityComparison(
        task_id=task_id,
        seed=seed,
        capx=load_capx_trial(capx_trial, task_id=task_id, seed=seed),
        capmas=load_capmas_episode(capmas_episode, task_id=task_id, seed=seed),
    )


def _load_json_mapping(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise ValueError(f"CAP-MAS episode must be a JSON file: {path}")
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid CAP-MAS JSON artifact: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"CAP-MAS artifact must contain a JSON object: {path}")
    return value


def _as_bool(value: object) -> bool:
    return value if isinstance(value, bool) else bool(value)


def _trace_count(payload: Mapping[str, Any]) -> int | None:
    traces = payload.get("traces")
    if isinstance(traces, list):
        return len(traces)
    episode_trace = payload.get("episode_trace")
    if isinstance(episode_trace, Mapping) and isinstance(episode_trace.get("traces"), list):
        return len(episode_trace["traces"])
    result = payload.get("result")
    if isinstance(result, Mapping) and isinstance(result.get("traces"), list):
        return len(result["traces"])
    return None


def _capmas_failure_reason(payload: Mapping[str, Any]) -> str | None:
    failure = payload.get("failure")
    result = payload.get("result")
    if isinstance(result, Mapping) and failure is None:
        failure = result.get("failure")
    if isinstance(failure, Mapping):
        return str(failure.get("failure_class") or failure.get("message") or "failure")
    stop_reason = payload.get("stop_reason")
    if stop_reason is None and isinstance(result, Mapping):
        stop_reason = result.get("stop_reason")
    return str(stop_reason) if stop_reason and stop_reason != "evaluator_success" else None


def _capx_action_count(summary: str) -> int | None:
    match = re.search(r"Num Code Blocks:\s*(\d+)", summary)
    return int(match.group(1)) if match else None


def _capx_failure_reason(summary: str) -> str | None:
    match = re.search(r"(?:Stderr|Environment response):\s*(.*?)(?:\n\s*Reward:|\Z)", summary, re.S)
    if match:
        text = " ".join(match.group(1).split())
        if text:
            return text[:500]
    return "task_not_completed"


__all__ = [
    "NormalizedEpisode",
    "ParityComparison",
    "compare_artifacts",
    "load_capmas_episode",
    "load_capx_trial",
]
