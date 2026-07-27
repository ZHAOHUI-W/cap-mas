from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class ArtifactRef:
    uri: str
    media_type: str
    sha256: str | None = None
    byte_size: int | None = None


@dataclass(frozen=True)
class SkillRef:
    skill_id: str
    version: str


@dataclass(frozen=True)
class EpisodeHandle:
    episode_id: str
    task_id: str
    suite_name: str
    backend_id: str
    seed: int | None
    episode_epoch: int
    started_at_ns: int
    status: str = "active"
    metadata: Mapping[str, str] = field(default_factory=dict)
