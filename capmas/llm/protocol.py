from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Mapping, Protocol, Sequence


@dataclass(frozen=True)
class LLMRequest:
    request_id: str
    agent_name: str
    messages: Sequence[Mapping[str, object]]
    response_schema: Mapping[str, object] | None = None
    deadline_ms: int = 30_000
    max_output_tokens: int = 4096


@dataclass(frozen=True)
class LLMResponse:
    request_id: str
    content: str
    structured: Mapping[str, object] | None = None
    model: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMCallTrace:
    """Sanitized telemetry for one logical LLM completion request."""

    request_id: str
    agent_name: str
    model: str
    status: str
    started_at_ns: int
    finished_at_ns: int
    latency_ms: float
    attempts: int = 1
    input_tokens: int | None = None
    output_tokens: int | None = None
    provider_status_code: int | None = None
    schema_mode: str = "none"
    schema_hash: str | None = None
    structured_output_fallback: bool = False
    error_type: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if not self.request_id or not self.agent_name or not self.model:
            raise ValueError("LLM call identity fields must not be empty")
        if self.status not in {"completed", "failed"}:
            raise ValueError("LLM call status must be completed or failed")
        if self.finished_at_ns < self.started_at_ns:
            raise ValueError("LLM call timestamps must be monotonic")
        if self.latency_ms < 0:
            raise ValueError("LLM call latency must not be negative")
        if self.attempts <= 0:
            raise ValueError("LLM call attempts must be positive")
        if self.schema_mode not in {"none", "strict_provider_schema", "local_json_validation"}:
            raise ValueError("unsupported LLM schema mode")


class LLMTraceSink(Protocol):
    def record(self, trace: LLMCallTrace) -> None: ...


class LLMTraceCollector:
    """Thread-safe trace sink with deterministic snapshots for artifacts."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._traces: list[LLMCallTrace] = []

    def record(self, trace: LLMCallTrace) -> None:
        with self._lock:
            self._traces.append(trace)

    def snapshot(self) -> tuple[LLMCallTrace, ...]:
        with self._lock:
            return tuple(
                sorted(
                    self._traces,
                    key=lambda item: (item.started_at_ns, item.request_id),
                )
            )


class LLMClient(Protocol):
    def complete(self, request: LLMRequest) -> LLMResponse: ...
