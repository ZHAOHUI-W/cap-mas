"""CAP-X-compatible OpenAI chat-completions client.

The client deliberately depends only on the Python standard library.  CAP-X
can therefore remain the source of truth for the endpoint and model settings,
while CAP-MAS records a normalized ``LLMResponse`` and enforces the typed
artifact boundary in ``MissionGraphDecoder``.
"""

from __future__ import annotations

import json
import hashlib
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request as HTTPRequest
from urllib.request import urlopen

from capmas.llm.protocol import LLMCallTrace, LLMRequest, LLMResponse, LLMTraceSink

try:  # CAP-X already uses requests; urllib remains the dependency-free fallback.
    import requests
except ImportError:  # pragma: no cover - exercised only in minimal installs
    requests = None  # type: ignore[assignment]


class LLMTransportError(RuntimeError):
    """A provider or transport failure that must not cross into execution."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class CAPXCompatibleConfig:
    endpoint: str
    model: str
    api_key_env: str = "CAPMAS_LLM_API_KEY"
    temperature: float = 0.0
    structured_outputs: bool = True
    max_retries: int = 0
    retry_backoff_ms: int = 100

    def __post_init__(self) -> None:
        if not self.endpoint:
            raise ValueError("LLM endpoint must not be empty")
        if not self.model:
            raise ValueError("LLM model must not be empty")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("LLM temperature must be in [0, 2]")
        if self.max_retries < 0:
            raise ValueError("LLM max_retries must not be negative")
        if self.retry_backoff_ms < 0:
            raise ValueError("LLM retry_backoff_ms must not be negative")


HTTPTransport = Callable[[HTTPRequest, float], object]


class CAPXCompatibleLLMClient:
    """Call a CAP-X/OpenAI-compatible chat-completions endpoint.

    ``transport`` is injectable for deterministic tests and replay adapters.
    The default transport is ``urllib.request.urlopen``; no provider SDK is
    imported into the CAP-MAS runtime.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        api_key: str | None = None,
        api_key_env: str = "CAPMAS_LLM_API_KEY",
        temperature: float = 0.0,
        structured_outputs: bool = True,
        max_retries: int = 0,
        retry_backoff_ms: int = 100,
        transport: HTTPTransport | None = None,
        trace_sink: LLMTraceSink | Callable[[LLMCallTrace], None] | None = None,
    ) -> None:
        self.config = CAPXCompatibleConfig(
            endpoint=endpoint,
            model=model,
            api_key_env=api_key_env,
            temperature=temperature,
            structured_outputs=structured_outputs,
            max_retries=max_retries,
            retry_backoff_ms=retry_backoff_ms,
        )
        self._api_key = api_key
        self._transport = transport or _default_transport
        self._trace_sink = trace_sink

    def complete(self, request: LLMRequest) -> LLMResponse:
        started_at_ns = time.time_ns()
        started = time.monotonic()
        state = {"attempts": 0}
        try:
            response = self._complete(request, started, state)
        except LLMTransportError as exc:
            self._record_trace(
                request,
                started_at_ns=started_at_ns,
                started=started,
                status="failed",
                attempts=max(1, int(state["attempts"])),
                provider_status_code=exc.status_code,
                error=exc,
            )
            raise
        self._record_trace(
            request,
            started_at_ns=started_at_ns,
            started=started,
            status="completed",
            attempts=max(1, int(state["attempts"])),
            provider_status_code=_provider_status(response),
            response=response,
        )
        return response

    def _complete(
        self,
        request: LLMRequest,
        started: float,
        state: dict[str, int],
    ) -> LLMResponse:
        if request.deadline_ms <= 0:
            raise LLMTransportError("LLM request deadline must be positive")

        http_request = self._http_request(request, structured_outputs=self.config.structured_outputs)
        structured_fallback_used = False

        last_error: LLMTransportError | None = None
        for attempt in range(self.config.max_retries + 1):
            remaining = request.deadline_ms / 1000.0 - (time.monotonic() - started)
            if remaining <= 0:
                break
            try:
                state["attempts"] += 1
                body, status_code = self._send(http_request, remaining)
                return self._response(
                    request,
                    body,
                    status_code,
                    started,
                    structured_output_fallback=structured_fallback_used,
                )
            except LLMTransportError as exc:
                if (
                    not structured_fallback_used
                    and request.response_schema is not None
                    and self.config.structured_outputs
                    and _is_schema_compatibility_error(exc)
                ):
                    structured_fallback_used = True
                    fallback_http_request = self._http_request(
                        request,
                        structured_outputs=False,
                    )
                    try:
                        state["attempts"] += 1
                        body, status_code = self._send(fallback_http_request, remaining)
                        return self._response(
                            request,
                            body,
                            status_code,
                            started,
                            structured_output_fallback=True,
                        )
                    except LLMTransportError as fallback_error:
                        exc = fallback_error
                last_error = exc
                retryable = exc.status_code is None or exc.status_code in {
                    408,
                    429,
                    500,
                    502,
                    503,
                    504,
                }
                if attempt >= self.config.max_retries or not retryable:
                    raise
                delay = self.config.retry_backoff_ms / 1000.0 * (2**attempt)
                remaining = request.deadline_ms / 1000.0 - (time.monotonic() - started)
                if delay >= remaining:
                    break
                time.sleep(delay)

        if last_error is not None:
            raise LLMTransportError(
                f"LLM request deadline exhausted: {last_error}",
                status_code=last_error.status_code,
            ) from last_error
        raise LLMTransportError("LLM request deadline exhausted")

    def _record_trace(
        self,
        request: LLMRequest,
        *,
        started_at_ns: int,
        started: float,
        status: str,
        attempts: int,
        provider_status_code: int | None = None,
        response: LLMResponse | None = None,
        error: LLMTransportError | None = None,
    ) -> None:
        if self._trace_sink is None:
            return
        finished_at_ns = time.time_ns()
        fallback = bool(response and response.metadata.get("structured_output_fallback"))
        trace = LLMCallTrace(
            request_id=request.request_id,
            agent_name=request.agent_name,
            model=self.config.model,
            status=status,
            started_at_ns=started_at_ns,
            finished_at_ns=finished_at_ns,
            latency_ms=(time.monotonic() - started) * 1000.0,
            attempts=attempts,
            input_tokens=response.input_tokens if response else None,
            output_tokens=response.output_tokens if response else None,
            provider_status_code=provider_status_code,
            schema_mode=_schema_mode(request, self.config.structured_outputs),
            schema_hash=_schema_hash(request.response_schema),
            structured_output_fallback=fallback,
            error_type=type(error).__name__ if error else None,
            error_message=_sanitize_error(str(error)) if error else None,
        )
        try:
            record = getattr(self._trace_sink, "record", None)
            if callable(record):
                record(trace)
            else:
                self._trace_sink(trace)  # type: ignore[operator]
        except Exception:
            # Telemetry is deliberately non-authoritative: a broken collector
            # must never turn a completed robot-planning request into a failure.
            return

    def _http_request(
        self,
        request: LLMRequest,
        *,
        structured_outputs: bool,
    ) -> HTTPRequest:
        payload = self._payload(request, structured_outputs=structured_outputs)
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "cap-mas/0.1",
        }
        api_key = self._api_key or os.getenv(self.config.api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return HTTPRequest(
            self.config.endpoint,
            data=encoded,
            headers=headers,
            method="POST",
        )

    def _payload(
        self,
        request: LLMRequest,
        *,
        structured_outputs: bool | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.config.model,
            "messages": list(request.messages),
        }
        # CAP-X uses max_completion_tokens for GPT-5/o-series endpoints and
        # max_tokens for the older OpenAI-compatible chat-completions surface.
        if self.config.model.startswith(("gpt-5", "o1", "o3", "o4")):
            payload["max_completion_tokens"] = request.max_output_tokens
        else:
            payload["max_tokens"] = request.max_output_tokens
            payload["temperature"] = self.config.temperature

        use_structured_outputs = (
            self.config.structured_outputs
            if structured_outputs is None
            else structured_outputs
        )
        if request.response_schema is not None and use_structured_outputs:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "capmas_artifact",
                    "strict": True,
                    "schema": request.response_schema,
                },
            }
        return payload

    def _send(
        self,
        request: HTTPRequest,
        timeout: float,
    ) -> tuple[Mapping[str, object], int]:
        try:
            response = self._transport(request, timeout)
            status_code = int(getattr(response, "status", 200))
            raw = response.read()  # type: ignore[union-attr]
            close = getattr(response, "close", None)
            if callable(close):
                close()
        except HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = str(exc)
            raise LLMTransportError(
                f"LLM provider returned HTTP {exc.code}: {detail[:500]}",
                status_code=exc.code,
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise LLMTransportError(f"LLM transport failed: {exc}") from exc
        except Exception as exc:
            if isinstance(exc, LLMTransportError):
                raise
            raise LLMTransportError(f"LLM transport failed: {exc}") from exc

        if status_code >= 400:
            detail = raw.decode("utf-8", errors="replace")[:500]
            raise LLMTransportError(
                f"LLM provider returned HTTP {status_code}: {detail}",
                status_code=status_code,
            )
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LLMTransportError("LLM provider returned invalid JSON") from exc
        if not isinstance(body, Mapping):
            raise LLMTransportError("LLM provider response must be a JSON object")
        return body, status_code

    @staticmethod
    def _response(
        request: LLMRequest,
        body: Mapping[str, object],
        status_code: int,
        started: float,
        *,
        structured_output_fallback: bool = False,
    ) -> LLMResponse:
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMTransportError(
                "LLM provider response has no choices",
                status_code=status_code,
            )
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise LLMTransportError(
                "LLM provider choice must be an object",
                status_code=status_code,
            )
        message = choice.get("message", {})
        if not isinstance(message, Mapping):
            raise LLMTransportError(
                "LLM provider message must be an object",
                status_code=status_code,
            )

        raw_content = message.get("content")
        structured = message.get("parsed")
        content = _content_text(raw_content)
        if not isinstance(structured, Mapping):
            structured = body.get("structured")
        if not isinstance(structured, Mapping) and isinstance(raw_content, Mapping):
            structured = raw_content
        if not isinstance(structured, Mapping) and content:
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, Mapping):
                structured = parsed

        usage = body.get("usage")
        if not isinstance(usage, Mapping):
            usage = {}
        input_tokens = _usage_int(usage, "prompt_tokens", "input_tokens")
        output_tokens = _usage_int(usage, "completion_tokens", "output_tokens")
        latency_ms = (time.monotonic() - started) * 1000.0
        metadata = {
            "provider_status_code": status_code,
            "finish_reason": choice.get("finish_reason"),
            "structured_output_fallback": structured_output_fallback,
        }
        return LLMResponse(
            request_id=request.request_id,
            content=content,
            structured=dict(structured) if isinstance(structured, Mapping) else None,
            model=str(body.get("model", "")),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            metadata=metadata,
        )


def _default_transport(request: HTTPRequest, timeout: float) -> object:
    if requests is not None:
        response = requests.post(  # type: ignore[union-attr]
            request.full_url,
            data=request.data,
            headers=dict(request.header_items()),
            timeout=timeout,
        )
        return _RequestsResponse(response)
    return urlopen(request, timeout=timeout)


class _RequestsResponse:
    """Adapt ``requests.Response`` to the small response surface used above."""

    def __init__(self, response: object) -> None:
        self._response = response
        self.status = int(getattr(response, "status_code", 200))

    def read(self) -> bytes:
        content = getattr(self._response, "content")
        return content if isinstance(content, bytes) else str(content).encode("utf-8")

    def close(self) -> None:
        close = getattr(self._response, "close", None)
        if callable(close):
            close()


def _content_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return ""


def _usage_int(usage: Mapping[str, object], *keys: str) -> int | None:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _is_schema_compatibility_error(error: LLMTransportError) -> bool:
    """Identify only provider schema errors eligible for local validation fallback."""
    if error.status_code != 400:
        return False
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "invalid schema",
            "response_format",
            "additionalproperties",
            "json_schema",
        )
    )


def _schema_mode(request: LLMRequest, structured_outputs: bool) -> str:
    if request.response_schema is None:
        return "none"
    return "strict_provider_schema" if structured_outputs else "local_json_validation"


def _schema_hash(schema: Mapping[str, object] | None) -> str | None:
    if schema is None:
        return None
    encoded = json.dumps(schema, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _provider_status(response: LLMResponse) -> int | None:
    value = response.metadata.get("provider_status_code")
    return value if isinstance(value, int) else None


def _sanitize_error(message: str) -> str:
    # Provider error bodies are already truncated by _send. Keep artifact
    # diagnostics bounded and avoid accidentally persisting authorization data
    # if a proxy echoes request headers.
    return message.replace("Authorization", "authorization")[:500]


__all__ = [
    "CAPXCompatibleConfig",
    "CAPXCompatibleLLMClient",
    "LLMTransportError",
]
