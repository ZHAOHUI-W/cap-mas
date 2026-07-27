from __future__ import annotations

import json

import pytest

from capmas.llm.capx_compatible import (
    CAPXCompatibleLLMClient,
    LLMTransportError,
)
from capmas.llm.protocol import LLMCallTrace, LLMRequest, LLMTraceCollector


class _Response:
    def __init__(self, body: dict[str, object], status: int = 200) -> None:
        self._body = json.dumps(body).encode()
        self.status = status
        self.headers = {"content-type": "application/json"}
        self.closed = False

    def read(self) -> bytes:
        return self._body

    def close(self) -> None:
        self.closed = True


def test_capx_compatible_client_sends_structured_chat_request_and_parses_usage() -> None:
    calls: list[tuple[object, float]] = []

    def transport(request: object, timeout: float) -> _Response:
        calls.append((request, timeout))
        return _Response(
            {
                "id": "completion-1",
                "model": "gpt-5.4",
                "choices": [
                    {
                        "message": {
                            "content": '{"ok": true}',
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            }
        )

    client = CAPXCompatibleLLMClient(
        endpoint="https://llm.example/v1/chat/completions",
        model="gpt-5.4",
        api_key="test-key",
        transport=transport,
    )
    response = client.complete(
        LLMRequest(
            "req-1",
            "manager",
            ({"role": "user", "content": "make a graph"},),
            response_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
            deadline_ms=500,
            max_output_tokens=128,
        )
    )

    assert response.request_id == "req-1"
    assert response.structured == {"ok": True}
    assert response.input_tokens == 11
    assert response.output_tokens == 7
    assert response.model == "gpt-5.4"
    assert len(calls) == 1
    request = calls[0][0]
    body = json.loads(request.data.decode())  # type: ignore[union-attr]
    assert body["model"] == "gpt-5.4"
    assert body["max_completion_tokens"] == 128
    assert body["response_format"]["type"] == "json_schema"
    assert request.headers["Authorization"] == "Bearer test-key"  # type: ignore[union-attr]
    assert request.headers["User-agent"] == "cap-mas/0.1"  # type: ignore[union-attr]


def test_capx_compatible_client_records_successful_call_trace() -> None:
    collector = LLMTraceCollector()

    def transport(_request: object, _timeout: float) -> _Response:
        return _Response(
            {
                "model": "gpt-5.5",
                "choices": [{"message": {"content": '{"ok": true}'}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 8},
            }
        )

    client = CAPXCompatibleLLMClient(
        endpoint="https://llm.example/v1/chat/completions",
        model="gpt-5.5",
        transport=transport,
        trace_sink=collector.record,
    )
    response = client.complete(
        LLMRequest(
            "trace-1",
            "policy-0",
            (),
            response_schema={"type": "object", "properties": {}},
            deadline_ms=500,
        )
    )

    assert response.metadata["structured_output_fallback"] is False
    traces = collector.snapshot()
    assert len(traces) == 1
    trace = traces[0]
    assert trace.request_id == "trace-1"
    assert trace.agent_name == "policy-0"
    assert trace.status == "completed"
    assert trace.schema_mode == "strict_provider_schema"
    assert trace.structured_output_fallback is False
    assert trace.attempts == 1
    assert trace.input_tokens == 12
    assert trace.output_tokens == 8
    assert trace.latency_ms >= 0
    assert trace.schema_hash


def test_capx_compatible_client_records_schema_fallback_as_one_trace() -> None:
    collector = LLMTraceCollector()
    calls = 0

    def transport(_request: object, _timeout: float) -> _Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _Response(
                {"error": {"message": "Invalid schema: response_format"}},
                status=400,
            )
        return _Response(
            {"model": "test-model", "choices": [{"message": {"content": '{"ok": true}'}}]}
        )

    client = CAPXCompatibleLLMClient(
        endpoint="https://llm.example/v1/chat/completions",
        model="test-model",
        transport=transport,
        trace_sink=collector.record,
    )
    client.complete(
        LLMRequest(
            "trace-fallback",
            "manager",
            (),
            response_schema={"type": "object", "properties": {}},
            deadline_ms=500,
        )
    )

    traces = collector.snapshot()
    assert len(traces) == 1
    assert traces[0].status == "completed"
    assert traces[0].structured_output_fallback is True
    assert traces[0].attempts == 2


def test_capx_compatible_client_records_terminal_failure() -> None:
    collector = LLMTraceCollector()

    def transport(_request: object, _timeout: float) -> _Response:
        return _Response({"error": {"message": "provider unavailable"}}, status=503)

    client = CAPXCompatibleLLMClient(
        endpoint="https://llm.example/v1/chat/completions",
        model="test-model",
        transport=transport,
        max_retries=0,
        trace_sink=collector.record,
    )

    with pytest.raises(LLMTransportError):
        client.complete(LLMRequest("trace-failure", "policy-1", (), deadline_ms=500))

    traces = collector.snapshot()
    assert len(traces) == 1
    assert traces[0].status == "failed"
    assert traces[0].provider_status_code == 503
    assert traces[0].error_type == "LLMTransportError"


def test_llm_trace_collector_sorts_concurrent_records_deterministically() -> None:
    collector = LLMTraceCollector()
    collector.record(
        LLMCallTrace(
            request_id="b",
            agent_name="policy-b",
            model="test-model",
            status="completed",
            started_at_ns=20,
            finished_at_ns=21,
            latency_ms=1,
        )
    )
    collector.record(
        LLMCallTrace(
            request_id="a",
            agent_name="policy-a",
            model="test-model",
            status="completed",
            started_at_ns=10,
            finished_at_ns=11,
            latency_ms=1,
        )
    )

    assert [trace.request_id for trace in collector.snapshot()] == ["a", "b"]


def test_capx_compatible_client_rejects_malformed_provider_response() -> None:
    def transport(_request: object, _timeout: float) -> _Response:
        return _Response({"choices": []})

    client = CAPXCompatibleLLMClient(
        endpoint="https://llm.example/v1/chat/completions",
        model="test-model",
        transport=transport,
    )

    with pytest.raises(LLMTransportError, match="no choices"):
        client.complete(LLMRequest("req-1", "manager", ()))


def test_capx_compatible_client_falls_back_only_for_schema_compatibility_400() -> None:
    calls: list[object] = []

    def transport(request: object, _timeout: float) -> _Response:
        calls.append(request)
        if len(calls) == 1:
            return _Response(
                {
                    "error": {
                        "message": "Invalid schema: additionalProperties is required to be false"
                    }
                },
                status=400,
            )
        return _Response(
            {
                "model": "test-model",
                "choices": [{"message": {"content": '{"ok": true}'}}],
            }
        )

    client = CAPXCompatibleLLMClient(
        endpoint="https://llm.example/v1/chat/completions",
        model="test-model",
        api_key="test-key",
        transport=transport,
    )
    response = client.complete(
        LLMRequest(
            "req-1",
            "manager",
            (),
            response_schema={"type": "object"},
            deadline_ms=500,
        )
    )

    first = json.loads(calls[0].data.decode())  # type: ignore[union-attr]
    second = json.loads(calls[1].data.decode())  # type: ignore[union-attr]
    assert "response_format" in first
    assert "response_format" not in second
    assert response.structured == {"ok": True}
    assert response.metadata["structured_output_fallback"] is True


def test_capx_compatible_client_preserves_transport_error_without_default_plan() -> None:
    def transport(_request: object, _timeout: float) -> _Response:
        raise TimeoutError("deadline")

    client = CAPXCompatibleLLMClient(
        endpoint="https://llm.example/v1/chat/completions",
        model="test-model",
        transport=transport,
    )

    with pytest.raises(LLMTransportError, match="transport failed"):
        client.complete(LLMRequest("req-1", "manager", (), deadline_ms=10))
