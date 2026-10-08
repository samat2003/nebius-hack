"""Integration tests for the Alienese HTTP API and provider fault handling."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx
import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from alienese.api.app import create_app
from alienese.observability.tracing import RuntimeTracer
from alienese.providers.base import ProviderFaultMode
from alienese.providers.fake import FakeController, FakeGenerator, FakeRetriever


class _FailingSpanExporter(SpanExporter):
    """SpanExporter that always raises an exception to verify observability isolation."""

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        raise RuntimeError("Simulated OTLP collector outage")

    def shutdown(self) -> None:
        return None


async def test_health_and_models_endpoints() -> None:
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        health_resp = await client.get("/health")
        assert health_resp.status_code == 200
        assert health_resp.json() == {
            "status": "ok",
            "version": "0.1.0",
            "provider_mode": "fake",
        }

        models_resp = await client.get("/v1/models")
        assert models_resp.status_code == 200
        body = models_resp.json()
        assert body["object"] == "list"
        model_ids = [m["id"] for m in body["data"]]
        assert model_ids == ["alienese-default"]


async def test_chat_completions_fake_assistant_response() -> None:
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            headers={
                "X-Request-ID": "req_caller_001",
                "X-Trace-ID": "trc_caller_001",
                "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
            },
            json={
                "model": "alienese-default",
                "messages": [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "Summarize the runtime status."},
                ],
                "temperature": 0.2,
                "max_tokens": 128,
            },
        )
        assert resp.status_code == 200
        assert resp.headers["X-Request-ID"] == "req_caller_001"
        assert resp.headers["X-Trace-ID"] == "trc_caller_001"
        assert resp.headers["X-Operation-ID"].startswith("op_")
        assert resp.headers["X-Idempotent-Replay"] == "false"

        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["model"] == "alienese-default"
        assert len(data["choices"]) == 1
        choice = data["choices"][0]
        assert choice["finish_reason"] == "stop"
        assert choice["message"]["role"] == "assistant"
        assert "[fake:answer]" in choice["message"]["content"]
        assert choice["message"].get("tool_calls") is None


async def test_chat_completions_fake_tool_call_when_arguments_complete() -> None:
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    tool_def = {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "default": "."},
                },
                "required": ["directory"],
            },
        },
    }
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # In 'auto' mode, FakeController conservatively responds rather than auto-calling
        auto_resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "alienese-default",
                "messages": [{"role": "user", "content": "List repository files"}],
                "tools": [tool_def],
            },
        )
        assert auto_resp.status_code == 200
        assert auto_resp.json()["choices"][0]["finish_reason"] == "stop"

        # When explicitly requested via tool_choice, low-risk tool with defaults is selected
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "alienese-default",
                "messages": [{"role": "user", "content": "List repository files"}],
                "tools": [tool_def],
                "tool_choice": {"type": "function", "function": {"name": "list_files"}},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["choices"]) == 1
        choice = data["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        msg = choice["message"]
        assert msg["content"] is None
        assert len(msg["tool_calls"]) == 1
        tc = msg["tool_calls"][0]
        assert tc["function"]["name"] == "list_files"
        assert tc["function"]["arguments"] == '{"directory":"."}'


async def test_chat_completions_does_not_fabricate_missing_required_tool_arguments() -> None:
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    tool_requiring_path: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # In auto mode, incomplete tool candidate is skipped and assistant response is returned
        auto_resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "alienese-default",
                "messages": [{"role": "user", "content": "Read some file"}],
                "tools": [tool_requiring_path],
                "tool_choice": "auto",
            },
        )
        assert auto_resp.status_code == 200
        assert auto_resp.json()["choices"][0]["finish_reason"] == "stop"

        # In required mode, fails with CompatibilityError rather than fabricating `path`
        req_resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "alienese-default",
                "messages": [{"role": "user", "content": "Read some file"}],
                "tools": [tool_requiring_path],
                "tool_choice": "required",
            },
        )
        assert req_resp.status_code == 400
        err = req_resp.json()["error"]
        assert err["type"] == "compatibility_error"
        assert err["code"] == "ungrounded_required_tool_arguments"


async def test_chat_completions_unsupported_and_malformed_requests() -> None:
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # stream=True -> 400 compatibility_error
        stream_resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "alienese-default",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )
        assert stream_resp.status_code == 400
        err = stream_resp.json()["error"]
        assert err["type"] == "compatibility_error"
        assert err["code"] == "streaming_not_supported"
        assert err["param"] == "stream"

        # Invalid JSON -> 400 invalid_request_error
        bad_json_resp = await client.post(
            "/v1/chat/completions",
            content=b"{not valid json",
            headers={"Content-Type": "application/json"},
        )
        assert bad_json_resp.status_code == 400
        assert bad_json_resp.json()["error"]["code"] == "invalid_json"

        # Empty messages -> 400 invalid_request_error
        empty_msg_resp = await client.post(
            "/v1/chat/completions",
            json={"model": "alienese-default", "messages": []},
        )
        assert empty_msg_resp.status_code == 400
        assert empty_msg_resp.json()["error"]["type"] == "invalid_request_error"


async def test_chat_completions_idempotency_over_http() -> None:
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    payload = {
        "model": "alienese-default",
        "messages": [{"role": "user", "content": "Idempotent HTTP test"}],
    }
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r1 = await client.post(
            "/v1/chat/completions",
            headers={"Idempotency-Key": "http-idem-1", "X-Request-ID": "req_http_1"},
            json=payload,
        )
        assert r1.status_code == 200
        assert r1.headers["X-Idempotent-Replay"] == "false"
        op_id_1 = r1.headers["X-Operation-ID"]

        # Duplicate with new HTTP attempt ID
        r2 = await client.post(
            "/v1/chat/completions",
            headers={"Idempotency-Key": "http-idem-1", "X-Request-ID": "req_http_2"},
            json=payload,
        )
        assert r2.status_code == 200
        assert r2.headers["X-Idempotent-Replay"] == "true"
        assert r2.headers["X-Request-ID"] == "req_http_2"
        assert r2.headers["X-Operation-ID"] == op_id_1
        assert r2.json() == r1.json()

        # Conflict with different body
        r3 = await client.post(
            "/v1/chat/completions",
            headers={"Idempotency-Key": "http-idem-1"},
            json={
                "model": "alienese-default",
                "messages": [{"role": "user", "content": "Different body"}],
            },
        )
        assert r3.status_code == 409
        assert r3.json()["error"]["type"] == "idempotency_conflict"


@pytest.mark.parametrize(
    ("fault_mode", "expected_status", "expected_error_type"),
    [
        (ProviderFaultMode.TIMEOUT, 504, "provider_timeout"),
        (ProviderFaultMode.UNAVAILABLE, 503, "provider_unavailable"),
        (ProviderFaultMode.PROVIDER_ERROR, 502, "provider_error"),
        (ProviderFaultMode.MALFORMED_RESPONSE, 502, "invalid_provider_response"),
    ],
)
async def test_controller_provider_fault_modes(
    fault_mode: ProviderFaultMode,
    expected_status: int,
    expected_error_type: str,
) -> None:
    app = create_app(controller=FakeController(fault_mode=fault_mode))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "alienese-default",
                "messages": [{"role": "user", "content": "Trigger controller fault"}],
            },
        )
        assert resp.status_code == expected_status
        err = resp.json()["error"]
        assert err["type"] == expected_error_type
        assert "Traceback" not in str(resp.text)


@pytest.mark.parametrize(
    ("fault_mode", "expected_status", "expected_error_type"),
    [
        (ProviderFaultMode.TIMEOUT, 504, "provider_timeout"),
        (ProviderFaultMode.UNAVAILABLE, 503, "provider_unavailable"),
        (ProviderFaultMode.PROVIDER_ERROR, 502, "provider_error"),
        (ProviderFaultMode.MALFORMED_RESPONSE, 502, "invalid_provider_response"),
    ],
)
async def test_generator_provider_fault_modes(
    fault_mode: ProviderFaultMode,
    expected_status: int,
    expected_error_type: str,
) -> None:
    app = create_app(generator=FakeGenerator(fault_mode=fault_mode))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "alienese-default",
                "messages": [{"role": "user", "content": "Trigger generator fault"}],
            },
        )
        assert resp.status_code == expected_status
        err = resp.json()["error"]
        assert err["type"] == expected_error_type


async def test_retriever_provider_timeout_fault_mode() -> None:
    app = create_app(retriever=FakeRetriever(fault_mode=ProviderFaultMode.TIMEOUT))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "alienese-default",
                "messages": [{"role": "user", "content": "Trigger retriever fault"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "list_files",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["finish_reason"] == "stop"


async def test_tracing_exporter_failure_does_not_fail_inference() -> None:
    tracer = RuntimeTracer(exporter=_FailingSpanExporter())
    app = create_app(tracer=tracer)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "alienese-default",
                "messages": [
                    {"role": "user", "content": "Should succeed despite exporter failure"}
                ],
            },
        )
        assert resp.status_code == 200
