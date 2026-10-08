"""FastAPI router for POST /v1/chat/completions."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, Header, Request, Response
from starlette.requests import ClientDisconnect

from alienese.api.errors import ProtocolError, ProviderTimeout
from alienese.api.models import ChatCompletionRequest, ChatCompletionResponse
from alienese.config import Settings
from alienese.contracts.context import (
    RequestContext,
    generate_correlation_trace_id,
    generate_operation_id,
    generate_request_id,
    normalize_caller_id,
)
from alienese.engine.idempotency import IdempotencyCoordinator
from alienese.engine.turn import TurnEngine
from alienese.providers.runtime.deadlines import DeadlineBudget, ensure_context_deadline

router = APIRouter()


def build_request_context(
    *,
    x_request_id: str | None,
    x_trace_id: str | None,
    traceparent: str | None,
    idempotency_key: str | None,
) -> RequestContext:
    """Create a RequestContext from incoming HTTP headers."""
    req_id = normalize_caller_id(x_request_id, fallback_factory=generate_request_id)
    corr_trace_id = normalize_caller_id(
        x_trace_id,
        fallback_factory=generate_correlation_trace_id,
    )
    clean_idem_key = (
        idempotency_key.strip() if idempotency_key and idempotency_key.strip() else None
    )
    return RequestContext(
        request_id=req_id,
        operation_id=generate_operation_id(),
        correlation_trace_id=corr_trace_id,
        traceparent=traceparent,
        idempotency_key=clean_idem_key,
    )


async def _read_bounded_request_body(
    request: Request,
    *,
    max_body_bytes: int,
    deadline: DeadlineBudget,
) -> bytes:
    """Incrementally read the ASGI request stream with hard byte and deadline bounds.

    Rejects oversized requests immediately as soon as the byte budget is exceeded,
    without buffering the remainder of the stream into memory (`request.body()`).
    """
    content_length_header = request.headers.get("content-length")
    if content_length_header is not None:
        try:
            declared_length = int(content_length_header.strip())
        except ValueError as exc:
            raise ProtocolError(
                "Malformed Content-Length header; must be a non-negative integer.",
                code="invalid_content_length",
                status_code=400,
            ) from exc
        if declared_length < 0:
            raise ProtocolError(
                "Content-Length header cannot be negative.",
                code="invalid_content_length",
                status_code=400,
            )
        if declared_length > max_body_bytes:
            raise ProtocolError(
                f"Request body exceeds maximum allowed size of {max_body_bytes} bytes.",
                code="request_body_too_large",
                status_code=413,
            )

    remaining_budget = deadline.require_remaining(
        provider_name="http_ingress",
        phase="request_body_read",
    )
    chunks: list[bytes] = []
    total_bytes = 0
    try:
        async with asyncio.timeout(remaining_budget):
            async for chunk in request.stream():
                deadline.require_remaining(
                    provider_name="http_ingress",
                    phase="request_body_read",
                )
                if not chunk:
                    continue
                total_bytes += len(chunk)
                if total_bytes > max_body_bytes:
                    raise ProtocolError(
                        f"Request body exceeds maximum allowed size of {max_body_bytes} bytes.",
                        code="request_body_too_large",
                        status_code=413,
                    )
                chunks.append(chunk)
    except ClientDisconnect as exc:
        raise ProtocolError(
            "Client disconnected while sending request body.",
            code="client_disconnected",
            status_code=400,
        ) from exc
    except TimeoutError as exc:
        raise ProviderTimeout(
            "Timed out reading HTTP request body before turn deadline.",
            code="turn_deadline_exceeded",
        ) from exc

    return b"".join(chunks)


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: Request,
    response: Response,
    x_request_id: str | None = Header(default=None, alias="X-Request-ID"),
    x_trace_id: str | None = Header(default=None, alias="X-Trace-ID"),
    traceparent: str | None = Header(default=None, alias="traceparent"),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> ChatCompletionResponse:
    """Execute a single OpenAI-compatible turn through the Alienese TurnEngine."""
    settings: Settings = request.app.state.settings
    engine: TurnEngine = request.app.state.turn_engine
    idempotency: IdempotencyCoordinator = request.app.state.idempotency

    ctx = build_request_context(
        x_request_id=x_request_id,
        x_trace_id=x_trace_id,
        traceparent=traceparent,
        idempotency_key=idempotency_key,
    )
    ctx = ensure_context_deadline(
        ctx,
        timeout_seconds=settings.request_deadline_seconds,
        clock=engine.monotonic_clock,
    )
    request.state.request_context = ctx

    request_deadline = DeadlineBudget.from_context(
        ctx,
        default_timeout_seconds=settings.request_deadline_seconds,
        clock=engine.monotonic_clock,
    )
    raw_bytes = await _read_bounded_request_body(
        request,
        max_body_bytes=settings.max_request_body_bytes,
        deadline=request_deadline,
    )

    try:
        raw_body: Any = json.loads(raw_bytes.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ProtocolError(
            "Request body must be valid UTF-8 encoded JSON.",
            code="invalid_encoding",
        ) from exc
    except json.JSONDecodeError as exc:
        raise ProtocolError(
            f"Malformed JSON request body: {exc.msg}",
            code="invalid_json",
        ) from exc

    if not isinstance(raw_body, dict):
        raise ProtocolError(
            "Request body must be a JSON object.",
            code="invalid_request_body",
        )

    chat_request = ChatCompletionRequest.model_validate(raw_body)

    async def _run_turn(turn_ctx: RequestContext) -> ChatCompletionResponse:
        completion_response, _artifact = await engine.execute_turn(turn_ctx, chat_request)
        return completion_response

    effective_ctx, completion, replayed = await idempotency.execute(ctx, chat_request, _run_turn)
    request.state.request_context = effective_ctx

    response.headers["X-Request-ID"] = effective_ctx.request_id
    response.headers["X-Operation-ID"] = effective_ctx.operation_id
    response.headers["X-Trace-ID"] = effective_ctx.correlation_trace_id
    response.headers["X-Idempotent-Replay"] = "true" if replayed else "false"
    return completion
