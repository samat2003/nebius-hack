"""FastAPI router for POST /v1/chat/completions."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Header, Request, Response

from alienese.api.errors import ProtocolError
from alienese.api.models import ChatCompletionRequest, ChatCompletionResponse
from alienese.contracts.context import (
    RequestContext,
    generate_correlation_trace_id,
    generate_operation_id,
    generate_request_id,
    normalize_caller_id,
)
from alienese.engine.idempotency import IdempotencyCoordinator
from alienese.engine.turn import TurnEngine

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
    ctx = build_request_context(
        x_request_id=x_request_id,
        x_trace_id=x_trace_id,
        traceparent=traceparent,
        idempotency_key=idempotency_key,
    )
    request.state.request_context = ctx

    try:
        raw_body: Any = await request.json()
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

    engine: TurnEngine = request.app.state.turn_engine
    idempotency: IdempotencyCoordinator = request.app.state.idempotency

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
