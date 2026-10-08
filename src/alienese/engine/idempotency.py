"""Idempotent request coordinator and canonical request fingerprinting.

Enforces Refinement 7:
- Separates HTTP attempt `request_id` from logical `operation_id`.
- For the same `Idempotency-Key` and identical `request_fingerprint`, returns the
  exact completed logical response while preserving the retry attempt's `request_id`
  in structured logs.
- For the same `Idempotency-Key` with a different `request_fingerprint`, raises
  `IdempotencyConflict` (HTTP 409).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections.abc import Awaitable, Callable

from alienese.api.errors import IdempotencyConflict, ProtocolError, ProviderTimeout
from alienese.api.models import ChatCompletionRequest, ChatCompletionResponse, NamedToolChoice
from alienese.contracts.context import RequestContext
from alienese.observability.logging import get_request_logger
from alienese.providers.runtime.deadlines import (
    DEFAULT_TURN_TIMEOUT_SECONDS,
    DeadlineBudget,
    ensure_context_deadline,
)
from alienese.storage.idempotency import IdempotencyRecord, IdempotencyStore

_VALID_IDEMPOTENCY_KEY_RE = re.compile(r"^[\x21-\x7E]{1,255}$")


def validate_idempotency_key(raw_key: str) -> str:
    """Validate that an Idempotency-Key is 1..255 printable ASCII characters."""
    cleaned = raw_key.strip()
    if not _VALID_IDEMPOTENCY_KEY_RE.match(cleaned):
        raise ProtocolError(
            "Header 'Idempotency-Key' must be 1 to 255 printable non-whitespace ASCII characters.",
            param="Idempotency-Key",
            code="invalid_idempotency_key",
        )
    return cleaned


def compute_request_fingerprint(request: ChatCompletionRequest) -> str:
    """Compute a canonical SHA-256 fingerprint over the semantic request body."""
    canonical_messages = [
        {
            "role": m.role,
            "content": m.normalized_text_content(),
            "name": m.name,
            "tool_call_id": m.tool_call_id,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments)
                        if tc.function.arguments.strip()
                        else {},
                    },
                }
                for tc in m.tool_calls
            ]
            if m.tool_calls
            else None,
        }
        for m in request.messages
    ]
    canonical_tools = (
        [t.model_dump(mode="json", exclude_none=True) for t in request.tools]
        if request.tools
        else None
    )
    tool_choice_val = (
        request.tool_choice.model_dump(mode="json")
        if isinstance(request.tool_choice, NamedToolChoice)
        else (request.tool_choice or "auto")
    )
    semantic_payload = {
        "model": request.model,
        "messages": canonical_messages,
        "tools": canonical_tools,
        "tool_choice": tool_choice_val if canonical_tools else "none",
        "temperature": request.temperature,
        "effective_max_tokens": request.effective_max_tokens,
    }
    encoded = json.dumps(
        semantic_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class IdempotencyCoordinator:
    """Coordinates idempotent execution of logical turn operations.

    Distinguishes:
    1. Incoming HTTP request deadline (`ctx.deadline_monotonic` / `request_deadline_seconds`).
    2. Duplicate waiter lock-acquisition budget (`min(remaining_request_deadline, wait_timeout)`).
    3. Logical operation execution deadline inside `runner(ctx)`.
    """

    def __init__(
        self,
        store: IdempotencyStore,
        *,
        request_deadline_seconds: float = DEFAULT_TURN_TIMEOUT_SECONDS,
        wait_timeout_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._store = store
        self._request_deadline_seconds = request_deadline_seconds
        self._wait_timeout_seconds = wait_timeout_seconds
        self._clock = clock

    async def execute(
        self,
        ctx: RequestContext,
        request: ChatCompletionRequest,
        runner: Callable[[RequestContext], Awaitable[ChatCompletionResponse]],
    ) -> tuple[RequestContext, ChatCompletionResponse, bool]:
        """Execute `runner` idempotently while bounding duplicate waiter lock acquisition."""
        ctx = ensure_context_deadline(
            ctx,
            timeout_seconds=self._request_deadline_seconds,
            clock=self._clock,
        )
        incoming_deadline = DeadlineBudget.from_context(
            ctx,
            default_timeout_seconds=self._request_deadline_seconds,
            clock=self._clock,
        )
        incoming_deadline.require_remaining(
            provider_name="idempotency",
            phase="pre_coordination",
        )

        raw_key = ctx.idempotency_key
        if raw_key is None or not raw_key.strip():
            response = await runner(ctx)
            return ctx, response, False

        key = validate_idempotency_key(raw_key)
        fingerprint = compute_request_fingerprint(request)
        lock = self._store.key_lock(key)

        wait_budget = incoming_deadline.require_remaining(
            provider_name="idempotency",
            phase="lock_wait",
        )
        if self._wait_timeout_seconds is not None:
            wait_budget = min(wait_budget, self._wait_timeout_seconds)

        try:
            await asyncio.wait_for(lock.__aenter__(), timeout=wait_budget)
        except TimeoutError as exc:
            logger = get_request_logger(ctx, "engine.idempotency")
            logger.warning(
                "idempotency_wait_timeout",
                idempotency_key=key,
                wait_budget_seconds=round(wait_budget, 3),
            )
            raise ProviderTimeout(
                f"Timed out waiting ({wait_budget:.2f}s) for in-flight idempotent "
                f"operation on key '{key}'.",
                code="idempotency_wait_timeout",
                status_code=504,
            ) from exc

        try:
            incoming_deadline.require_remaining(
                provider_name="idempotency",
                phase="post_lock_acquire",
            )
            existing = await self._store.get(key)
            if existing is not None:
                if existing.request_fingerprint != fingerprint:
                    logger = get_request_logger(ctx, "engine.idempotency")
                    logger.warning(
                        "idempotency_conflict",
                        idempotency_key=key,
                        existing_operation_id=existing.operation_id,
                    )
                    raise IdempotencyConflict(
                        f"Idempotency-Key '{key}' was already used with a different "
                        "request payload.",
                        param="Idempotency-Key",
                    )

                replayed_ctx = ctx.with_operation_id(existing.operation_id)
                logger = get_request_logger(replayed_ctx, "engine.idempotency")
                logger.info(
                    "idempotency_replay_hit",
                    idempotency_key=key,
                    first_request_id=existing.first_request_id,
                )
                return replayed_ctx, existing.response, True

            response = await runner(ctx)
            record = IdempotencyRecord(
                idempotency_key=key,
                request_fingerprint=fingerprint,
                operation_id=ctx.operation_id,
                first_request_id=ctx.request_id,
                response=response,
            )
            await self._store.put(record)
            logger = get_request_logger(ctx, "engine.idempotency")
            logger.info("idempotency_recorded", idempotency_key=key)
            return ctx, response, False
        finally:
            await lock.__aexit__(None, None, None)
