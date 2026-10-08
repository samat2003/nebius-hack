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

import hashlib
import json
from collections.abc import Awaitable, Callable

from alienese.api.errors import IdempotencyConflict
from alienese.api.models import ChatCompletionRequest, ChatCompletionResponse
from alienese.contracts.context import RequestContext
from alienese.observability.logging import get_request_logger
from alienese.storage.idempotency import IdempotencyRecord, IdempotencyStore


def compute_request_fingerprint(request: ChatCompletionRequest) -> str:
    """Compute a canonical SHA-256 fingerprint over the semantic request body."""
    payload = request.model_dump(mode="json", exclude_none=True)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class IdempotencyCoordinator:
    """Coordinates idempotent execution of logical turn operations."""

    def __init__(self, store: IdempotencyStore) -> None:
        self._store = store

    async def execute(
        self,
        ctx: RequestContext,
        request: ChatCompletionRequest,
        runner: Callable[[RequestContext], Awaitable[ChatCompletionResponse]],
    ) -> tuple[RequestContext, ChatCompletionResponse, bool]:
        """Execute `runner` idempotently.

        Returns `(effective_context, response, was_replayed)`.
        When replayed, `effective_context.operation_id` is updated to the original
        logical `operation_id`, while `effective_context.request_id` remains the
        current HTTP attempt's ID.
        """
        raw_key = ctx.idempotency_key
        if raw_key is None or not raw_key.strip():
            response = await runner(ctx)
            return ctx, response, False

        key = raw_key.strip()
        fingerprint = compute_request_fingerprint(request)
        lock = self._store.key_lock(key)

        async with lock:
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
