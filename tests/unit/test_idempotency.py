"""Unit tests for idempotent request handling and logical operation separation."""

from __future__ import annotations

import asyncio

import pytest

from alienese.api.errors import IdempotencyConflict
from alienese.api.models import (
    AssistantMessageOutput,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    UsageInfo,
)
from alienese.contracts.context import RequestContext
from alienese.engine.idempotency import IdempotencyCoordinator, compute_request_fingerprint
from alienese.storage.idempotency import InMemoryIdempotencyStore


def _make_sample_request(text: str = "Hello Alienese") -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(
        {
            "model": "alienese-default",
            "messages": [{"role": "user", "content": text}],
        }
    )


def _make_sample_response(operation_id: str, content: str = "Ack") -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id=f"chatcmpl-{operation_id}",
        created=1728345600,
        model="alienese-default",
        choices=[
            ChatCompletionChoice(
                index=0,
                message=AssistantMessageOutput(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(prompt_tokens=2, completion_tokens=1, total_tokens=3),
    )


async def test_idempotency_first_and_duplicate_request_preserve_operation_id() -> None:
    store = InMemoryIdempotencyStore()
    coordinator = IdempotencyCoordinator(store)
    req = _make_sample_request("Check idempotency")

    execution_count = 0

    async def _runner(ctx: RequestContext) -> ChatCompletionResponse:
        nonlocal execution_count
        execution_count += 1
        return _make_sample_response(ctx.operation_id)

    ctx1 = RequestContext(
        request_id="req_attempt_1",
        operation_id="op_logical_1",
        idempotency_key="idem-key-100",
    )
    eff_ctx1, resp1, replayed1 = await coordinator.execute(ctx1, req, _runner)
    assert replayed1 is False
    assert execution_count == 1
    assert eff_ctx1.request_id == "req_attempt_1"
    assert eff_ctx1.operation_id == "op_logical_1"

    # Second HTTP attempt with the same Idempotency-Key gets a new request_id
    # and a newly generated initial operation_id, which is rebound to op_logical_1.
    ctx2 = RequestContext(
        request_id="req_attempt_2",
        operation_id="op_logical_2_unused",
        idempotency_key="idem-key-100",
    )
    eff_ctx2, resp2, replayed2 = await coordinator.execute(ctx2, req, _runner)
    assert replayed2 is True
    assert execution_count == 1
    # Separate HTTP attempt ID is preserved while logical operation ID matches first completion
    assert eff_ctx2.request_id == "req_attempt_2"
    assert eff_ctx2.operation_id == "op_logical_1"
    assert resp2 == resp1
    assert resp2.id == "chatcmpl-op_logical_1"


async def test_idempotency_conflict_on_mismatched_request_body() -> None:
    store = InMemoryIdempotencyStore()
    coordinator = IdempotencyCoordinator(store)
    req1 = _make_sample_request("First payload")
    req2 = _make_sample_request("Materially different payload")

    async def _runner(ctx: RequestContext) -> ChatCompletionResponse:
        return _make_sample_response(ctx.operation_id)

    ctx1 = RequestContext(idempotency_key="idem-conflict-key")
    await coordinator.execute(ctx1, req1, _runner)

    ctx2 = RequestContext(idempotency_key="idem-conflict-key")
    with pytest.raises(IdempotencyConflict, match="already used with a different request payload"):
        await coordinator.execute(ctx2, req2, _runner)


async def test_idempotency_concurrent_duplicates_execute_runner_only_once() -> None:
    store = InMemoryIdempotencyStore()
    coordinator = IdempotencyCoordinator(store)
    req = _make_sample_request("Concurrent payload")

    execution_count = 0

    async def _slow_runner(ctx: RequestContext) -> ChatCompletionResponse:
        nonlocal execution_count
        execution_count += 1
        await asyncio.sleep(0.02)
        return _make_sample_response(ctx.operation_id, content="Concurrent result")

    contexts = [
        RequestContext(
            request_id=f"req_concurrent_{i}",
            operation_id=f"op_concurrent_{i}",
            idempotency_key="idem-concurrent-key",
        )
        for i in range(5)
    ]

    results = await asyncio.gather(*(coordinator.execute(c, req, _slow_runner) for c in contexts))

    assert execution_count == 1
    replayed_flags = [r[2] for r in results]
    assert replayed_flags.count(False) == 1
    assert replayed_flags.count(True) == 4

    first_resp = results[0][1]
    for eff_ctx, resp, _ in results:
        assert resp == first_resp
        assert eff_ctx.operation_id == "op_concurrent_0"


def test_request_fingerprint_is_deterministic() -> None:
    r1 = _make_sample_request("Same text")
    r2 = _make_sample_request("Same text")
    assert compute_request_fingerprint(r1) == compute_request_fingerprint(r2)
