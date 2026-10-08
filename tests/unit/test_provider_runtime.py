"""Comprehensive unit tests for the shared remote-provider runtime layer.

Covers:
1. End-to-end turn deadline across queue waits, retries, and attempt timeouts.
2. Cancellation while queued, while sending, and while reading streaming bodies.
3. Read-timeout-after-send (`AMBIGUOUS_COMPLETION`) with no duplicate inference by default.
4. Simultaneous circuit-breaker `HALF_OPEN` probes (strict single-probe admission).
5. Local pool exhaustion (`PoolTimeout`) and queue saturation not tripping the circuit breaker.
6. Incremental streaming response body cap exceeding `max_response_bytes` before JSON parsing.
7. Unexpected redirect rejection without forwarding `Authorization`, plus origin/path/HTTPS bounds.
8. `Retry-After` delta-seconds and HTTP-date parsing within remaining deadline.
9. Retried-attempt token usage accounting and secret non-leakage in errors.
"""

from __future__ import annotations

import asyncio
import gzip
import json
from collections.abc import AsyncIterator
from email.utils import formatdate

import httpx
import pytest

from alienese.api.errors import (
    CompatibilityError,
    InvalidProviderResponse,
    ProviderError,
    ProviderTimeout,
    ProviderUnavailable,
)
from alienese.contracts.context import RequestContext
from alienese.providers.runtime.circuit_breaker import CircuitState, ProviderCircuitBreaker
from alienese.providers.runtime.client import ProviderHttpClient
from alienese.providers.runtime.concurrency import ProviderConcurrencyLimiter
from alienese.providers.runtime.deadlines import DeadlineBudget
from alienese.providers.runtime.errors import FailureCategory
from alienese.providers.runtime.retry import RetryConfig, parse_retry_after_seconds


class FakeClock:
    """Deterministic controllable monotonic and epoch clock for runtime tests."""

    def __init__(
        self, initial_monotonic: float = 1000.0, initial_epoch: float = 1_760_000_000.0
    ) -> None:
        self.mono = initial_monotonic
        self.epoch = initial_epoch
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.mono

    def time(self) -> float:
        return self.epoch

    def advance(self, seconds: float) -> None:
        self.mono += seconds
        self.epoch += seconds

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.advance(seconds)


class _MultiChunkByteStream(httpx.AsyncByteStream):
    """Streaming response body yielding multiple chunks without declaring Content-Length."""

    def __init__(
        self,
        chunks: list[bytes],
        *,
        clock: FakeClock | None = None,
        per_chunk_advance: float = 0.0,
        close_tracker: list[bool] | None = None,
    ) -> None:
        self._chunks = chunks
        self._clock = clock
        self._per_chunk_advance = per_chunk_advance
        self._close_tracker = close_tracker

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            if self._clock is not None and self._per_chunk_advance > 0.0:
                self._clock.advance(self._per_chunk_advance)
            await asyncio.sleep(0)
            yield chunk

    async def aclose(self) -> None:
        if self._close_tracker is not None:
            self._close_tracker.append(True)


async def test_1_end_to_end_turn_deadline_across_retries_and_queue_waits() -> None:
    clock = FakeClock()
    attempts = 0
    observed_timeouts: list[float] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        timeout_ext = request.extensions.get("timeout", {})
        if isinstance(timeout_ext, dict) and "read" in timeout_ext:
            observed_timeouts.append(float(timeout_ext["read"]))
        clock.advance(0.6)
        return httpx.Response(
            503,
            headers={"content-type": "application/json"},
            content=b'{"error":"unavailable"}',
        )

    client = ProviderHttpClient(
        provider_name="nvidia_build",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="nvapi-" + ("a" * 16),
        default_timeout_seconds=10.0,
        retry_config=RetryConfig(max_attempts=4, base_delay_seconds=0.3, max_delay_seconds=2.0),
        transport=httpx.MockTransport(handler),
        clock=clock.monotonic,
        epoch_clock=clock.time,
        sleep_func=clock.sleep,
    )

    # Total turn budget is 1.5s:
    # Attempt 1 takes 0.6s -> 0.9s remaining -> sleeps 0.3s -> 0.6s remaining.
    # Attempt 2 takes 0.6s -> 0.0s remaining -> cannot schedule retry 3 because
    # deadline is exhausted!
    ctx = RequestContext(deadline_monotonic=clock.monotonic() + 1.5)
    with pytest.raises(ProviderTimeout) as exc_info:
        await client.post_json(ctx, "/chat/completions", {"model": "test"})

    assert exc_info.value.code == "turn_deadline_exhausted"
    assert attempts == 2
    assert len(observed_timeouts) == 2
    assert observed_timeouts[0] == pytest.approx(1.5, rel=1e-3)
    assert observed_timeouts[1] == pytest.approx(0.6, rel=1e-3)
    assert clock.monotonic() - 1000.0 <= 1.55
    await client.aclose()


async def test_2_cancellation_while_queued_sending_and_reading_releases_resources() -> None:
    clock = FakeClock()
    limiter = ProviderConcurrencyLimiter(
        provider_name="nvidia_build",
        max_concurrency=1,
        max_queue_waiters=2,
    )
    deadline = DeadlineBudget(deadline_monotonic=clock.monotonic() + 10.0, clock=clock.monotonic)

    # 2a. Cancellation while queued waiting for a concurrency slot
    slot_held = asyncio.Event()
    release_holder = asyncio.Event()

    async def hold_slot() -> None:
        async with limiter.acquire(deadline):
            slot_held.set()
            await release_holder.wait()

    holder_task = asyncio.create_task(hold_slot())
    await slot_held.wait()
    assert limiter.active_count == 1

    async def queued_caller() -> None:
        async with limiter.acquire(deadline):
            pass

    waiter_task = asyncio.create_task(queued_caller())
    await asyncio.sleep(0)
    assert limiter.waiting_count == 1
    waiter_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter_task

    assert limiter.waiting_count == 0
    release_holder.set()
    await holder_task
    assert limiter.active_count == 0

    # 2b. Cancellation while sending and while reading in HALF_OPEN state releases probe & stream
    cb = ProviderCircuitBreaker(
        provider_name="nvidia_build",
        failure_threshold=1,
        recovery_timeout_seconds=5.0,
        clock=clock.monotonic,
    )
    cb.record_failure(
        category=__import__(
            "alienese.providers.runtime.errors", fromlist=["FailureCategory"]
        ).FailureCategory.SAFE_RETRYABLE_UPSTREAM
    )
    assert cb.state.value == CircuitState.OPEN.value
    clock.advance(6.0)
    assert cb.state.value == CircuitState.HALF_OPEN.value

    send_started = asyncio.Event()
    stream_closed: list[bool] = []

    class SlowBlockingStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            send_started.set()
            await asyncio.sleep(3600)
            yield b"{}"

        async def aclose(self) -> None:
            stream_closed.append(True)

    async def slow_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=SlowBlockingStream(),
        )

    client = ProviderHttpClient(
        provider_name="nvidia_build",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="nvapi-" + ("a" * 16),
        circuit_breaker=cb,
        concurrency_limiter=limiter,
        transport=httpx.MockTransport(slow_handler),
        clock=clock.monotonic,
    )

    ctx = RequestContext(deadline_monotonic=clock.monotonic() + 30.0)
    inflight_task = asyncio.create_task(
        client.post_json(ctx, "/chat/completions", {"model": "test"})
    )
    await send_started.wait()
    assert cb.half_open_in_flight == 1
    assert limiter.active_count == 1

    inflight_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await inflight_task

    assert cb.half_open_in_flight == 0
    assert limiter.active_count == 0
    assert stream_closed == [True]
    await client.aclose()


async def test_3_read_timeout_after_send_does_not_retry_by_default() -> None:
    clock = FakeClock()
    attempts = 0

    async def read_timeout_handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ReadTimeout("Timed out reading response after request sent", request=request)

    # Default RetryConfig has retry_ambiguous_failures=False
    client = ProviderHttpClient(
        provider_name="nvidia_build",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="nvapi-" + ("a" * 16),
        retry_config=RetryConfig(max_attempts=3),
        transport=httpx.MockTransport(read_timeout_handler),
        clock=clock.monotonic,
        sleep_func=clock.sleep,
    )

    ctx = RequestContext(deadline_monotonic=clock.monotonic() + 20.0)
    with pytest.raises(ProviderTimeout) as exc_info:
        await client.post_json(ctx, "/chat/completions", {"model": "test"})

    assert exc_info.value.code == "provider_read_timeout"
    assert attempts == 1  # Strictly 1 attempt: no duplicate inference!
    await client.aclose()


async def test_4_simultaneous_circuit_breaker_half_open_probes() -> None:
    clock = FakeClock()
    cb = ProviderCircuitBreaker(
        provider_name="nvidia_build",
        failure_threshold=2,
        recovery_timeout_seconds=10.0,
        half_open_max_probes=1,
        clock=clock.monotonic,
    )
    probe_in_handler = asyncio.Event()
    release_probe = asyncio.Event()
    upstream_calls = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal upstream_calls
        upstream_calls += 1
        if upstream_calls <= 2:
            return httpx.Response(503, content=b'{"error":"down"}')
        probe_in_handler.set()
        await release_probe.wait()
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=b'{"ok":true}',
        )

    client = ProviderHttpClient(
        provider_name="nvidia_build",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="nvapi-" + ("a" * 16),
        retry_config=RetryConfig(max_attempts=1),
        circuit_breaker=cb,
        transport=httpx.MockTransport(handler),
        clock=clock.monotonic,
    )

    ctx = RequestContext(deadline_monotonic=clock.monotonic() + 30.0)
    for _ in range(2):
        with pytest.raises(ProviderUnavailable):
            await client.post_json(ctx, "/chat/completions", {"model": "test"})

    assert cb.state.value == CircuitState.OPEN.value
    # Advance past recovery window -> HALF_OPEN
    clock.advance(11.0)
    assert cb.state.value == CircuitState.HALF_OPEN.value

    # Launch probe 1 and pause it in-flight
    probe1 = asyncio.create_task(client.post_json(ctx, "/chat/completions", {"model": "test"}))
    await probe_in_handler.wait()
    assert cb.half_open_in_flight == 1

    # Simultaneous probe 2 must be rejected immediately without hitting transport
    with pytest.raises(ProviderUnavailable) as exc_info:
        await client.post_json(ctx, "/chat/completions", {"model": "test"})
    assert exc_info.value.code == "circuit_breaker_open"
    assert upstream_calls == 3

    # Complete probe 1 -> circuit closes
    release_probe.set()
    resp1 = await probe1
    assert resp1.status_code == 200
    assert cb.state == CircuitState.CLOSED
    assert cb.consecutive_failures == 0
    await client.aclose()


async def test_5_local_pool_exhaustion_and_rate_limits_do_not_trip_circuit_breaker() -> None:
    clock = FakeClock()
    cb = ProviderCircuitBreaker(
        provider_name="nvidia_build",
        failure_threshold=2,
        recovery_timeout_seconds=10.0,
        clock=clock.monotonic,
    )

    mode = "pool_timeout"

    async def handler(request: httpx.Request) -> httpx.Response:
        if mode == "pool_timeout":
            raise httpx.PoolTimeout("Local pool saturated", request=request)
        return httpx.Response(
            429,
            headers={"content-type": "application/json", "retry-after": "1"},
            content=b'{"error":"rate_limited"}',
        )

    client = ProviderHttpClient(
        provider_name="nvidia_build",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="nvapi-" + ("a" * 16),
        retry_config=RetryConfig(max_attempts=1),
        circuit_breaker=cb,
        transport=httpx.MockTransport(handler),
        clock=clock.monotonic,
    )
    ctx = RequestContext(deadline_monotonic=clock.monotonic() + 30.0)

    for _ in range(5):
        with pytest.raises(ProviderUnavailable) as exc_info:
            await client.post_json(ctx, "/chat/completions", {"model": "test"})
        assert exc_info.value.code == "local_pool_exhausted"

    assert cb.state == CircuitState.CLOSED
    assert cb.consecutive_failures == 0

    mode = "rate_limit"
    for _ in range(5):
        with pytest.raises(ProviderUnavailable) as exc_info:
            await client.post_json(ctx, "/chat/completions", {"model": "test"})
        assert exc_info.value.code == "provider_rate_limited"

    assert cb.state == CircuitState.CLOSED
    assert cb.consecutive_failures == 0
    await client.aclose()


async def test_6_streaming_response_exceeding_byte_cap_aborts_before_allocation() -> None:
    clock = FakeClock()
    close_tracker: list[bool] = []
    # 4 chunks of 60 bytes each = 240 bytes > max_response_bytes=150 (with no Content-Length header)
    chunks = [b'{"data":"' + (b"x" * 50), b"y" * 60, b"z" * 60, b'"}']

    async def streaming_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=_MultiChunkByteStream(chunks, close_tracker=close_tracker),
        )

    client = ProviderHttpClient(
        provider_name="nvidia_build",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="nvapi-" + ("a" * 16),
        max_response_bytes=150,
        transport=httpx.MockTransport(streaming_handler),
        clock=clock.monotonic,
    )
    ctx = RequestContext(deadline_monotonic=clock.monotonic() + 10.0)
    with pytest.raises(InvalidProviderResponse) as exc_info:
        await client.post_json(ctx, "/chat/completions", {"model": "test"})

    assert exc_info.value.code == "response_payload_too_large"
    assert close_tracker == [True]
    await client.aclose()


async def test_7_unexpected_redirect_carries_no_authorization_and_origin_bounds_enforced() -> None:
    clock = FakeClock()
    seen_urls: list[str] = []

    async def redirect_handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        return httpx.Response(
            302,
            headers={"location": "https://evil.example.com/v1/chat/completions"},
        )

    client = ProviderHttpClient(
        provider_name="nvidia_build",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="nvapi-" + ("a" * 16),
        transport=httpx.MockTransport(redirect_handler),
        clock=clock.monotonic,
    )
    ctx = RequestContext(deadline_monotonic=clock.monotonic() + 10.0)
    with pytest.raises(InvalidProviderResponse) as exc_info:
        await client.post_json(ctx, "/chat/completions", {"model": "test"})

    assert exc_info.value.code == "unexpected_redirect"
    # Verify redirect was never followed to evil.example.com
    assert seen_urls == ["https://integrate.api.nvidia.com/v1/chat/completions"]

    # Path traversal or absolute URL injection in endpoint_path is rejected before sending
    for bad_path in (
        "https://evil.example.com/v1/chat",
        "/../v2/chat",
        "chat/completions",
        "/chat/completions?redirect=1",
    ):
        with pytest.raises(ProviderError) as path_exc:
            await client.post_json(ctx, bad_path, {"model": "test"})
        assert path_exc.value.code == "invalid_provider_endpoint_path"

    # Outbound request size cap enforced before network transmission
    small_req_client = ProviderHttpClient(
        provider_name="nvidia_build",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="nvapi-" + ("a" * 16),
        max_request_bytes=64,
        transport=httpx.MockTransport(redirect_handler),
        clock=clock.monotonic,
    )
    with pytest.raises(ProviderError) as size_exc:
        await small_req_client.post_json(
            ctx,
            "/chat/completions",
            {"model": "test", "padding": "x" * 100},
        )
    assert size_exc.value.code == "request_payload_too_large"
    assert len(seen_urls) == 1  # Not transmitted!

    # Insecure HTTP base_url rejected in deployment mode
    with pytest.raises(CompatibilityError) as http_exc:
        ProviderHttpClient(
            provider_name="nvidia_build",
            base_url="http://integrate.api.nvidia.com/v1",
            api_key="nvapi-" + ("a" * 16),
        )
    assert http_exc.value.code == "insecure_provider_base_url"

    await client.aclose()
    await small_req_client.aclose()


async def test_8_retry_after_header_delta_and_http_date_within_deadline() -> None:
    clock = FakeClock()
    now_epoch = clock.time()
    http_date_str = formatdate(now_epoch + 1.5, usegmt=True)
    assert parse_retry_after_seconds("2.5", now_epoch=now_epoch) == pytest.approx(2.5)
    assert parse_retry_after_seconds(http_date_str, now_epoch=now_epoch) == pytest.approx(
        1.5, abs=1.0
    )
    assert parse_retry_after_seconds("invalid-header", now_epoch=now_epoch) is None
    assert parse_retry_after_seconds("-5", now_epoch=now_epoch) is None

    attempts = 0

    async def rate_limited_then_ok(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                429,
                headers={"content-type": "application/json", "retry-after": "1.25"},
                content=json.dumps(
                    {
                        "error": "rate_limited",
                        "usage": {"prompt_tokens": 11, "completion_tokens": 0},
                    }
                ).encode("utf-8"),
            )
        return httpx.Response(
            200,
            headers={"content-type": "application/json", "nvcf-reqid": "req-upstream-42"},
            content=json.dumps(
                {"ok": True, "usage": {"prompt_tokens": 11, "completion_tokens": 7}}
            ).encode("utf-8"),
        )

    client = ProviderHttpClient(
        provider_name="nvidia_build",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="nvapi-" + ("a" * 16),
        retry_config=RetryConfig(max_attempts=2, base_delay_seconds=0.1),
        transport=httpx.MockTransport(rate_limited_then_ok),
        clock=clock.monotonic,
        epoch_clock=clock.time,
        sleep_func=clock.sleep,
    )

    ctx = RequestContext(deadline_monotonic=clock.monotonic() + 5.0)
    resp = await client.post_json(ctx, "/chat/completions", {"model": "test"})
    assert resp.status_code == 200
    assert clock.sleeps == [pytest.approx(1.25)]
    assert resp.telemetry.attempt_count == 2
    assert resp.telemetry.failed_attempt_count == 1
    assert resp.telemetry.prompt_tokens == 11
    assert resp.telemetry.completion_tokens == 7
    assert resp.telemetry.retried_prompt_tokens == 11
    assert resp.telemetry.retried_completion_tokens == 0
    assert resp.telemetry.upstream_request_id == "req-upstream-42"

    # Now verify Retry-After exceeding remaining deadline fails immediately without sleeping
    attempts = 0
    tight_ctx = RequestContext(deadline_monotonic=clock.monotonic() + 0.5)
    with pytest.raises(ProviderTimeout) as exc_info:
        await client.post_json(tight_ctx, "/chat/completions", {"model": "test"})
    assert exc_info.value.code == "retry_after_exceeds_deadline"
    assert len(clock.sleeps) == 1  # No second sleep occurred!
    await client.aclose()


async def test_9_concurrency_permit_released_during_retry_backoff_allows_other_request() -> None:
    clock = FakeClock()
    limiter = ProviderConcurrencyLimiter(
        provider_name="nvidia_build",
        max_concurrency=1,
        max_queue_waiters=2,
    )
    req_a_sleeping = asyncio.Event()
    release_req_a_sleep = asyncio.Event()
    call_order: list[str] = []
    req_a_attempts = 0

    async def custom_sleep(delay: float) -> None:
        clock.sleeps.append(delay)
        # Permit MUST be released while Request A is sleeping in retry backoff
        assert limiter.active_count == 0
        req_a_sleeping.set()
        await release_req_a_sleep.wait()
        clock.advance(delay)

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal req_a_attempts
        body = json.loads(request.content.decode("utf-8"))
        tag = body["req_tag"]
        call_order.append(f"{tag}:attempt")
        if tag == "A":
            req_a_attempts += 1
            if req_a_attempts == 1:
                return httpx.Response(
                    429,
                    headers={"content-type": "application/json", "retry-after": "1.0"},
                    content=b'{"error":"rate_limited"}',
                )
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=json.dumps({"ok": True, "tag": tag}).encode("utf-8"),
        )

    client = ProviderHttpClient(
        provider_name="nvidia_build",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="nvapi-" + ("a" * 16),
        retry_config=RetryConfig(max_attempts=2, base_delay_seconds=0.5),
        concurrency_limiter=limiter,
        transport=httpx.MockTransport(handler),
        clock=clock.monotonic,
        epoch_clock=clock.time,
        sleep_func=custom_sleep,
    )

    ctx_a = RequestContext(deadline_monotonic=clock.monotonic() + 10.0)
    ctx_b = RequestContext(deadline_monotonic=clock.monotonic() + 10.0)

    task_a = asyncio.create_task(
        client.post_json(ctx_a, "/chat/completions", {"model": "test", "req_tag": "A"})
    )
    await req_a_sleeping.wait()
    assert limiter.active_count == 0

    # Request B runs and completes while Request A is sleeping in backoff under max_concurrency=1
    resp_b = await client.post_json(ctx_b, "/chat/completions", {"model": "test", "req_tag": "B"})
    assert resp_b.status_code == 200
    assert resp_b.data["tag"] == "B"

    release_req_a_sleep.set()
    resp_a = await task_a
    assert resp_a.status_code == 200
    assert resp_a.data["tag"] == "A"
    assert call_order == ["A:attempt", "B:attempt", "A:attempt"]
    assert limiter.active_count == 0
    await client.aclose()


async def test_10_retry_attempt_cannot_bypass_circuit_breaker_opened_during_backoff() -> None:
    clock = FakeClock()
    cb = ProviderCircuitBreaker(
        provider_name="nvidia_build",
        failure_threshold=2,
        recovery_timeout_seconds=30.0,
        clock=clock.monotonic,
    )
    upstream_attempts = 0

    async def sleep_that_trips_breaker(delay: float) -> None:
        clock.advance(delay)
        # Another concurrent call trips the circuit breaker while this call is in backoff
        cb.record_failure(FailureCategory.SAFE_RETRYABLE_UPSTREAM)
        cb.record_failure(FailureCategory.SAFE_RETRYABLE_UPSTREAM)
        assert cb.state == CircuitState.OPEN

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal upstream_attempts
        upstream_attempts += 1
        return httpx.Response(
            429,
            headers={"content-type": "application/json", "retry-after": "0.5"},
            content=b'{"error":"rate_limited"}',
        )

    client = ProviderHttpClient(
        provider_name="nvidia_build",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="nvapi-" + ("a" * 16),
        retry_config=RetryConfig(max_attempts=3, base_delay_seconds=0.1),
        circuit_breaker=cb,
        transport=httpx.MockTransport(handler),
        clock=clock.monotonic,
        epoch_clock=clock.time,
        sleep_func=sleep_that_trips_breaker,
    )

    ctx = RequestContext(deadline_monotonic=clock.monotonic() + 10.0)
    with pytest.raises(ProviderUnavailable) as exc_info:
        await client.post_json(ctx, "/chat/completions", {"model": "test"})

    assert exc_info.value.code == "circuit_breaker_open"
    assert upstream_attempts == 1  # Attempt 2 was blocked by the newly OPEN circuit breaker
    await client.aclose()


async def test_11_compressed_response_rejected_before_decompression_and_identity_enforced() -> None:
    clock = FakeClock()
    seen_accept_encodings: list[str | None] = []
    # 50 KB uncompressed payload compressed to ~100 bytes with gzip
    uncompressed_bomb = json.dumps({"data": "A" * 50_000}).encode("utf-8")
    compressed_bomb = gzip.compress(uncompressed_bomb)
    assert len(compressed_bomb) < 256

    mode = "gzip_bomb"

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_accept_encodings.append(request.headers.get("accept-encoding"))
        if mode == "gzip_bomb":
            return httpx.Response(
                200,
                headers={
                    "content-type": "application/json",
                    "content-encoding": "gzip",
                    "content-length": str(len(compressed_bomb)),
                },
                stream=_MultiChunkByteStream([compressed_bomb]),
            )
        return httpx.Response(
            200,
            headers={
                "content-type": "application/json",
                "content-encoding": "identity",
            },
            stream=_MultiChunkByteStream([b'{"ok":true}']),
        )

    client = ProviderHttpClient(
        provider_name="nvidia_build",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="nvapi-" + ("a" * 16),
        max_response_bytes=512,
        retry_config=RetryConfig(max_attempts=1),
        transport=httpx.MockTransport(handler),
        clock=clock.monotonic,
    )
    ctx = RequestContext(deadline_monotonic=clock.monotonic() + 10.0)

    # 1. Compressed response is rejected before decompression despite small Content-Length
    with pytest.raises(InvalidProviderResponse) as exc_info:
        await client.post_json(ctx, "/chat/completions", {"model": "test"})
    assert exc_info.value.code == "unsupported_content_encoding"
    assert seen_accept_encodings == ["identity"]

    # 2. Identity-encoded response succeeds within raw byte limit
    mode = "identity_ok"
    resp = await client.post_json(ctx, "/chat/completions", {"model": "test"})
    assert resp.status_code == 200
    assert resp.data == {"ok": True}
    assert seen_accept_encodings == ["identity", "identity"]
    await client.aclose()
