"""Contract tests for the Hybrid Gateway mode, provider configuration matrix, and error boundaries.

Covers:
1. Fake mode with populated `.env` makes zero outbound calls.
2. Invalid provider-mode combinations fail before serving.
3. Hybrid mode end-to-end turn execution via `POST /v1/chat/completions`
   preserving `alienese-default` external model identity, one-action-per-turn,
   and idempotent replay without duplicate remote calls.
4. Missing usage accounting in remote generator response produces `usage=None`
   on wire and `None` in telemetry (never fabricated as zero).
5. Remote NVIDIA generator failure in hybrid mode never silently falls back to
   `FakeGenerator` output.
6. Zero secret leakage in telemetry, error envelopes, and `Settings.safe_dump()`.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from alienese.api.app import create_app
from alienese.api.errors import CompatibilityError
from alienese.config import Settings
from alienese.providers.fake import FakeGenerator
from alienese.providers.generator.nebius_token_factory import (
    NEBIUS_DEFAULT_BASE_URL,
    NebiusTokenFactoryGenerator,
)
from alienese.providers.generator.nvidia_build import (
    NVIDIA_DEFAULT_BASE_URL,
    NVIDIA_NEMOTRON_SUPER_MODEL,
    NvidiaBuildGenerator,
)
from alienese.providers.runtime.client import ProviderHttpClient
from alienese.providers.runtime.retry import RetryConfig


async def test_fake_mode_with_populated_env_makes_zero_outbound_calls() -> None:
    cfg = Settings(
        _env_file=None,
        alienese_provider_mode="fake",
        generator_provider="nvidia_build",
        generator_api_key=SecretStr("nvapi-" + ("a" * 16)),
        generator_base_url=NVIDIA_DEFAULT_BASE_URL,
        generator_model=NVIDIA_NEMOTRON_SUPER_MODEL,
        nebius_token_factory_key=SecretStr("nebius-" + ("b" * 16)),
        nebius_token_factory_base_url=NEBIUS_DEFAULT_BASE_URL,
        nebius_token_factory_model=NVIDIA_NEMOTRON_SUPER_MODEL,
    )
    app = create_app(settings=cfg)
    assert isinstance(app.state.generator, FakeGenerator)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "alienese-default",
                "messages": [{"role": "user", "content": "Hello in fake mode."}],
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "alienese-default"
    assert body["choices"][0]["message"]["content"].startswith("[fake:answer]")


@pytest.mark.parametrize(
    ("kwargs", "expected_code"),
    [
        # 1. 'remote' mode is not yet enabled
        ({"alienese_provider_mode": "remote"}, "unsupported_provider_mode"),
        # 2. hybrid mode with generator_provider='fake'
        (
            {
                "alienese_provider_mode": "hybrid",
                "generator_provider": "fake",
                "generator_api_key": SecretStr("nvapi-" + ("a" * 16)),
            },
            "invalid_provider_mode_combination",
        ),
        # 3. hybrid nvidia_build with missing generator_api_key (even if nebius key present)
        (
            {
                "alienese_provider_mode": "hybrid",
                "generator_provider": "nvidia_build",
                "generator_api_key": None,
                "nebius_token_factory_key": SecretStr("nebius-" + ("b" * 16)),
            },
            "missing_provider_api_key",
        ),
        # 4. hybrid nebius_token_factory missing nebius_token_factory_key (even if nvidia key set)
        (
            {
                "alienese_provider_mode": "hybrid",
                "generator_provider": "nebius_token_factory",
                "generator_api_key": SecretStr("nvapi-" + ("a" * 16)),
                "nebius_token_factory_key": None,
            },
            "missing_provider_api_key",
        ),
        # 5. nvidia_build configured with Nebius URL
        (
            {
                "alienese_provider_mode": "hybrid",
                "generator_provider": "nvidia_build",
                "generator_api_key": SecretStr("nvapi-" + ("a" * 16)),
                "generator_base_url": NEBIUS_DEFAULT_BASE_URL,
                "generator_model": NVIDIA_NEMOTRON_SUPER_MODEL,
            },
            "provider_origin_mismatch",
        ),
        # 6. nvidia_build configured with unallowlisted wildcard subdomain
        (
            {
                "alienese_provider_mode": "hybrid",
                "generator_provider": "nvidia_build",
                "generator_api_key": SecretStr("nvapi-" + ("a" * 16)),
                "generator_base_url": "https://untrusted.api.nvidia.com/v1",
                "generator_model": NVIDIA_NEMOTRON_SUPER_MODEL,
            },
            "provider_origin_mismatch",
        ),
        # 7. nebius_token_factory reusing an NVIDIA nvapi-* key
        (
            {
                "alienese_provider_mode": "hybrid",
                "generator_provider": "nebius_token_factory",
                "nebius_token_factory_key": SecretStr("nvapi-" + ("a" * 16)),
                "nebius_token_factory_base_url": NEBIUS_DEFAULT_BASE_URL,
                "nebius_token_factory_model": NVIDIA_NEMOTRON_SUPER_MODEL,
            },
            "cross_provider_credential_reuse",
        ),
        # 8. nvidia_build reusing a Nebius key
        (
            {
                "alienese_provider_mode": "hybrid",
                "generator_provider": "nvidia_build",
                "generator_api_key": SecretStr("nebius-" + ("b" * 16)),
                "generator_base_url": NVIDIA_DEFAULT_BASE_URL,
                "generator_model": NVIDIA_NEMOTRON_SUPER_MODEL,
            },
            "cross_provider_credential_reuse",
        ),
        # 9. nebius_token_factory pointing to NVIDIA URL
        (
            {
                "alienese_provider_mode": "hybrid",
                "generator_provider": "nebius_token_factory",
                "nebius_token_factory_key": SecretStr("nebius-" + ("b" * 16)),
                "nebius_token_factory_base_url": NVIDIA_DEFAULT_BASE_URL,
                "nebius_token_factory_model": NVIDIA_NEMOTRON_SUPER_MODEL,
            },
            "provider_origin_mismatch",
        ),
        # 10. nebius_token_factory pointing to unallowlisted wildcard subdomain
        (
            {
                "alienese_provider_mode": "hybrid",
                "generator_provider": "nebius_token_factory",
                "nebius_token_factory_key": SecretStr("nebius-" + ("b" * 16)),
                "nebius_token_factory_base_url": "https://untrusted.nebius.com/v1",
                "nebius_token_factory_model": NVIDIA_NEMOTRON_SUPER_MODEL,
            },
            "provider_origin_mismatch",
        ),
        # 11. non-fake retriever_provider in Phase 2
        (
            {
                "alienese_provider_mode": "hybrid",
                "retriever_provider": "embeddinggemma",
                "generator_provider": "nvidia_build",
                "generator_api_key": SecretStr("nvapi-" + ("a" * 16)),
            },
            "invalid_provider_mode_combination",
        ),
        # 12. non-fake controller_provider in Phase 2
        (
            {
                "alienese_provider_mode": "hybrid",
                "controller_provider": "mini_jev",
                "generator_provider": "nvidia_build",
                "generator_api_key": SecretStr("nvapi-" + ("a" * 16)),
            },
            "invalid_provider_mode_combination",
        ),
    ],
)
def test_invalid_provider_mode_combinations_fail_before_serving(
    kwargs: dict[str, Any],
    expected_code: str,
) -> None:
    with pytest.raises(CompatibilityError) as exc_info:
        Settings(_env_file=None, **kwargs)
    assert exc_info.value.code == expected_code


async def test_hybrid_gateway_turn_idempotency_one_action_and_missing_usage_handling() -> None:
    upstream_calls = 0
    include_usage = True

    async def nvidia_mock(request: httpx.Request) -> httpx.Response:
        nonlocal upstream_calls
        upstream_calls += 1
        assert request.headers.get("authorization") == "Bearer nvapi-" + ("a" * 16)
        payload: dict[str, Any] = {
            "id": f"chatcmpl-nv-{upstream_calls}",
            "object": "chat.completion",
            "model": NVIDIA_NEMOTRON_SUPER_MODEL,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "NVIDIA Nemotron synthesized this response.",
                    },
                }
            ],
        }
        if include_usage:
            payload["usage"] = {
                "prompt_tokens": 19,
                "completion_tokens": 6,
                "total_tokens": 25,
            }
        return httpx.Response(
            200,
            headers={"content-type": "application/json", "nvcf-reqid": "nvcf-mock-01"},
            content=json.dumps(payload).encode("utf-8"),
        )

    cfg = Settings(
        alienese_provider_mode="hybrid",
        generator_provider="nvidia_build",
        generator_api_key=SecretStr("nvapi-" + ("a" * 16)),
        generator_base_url=NVIDIA_DEFAULT_BASE_URL,
        generator_model=NVIDIA_NEMOTRON_SUPER_MODEL,
    )
    http_client = ProviderHttpClient(
        provider_name="nvidia_build",
        base_url=NVIDIA_DEFAULT_BASE_URL,
        api_key="nvapi-" + ("a" * 16),
        retry_config=RetryConfig(max_attempts=1),
        transport=httpx.MockTransport(nvidia_mock),
    )
    gen = NvidiaBuildGenerator(http_client=http_client, model_id=NVIDIA_NEMOTRON_SUPER_MODEL)
    app = create_app(settings=cfg, generator=gen)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        req_body = {
            "model": "alienese-default",
            "messages": [{"role": "user", "content": "What does a software test do?"}],
        }
        r1 = await client.post(
            "/v1/chat/completions",
            headers={"Idempotency-Key": "idem-hybrid-001"},
            json=req_body,
        )
        assert r1.status_code == 200
        assert r1.headers["X-Idempotent-Replay"] == "false"
        b1 = r1.json()
        assert b1["model"] == "alienese-default"
        assert len(b1["choices"]) == 1
        assert (
            b1["choices"][0]["message"]["content"] == "NVIDIA Nemotron synthesized this response."
        )
        assert b1["choices"][0]["message"]["tool_calls"] is None
        assert b1["usage"] == {"prompt_tokens": 19, "completion_tokens": 6, "total_tokens": 25}
        assert upstream_calls == 1

        # Duplicate request with same Idempotency-Key replays without a second NVIDIA call
        r2 = await client.post(
            "/v1/chat/completions",
            headers={"Idempotency-Key": "idem-hybrid-001"},
            json=req_body,
        )
        assert r2.status_code == 200
        assert r2.headers["X-Idempotent-Replay"] == "true"
        assert r2.json() == b1
        assert upstream_calls == 1

        # Now omit usage from upstream response -> usage is None (not fabricated as 0)
        include_usage = False
        r3 = await client.post(
            "/v1/chat/completions",
            json=req_body,
        )
        assert r3.status_code == 200
        b3 = r3.json()
        assert b3["usage"] is None
        op_id_3 = r3.headers["X-Operation-ID"]
        stored_art = await app.state.trace_store.get_by_operation_id(op_id_3)
        assert stored_art is not None
        assert stored_art.telemetry.generation_telemetry is not None
        assert stored_art.telemetry.generation_telemetry.prompt_tokens is None
        assert stored_art.telemetry.generation_telemetry.completion_tokens is None

    await gen.aclose()


async def test_hybrid_generator_failure_never_falls_back_to_fake_and_redacts_secrets() -> None:
    secret_key = "nvapi-SuperSecretKeyDoNotLeak999999"

    async def failing_nvidia_mock(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            headers={"content-type": "application/json"},
            content=json.dumps({"error": f"Invalid key {secret_key}"}).encode("utf-8"),
        )

    cfg = Settings(
        alienese_provider_mode="hybrid",
        generator_provider="nvidia_build",
        generator_api_key=SecretStr(secret_key),
        generator_base_url=NVIDIA_DEFAULT_BASE_URL,
        generator_model=NVIDIA_NEMOTRON_SUPER_MODEL,
    )
    assert secret_key not in json.dumps(cfg.safe_dump())

    http_client = ProviderHttpClient(
        provider_name="nvidia_build",
        base_url=NVIDIA_DEFAULT_BASE_URL,
        api_key=secret_key,
        retry_config=RetryConfig(max_attempts=1),
        transport=httpx.MockTransport(failing_nvidia_mock),
    )
    gen = NvidiaBuildGenerator(http_client=http_client, model_id=NVIDIA_NEMOTRON_SUPER_MODEL)
    app = create_app(settings=cfg, generator=gen)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "alienese-default",
                "messages": [{"role": "user", "content": "Trigger auth failure."}],
            },
        )

    assert resp.status_code == 503
    raw_err_text = resp.text
    err_body = resp.json()
    assert err_body["error"]["code"] == "provider_auth_failed"
    assert "[fake:" not in raw_err_text
    assert secret_key not in raw_err_text
    await gen.aclose()


async def test_chunked_oversized_request_without_content_length_and_misleading_content_length() -> (
    None
):
    cfg = Settings(max_request_body_bytes=1024)
    app = create_app(settings=cfg)

    yielded_chunks = 0

    async def oversized_chunk_stream() -> AsyncIterator[bytes]:
        nonlocal yielded_chunks
        for _ in range(10):
            yielded_chunks += 1
            yield b"x" * 400

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. Chunked request without Content-Length stops reading as soon as 1024 bytes is exceeded
        resp_chunked = await client.post(
            "/v1/chat/completions",
            content=oversized_chunk_stream(),
            headers={"content-type": "application/json"},
        )
        assert resp_chunked.status_code == 413
        assert resp_chunked.json()["error"]["code"] == "request_body_too_large"
        # 3 chunks * 400 bytes = 1200 > 1024: stops at chunk 3 without buffering all 10 chunks
        assert yielded_chunks == 3

        # 2. Misleadingly small Content-Length header with oversized streamed body
        misleading_chunks = 0

        async def misleading_body_stream() -> AsyncIterator[bytes]:
            nonlocal misleading_chunks
            for _ in range(5):
                misleading_chunks += 1
                yield b"y" * 400

        req_misleading = client.build_request(
            "POST",
            "/v1/chat/completions",
            content=misleading_body_stream(),
            headers={"content-type": "application/json"},
        )
        req_misleading.headers["content-length"] = "10"
        resp_misleading = await client.send(req_misleading)
        assert resp_misleading.status_code == 413
        assert resp_misleading.json()["error"]["code"] == "request_body_too_large"
        assert misleading_chunks == 3

        # 3. Malformed Content-Length header rejected with HTTP 400
        req_invalid_cl = client.build_request(
            "POST",
            "/v1/chat/completions",
            content=b'{"model":"alienese-default","messages":[{"role":"user","content":"hi"}]}',
            headers={"content-type": "application/json"},
        )
        req_invalid_cl.headers["content-length"] = "not-an-integer"
        resp_invalid_cl = await client.send(req_invalid_cl)
        assert resp_invalid_cl.status_code == 400
        assert resp_invalid_cl.json()["error"]["code"] == "invalid_content_length"


async def test_duplicate_idempotent_waiter_timeout_does_not_cancel_slow_leader() -> None:
    leader_in_flight = asyncio.Event()
    release_leader = asyncio.Event()
    upstream_calls = 0

    async def slow_nvidia_mock(_request: httpx.Request) -> httpx.Response:
        nonlocal upstream_calls
        upstream_calls += 1
        leader_in_flight.set()
        await release_leader.wait()
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=json.dumps(
                {
                    "id": "chatcmpl-slow-leader",
                    "object": "chat.completion",
                    "model": NVIDIA_NEMOTRON_SUPER_MODEL,
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": "Slow leader completed normally.",
                            },
                        }
                    ],
                }
            ).encode("utf-8"),
        )

    cfg = Settings(
        alienese_provider_mode="hybrid",
        generator_provider="nvidia_build",
        generator_api_key=SecretStr("nvapi-" + ("a" * 16)),
        generator_base_url=NVIDIA_DEFAULT_BASE_URL,
        generator_model=NVIDIA_NEMOTRON_SUPER_MODEL,
        request_deadline_seconds=5.0,
        provider_timeout_seconds=5.0,
        idempotency_wait_timeout_seconds=0.05,
    )
    http_client = ProviderHttpClient(
        provider_name="nvidia_build",
        base_url=NVIDIA_DEFAULT_BASE_URL,
        api_key="nvapi-" + ("a" * 16),
        retry_config=RetryConfig(max_attempts=1),
        transport=httpx.MockTransport(slow_nvidia_mock),
    )
    gen = NvidiaBuildGenerator(http_client=http_client, model_id=NVIDIA_NEMOTRON_SUPER_MODEL)
    app = create_app(settings=cfg, generator=gen)

    req_payload = {
        "model": "alienese-default",
        "messages": [{"role": "user", "content": "Run slow idempotent turn."}],
    }
    headers = {"Idempotency-Key": "idem-slow-leader-key"}

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        leader_task = asyncio.create_task(
            client.post("/v1/chat/completions", headers=headers, json=req_payload)
        )
        await leader_in_flight.wait()

        # Duplicate request arrives while leader holds the per-key lock and exhausts wait budget
        dup_resp = await client.post("/v1/chat/completions", headers=headers, json=req_payload)
        assert dup_resp.status_code == 504
        assert dup_resp.json()["error"]["code"] == "idempotency_wait_timeout"
        assert not leader_task.done()

        # Allow the original slow leader operation to finish -> completes and caches result
        release_leader.set()
        leader_resp = await leader_task
        assert leader_resp.status_code == 200
        assert leader_resp.headers["X-Idempotent-Replay"] == "false"
        assert (
            leader_resp.json()["choices"][0]["message"]["content"]
            == "Slow leader completed normally."
        )
        assert upstream_calls == 1

        # Subsequent duplicate request replays cached leader result without another upstream call
        replay_resp = await client.post("/v1/chat/completions", headers=headers, json=req_payload)
        assert replay_resp.status_code == 200
        assert replay_resp.headers["X-Idempotent-Replay"] == "true"
        assert replay_resp.json() == leader_resp.json()
        assert upstream_calls == 1

    await gen.aclose()


async def test_nebius_hybrid_gateway_turn_uses_nebius_key_and_never_transmits_nvidia_key() -> None:
    nvidia_secret = "nvapi-" + ("a" * 16)
    nebius_secret = "nebius-" + ("b" * 16)
    seen_auth_headers: list[str | None] = []
    seen_urls: list[str] = []

    async def nebius_mock(request: httpx.Request) -> httpx.Response:
        seen_auth_headers.append(request.headers.get("authorization"))
        seen_urls.append(str(request.url))
        return httpx.Response(
            200,
            headers={
                "content-type": "application/json",
                "x-request-id": "nebius-req-mock-01",
            },
            content=json.dumps(
                {
                    "id": "chatcmpl-neb-01",
                    "object": "chat.completion",
                    "model": NVIDIA_NEMOTRON_SUPER_MODEL,
                    "system_fingerprint": "vllm-0.1.dev1+g5001743e3-dp2-9bbaa064",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": "Nebius Token Factory synthesized this response.",
                                "reasoning": None,
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 22,
                        "completion_tokens": 8,
                        "total_tokens": 30,
                    },
                }
            ).encode("utf-8"),
        )

    cfg = Settings(
        _env_file=None,
        alienese_provider_mode="hybrid",
        generator_provider="nebius_token_factory",
        generator_api_key=SecretStr(nvidia_secret),
        generator_base_url=NVIDIA_DEFAULT_BASE_URL,
        generator_model=NVIDIA_NEMOTRON_SUPER_MODEL,
        nebius_token_factory_key=SecretStr(nebius_secret),
        nebius_token_factory_base_url=NEBIUS_DEFAULT_BASE_URL,
        nebius_token_factory_model=NVIDIA_NEMOTRON_SUPER_MODEL,
    )
    assert cfg.effective_generator_api_key == SecretStr(nebius_secret)
    assert cfg.effective_generator_base_url == NEBIUS_DEFAULT_BASE_URL
    assert cfg.effective_generator_model == NVIDIA_NEMOTRON_SUPER_MODEL
    safe_cfg_str = json.dumps(cfg.safe_dump())
    assert nvidia_secret not in safe_cfg_str
    assert nebius_secret not in safe_cfg_str

    assert cfg.effective_generator_api_key is not None
    http_client = ProviderHttpClient(
        provider_name="nebius_token_factory",
        base_url=cfg.effective_generator_base_url,
        api_key=cfg.effective_generator_api_key,
        retry_config=RetryConfig(max_attempts=1),
        transport=httpx.MockTransport(nebius_mock),
    )
    gen = NebiusTokenFactoryGenerator(
        http_client=http_client,
        model_id=cfg.effective_generator_model,
    )
    app = create_app(settings=cfg, generator=gen)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "alienese-default",
                "messages": [{"role": "user", "content": "Verify Nebius credential isolation."}],
            },
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "alienese-default"
    assert (
        body["choices"][0]["message"]["content"]
        == "Nebius Token Factory synthesized this response."
    )
    assert seen_urls == [f"{NEBIUS_DEFAULT_BASE_URL}/chat/completions"]
    assert seen_auth_headers == [f"Bearer {nebius_secret}"]
    assert nvidia_secret not in str(seen_auth_headers)
    await gen.aclose()
