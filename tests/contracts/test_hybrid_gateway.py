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

import json
from typing import Any

import httpx
import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from alienese.api.app import create_app
from alienese.api.errors import CompatibilityError
from alienese.config import Settings
from alienese.providers.fake import FakeGenerator
from alienese.providers.generator.nvidia_build import (
    NVIDIA_DEFAULT_BASE_URL,
    NVIDIA_NEMOTRON_SUPER_MODEL,
    NvidiaBuildGenerator,
)
from alienese.providers.runtime.client import ProviderHttpClient
from alienese.providers.runtime.retry import RetryConfig


async def test_fake_mode_with_populated_env_makes_zero_outbound_calls() -> None:
    cfg = Settings(
        alienese_provider_mode="fake",
        generator_provider="nvidia_build",
        generator_api_key=SecretStr("nvapi-" + ("a" * 16)),
        generator_base_url=NVIDIA_DEFAULT_BASE_URL,
        generator_model=NVIDIA_NEMOTRON_SUPER_MODEL,
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
        # 3. hybrid mode with missing generator_api_key
        (
            {
                "alienese_provider_mode": "hybrid",
                "generator_provider": "nvidia_build",
                "generator_api_key": None,
            },
            "missing_provider_api_key",
        ),
        # 4. nvidia_build silently inheriting or configured with Nebius URL
        (
            {
                "alienese_provider_mode": "hybrid",
                "generator_provider": "nvidia_build",
                "generator_api_key": SecretStr("nvapi-" + ("a" * 16)),
                "generator_base_url": "https://api.tokenfactory.nebius.com/v1",
                "generator_model": NVIDIA_NEMOTRON_SUPER_MODEL,
            },
            "provider_origin_mismatch",
        ),
        # 5. nebius_token_factory reusing an NVIDIA nvapi-* key
        (
            {
                "alienese_provider_mode": "hybrid",
                "generator_provider": "nebius_token_factory",
                "generator_api_key": SecretStr("nvapi-" + ("a" * 16)),
                "generator_base_url": "https://api.tokenfactory.nebius.com/v1",
                "generator_model": NVIDIA_NEMOTRON_SUPER_MODEL,
            },
            "cross_provider_credential_reuse",
        ),
        # 6. nebius_token_factory pointing to NVIDIA URL
        (
            {
                "alienese_provider_mode": "hybrid",
                "generator_provider": "nebius_token_factory",
                "generator_api_key": SecretStr("nebius-testSecretKey1234567890"),
                "generator_base_url": NVIDIA_DEFAULT_BASE_URL,
                "generator_model": NVIDIA_NEMOTRON_SUPER_MODEL,
            },
            "provider_origin_mismatch",
        ),
        # 7. non-fake retriever_provider in Phase 2
        (
            {
                "alienese_provider_mode": "hybrid",
                "retriever_provider": "embeddinggemma",
                "generator_provider": "nvidia_build",
                "generator_api_key": SecretStr("nvapi-" + ("a" * 16)),
            },
            "invalid_provider_mode_combination",
        ),
        # 8. non-fake controller_provider in Phase 2
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
        Settings(**kwargs)
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
