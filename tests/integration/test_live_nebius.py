from __future__ import annotations

import os
import time

import httpx
import pytest
from pydantic import SecretStr

from alienese.api.app import create_app
from alienese.config import Settings
from alienese.contracts.traces import TraceMode
from alienese.observability.redaction import contains_sensitive_material
from alienese.providers.generator.nebius_token_factory import (
    NEBIUS_DEFAULT_BASE_URL,
    NebiusTokenFactoryGenerator,
)
from alienese.providers.generator.nvidia_build import NVIDIA_NEMOTRON_SUPER_MODEL
from alienese.storage.traces import InMemoryTraceStore

# Explicit Live Test Budget Governance
MAX_LIVE_REQUESTS = 1
MAX_OUTPUT_TOKENS_PER_REQUEST = 64
MAX_TOTAL_OUTPUT_TOKENS = 64
MAX_ATTEMPTS_PER_REQUEST = 1
MAX_TEST_DURATION_SECONDS = 30.0


@pytest.mark.live_nebius
@pytest.mark.asyncio
async def test_live_nebius_hybrid_gateway_turn() -> None:
    """Opt-in live Nebius Token Factory verification of the full Alienese hybrid gateway turn."""
    if os.getenv("RUN_LIVE_NEBIUS_TESTS") != "1":
        pytest.skip("Set RUN_LIVE_NEBIUS_TESTS=1 to run live Nebius Token Factory tests.")

    base_settings = Settings()
    raw_nebius_key = (
        base_settings.nebius_token_factory_key.get_secret_value().strip()
        if base_settings.nebius_token_factory_key is not None
        else ""
    )
    if not raw_nebius_key:
        pytest.fail("NEBIUS_TOKEN_FACTORY_KEY is missing in environment/.env for live Nebius test.")

    raw_nvidia_key = (
        base_settings.generator_api_key.get_secret_value().strip()
        if base_settings.generator_api_key is not None
        else ""
    )
    resolved_base_url = base_settings.nebius_token_factory_base_url or NEBIUS_DEFAULT_BASE_URL
    resolved_model = base_settings.nebius_token_factory_model or NVIDIA_NEMOTRON_SUPER_MODEL

    settings = Settings(
        _env_file=None,
        alienese_provider_mode="hybrid",
        generator_provider="nebius_token_factory",
        generator_api_key=None,
        nebius_token_factory_key=SecretStr(raw_nebius_key),
        nebius_token_factory_base_url=resolved_base_url,
        nebius_token_factory_model=resolved_model,
        generator_max_tokens=MAX_OUTPUT_TOKENS_PER_REQUEST,
        generator_enable_thinking=False,
        provider_max_attempts=MAX_ATTEMPTS_PER_REQUEST,
        provider_timeout_seconds=MAX_TEST_DURATION_SECONDS,
        request_deadline_seconds=MAX_TEST_DURATION_SECONDS,
        alienese_trace_content=False,
    )

    trace_store = InMemoryTraceStore(allow_full_fidelity=False)
    app = create_app(settings=settings, trace_store=trace_store)
    assert isinstance(app.state.generator, NebiusTokenFactoryGenerator)

    live_requests_executed = 0
    total_output_tokens = 0
    t_start = time.monotonic()

    req_payload = {
        "model": "alienese-default",
        "temperature": 0.7,
        "max_tokens": MAX_OUTPUT_TOKENS_PER_REQUEST,
        "messages": [
            {
                "role": "system",
                "content": "Reply with one short factual sentence.",
            },
            {
                "role": "user",
                "content": (
                    "State in one short sentence what deterministic "
                    "state reconstruction guarantees."
                ),
            },
        ],
    }

    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://testserver") as client,
    ):
        live_requests_executed += 1
        assert live_requests_executed <= MAX_LIVE_REQUESTS

        resp = await client.post(
            "/v1/chat/completions",
            headers={"Idempotency-Key": "idem-live-nebius-gateway-01"},
            json=req_payload,
        )

        # Verify idempotent replay serves cached response with zero additional remote requests
        replay_resp = await client.post(
            "/v1/chat/completions",
            headers={"Idempotency-Key": "idem-live-nebius-gateway-01"},
            json=req_payload,
        )

    elapsed = time.monotonic() - t_start
    assert elapsed <= MAX_TEST_DURATION_SECONDS, (
        f"Live test duration {elapsed:.2f}s exceeded cap {MAX_TEST_DURATION_SECONDS}s"
    )

    assert resp.status_code == 200, f"Unexpected HTTP status {resp.status_code}: {resp.json()}"
    assert resp.headers["X-Idempotent-Replay"] == "false"
    assert replay_resp.status_code == 200
    assert replay_resp.headers["X-Idempotent-Replay"] == "true"
    assert replay_resp.json() == resp.json()

    body = resp.json()
    assert body["model"] == "alienese-default"
    assert len(body["choices"]) == 1
    choice = body["choices"][0]
    assert choice["finish_reason"] == "stop"
    content = choice["message"]["content"]
    assert isinstance(content, str) and len(content.strip()) > 0
    assert not content.startswith("[fake:")

    usage = body.get("usage")
    assert usage is not None
    assert usage["prompt_tokens"] > 0
    assert 0 < usage["completion_tokens"] <= MAX_OUTPUT_TOKENS_PER_REQUEST
    total_output_tokens += usage["completion_tokens"]
    assert total_output_tokens <= MAX_TOTAL_OUTPUT_TOKENS
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    # Verify trace artifact metadata-only safety and provider telemetry
    artifacts = trace_store.list_all()
    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.trace_mode == TraceMode.METADATA_ONLY
    assert artifact.replayable is False
    assert artifact.semantics.decision_semantics.provider_name == "fake_controller"
    assert artifact.semantics.generation_semantics is None

    gen_tel = artifact.telemetry.generation_telemetry
    assert gen_tel is not None
    assert gen_tel.upstream_request_id is not None and len(gen_tel.upstream_request_id) > 0
    assert gen_tel.serving_fingerprint is not None and len(gen_tel.serving_fingerprint) > 0
    assert gen_tel.attempt_count == 1
    assert gen_tel.failed_attempt_count == 0
    assert gen_tel.prompt_tokens == usage["prompt_tokens"]
    assert gen_tel.completion_tokens == usage["completion_tokens"]
    assert gen_tel.total_tokens == usage["total_tokens"]
    assert gen_tel.estimated_cost_usd is None

    serialized_artifact = artifact.model_dump_json()
    assert not contains_sensitive_material(artifact.model_dump(mode="json"))
    assert raw_nebius_key not in serialized_artifact
    if raw_nvidia_key:
        assert raw_nvidia_key not in serialized_artifact
    assert "deterministic state reconstruction" not in serialized_artifact
