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
from alienese.storage.traces import InMemoryTraceStore

# Explicit Live Test Budget Governance (Addendum Section 10)
MAX_LIVE_REQUESTS = 2
MAX_OUTPUT_TOKENS_PER_REQUEST = 64
MAX_TOTAL_OUTPUT_TOKENS = 128
MAX_ATTEMPTS_PER_REQUEST = 1
MAX_TEST_DURATION_SECONDS = 30.0


@pytest.mark.live_nvidia
@pytest.mark.asyncio
async def test_live_nvidia_hybrid_gateway_turn() -> None:
    """Opt-in live NVIDIA verification of the full Alienese hybrid gateway turn."""
    if os.getenv("RUN_LIVE_NVIDIA_TESTS") != "1":
        pytest.skip("Set RUN_LIVE_NVIDIA_TESTS=1 to run live NVIDIA API Catalog tests.")

    # Load from local .env while overriding mode and enforcing strict budget caps
    base_settings = Settings()
    raw_key = (
        base_settings.generator_api_key.get_secret_value().strip()
        if base_settings.generator_api_key is not None
        else ""
    )
    if not raw_key:
        pytest.fail("GENERATOR_API_KEY is missing in environment/.env for live NVIDIA test.")

    settings = Settings(
        _env_file=None,
        alienese_provider_mode="hybrid",
        generator_provider="nvidia_build",
        generator_api_key=SecretStr(raw_key),
        generator_base_url="https://integrate.api.nvidia.com/v1",
        generator_model="nvidia/nemotron-3-super-120b-a12b",
        generator_max_tokens=MAX_OUTPUT_TOKENS_PER_REQUEST,
        generator_enable_thinking=False,
        provider_max_attempts=MAX_ATTEMPTS_PER_REQUEST,
        provider_timeout_seconds=MAX_TEST_DURATION_SECONDS,
        alienese_trace_content=False,
    )

    trace_store = InMemoryTraceStore(allow_full_fidelity=False)
    app = create_app(settings=settings, trace_store=trace_store)

    live_requests_executed = 0
    total_output_tokens = 0
    t_start = time.monotonic()

    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://testserver") as client,
    ):
        live_requests_executed += 1
        assert live_requests_executed <= MAX_LIVE_REQUESTS

        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "alienese-default",
                "temperature": 1.0,
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
            },
        )

    elapsed = time.monotonic() - t_start
    assert elapsed <= MAX_TEST_DURATION_SECONDS, (
        f"Live test duration {elapsed:.2f}s exceeded cap {MAX_TEST_DURATION_SECONDS}s"
    )

    assert resp.status_code == 200, f"Unexpected HTTP status {resp.status_code}: {resp.json()}"
    body = resp.json()
    assert body["model"] == "alienese-default"
    assert len(body["choices"]) == 1
    choice = body["choices"][0]
    assert choice["finish_reason"] == "stop"
    content = choice["message"]["content"]
    assert isinstance(content, str) and len(content.strip()) > 0

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
    gen_tel = artifact.telemetry.generation_telemetry
    assert gen_tel is not None
    assert gen_tel.upstream_request_id is not None and len(gen_tel.upstream_request_id) > 0
    assert gen_tel.attempt_count == 1
    assert gen_tel.failed_attempt_count == 0
    assert gen_tel.prompt_tokens == usage["prompt_tokens"]
    assert gen_tel.completion_tokens == usage["completion_tokens"]
    assert gen_tel.total_tokens == usage["total_tokens"]
    assert gen_tel.estimated_cost_usd is None
    assert not contains_sensitive_material(artifact.model_dump(mode="json"))
    assert raw_key not in artifact.model_dump_json()
