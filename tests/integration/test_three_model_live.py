"""Live integration tests for Alienese Three-Model architecture.

Verifies end-to-end execution with:
1. EmbeddingGemma 2 (google/embeddinggemma-2) @ http://127.0.0.1:8001/embeddings
2. mini-Jev (samatv256/mini-Jev step-010626) @ http://127.0.0.1:8001/predict
3. Alienese Gateway & State Engine

Tests run against the live local model server on port 8001.
"""

from __future__ import annotations

import httpx
import pytest
from pydantic import SecretStr

from alienese.api.app import create_app
from alienese.config import Settings
from alienese.contracts.context import RequestContext
from alienese.contracts.providers import RetrievalCandidateItem, RetrievalRequest
from alienese.providers.retriever.embeddinggemma import EmbeddingGemmaRetriever
from alienese.providers.runtime.client import ProviderHttpClient


@pytest.fixture
def local_model_server_available() -> bool:
    """Check if the local model server is running on port 8001."""
    try:
        resp = httpx.get("http://127.0.0.1:8001/health", timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False


@pytest.mark.asyncio
async def test_live_embeddinggemma_retriever_ranking(local_model_server_available: bool) -> None:
    if not local_model_server_available:
        pytest.skip("Local model server is not running on http://127.0.0.1:8001")

    client = ProviderHttpClient(
        provider_name="embeddinggemma",
        base_url="http://127.0.0.1:8001",
        api_key=SecretStr("local-test"),
        allow_localhost_http=True,
    )
    retriever = EmbeddingGemmaRetriever(
        http_client=client,
        model_id="google/embeddinggemma-2",
        target_dimension=768,
    )

    req = RetrievalRequest(
        query="Fix token expiration error in session manager",
        items=(
            RetrievalCandidateItem(
                item_id="doc_auth",
                content="auth/session_manager.py: manages token refresh and expiration",
                mandatory=False,
            ),
            RetrievalCandidateItem(
                item_id="doc_unrelated",
                content="docs/conf.py: Sphinx documentation builder configuration",
                mandatory=False,
            ),
            RetrievalCandidateItem(
                item_id="doc_rules",
                content="AGENTS.md: Repository engineering invariants",
                mandatory=True,
            ),
        ),
        max_results=3,
    )

    ctx = RequestContext()
    result = await retriever.rank(ctx, req)
    await retriever.aclose()

    assert len(result.ranked_items) == 3
    # Mandatory item must always be preserved first
    assert result.ranked_items[0].item_id == "doc_rules"
    # Semantic similarity: doc_auth should score significantly higher than doc_unrelated
    scores = {item.item_id: item.score for item in result.ranked_items}
    assert scores["doc_auth"] > scores["doc_unrelated"]
    assert result.semantics.provider_name == "embeddinggemma"
    assert result.semantics.model_id == "google/embeddinggemma-2"


@pytest.mark.asyncio
async def test_live_three_model_chat_completions(local_model_server_available: bool) -> None:
    if not local_model_server_available:
        pytest.skip("Local model server is not running on http://127.0.0.1:8001")

    settings = Settings(
        alienese_provider_mode="three_model",
        retriever_provider="embeddinggemma",
        retriever_base_url="http://127.0.0.1:8001",
        controller_provider="mini_jev",
        controller_base_url="http://127.0.0.1:8001",
        generator_provider="fake",
    )

    app = create_app(settings=settings)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Check health
        health_resp = await client.get("/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["provider_mode"] == "three_model"

        # Execute a turn with multiple available tool candidates
        payload = {
            "model": "alienese-default",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "A pytest test failed with AssertionError: TokenExpiredError. "
                        "Inspect auth/session_manager.py and run the test."
                    ),
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "run_test",
                                "arguments": '{"test_path": "tests/test_session.py"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": (
                        "FAILED tests/test_session.py::test_refresh_expired - "
                        "AssertionError: session token expired"
                    ),
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read file contents",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "run_test",
                        "description": "Execute pytest tests",
                        "parameters": {
                            "type": "object",
                            "properties": {"test_path": {"type": "string"}},
                            "required": ["test_path"],
                        },
                    },
                },
            ],
        }

        resp = await client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert "choices" in data
        assert len(data["choices"]) == 1
        choice = data["choices"][0]
        # Must produce a tool call or assistant response
        assert choice["finish_reason"] in ("tool_calls", "stop")
        if choice["finish_reason"] == "tool_calls":
            tool_call = choice["message"]["tool_calls"][0]
            assert tool_call["function"]["name"] in ("read_file", "run_test")


@pytest.mark.asyncio
async def test_live_three_model_full_generation_with_nebius(
    local_model_server_available: bool,
) -> None:
    if not local_model_server_available:
        pytest.skip("Local model server is not running on http://127.0.0.1:8001")

    base_settings = Settings()
    if not base_settings.nebius_token_factory_key:
        pytest.skip("NEBIUS_TOKEN_FACTORY_KEY is not configured")

    settings = Settings(
        alienese_provider_mode="three_model",
        retriever_provider="embeddinggemma",
        retriever_base_url="http://127.0.0.1:8001",
        controller_provider="mini_jev",
        controller_base_url="http://127.0.0.1:8001",
        generator_provider="nebius_token_factory",
    )

    app = create_app(settings=settings)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        payload = {
            "model": "alienese-default",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Answer in one concise sentence: Why did token refresh "
                        "fail when old token was revoked?"
                    ),
                }
            ],
            "max_tokens": 2048,
            "tools": [],
        }
        resp = await client.post("/v1/chat/completions", json=payload, timeout=30.0)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert len(data["choices"]) == 1
        choice = data["choices"][0]
        assert choice["finish_reason"] == "stop"
        content = choice["message"]["content"]
        assert len(content) > 10
