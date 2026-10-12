"""Live integration tests for Alienese Three-Model architecture.

Verifies end-to-end execution with:
1. EmbeddingGemma 2 (google/embeddinggemma-2) @ http://127.0.0.1:8001/embeddings
2. mini-Jev (samatv256/mini-Jev step-010626) @ http://127.0.0.1:8001/predict
3. Alienese Gateway & State Engine

Tests run against the live local model server on port 8001.
"""

from __future__ import annotations

import time

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

    # 1. Verify mini-Jev does NOT bypass when multiple candidates are provided
    from alienese.contracts.candidates import CandidateAction, CandidateDisposition
    from alienese.contracts.context import RequestContext
    from alienese.contracts.state import CanonicalCapability, WorkingState
    from alienese.providers.controller.mini_jev import MiniJevController
    from alienese.providers.runtime.client import ProviderHttpClient

    http_client = ProviderHttpClient(
        provider_name="mini_jev",
        base_url="http://127.0.0.1:8001",
        api_key="test-key",
        default_timeout_seconds=5.0,
        allow_localhost_http=True,
    )
    controller = MiniJevController(http_client=http_client)
    try:
        cand_a = CandidateAction(
            candidate_id="cand_read",
            canonical_intent=CanonicalCapability.READ_FILE,
            disposition=CandidateDisposition.EXTERNAL_TOOL,
            external_tool_name="read_file",
            arguments={"path": "auth/session_manager.py"},
            arguments_complete=True,
            rationale="Inspect session manager",
        )
        cand_b = CandidateAction(
            candidate_id="cand_test",
            canonical_intent=CanonicalCapability.RUN_TEST,
            disposition=CandidateDisposition.EXTERNAL_TOOL,
            external_tool_name="run_test",
            arguments={"test_path": "tests/test_session.py"},
            arguments_complete=True,
            rationale="Run tests",
        )
        ws = WorkingState(
            state_digest="0" * 64,
            event_count=1,
            user_messages=("Analyze token refresh failure",),
        )
        ctx = RequestContext(
            request_id="req-test-multi",
            deadline_monotonic=time.monotonic() + 5.0,
        )
        decision = await controller.decide(ctx, ws, [cand_a, cand_b])
        # Invariant 4 & Rule 8: mini-Jev must NOT be bypassed when multiple candidates exist
        assert decision.semantics.guard_metadata.bypassed_controller is False
        assert len(decision.semantics.scores) == 2
        assert all(s.score > 0.0 for s in decision.semantics.scores)
    finally:
        await controller.aclose()

    # 2. Verify end-to-end Gateway turn with Nebius Token Factory Nemotron 3 Super
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
        # Multi-candidate Gateway turn with tools
        payload_tools = {
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
        resp_tools = await client.post("/v1/chat/completions", json=payload_tools, timeout=30.0)
        assert resp_tools.status_code == 200, resp_tools.text
        data_tools = resp_tools.json()
        assert len(data_tools["choices"]) == 1
        assert data_tools["choices"][0]["finish_reason"] in ("tool_calls", "stop")

        # Text generation turn verified against live Nebius Token Factory
        payload_gen = {
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
        resp_gen = await client.post("/v1/chat/completions", json=payload_gen, timeout=30.0)
        assert resp_gen.status_code == 200, resp_gen.text
        data_gen = resp_gen.json()
        assert len(data_gen["choices"]) == 1
        choice_gen = data_gen["choices"][0]
        assert choice_gen["finish_reason"] == "stop"
        content = choice_gen["message"]["content"]
        assert len(content) > 10
