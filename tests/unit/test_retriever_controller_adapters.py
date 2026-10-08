"""Unit tests for EmbeddingGemmaRetriever and MiniJevController adapter contracts.

Covers:
1. EmbeddingGemmaRetriever.rank() against repository RetrievalRequest/RetrievalResult contracts:
   - Documented query/document task prefixes
   - Matryoshka truncation + L2 renormalization
   - Mandatory item preservation and deterministic tie-breakers
   - Duplicate/unknown/missing item IDs, dimension mismatch, NaN/Inf, and zero-norm vectors
2. MiniJevController.decide() against pinned step-010626 finite-choice contract:
   - Single-candidate deterministic bypass (0 HTTP calls)
   - Disposition preservation (internal_transition vs external_tool vs generation_job)
   - Duplicate/unknown/missing candidate IDs, unnormalized/non-finite probabilities,
     inconsistent selected_id, and model/revision mismatch
"""

from __future__ import annotations

import json
import math
from typing import Any

import httpx
import pytest

from alienese.api.errors import CompatibilityError, InvalidProviderResponse
from alienese.contracts.candidates import (
    CandidateAction,
    CandidateDisposition,
    CostClass,
    RiskClass,
)
from alienese.contracts.context import RequestContext
from alienese.contracts.events import (
    EventKind,
    EventProvenance,
    NormalizedEvent,
    SourceRole,
    TrustLevel,
)
from alienese.contracts.generation import GenerationJobType
from alienese.contracts.providers import RetrievalCandidateItem, RetrievalRequest
from alienese.contracts.state import CanonicalCapability, WorkingState
from alienese.engine.reconstruct import reconstruct
from alienese.providers.controller.mini_jev import (
    MINI_JEV_DEFAULT_MODEL,
    MINI_JEV_PINNED_REVISION,
    MiniJevController,
    serialize_mini_jev_option,
)
from alienese.providers.retriever.embeddinggemma import (
    EMBEDDINGGEMMA_DEFAULT_MODEL,
    EmbeddingGemmaRetriever,
    format_embeddinggemma_document,
    format_embeddinggemma_query,
    truncate_and_l2_normalize,
)
from alienese.providers.runtime.client import ProviderHttpClient
from alienese.providers.runtime.retry import RetryConfig


def _build_working_state() -> WorkingState:
    events = (
        NormalizedEvent(
            sequence_no=0,
            event_id="ev_0",
            kind=EventKind.SYSTEM_MESSAGE,
            trust=TrustLevel.SYSTEM_TRUSTED,
            content="Follow repository safety policy.",
            provenance=EventProvenance(message_index=0, source_role=SourceRole.SYSTEM),
        ),
        NormalizedEvent(
            sequence_no=1,
            event_id="ev_1",
            kind=EventKind.USER_MESSAGE,
            trust=TrustLevel.USER,
            content="Check the test results.",
            provenance=EventProvenance(message_index=1, source_role=SourceRole.USER),
        ),
    )
    return reconstruct(events, available_tools=())


async def test_embeddinggemma_prefixes_matryoshka_renormalization_and_mandatory_preservation() -> (
    None
):
    assert (
        format_embeddinggemma_query("find bug", task_type="code_retrieval")
        == "task: code retrieval | query: find bug"
    )
    assert (
        format_embeddinggemma_document("def add(a, b): return a + b")
        == "title: none | text: def add(a, b): return a + b"
    )

    # Verify Matryoshka truncation renormalizes to unit L2 length
    raw_768 = [3.0, 4.0] + [0.0] * 766
    truncated_128 = truncate_and_l2_normalize(
        raw_768, target_dim=128, provider_name="embeddinggemma"
    )
    assert len(truncated_128) == 128
    assert math.sqrt(sum(x * x for x in truncated_128)) == pytest.approx(1.0)
    assert truncated_128[0] == pytest.approx(0.6)
    assert truncated_128[1] == pytest.approx(0.8)

    # Unsupported dimension rejected
    dummy_client = ProviderHttpClient(
        provider_name="embeddinggemma",
        base_url="https://embeddings.example.com/v1",
        api_key="emb-" + ("c" * 12),
        transport=httpx.MockTransport(lambda req: httpx.Response(200, content=b"{}")),
    )
    with pytest.raises(CompatibilityError) as dim_exc:
        EmbeddingGemmaRetriever(http_client=dummy_client, target_dimension=300)
    assert dim_exc.value.code == "unsupported_embedding_dimension"
    await dummy_client.aclose()

    captured_inputs: list[str] = []

    def _vec128(x: float, y: float) -> list[float]:
        return [x, y] + [0.0] * 126

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured_inputs.extend(body["input"])
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=json.dumps(
                {
                    "model": EMBEDDINGGEMMA_DEFAULT_MODEL,
                    "model_revision": "gemma2-emb-v1",
                    "data": [
                        {"index": 0, "embedding": _vec128(1.0, 0.0)},  # Query along +X
                        {
                            "index": 1,
                            "embedding": _vec128(0.0, 1.0),
                        },  # item_mandatory (sim=0.0, mandatory=True!)
                        {"index": 2, "embedding": _vec128(1.0, 0.0)},  # item_high (sim=1.0)
                        {"index": 3, "embedding": _vec128(0.6, 0.8)},  # item_mid (sim=0.6)
                    ],
                    "usage": {"prompt_tokens": 40, " total_tokens": 40},
                }
            ).encode("utf-8"),
        )

    client = ProviderHttpClient(
        provider_name="embeddinggemma",
        base_url="https://embeddings.example.com/v1",
        api_key="emb-" + ("c" * 12),
        retry_config=RetryConfig(max_attempts=1),
        transport=httpx.MockTransport(handler),
    )
    retriever = EmbeddingGemmaRetriever(
        http_client=client,
        target_dimension=128,
        task_type="search",
    )

    req = RetrievalRequest(
        query="how to run tests",
        items=(
            RetrievalCandidateItem(
                item_id="item_mandatory", content="Mandatory rule", mandatory=True
            ),
            RetrievalCandidateItem(item_id="item_high", content="pytest command", mandatory=False),
            RetrievalCandidateItem(item_id="item_mid", content="other doc", mandatory=False),
        ),
        max_results=2,
    )
    result = await retriever.rank(RequestContext(), req)

    # Mandatory item is preserved first even though its cosine similarity was 0.0!
    assert [r.item_id for r in result.ranked_items] == ["item_mandatory", "item_high"]
    assert result.ranked_items[0].score == pytest.approx(0.0)
    assert result.ranked_items[1].score == pytest.approx(1.0)
    assert captured_inputs[0] == "task: search result | query: how to run tests"
    assert captured_inputs[1] == "title: none | text: Mandatory rule"
    await retriever.aclose()


async def test_embeddinggemma_rejects_duplicate_ids_dimension_mismatch_and_zero_vectors() -> None:
    response_data: dict[str, Any] = {}

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=json.dumps(response_data).encode("utf-8"),
        )

    client = ProviderHttpClient(
        provider_name="embeddinggemma",
        base_url="https://embeddings.example.com/v1",
        api_key="emb-" + ("c" * 12),
        retry_config=RetryConfig(max_attempts=1),
        transport=httpx.MockTransport(handler),
    )
    retriever = EmbeddingGemmaRetriever(http_client=client, target_dimension=128)

    # 1. Duplicate input item_id rejected before network call
    dup_req = RetrievalRequest(
        query="q",
        items=(
            RetrievalCandidateItem(item_id="dup_1", content="a"),
            RetrievalCandidateItem(item_id="dup_1", content="b"),
        ),
    )
    with pytest.raises(CompatibilityError) as dup_exc:
        await retriever.rank(RequestContext(), dup_req)
    assert dup_exc.value.code == "duplicate_retrieval_item_id"

    valid_req = RetrievalRequest(
        query="q",
        items=(RetrievalCandidateItem(item_id="item_1", content="a"),),
    )

    # 2. Zero-norm vector rejected
    response_data = {
        "model": EMBEDDINGGEMMA_DEFAULT_MODEL,
        "data": [
            {"index": 0, "embedding": [0.0] * 128},
            {"index": 1, "embedding": [1.0] + [0.0] * 127},
        ],
    }
    with pytest.raises(InvalidProviderResponse) as zero_exc:
        await retriever.rank(RequestContext(), valid_req)
    assert zero_exc.value.code == "zero_norm_embedding_vector"

    # 3. Dimension mismatch across vectors rejected
    response_data = {
        "model": EMBEDDINGGEMMA_DEFAULT_MODEL,
        "data": [
            {"index": 0, "embedding": [1.0] + [0.0] * 127},
            {"index": 1, "embedding": [1.0] + [0.0] * 255},
        ],
    }
    with pytest.raises(InvalidProviderResponse) as dim_exc:
        await retriever.rank(RequestContext(), valid_req)
    assert dim_exc.value.code == "embedding_dimension_mismatch"

    # 4. Duplicate index in response rejected
    response_data = {
        "model": EMBEDDINGGEMMA_DEFAULT_MODEL,
        "data": [
            {"index": 0, "embedding": [1.0] + [0.0] * 127},
            {"index": 0, "embedding": [1.0] + [0.0] * 127},
        ],
    }
    with pytest.raises(InvalidProviderResponse) as dup_idx_exc:
        await retriever.rank(RequestContext(), valid_req)
    assert dup_idx_exc.value.code == "duplicate_embedding_entry"

    await retriever.aclose()


async def test_mini_jev_single_candidate_bypass_and_disposition_preservation() -> None:
    http_calls = 0
    captured_payloads: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal http_calls
        http_calls += 1
        captured_payloads.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=json.dumps(
                {
                    "model": MINI_JEV_DEFAULT_MODEL,
                    "model_revision": MINI_JEV_PINNED_REVISION,
                    "selected_id": "cand_answer",
                    "selected_label": "generation_job:respond",
                    "options": [
                        {
                            "id": "cand_internal",
                            "label": "internal_transition:expand_search",
                            "probability": 0.2,
                        },
                        {
                            "id": "cand_answer",
                            "label": "generation_job:respond",
                            "probability": 0.8,
                        },
                    ],
                }
            ).encode("utf-8"),
        )

    client = ProviderHttpClient(
        provider_name="mini_jev",
        base_url="https://controller.example.com/v1",
        api_key="ctrl-" + ("d" * 12),
        retry_config=RetryConfig(max_attempts=1),
        transport=httpx.MockTransport(handler),
    )
    controller = MiniJevController(http_client=client)
    state = _build_working_state()

    cand_answer = CandidateAction(
        candidate_id="cand_answer",
        canonical_intent=CanonicalCapability.RESPOND,
        disposition=CandidateDisposition.GENERATION_JOB,
        requires_generation=True,
        generation_job_type=GenerationJobType.ANSWER,
        risk_class=RiskClass.LOW,
        cost_class=CostClass.LOW,
        rationale="Respond to user.",
    )
    cand_internal = CandidateAction(
        candidate_id="cand_internal",
        canonical_intent=CanonicalCapability.EXPAND_SEARCH,
        disposition=CandidateDisposition.INTERNAL_TRANSITION,
        requires_generation=False,
        risk_class=RiskClass.LOW,
        cost_class=CostClass.LOW,
        rationale="Internal search expansion.",
    )

    # 1. Single eligible candidate takes deterministic bypass with 0 HTTP calls!
    single_res = await controller.decide(RequestContext(), state, (cand_answer,))
    assert http_calls == 0
    assert single_res.selected_candidate_id == "cand_answer"
    assert single_res.guard_metadata.bypassed_controller is True
    assert single_res.guard_metadata.fallback_reason == "single_eligible_candidate"

    # 2. Multi-candidate call preserves disposition types and verifies response
    multi_res = await controller.decide(RequestContext(), state, (cand_internal, cand_answer))
    assert http_calls == 1
    assert multi_res.selected_candidate_id == "cand_answer"
    assert multi_res.model_revision == MINI_JEV_PINNED_REVISION

    sent_options = captured_payloads[0]["answer_options"]
    assert sent_options[0]["type"] == "internal_transition"
    assert sent_options[1]["type"] == "generation_job"
    assert serialize_mini_jev_option(cand_internal)["type"] == "internal_transition"

    await controller.aclose()


@pytest.mark.parametrize(
    ("response_payload", "expected_code"),
    [
        # Unnormalized probabilities (sum = 0.5 != 1.0)
        (
            {
                "selected_id": "cand_b",
                "options": [
                    {"id": "cand_a", "probability": 0.2},
                    {"id": "cand_b", "probability": 0.3},
                ],
            },
            "unnormalized_candidate_probabilities",
        ),
        # Duplicate candidate IDs in response
        (
            {
                "selected_id": "cand_a",
                "options": [
                    {"id": "cand_a", "probability": 0.5},
                    {"id": "cand_a", "probability": 0.5},
                ],
            },
            "duplicate_candidate_id",
        ),
        # Unknown candidate ID in response
        (
            {
                "selected_id": "cand_a",
                "options": [
                    {"id": "cand_a", "probability": 0.6},
                    {"id": "cand_unknown", "probability": 0.4},
                ],
            },
            "unknown_candidate_id",
        ),
        # Selected candidate inconsistent with argmax probability
        (
            {
                "selected_id": "cand_a",
                "options": [
                    {"id": "cand_a", "probability": 0.1},
                    {"id": "cand_b", "probability": 0.9},
                ],
            },
            "inconsistent_selected_candidate",
        ),
        # Revision mismatch
        (
            {
                "model_revision": "step-999999-wrong",
                "selected_id": "cand_b",
                "options": [
                    {"id": "cand_a", "probability": 0.1},
                    {"id": "cand_b", "probability": 0.9},
                ],
            },
            "provider_revision_mismatch",
        ),
    ],
)
async def test_mini_jev_rejects_malformed_distributions_and_mismatches(
    response_payload: dict[str, Any],
    expected_code: str,
) -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=json.dumps(response_payload).encode("utf-8"),
        )

    client = ProviderHttpClient(
        provider_name="mini_jev",
        base_url="https://controller.example.com/v1",
        api_key="ctrl-" + ("d" * 12),
        retry_config=RetryConfig(max_attempts=1),
        transport=httpx.MockTransport(handler),
    )
    controller = MiniJevController(http_client=client)
    state = _build_working_state()

    cands = (
        CandidateAction(
            candidate_id="cand_a",
            canonical_intent=CanonicalCapability.RESPOND,
            disposition=CandidateDisposition.GENERATION_JOB,
            requires_generation=True,
            generation_job_type=GenerationJobType.ANSWER,
            risk_class=RiskClass.LOW,
            cost_class=CostClass.LOW,
            rationale="A",
        ),
        CandidateAction(
            candidate_id="cand_b",
            canonical_intent=CanonicalCapability.RESPOND,
            disposition=CandidateDisposition.GENERATION_JOB,
            requires_generation=True,
            generation_job_type=GenerationJobType.ANSWER,
            risk_class=RiskClass.LOW,
            cost_class=CostClass.LOW,
            rationale="B",
        ),
    )

    with pytest.raises(InvalidProviderResponse) as exc_info:
        await controller.decide(RequestContext(), state, cands)
    assert exc_info.value.code == expected_code

    # Duplicate input candidate IDs also rejected before network call
    with pytest.raises(CompatibilityError) as dup_exc:
        await controller.decide(RequestContext(), state, (cands[0], cands[0]))
    assert dup_exc.value.code == "duplicate_candidate_id"

    await controller.aclose()
