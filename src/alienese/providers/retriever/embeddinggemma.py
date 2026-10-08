"""EmbeddingGemma 2 (`google/embeddinggemma-2`) Retriever adapter boundary.

Implements `Retriever.rank(ctx, request) -> RetrievalResult` against Alienese's
existing retrieval contracts (`RetrievalCandidateItem`, `RankedItem`,
`RetrievalRequest`, `RetrievalSemantics`, `RetrievalResult`).

Enforces:
- Verified Matryoshka dimensions (`128`, `256`, `512`, `768`).
- Asymmetric task instruction prefixes (`task: search result | query: ...` and
  `title: none | text: ...`).
- Duplicate/missing/unknown item ID rejection.
- Finite vector validation, dimension consistency, zero-norm rejection, and
  post-truncation L2 renormalization.
- Mandatory item preservation and deterministic tie-breaking.

Hosting status in Phase 2: `BLOCKED_HOSTING` / `PENDING_CREDENTIALS` (verified
offline via `httpx.MockTransport`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from alienese.api.errors import CompatibilityError, InvalidProviderResponse
from alienese.contracts.context import RequestContext
from alienese.contracts.providers import (
    RankedItem,
    RetrievalCandidateItem,
    RetrievalRequest,
    RetrievalResult,
    RetrievalSemantics,
)
from alienese.providers.runtime.client import ProviderHttpClient

EMBEDDINGGEMMA_DEFAULT_MODEL = "google/embeddinggemma-2"
SUPPORTED_EMBEDDING_DIMENSIONS: frozenset[int] = frozenset({128, 256, 512, 768})
MAX_RETRIEVAL_CANDIDATES = 64
_ZERO_NORM_EPSILON = 1e-12


def format_embeddinggemma_query(
    query: str,
    *,
    task_type: Literal["search", "code_retrieval", "question_answering"] = "search",
) -> str:
    """Format a query string with the documented EmbeddingGemma 2 asymmetric task prefix."""
    cleaned = query.strip()
    if task_type == "code_retrieval":
        return f"task: code retrieval | query: {cleaned}"
    if task_type == "question_answering":
        return f"task: question answering | query: {cleaned}"
    return f"task: search result | query: {cleaned}"


def format_embeddinggemma_document(content: str, *, title: str | None = None) -> str:
    """Format a document string with the documented EmbeddingGemma 2 document prefix."""
    clean_title = title.strip() if title and title.strip() else "none"
    return f"title: {clean_title} | text: {content.strip()}"


def truncate_and_l2_normalize(
    vector: Sequence[Any],
    *,
    target_dim: int,
    provider_name: str,
) -> tuple[float, ...]:
    """Validate finite numeric values, apply Matryoshka truncation, and L2-renormalize."""
    if not isinstance(vector, (list, tuple)) or len(vector) < target_dim:
        actual_len = len(vector) if isinstance(vector, (list, tuple)) else "non-sequence"
        raise InvalidProviderResponse(
            f"Provider '{provider_name}' returned embedding vector of length {actual_len}; "
            f"requires at least target_dim={target_dim}.",
            code="embedding_dimension_mismatch",
        )

    floats: list[float] = []
    for raw_val in vector[:target_dim]:
        if isinstance(raw_val, bool) or not isinstance(raw_val, (int, float)):
            raise InvalidProviderResponse(
                f"Provider '{provider_name}' returned non-numeric embedding coordinate.",
                code="invalid_embedding_vector",
            )
        fval = float(raw_val)
        if not math.isfinite(fval):
            raise InvalidProviderResponse(
                f"Provider '{provider_name}' returned non-finite (NaN/Inf) embedding coordinate.",
                code="non_finite_embedding_vector",
            )
        floats.append(fval)

    sq_sum = sum(v * v for v in floats)
    norm = math.sqrt(sq_sum)
    if not math.isfinite(norm) or norm <= _ZERO_NORM_EPSILON:
        raise InvalidProviderResponse(
            f"Provider '{provider_name}' returned a zero-norm embedding vector.",
            code="zero_norm_embedding_vector",
        )

    return tuple(v / norm for v in floats)


def _cosine_similarity(vec_a: tuple[float, ...], vec_b: tuple[float, ...]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
    if not math.isfinite(dot):
        raise InvalidProviderResponse(
            "Computed non-finite cosine similarity score.",
            code="non_finite_retrieval_score",
        )
    # Clamp to [-1.0, 1.0] for numerical precision safety
    return max(-1.0, min(1.0, dot))


class EmbeddingGemmaRetriever:
    """Typed remote Retriever adapter for `google/embeddinggemma-2`."""

    def __init__(
        self,
        *,
        http_client: ProviderHttpClient,
        model_id: str = EMBEDDINGGEMMA_DEFAULT_MODEL,
        target_dimension: int = 768,
        task_type: Literal["search", "code_retrieval", "question_answering"] = "search",
        max_candidates: int = MAX_RETRIEVAL_CANDIDATES,
    ) -> None:
        self._provider_name = "embeddinggemma"
        cleaned_model = model_id.strip()
        if not cleaned_model:
            raise CompatibilityError(
                "EmbeddingGemmaRetriever requires a non-empty model_id.",
                param="retriever_model",
                code="invalid_retriever_model",
            )
        if target_dimension not in SUPPORTED_EMBEDDING_DIMENSIONS:
            raise CompatibilityError(
                f"Unsupported EmbeddingGemma 2 dimension {target_dimension}; "
                f"supported Matryoshka dimensions are {sorted(SUPPORTED_EMBEDDING_DIMENSIONS)}.",
                param="target_dimension",
                code="unsupported_embedding_dimension",
            )
        if max_candidates < 1 or max_candidates > MAX_RETRIEVAL_CANDIDATES:
            raise CompatibilityError(
                f"max_candidates must be between 1 and {MAX_RETRIEVAL_CANDIDATES}.",
                param="max_candidates",
                code="invalid_retriever_config",
            )

        self._http = http_client
        self._model_id = cleaned_model
        self._target_dimension = target_dimension
        self._task_type: Literal["search", "code_retrieval", "question_answering"] = task_type
        self._max_candidates = max_candidates

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def target_dimension(self) -> int:
        return self._target_dimension

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._http.aclose()

    def _validate_candidate_items(
        self,
        items: Sequence[RetrievalCandidateItem],
    ) -> list[RetrievalCandidateItem]:
        if not items:
            raise CompatibilityError(
                "RetrievalRequest requires at least one candidate item.",
                param="items",
                code="empty_retrieval_items",
            )
        if len(items) > self._max_candidates:
            raise CompatibilityError(
                f"RetrievalRequest item count ({len(items)}) exceeds maximum "
                f"allowed ({self._max_candidates}).",
                param="items",
                code="too_many_retrieval_items",
            )

        seen_ids: set[str] = set()
        validated: list[RetrievalCandidateItem] = []
        for item in items:
            clean_id = item.item_id.strip()
            if not clean_id:
                raise CompatibilityError(
                    "RetrievalCandidateItem.item_id must be non-empty.",
                    param="items",
                    code="empty_retrieval_item_id",
                )
            if clean_id in seen_ids:
                raise CompatibilityError(
                    f"Duplicate RetrievalCandidateItem.item_id '{clean_id}'.",
                    param="items",
                    code="duplicate_retrieval_item_id",
                )
            seen_ids.add(clean_id)
            validated.append(item)
        return validated

    def _parse_embeddings_response(
        self,
        data: Mapping[str, Any],
        expected_item_ids: Sequence[str],
    ) -> tuple[tuple[float, ...], dict[str, tuple[float, ...]], str]:
        """Extract normalized `(query_vector, item_vectors_by_id, revision)` from response JSON."""
        reported_model = data.get("model")
        if (
            isinstance(reported_model, str)
            and reported_model.strip()
            and reported_model.strip() != self._model_id
        ):
            raise InvalidProviderResponse(
                f"Provider '{self._provider_name}' returned model '{reported_model.strip()}', "
                f"expected '{self._model_id}'.",
                code="provider_model_mismatch",
            )

        raw_list = data.get("data")
        expected_total = 1 + len(expected_item_ids)
        if not isinstance(raw_list, list) or len(raw_list) != expected_total:
            actual = len(raw_list) if isinstance(raw_list, list) else "non-list"
            raise InvalidProviderResponse(
                f"Provider '{self._provider_name}' must return {expected_total} embeddings "
                f"(1 query + {len(expected_item_ids)} items), got {actual}.",
                code="invalid_embedding_count",
            )

        expected_id_set = set(expected_item_ids)
        raw_vectors_by_index: dict[int, Sequence[Any]] = {}
        raw_vectors_by_item_id: dict[str, Sequence[Any]] = {}
        raw_query_vec: Sequence[Any] | None = None
        observed_raw_dim: int | None = None

        for entry in raw_list:
            if not isinstance(entry, Mapping):
                raise InvalidProviderResponse(
                    f"Provider '{self._provider_name}' embedding entry must be an object.",
                    code="invalid_provider_response",
                )
            vec = entry.get("embedding")
            if not isinstance(vec, (list, tuple)) or not vec:
                raise InvalidProviderResponse(
                    f"Provider '{self._provider_name}' embedding entry is missing vector data.",
                    code="invalid_embedding_vector",
                )
            if observed_raw_dim is None:
                observed_raw_dim = len(vec)
            elif len(vec) != observed_raw_dim:
                raise InvalidProviderResponse(
                    f"Provider '{self._provider_name}' returned inconsistent vector dimensions "
                    f"({observed_raw_dim} vs {len(vec)}).",
                    code="embedding_dimension_mismatch",
                )

            entry_item_id = entry.get("item_id")
            if isinstance(entry_item_id, str):
                if entry_item_id == "__query__":
                    if raw_query_vec is not None:
                        raise InvalidProviderResponse(
                            "Duplicate query embedding in response.",
                            code="duplicate_embedding_entry",
                        )
                    raw_query_vec = vec
                    continue
                if entry_item_id not in expected_id_set:
                    raise InvalidProviderResponse(
                        f"Provider '{self._provider_name}' returned unknown item_id "
                        f"'{entry_item_id}'.",
                        code="unknown_retrieval_item_id",
                    )
                if entry_item_id in raw_vectors_by_item_id:
                    raise InvalidProviderResponse(
                        f"Provider '{self._provider_name}' returned duplicate item_id "
                        f"'{entry_item_id}'.",
                        code="duplicate_retrieval_item_id",
                    )
                raw_vectors_by_item_id[entry_item_id] = vec
                continue

            idx = entry.get("index")
            if isinstance(idx, bool) or not isinstance(idx, int):
                raise InvalidProviderResponse(
                    f"Provider '{self._provider_name}' embedding entry missing integer 'index'.",
                    code="invalid_embedding_index",
                )
            if idx < 0 or idx >= expected_total:
                raise InvalidProviderResponse(
                    f"Provider '{self._provider_name}' returned out-of-range embedding "
                    f"index {idx}.",
                    code="invalid_embedding_index",
                )
            if idx in raw_vectors_by_index:
                raise InvalidProviderResponse(
                    f"Provider '{self._provider_name}' returned duplicate embedding index {idx}.",
                    code="duplicate_embedding_entry",
                )
            raw_vectors_by_index[idx] = vec

        if raw_vectors_by_item_id or raw_query_vec is not None:
            if raw_query_vec is None or set(raw_vectors_by_item_id.keys()) != expected_id_set:
                raise InvalidProviderResponse(
                    f"Provider '{self._provider_name}' response did not match expected "
                    "item_id set.",
                    code="missing_retrieval_item_id",
                )
            query_norm = truncate_and_l2_normalize(
                raw_query_vec,
                target_dim=self._target_dimension,
                provider_name=self._provider_name,
            )
            item_norms = {
                item_id: truncate_and_l2_normalize(
                    raw_vectors_by_item_id[item_id],
                    target_dim=self._target_dimension,
                    provider_name=self._provider_name,
                )
                for item_id in expected_item_ids
            }
        else:
            if len(raw_vectors_by_index) != expected_total:
                raise InvalidProviderResponse(
                    f"Provider '{self._provider_name}' missing expected embedding indices.",
                    code="missing_retrieval_item_id",
                )
            query_norm = truncate_and_l2_normalize(
                raw_vectors_by_index[0],
                target_dim=self._target_dimension,
                provider_name=self._provider_name,
            )
            item_norms = {
                item_id: truncate_and_l2_normalize(
                    raw_vectors_by_index[pos + 1],
                    target_dim=self._target_dimension,
                    provider_name=self._provider_name,
                )
                for pos, item_id in enumerate(expected_item_ids)
            }

        raw_rev = data.get("model_revision") or data.get("system_fingerprint")
        revision = raw_rev.strip() if isinstance(raw_rev, str) and raw_rev.strip() else "unknown"
        return query_norm, item_norms, revision

    async def rank(
        self,
        ctx: RequestContext,
        request: RetrievalRequest,
    ) -> RetrievalResult:
        """Rank candidate items using EmbeddingGemma 2 while preserving mandatory items."""
        if not request.query or not request.query.strip():
            raise CompatibilityError(
                "RetrievalRequest.query must be non-empty.",
                param="query",
                code="empty_retrieval_query",
            )

        items = self._validate_candidate_items(request.items)
        expected_ids = [item.item_id for item in items]

        formatted_inputs = [
            format_embeddinggemma_query(request.query, task_type=self._task_type),
            *[format_embeddinggemma_document(item.content) for item in items],
        ]

        payload: dict[str, Any] = {
            "model": self._model_id,
            "input": formatted_inputs,
            "dimensions": self._target_dimension,
        }

        http_resp = await self._http.post_json(ctx, "/embeddings", payload)
        query_vec, item_vecs, revision = self._parse_embeddings_response(
            http_resp.data,
            expected_ids,
        )

        mandatory_scored: list[tuple[float, str]] = []
        optional_scored: list[tuple[float, str]] = []
        for item in items:
            raw_sim = _cosine_similarity(query_vec, item_vecs[item.item_id])
            rounded_sim = round(raw_sim, 6)
            if item.mandatory:
                mandatory_scored.append((rounded_sim, item.item_id))
            else:
                optional_scored.append((rounded_sim, item.item_id))

        # Deterministic sort: highest score first, then lexicographical item_id tie-breaker
        mandatory_scored.sort(key=lambda pair: (-pair[0], pair[1]))
        optional_scored.sort(key=lambda pair: (-pair[0], pair[1]))

        # Mandatory items are NEVER dropped by semantic ranking or max_results
        remaining_slots = max(0, request.max_results - len(mandatory_scored))
        combined = mandatory_scored + optional_scored[:remaining_slots]

        ranked = tuple(RankedItem(item_id=item_id, score=score) for score, item_id in combined)

        return RetrievalResult(
            semantics=RetrievalSemantics(
                ranked_items=ranked,
                provider_name=self._provider_name,
                model_id=self._model_id,
                model_revision=revision,
            ),
            telemetry=http_resp.telemetry,
        )
