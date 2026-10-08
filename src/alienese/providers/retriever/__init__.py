"""Remote Retriever provider adapters."""

from __future__ import annotations

from alienese.providers.retriever.embeddinggemma import (
    EMBEDDINGGEMMA_DEFAULT_MODEL,
    SUPPORTED_EMBEDDING_DIMENSIONS,
    EmbeddingGemmaRetriever,
    format_embeddinggemma_document,
    format_embeddinggemma_query,
    truncate_and_l2_normalize,
)

__all__ = [
    "EMBEDDINGGEMMA_DEFAULT_MODEL",
    "SUPPORTED_EMBEDDING_DIMENSIONS",
    "EmbeddingGemmaRetriever",
    "format_embeddinggemma_document",
    "format_embeddinggemma_query",
    "truncate_and_l2_normalize",
]
