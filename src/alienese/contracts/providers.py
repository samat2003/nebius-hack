"""Strict provider protocols for Retriever, Controller, and Generator.

Core domain modules depend only on these protocols and never on provider SDKs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from alienese.contracts.candidates import CandidateAction
from alienese.contracts.context import RequestContext
from alienese.contracts.decisions import DecisionResult, ProviderCallTelemetry
from alienese.contracts.generation import GenerationJob, GenerationResult
from alienese.contracts.state import WorkingState


class RetrievalCandidateItem(BaseModel):
    """An item submitted to the Retriever for ranking."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    item_id: str = Field(min_length=1)
    content: str
    mandatory: bool = False


class RankedItem(BaseModel):
    """A ranked item returned by the Retriever."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    item_id: str = Field(min_length=1)
    score: float


class RetrievalRequest(BaseModel):
    """Typed request for the Retriever provider."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str
    items: tuple[RetrievalCandidateItem, ...]
    max_results: int = Field(default=10, ge=1)


class RetrievalSemantics(BaseModel):
    """Deterministic semantic result of a retrieval ranking operation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ranked_items: tuple[RankedItem, ...]
    provider_name: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_revision: str = Field(default="fake-v1", min_length=1)


class RetrievalResult(BaseModel):
    """Complete result from a Retriever provider combining semantics and telemetry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    semantics: RetrievalSemantics
    telemetry: ProviderCallTelemetry = Field(default_factory=ProviderCallTelemetry)

    @property
    def ranked_items(self) -> tuple[RankedItem, ...]:
        return self.semantics.ranked_items


@runtime_checkable
class Retriever(Protocol):
    """Protocol for context and candidate ranking providers (e.g., EmbeddingGemma 2)."""

    async def rank(
        self,
        ctx: RequestContext,
        request: RetrievalRequest,
    ) -> RetrievalResult:
        """Rank candidate or context items for the given query."""
        ...


@runtime_checkable
class Controller(Protocol):
    """Protocol for finite-choice action selection providers (e.g., mini-Jev)."""

    async def decide(
        self,
        ctx: RequestContext,
        state: WorkingState,
        candidates: Sequence[CandidateAction],
    ) -> DecisionResult:
        """Score and select one CandidateAction from the provided finite set."""
        ...


@runtime_checkable
class Generator(Protocol):
    """Protocol for narrow open-ended synthesis providers (e.g., Nemotron)."""

    async def generate(
        self,
        ctx: RequestContext,
        job: GenerationJob,
    ) -> GenerationResult:
        """Execute a typed GenerationJob and return synthesized output."""
        ...
