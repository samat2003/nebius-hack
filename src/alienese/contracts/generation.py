"""Typed narrow synthesis contracts for the Generator provider.

Separates deterministic `GenerationSemantics` from operational `ProviderCallTelemetry`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from alienese.contracts.decisions import ProviderCallTelemetry


class GenerationJobType(StrEnum):
    """Bounded synthesis job categories supported by the Generator."""

    ANSWER = "ANSWER"
    EXPLAIN_FAILURE = "EXPLAIN_FAILURE"
    GENERATE_PATCH = "GENERATE_PATCH"
    WRITE_TEST = "WRITE_TEST"
    SYNTHESIZE_SEARCH = "SYNTHESIZE_SEARCH"


class GenerationJob(BaseModel):
    """Narrow, typed synthesis request passed to a Generator provider."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: str = Field(min_length=1)
    job_type: GenerationJobType
    candidate_id: str = Field(min_length=1)
    initial_user_request: str | None = None
    latest_user_request: str | None = None
    trusted_system_instructions: tuple[str, ...] = ()
    selected_evidence: tuple[str, ...] = ()
    target_tool_name: str | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)


class GenerationSemantics(BaseModel):
    """Deterministic semantic output of a generation job (used for replay equivalence)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: str = Field(min_length=1)
    job_type: GenerationJobType
    content: str
    structured_arguments: dict[str, Any] | None = None
    provider_name: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_revision: str = Field(default="unknown", min_length=1)


class GenerationResult(BaseModel):
    """Complete result from a Generator call combining semantics and telemetry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    semantics: GenerationSemantics
    telemetry: ProviderCallTelemetry = Field(default_factory=ProviderCallTelemetry)

    @property
    def job_id(self) -> str:
        return self.semantics.job_id

    @property
    def job_type(self) -> GenerationJobType:
        return self.semantics.job_type

    @property
    def content(self) -> str:
        return self.semantics.content

    @property
    def structured_arguments(self) -> dict[str, Any] | None:
        return self.semantics.structured_arguments

    @property
    def provider_name(self) -> str:
        return self.semantics.provider_name

    @property
    def model_id(self) -> str:
        return self.semantics.model_id

    @property
    def model_revision(self) -> str:
        return self.semantics.model_revision
