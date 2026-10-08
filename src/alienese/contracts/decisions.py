"""Controller decision contracts separating DecisionSemantics from ProviderCallTelemetry."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ProviderCallTelemetry(BaseModel):
    """Operational telemetry excluded from deterministic replay equality comparisons.

    Uses `None` for unknown token counts or unverified pricing rather than
    fabricating numerical zero measurements.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    latency_ms: float = Field(default=0.0, ge=0.0)
    request_attempt_id: str | None = None
    upstream_request_id: str | None = None
    attempt_count: int = Field(default=1, ge=1)
    failed_attempt_count: int = Field(default=0, ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    retried_prompt_tokens: int | None = Field(default=None, ge=0)
    retried_completion_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0.0)


class CandidateScore(BaseModel):
    """Controller score assigned to a specific CandidateAction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str = Field(min_length=1)
    score: float


class GuardMetadata(BaseModel):
    """Metadata describing policy guard or fallback intervention."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bypassed_controller: bool = False
    fallback_used: bool = False
    fallback_reason: str | None = None
    verification_required: bool = False


class DecisionSemantics(BaseModel):
    """Deterministic semantic content of a controller decision."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    selected_candidate_id: str = Field(min_length=1)
    scores: tuple[CandidateScore, ...] = ()
    provider_name: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_revision: str = Field(default="unknown", min_length=1)
    guard_metadata: GuardMetadata = Field(default_factory=GuardMetadata)


class DecisionResult(BaseModel):
    """Complete controller result combining deterministic semantics and runtime telemetry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    semantics: DecisionSemantics
    telemetry: ProviderCallTelemetry = Field(default_factory=ProviderCallTelemetry)

    @property
    def selected_candidate_id(self) -> str:
        return self.semantics.selected_candidate_id

    @property
    def scores(self) -> tuple[CandidateScore, ...]:
        return self.semantics.scores

    @property
    def provider_name(self) -> str:
        return self.semantics.provider_name

    @property
    def model_id(self) -> str:
        return self.semantics.model_id

    @property
    def model_revision(self) -> str:
        return self.semantics.model_revision

    @property
    def guard_metadata(self) -> GuardMetadata:
        return self.semantics.guard_metadata
