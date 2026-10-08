"""Replay artifact contracts separating deterministic ReplaySemantics from ReplayTelemetry."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from alienese.contracts.candidates import CandidateAction
from alienese.contracts.decisions import DecisionSemantics, ProviderCallTelemetry
from alienese.contracts.events import NormalizedEvent
from alienese.contracts.generation import GenerationJob, GenerationSemantics
from alienese.contracts.state import ExternalToolBinding


class ComponentVersions(BaseModel):
    """Version identifiers for deterministic replay compatibility verification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    runtime_version: str = "0.1.0"
    normalizer_version: str = "v1"
    state_schema_version: str = "v1"
    policy_version: str = "v1"


class ReplaySemantics(BaseModel):
    """Equality-critical semantic content of a recorded decision turn."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    versions: ComponentVersions = Field(default_factory=ComponentVersions)
    normalized_events: tuple[NormalizedEvent, ...]
    available_tools: tuple[ExternalToolBinding, ...] = ()
    state_digest: str = Field(min_length=64, max_length=64)
    candidates: tuple[CandidateAction, ...]
    decision_semantics: DecisionSemantics
    generation_job: GenerationJob | None = None
    generation_semantics: GenerationSemantics | None = None
    logical_response: dict[str, Any]


class ReplayTelemetry(BaseModel):
    """Non-deterministic operational metadata excluded from replay equivalence checks."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str
    operation_id: str
    correlation_trace_id: str
    decision_telemetry: ProviderCallTelemetry = Field(default_factory=ProviderCallTelemetry)
    generation_telemetry: ProviderCallTelemetry | None = None


class ReplayArtifact(BaseModel):
    """Complete replayable trace artifact for a single TurnEngine execution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    semantics: ReplaySemantics
    telemetry: ReplayTelemetry

    def is_semantically_equivalent(self, other: ReplayArtifact) -> bool:
        """Return True if two replay artifacts have identical deterministic semantics."""
        return self.semantics == other.semantics
