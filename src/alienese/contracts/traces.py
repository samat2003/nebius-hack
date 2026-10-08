"""Replay and telemetry trace contracts separating privacy-safe telemetry from replayable content.

Implements two explicit modes:
1. `TraceMode.METADATA_ONLY` (default, `replayable=False`):
   Stores hashes, counts, timings, action types/dispositions, model identifiers,
   and decision metrics; stores no raw user prompts, repository content, tool
   arguments, or tool outputs.
2. `TraceMode.FULL_FIDELITY_REPLAY` (disabled by default, `replayable=True`):
   Explicitly enabled only for controlled local development/evaluation; stores
   exact unmutated `normalized_events`, `available_tools`, `candidates`,
   `decision_semantics`, `generation_job`, `generation_semantics`, and
   `logical_response` so `WorkingState` can be reconstructed with the exact
   `state_digest`.

A sanitized/redacted or metadata-only artifact is never marked `replayable=True`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alienese.api.errors import InvariantViolation
from alienese.contracts.candidates import CandidateAction, CandidateDisposition, RiskClass
from alienese.contracts.decisions import DecisionSemantics, ProviderCallTelemetry
from alienese.contracts.events import EventKind, NormalizedEvent, SourceRole, TrustLevel
from alienese.contracts.generation import GenerationJob, GenerationJobType, GenerationSemantics
from alienese.contracts.state import CanonicalCapability, ExternalToolBinding


class TraceMode(StrEnum):
    """Explicit persistence mode for turn trace artifacts."""

    METADATA_ONLY = "metadata_only"
    FULL_FIDELITY_REPLAY = "full_fidelity_replay"


class ComponentVersions(BaseModel):
    """Version identifiers for deterministic replay compatibility verification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    runtime_version: str = "0.1.0"
    normalizer_version: str = "v1"
    state_schema_version: str = "v1"
    policy_version: str = "v1"


class EventMetadataDigest(BaseModel):
    """Privacy-safe digest of a NormalizedEvent with no raw prompt or tool content."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sequence_no: int = Field(ge=0)
    event_id: str = Field(min_length=1)
    kind: EventKind
    trust: TrustLevel
    source_role: SourceRole
    content_sha256: str = Field(min_length=64, max_length=64)
    content_length: int = Field(ge=0)
    tool_name: str | None = None
    tool_call_id: str | None = None
    arguments_sha256: str | None = None


class CandidateMetadataDigest(BaseModel):
    """Privacy-safe summary of a CandidateAction without raw tool argument values."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str = Field(min_length=1)
    canonical_intent: CanonicalCapability
    disposition: CandidateDisposition
    external_tool_name: str | None = None
    arguments_complete: bool
    requires_generation: bool
    generation_job_type: GenerationJobType | None = None
    risk_class: RiskClass = RiskClass.LOW


class ReplaySemantics(BaseModel):
    """Equality-critical semantic record of a turn execution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trace_mode: TraceMode = TraceMode.METADATA_ONLY
    replayable: bool = False
    versions: ComponentVersions = Field(default_factory=ComponentVersions)
    state_digest: str = Field(min_length=64, max_length=64)
    event_count: int = Field(default=0, ge=0)
    tool_count: int = Field(default=0, ge=0)
    candidate_count: int = Field(default=0, ge=0)
    event_digests: tuple[EventMetadataDigest, ...] = ()
    candidate_digests: tuple[CandidateMetadataDigest, ...] = ()
    decision_semantics: DecisionSemantics
    finish_reason: str = "stop"
    response_sha256: str = Field(
        default="0" * 64,
        min_length=64,
        max_length=64,
    )

    # Full-fidelity fields (populated ONLY when FULL_FIDELITY_REPLAY and replayable=True)
    normalized_events: tuple[NormalizedEvent, ...] = ()
    available_tools: tuple[ExternalToolBinding, ...] = ()
    candidates: tuple[CandidateAction, ...] = ()
    generation_job: GenerationJob | None = None
    generation_semantics: GenerationSemantics | None = None
    logical_response: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _enforce_replayability_invariants(self) -> ReplaySemantics:
        if self.trace_mode == TraceMode.METADATA_ONLY:
            if self.replayable:
                raise InvariantViolation("METADATA_ONLY trace cannot be marked replayable=True.")
            if (
                self.normalized_events
                or self.available_tools
                or self.candidates
                or self.generation_job is not None
                or self.generation_semantics is not None
                or self.logical_response is not None
            ):
                raise InvariantViolation(
                    "METADATA_ONLY trace must not retain raw events, tools, candidates, "
                    "generation content, or logical_response payloads."
                )
        elif self.trace_mode == TraceMode.FULL_FIDELITY_REPLAY:
            if not self.replayable:
                raise InvariantViolation("FULL_FIDELITY_REPLAY trace must have replayable=True.")
            if not self.normalized_events or not self.candidates or self.logical_response is None:
                raise InvariantViolation(
                    "FULL_FIDELITY_REPLAY trace requires normalized_events, candidates, "
                    "and logical_response."
                )
        return self


class ReplayTelemetry(BaseModel):
    """Non-deterministic operational metadata excluded from replay equivalence checks."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str
    operation_id: str
    correlation_trace_id: str
    created_timestamp: int = Field(default=0, ge=0)
    decision_telemetry: ProviderCallTelemetry = Field(default_factory=ProviderCallTelemetry)
    generation_telemetry: ProviderCallTelemetry | None = None


class ReplayArtifact(BaseModel):
    """Complete trace artifact for a single TurnEngine execution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    semantics: ReplaySemantics
    telemetry: ReplayTelemetry

    @property
    def trace_mode(self) -> TraceMode:
        return self.semantics.trace_mode

    @property
    def replayable(self) -> bool:
        return self.semantics.replayable

    def is_semantically_equivalent(self, other: ReplayArtifact) -> bool:
        """Return True if two artifacts have identical deterministic semantics."""
        return self.semantics == other.semantics
