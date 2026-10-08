"""WorkingState and canonical capability domain contracts.

WorkingState is a deterministic projection over NormalizedEvent sequences:
- Uses `initial_user_request` and `trusted_system_instructions` rather than
  claiming semantic NLP extraction of objectives or constraints in Phase 1.
- Preserves unknown tools as `CUSTOM_TOOL` rather than guessing `RUN_COMMAND`.
- Distinguishes `mutation_attempted` vs `mutation_confirmed` and
  `verification_attempted` vs `verification_confirmed`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alienese.api.errors import InvariantViolation
from alienese.contracts.events import EventProvenance, TrustLevel


class CanonicalCapability(StrEnum):
    """Canonical internal action intents decoupled from harness-specific tool names."""

    READ_FILE = "READ_FILE"
    SEARCH_TEXT = "SEARCH_TEXT"
    LIST_FILES = "LIST_FILES"
    RUN_COMMAND = "RUN_COMMAND"
    RUN_TEST = "RUN_TEST"
    APPLY_PATCH = "APPLY_PATCH"
    WRITE_FILE = "WRITE_FILE"
    EXPLAIN_FAILURE = "EXPLAIN_FAILURE"
    GENERATE_PATCH = "GENERATE_PATCH"
    WRITE_TEST = "WRITE_TEST"
    SYNTHESIZE_SEARCH = "SYNTHESIZE_SEARCH"
    RESPOND = "RESPOND"
    EXPAND_SEARCH = "EXPAND_SEARCH"
    REQUEST_EVIDENCE = "REQUEST_EVIDENCE"
    FINISH = "FINISH"
    CUSTOM_TOOL = "CUSTOM_TOOL"
    UNKNOWN = "UNKNOWN"


class ExternalToolBinding(BaseModel):
    """Preserves external tool definition and its deterministic capability mapping."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    external_name: str = Field(min_length=1)
    description: str = ""
    parameters_schema: dict[str, Any] = Field(default_factory=dict)
    canonical_capability: CanonicalCapability = CanonicalCapability.CUSTOM_TOOL


class RecordedAction(BaseModel):
    """A tool call action observed in the conversation event stream."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sequence_no: int = Field(ge=0)
    tool_call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    canonical_capability: CanonicalCapability
    arguments: dict[str, Any] = Field(default_factory=dict)
    provenance: EventProvenance


class ToolObservation(BaseModel):
    """An untrusted external observation produced by a tool execution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sequence_no: int = Field(ge=0)
    tool_call_id: str = Field(min_length=1)
    tool_name: str | None = None
    content: str
    trust: TrustLevel = TrustLevel.UNTRUSTED_EXTERNAL
    provenance: EventProvenance

    @model_validator(mode="after")
    def _enforce_untrusted(self) -> ToolObservation:
        if self.trust != TrustLevel.UNTRUSTED_EXTERNAL:
            raise InvariantViolation(
                f"ToolObservation must have UNTRUSTED_EXTERNAL trust, got {self.trust}"
            )
        return self


class MutationVerificationState(BaseModel):
    """Distinguishes attempted vs confirmed mutations and verifications."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mutation_attempted: bool = False
    mutation_confirmed: bool = False
    last_mutation_attempt_seq: int | None = None
    last_mutation_confirmed_seq: int | None = None

    verification_attempted: bool = False
    verification_confirmed: bool = False
    last_verification_attempt_seq: int | None = None
    last_verification_confirmed_seq: int | None = None


class WorkingState(BaseModel):
    """Deterministic, immutable projection of the normalized event history."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    state_digest: str = Field(min_length=64, max_length=64)
    event_count: int = Field(ge=0)
    last_sequence_no: int | None = None
    initial_user_request: str | None = None
    latest_user_request: str | None = None
    user_messages: tuple[str, ...] = ()
    trusted_system_instructions: tuple[str, ...] = ()
    available_tools: tuple[ExternalToolBinding, ...] = ()
    recent_actions: tuple[RecordedAction, ...] = ()
    pending_tool_call_ids: tuple[str, ...] = ()
    latest_tool_observation: ToolObservation | None = None
    mutation_verification: MutationVerificationState = Field(
        default_factory=MutationVerificationState
    )
    provenance_refs: tuple[EventProvenance, ...] = ()
