"""Typed immutable event taxonomy and trust classification contracts.

Enforces the 4-tier trust boundary:
- SYSTEM_TRUSTED: trusted system and developer policy instructions.
- USER: direct user instructions and prompts.
- MODEL_GENERATED: assistant messages and model-emitted tool calls.
- UNTRUSTED_EXTERNAL: tool outputs, repository contents, shell outputs, external data.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alienese.api.errors import InvariantViolation


class TrustLevel(StrEnum):
    """Trust boundary classification for protocol and runtime events."""

    SYSTEM_TRUSTED = "SYSTEM_TRUSTED"
    USER = "USER"
    MODEL_GENERATED = "MODEL_GENERATED"
    UNTRUSTED_EXTERNAL = "UNTRUSTED_EXTERNAL"


class SourceRole(StrEnum):
    """External protocol message role."""

    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class EventKind(StrEnum):
    """Extensible normalized event taxonomy."""

    # Core protocol events (Phase 1)
    SYSTEM_MESSAGE = "SYSTEM_MESSAGE"
    USER_MESSAGE = "USER_MESSAGE"
    ASSISTANT_MESSAGE = "ASSISTANT_MESSAGE"
    TOOL_CALL = "TOOL_CALL"
    TOOL_RESULT = "TOOL_RESULT"

    # Future specialization hooks (Phase 3+)
    TEST_RESULT = "TEST_RESULT"
    FAILURE = "FAILURE"
    FILE_OBSERVATION = "FILE_OBSERVATION"
    MUTATION = "MUTATION"
    VERIFICATION = "VERIFICATION"


_REQUIRED_TRUST_BY_KIND: dict[EventKind, frozenset[TrustLevel]] = {
    EventKind.SYSTEM_MESSAGE: frozenset({TrustLevel.SYSTEM_TRUSTED}),
    EventKind.USER_MESSAGE: frozenset({TrustLevel.USER}),
    EventKind.ASSISTANT_MESSAGE: frozenset({TrustLevel.MODEL_GENERATED}),
    EventKind.TOOL_CALL: frozenset({TrustLevel.MODEL_GENERATED}),
    EventKind.TOOL_RESULT: frozenset({TrustLevel.UNTRUSTED_EXTERNAL}),
    EventKind.TEST_RESULT: frozenset({TrustLevel.UNTRUSTED_EXTERNAL}),
    EventKind.FAILURE: frozenset({TrustLevel.UNTRUSTED_EXTERNAL}),
    EventKind.FILE_OBSERVATION: frozenset({TrustLevel.UNTRUSTED_EXTERNAL}),
    EventKind.MUTATION: frozenset({TrustLevel.UNTRUSTED_EXTERNAL}),
    EventKind.VERIFICATION: frozenset({TrustLevel.UNTRUSTED_EXTERNAL}),
}

_REQUIRED_TRUST_BY_ROLE: dict[SourceRole, TrustLevel] = {
    SourceRole.SYSTEM: TrustLevel.SYSTEM_TRUSTED,
    SourceRole.DEVELOPER: TrustLevel.SYSTEM_TRUSTED,
    SourceRole.USER: TrustLevel.USER,
    SourceRole.ASSISTANT: TrustLevel.MODEL_GENERATED,
    SourceRole.TOOL: TrustLevel.UNTRUSTED_EXTERNAL,
}


class EventProvenance(BaseModel):
    """Immutable provenance link back to the originating protocol message."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    message_index: int = Field(ge=0)
    sub_index: int = Field(default=0, ge=0)
    source_role: SourceRole
    external_tool_call_id: str | None = None
    tool_name: str | None = None


class NormalizedEvent(BaseModel):
    """Immutable normalized observation derived from protocol input."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sequence_no: int = Field(ge=0)
    event_id: str = Field(min_length=1)
    kind: EventKind
    trust: TrustLevel
    provenance: EventProvenance
    content: str = ""
    tool_name: str | None = None
    tool_call_id: str | None = None
    tool_arguments: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _enforce_trust_boundary(self) -> NormalizedEvent:
        allowed_trust = _REQUIRED_TRUST_BY_KIND.get(self.kind)
        if allowed_trust is not None and self.trust not in allowed_trust:
            raise InvariantViolation(
                f"Event kind {self.kind} cannot have trust level {self.trust}; "
                f"allowed: {sorted(t.value for t in allowed_trust)}"
            )
        expected_role_trust = _REQUIRED_TRUST_BY_ROLE[self.provenance.source_role]
        if self.trust != expected_role_trust:
            raise InvariantViolation(
                f"Source role {self.provenance.source_role} must have trust "
                f"{expected_role_trust}, got {self.trust}"
            )
        if self.kind == EventKind.TOOL_CALL and (
            not self.tool_call_id or not self.tool_name or self.tool_arguments is None
        ):
            raise InvariantViolation(
                "TOOL_CALL event requires tool_call_id, tool_name, and tool_arguments"
            )
        if self.kind == EventKind.TOOL_RESULT and not self.tool_call_id:
            raise InvariantViolation("TOOL_RESULT event requires tool_call_id")
        return self
