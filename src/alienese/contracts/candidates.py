"""CandidateAction domain contracts with explicit execution dispositions.

Separates external actions (`EXTERNAL_TOOL`, `ASSISTANT_RESPONSE`, `GENERATION_JOB`)
from internal state transitions (`INTERNAL_TRANSITION` such as `EXPAND_SEARCH` or
`REQUEST_EVIDENCE`), preventing internal transitions or incomplete tool candidates
from accidentally serializing as external tool calls.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alienese.api.errors import InvariantViolation
from alienese.contracts.events import EventProvenance
from alienese.contracts.generation import GenerationJobType
from alienese.contracts.state import CanonicalCapability


class CandidateDisposition(StrEnum):
    """Explicit execution disposition for a CandidateAction."""

    EXTERNAL_TOOL = "EXTERNAL_TOOL"
    ASSISTANT_RESPONSE = "ASSISTANT_RESPONSE"
    GENERATION_JOB = "GENERATION_JOB"
    INTERNAL_TRANSITION = "INTERNAL_TRANSITION"


class RiskClass(StrEnum):
    """Risk classification for candidate actions."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class CostClass(StrEnum):
    """Estimated execution/token cost classification for candidate actions."""

    FREE = "FREE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


_INTERNAL_META_INTENTS: frozenset[CanonicalCapability] = frozenset(
    {
        CanonicalCapability.EXPAND_SEARCH,
        CanonicalCapability.REQUEST_EVIDENCE,
    }
)


class CandidateAction(BaseModel):
    """A candidate next action evaluated by the controller policy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str = Field(min_length=1)
    canonical_intent: CanonicalCapability
    disposition: CandidateDisposition
    external_tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    arguments_complete: bool = True
    evidence_refs: tuple[EventProvenance, ...] = ()
    requires_generation: bool = False
    generation_job_type: GenerationJobType | None = None
    risk_class: RiskClass = RiskClass.LOW
    cost_class: CostClass = CostClass.LOW
    rationale: str = ""

    @model_validator(mode="after")
    def _validate_disposition_invariants(self) -> CandidateAction:
        if (
            self.canonical_intent in _INTERNAL_META_INTENTS
            and self.disposition != CandidateDisposition.INTERNAL_TRANSITION
        ):
            raise InvariantViolation(
                f"Meta-action {self.canonical_intent} must have disposition "
                f"INTERNAL_TRANSITION, got {self.disposition}"
            )
        if self.disposition == CandidateDisposition.INTERNAL_TRANSITION and (
            self.external_tool_name is not None
        ):
            raise InvariantViolation(
                "INTERNAL_TRANSITION candidate must not bind an external_tool_name"
            )
        if self.disposition == CandidateDisposition.EXTERNAL_TOOL and not self.external_tool_name:
            raise InvariantViolation("EXTERNAL_TOOL candidate requires external_tool_name")
        if self.disposition == CandidateDisposition.GENERATION_JOB and (
            not self.requires_generation or self.generation_job_type is None
        ):
            raise InvariantViolation(
                "GENERATION_JOB candidate requires requires_generation=True and generation_job_type"
            )
        return self
