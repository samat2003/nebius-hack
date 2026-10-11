"""Immutable typed grounding evidence domain contracts.

Represents discrete facts extracted from normalized event history and working state.
Evidence is kept strictly distinct from executable CandidateActions and preserves
source event provenance and the 4-level trust boundary.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from alienese.api.errors import InvariantViolation
from alienese.contracts.events import EventProvenance, TrustLevel


class EvidenceCategory(StrEnum):
    """Categorization of extracted grounding evidence."""

    FILE_PATH = "FILE_PATH"
    SYMBOL = "SYMBOL"
    TEST_TARGET = "TEST_TARGET"
    TEST_COMMAND = "TEST_COMMAND"
    SEARCH_PATTERN = "SEARCH_PATTERN"
    STACK_FRAME = "STACK_FRAME"
    FAILURE_MESSAGE = "FAILURE_MESSAGE"
    EXIT_STATUS = "EXIT_STATUS"
    MUTATION_TARGET = "MUTATION_TARGET"
    VERIFICATION_RESULT = "VERIFICATION_RESULT"


class EvidenceStatus(StrEnum):
    """Validation status of extracted evidence."""

    OBSERVED = "OBSERVED"  # Stated or referenced in event history
    CONFIRMED = "CONFIRMED"  # Validated by tool execution or direct observation
    FAILED = "FAILED"  # Associated with failure or execution error
    INFERRED = "INFERRED"  # Structurally inferred rather than explicitly named


def deterministic_evidence_id(
    category: EvidenceCategory,
    value: str,
    provenance: EventProvenance,
    sequence_no: int,
    sub_id: str = "",
) -> str:
    """Compute a deterministic stable evidence ID."""
    seed = (
        f"{category.value}:{value}:{provenance.source_role.value}:"
        f"{provenance.message_index}:{provenance.sub_index}:{sequence_no}:{sub_id}"
    )
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    return f"evi_{digest}"


class GroundingEvidence(BaseModel):
    """An immutable, typed fact derived deterministically from event history."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_id: str = Field(min_length=1)
    category: EvidenceCategory
    value: str = Field(min_length=1)
    source_provenance: EventProvenance
    trust: TrustLevel
    sequence_no: int = Field(ge=0)
    is_direct: bool = True
    status: EvidenceStatus = EvidenceStatus.OBSERVED
    tool_call_id: str | None = None
    line_number: int | None = None
    context_snippet: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _enforce_trust_and_semantics(self) -> GroundingEvidence:
        # Untrusted tool results must never produce trusted evidence
        if (
            self.source_provenance.source_role.value == "tool"
            and self.trust != TrustLevel.UNTRUSTED_EXTERNAL
        ):
            raise InvariantViolation(
                f"Evidence from tool output must have trust level UNTRUSTED_EXTERNAL, "
                f"got {self.trust}"
            )

        # Do not claim confirmed verification result unless explicitly marked
        if (
            self.category == EvidenceCategory.VERIFICATION_RESULT
            and self.status == EvidenceStatus.CONFIRMED
            and not self.tool_call_id
        ):
            raise InvariantViolation(
                "CONFIRMED VERIFICATION_RESULT evidence must be associated with a tool_call_id"
            )

        return self

    def canonical_digest(self) -> str:
        """Compute stable canonical digest for deduplication."""
        canonical_dict = {
            "category": self.category.value,
            "value": self.value,
            "status": self.status.value,
            "line_number": self.line_number,
        }
        raw = json.dumps(canonical_dict, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
