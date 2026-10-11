"""Base extractor protocol and resource bounds."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from alienese.contracts.events import NormalizedEvent
from alienese.contracts.state import ExternalToolBinding, WorkingState
from alienese.grounding.evidence import GroundingEvidence

MAX_EVENTS_SCANNED = 50
MAX_CHARS_PER_OBSERVATION = 32_768
MAX_EVIDENCE_RECORDS = 100


class EvidenceExtractor(Protocol):
    """Protocol for specialized deterministic evidence extractors."""

    def extract(
        self,
        events: Sequence[NormalizedEvent],
        state: WorkingState,
        tools: Sequence[ExternalToolBinding],
    ) -> Sequence[GroundingEvidence]:
        """Extract evidence records deterministically from events and state."""
        ...
