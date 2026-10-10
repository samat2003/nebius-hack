"""Deterministic evidence extractors package."""

from __future__ import annotations

from collections.abc import Sequence

from alienese.contracts.events import NormalizedEvent
from alienese.contracts.state import ExternalToolBinding, WorkingState
from alienese.grounding.evidence import GroundingEvidence
from alienese.grounding.extractors.base import MAX_EVIDENCE_RECORDS, EvidenceExtractor
from alienese.grounding.extractors.commands import CommandExtractor
from alienese.grounding.extractors.failures import FailureExtractor
from alienese.grounding.extractors.mutation import MutationExtractor
from alienese.grounding.extractors.paths import PathExtractor
from alienese.grounding.extractors.symbols import SymbolExtractor
from alienese.grounding.extractors.tests import TestExtractor
from alienese.grounding.normalization import deduplicate_evidence

__all__ = [
    "CommandExtractor",
    "EvidenceExtractor",
    "FailureExtractor",
    "MutationExtractor",
    "PathExtractor",
    "SymbolExtractor",
    "TestExtractor",
    "extract_all_evidence",
]


def extract_all_evidence(
    events: Sequence[NormalizedEvent],
    state: WorkingState,
    tools: Sequence[ExternalToolBinding],
) -> tuple[GroundingEvidence, ...]:
    """Execute all deterministic extractors in fixed sequence, deduplicate and cap."""
    extractors: tuple[EvidenceExtractor, ...] = (
        PathExtractor(),
        TestExtractor(),
        SymbolExtractor(),
        CommandExtractor(),
        FailureExtractor(),
        MutationExtractor(),
    )

    all_extracted: list[GroundingEvidence] = []
    for ext in extractors:
        all_extracted.extend(ext.extract(events, state, tools))

    deduped = deduplicate_evidence(all_extracted)
    return deduped[:MAX_EVIDENCE_RECORDS]
