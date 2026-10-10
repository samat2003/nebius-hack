"""Deterministic symbol and search pattern extractor."""

from __future__ import annotations

import re
from collections.abc import Sequence

from alienese.contracts.events import NormalizedEvent
from alienese.contracts.state import ExternalToolBinding, WorkingState
from alienese.grounding.evidence import (
    EvidenceCategory,
    EvidenceStatus,
    GroundingEvidence,
    deterministic_evidence_id,
)
from alienese.grounding.extractors.base import (
    MAX_CHARS_PER_OBSERVATION,
    MAX_EVENTS_SCANNED,
)
from alienese.grounding.normalization import normalize_symbol_name

_TRACEBACK_FUNC_RE = re.compile(r"line\s+\d+,\s+in\s+([a-zA-Z_][a-zA-Z0-9_]*)")
_QUALIFIED_SYMBOL_RE = re.compile(
    r'(?:^|[\s"\'`(<])([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]+)(?:$|[\s"\'`>)])'
)
_DEF_OR_CLASS_RE = re.compile(r"(?:def|class)\s+([a-zA-Z_][a-zA-Z0-9_]*)")


class SymbolExtractor:
    """Extracts function, class, qualified symbols, and search patterns."""

    def extract(
        self,
        events: Sequence[NormalizedEvent],
        state: WorkingState,
        tools: Sequence[ExternalToolBinding],
    ) -> Sequence[GroundingEvidence]:
        _ = (state, tools)
        results: list[GroundingEvidence] = []
        scanned_events = events[-MAX_EVENTS_SCANNED:]

        for ev in scanned_events:
            bounded_content = ev.content[:MAX_CHARS_PER_OBSERVATION]

            # 1. Traceback function names
            for match in _TRACEBACK_FUNC_RE.finditer(bounded_content):
                raw_func = match.group(1)
                norm = normalize_symbol_name(raw_func)
                if norm and norm != "<module>":
                    results.append(
                        GroundingEvidence(
                            evidence_id=deterministic_evidence_id(
                                EvidenceCategory.SYMBOL,
                                norm,
                                ev.provenance,
                                ev.sequence_no,
                                sub_id="tb_func",
                            ),
                            category=EvidenceCategory.SYMBOL,
                            value=norm,
                            source_provenance=ev.provenance,
                            trust=ev.trust,
                            sequence_no=ev.sequence_no,
                            is_direct=True,
                            status=EvidenceStatus.FAILED,
                            tool_call_id=ev.tool_call_id,
                        )
                    )
                    # Also register as search pattern candidate
                    results.append(
                        GroundingEvidence(
                            evidence_id=deterministic_evidence_id(
                                EvidenceCategory.SEARCH_PATTERN,
                                norm,
                                ev.provenance,
                                ev.sequence_no,
                                sub_id="search_pat",
                            ),
                            category=EvidenceCategory.SEARCH_PATTERN,
                            value=norm,
                            source_provenance=ev.provenance,
                            trust=ev.trust,
                            sequence_no=ev.sequence_no,
                            is_direct=False,
                            status=EvidenceStatus.INFERRED,
                            tool_call_id=ev.tool_call_id,
                        )
                    )

            # 2. Qualified symbols (e.g. Module.func or Class.method)
            for match in _QUALIFIED_SYMBOL_RE.finditer(bounded_content):
                raw_sym = match.group(1)
                norm = normalize_symbol_name(raw_sym)
                if norm:
                    results.append(
                        GroundingEvidence(
                            evidence_id=deterministic_evidence_id(
                                EvidenceCategory.SYMBOL,
                                norm,
                                ev.provenance,
                                ev.sequence_no,
                                sub_id="qual_sym",
                            ),
                            category=EvidenceCategory.SYMBOL,
                            value=norm,
                            source_provenance=ev.provenance,
                            trust=ev.trust,
                            sequence_no=ev.sequence_no,
                            is_direct=True,
                            status=EvidenceStatus.OBSERVED,
                            tool_call_id=ev.tool_call_id,
                        )
                    )

            # 3. def / class declarations
            for match in _DEF_OR_CLASS_RE.finditer(bounded_content):
                raw_decl = match.group(1)
                norm = normalize_symbol_name(raw_decl)
                if norm:
                    results.append(
                        GroundingEvidence(
                            evidence_id=deterministic_evidence_id(
                                EvidenceCategory.SYMBOL,
                                norm,
                                ev.provenance,
                                ev.sequence_no,
                                sub_id="decl_sym",
                            ),
                            category=EvidenceCategory.SYMBOL,
                            value=norm,
                            source_provenance=ev.provenance,
                            trust=ev.trust,
                            sequence_no=ev.sequence_no,
                            is_direct=True,
                            status=EvidenceStatus.OBSERVED,
                            tool_call_id=ev.tool_call_id,
                        )
                    )

        return results
