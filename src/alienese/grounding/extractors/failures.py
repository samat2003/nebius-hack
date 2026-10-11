"""Deterministic failure, error type, and exit status extractor."""

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

_EXIT_CODE_RE = re.compile(
    r"(?:exit(?:ed)?\s+(?:with\s+)?(?:code|status)\s*[:=]?\s*|returncode\s*=\s*)([1-9][0-9]*)",
    re.IGNORECASE,
)
_PYTHON_ERROR_LINE_RE = re.compile(
    r"([a-zA-Z0-9_.]*(?:Error|Exception|Interrupt|Exit|Fault|Failure)):\s*([^\n\r]+)"
)


class FailureExtractor:
    """Extracts failures, stack frames, assertion messages, and nonzero exit statuses."""

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

            # 1. Nonzero exit statuses
            for match in _EXIT_CODE_RE.finditer(bounded_content):
                code_str = match.group(1)
                results.append(
                    GroundingEvidence(
                        evidence_id=deterministic_evidence_id(
                            EvidenceCategory.EXIT_STATUS,
                            code_str,
                            ev.provenance,
                            ev.sequence_no,
                            sub_id="exit_code",
                        ),
                        category=EvidenceCategory.EXIT_STATUS,
                        value=code_str,
                        source_provenance=ev.provenance,
                        trust=ev.trust,
                        sequence_no=ev.sequence_no,
                        is_direct=True,
                        status=EvidenceStatus.FAILED,
                        tool_call_id=ev.tool_call_id,
                    )
                )

            # 2. Python exception type and message
            for match in _PYTHON_ERROR_LINE_RE.finditer(bounded_content):
                err_type = match.group(1)
                err_msg = match.group(2).strip()
                full_val = f"{err_type}: {err_msg}"[:256]
                results.append(
                    GroundingEvidence(
                        evidence_id=deterministic_evidence_id(
                            EvidenceCategory.FAILURE_MESSAGE,
                            full_val,
                            ev.provenance,
                            ev.sequence_no,
                            sub_id="err_msg",
                        ),
                        category=EvidenceCategory.FAILURE_MESSAGE,
                        value=full_val,
                        source_provenance=ev.provenance,
                        trust=ev.trust,
                        sequence_no=ev.sequence_no,
                        is_direct=True,
                        status=EvidenceStatus.FAILED,
                        tool_call_id=ev.tool_call_id,
                        context_snippet=match.group(0),
                    )
                )

        return results
