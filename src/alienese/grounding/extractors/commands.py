"""Deterministic command extractor."""

from __future__ import annotations

import re
from collections.abc import Sequence

from alienese.contracts.events import EventKind, NormalizedEvent
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
from alienese.grounding.normalization import is_safe_verification_command

_USER_CMD_PATTERNS = [
    re.compile(r'(?:please\s+)?run\s+[`\'"]([^`\'"\n\r]+)[`\'"]', re.IGNORECASE),
    re.compile(r'(?:execute|command):\s*[`\'"]?([^\n\r`\'"]+)[`\'"]?', re.IGNORECASE),
]
_CMD_PARAM_NAMES = frozenset({"command", "cmd", "command_line", "script"})


class CommandExtractor:
    """Extracts verified, safe verification commands from history."""

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

            # 1. Exact command from previous tool calls (only if safe verification command)
            if ev.kind == EventKind.TOOL_CALL and ev.tool_arguments:
                for k in _CMD_PARAM_NAMES:
                    cmd_val = ev.tool_arguments.get(k)
                    if isinstance(cmd_val, str) and cmd_val.strip():
                        clean_cmd = cmd_val.strip()
                        is_safe, _ = is_safe_verification_command(clean_cmd)
                        if is_safe:
                            category = EvidenceCategory.TEST_COMMAND
                            results.append(
                                GroundingEvidence(
                                    evidence_id=deterministic_evidence_id(
                                        category,
                                        clean_cmd,
                                        ev.provenance,
                                        ev.sequence_no,
                                        sub_id="tool_cmd",
                                    ),
                                    category=category,
                                    value=clean_cmd,
                                    source_provenance=ev.provenance,
                                    trust=ev.trust,
                                    sequence_no=ev.sequence_no,
                                    is_direct=True,
                                    status=EvidenceStatus.OBSERVED,
                                    tool_call_id=ev.tool_call_id,
                                )
                            )

            # 2. Explicit user requested command (only if safe verification command)
            if ev.kind == EventKind.USER_MESSAGE:
                for pat in _USER_CMD_PATTERNS:
                    for match in pat.finditer(bounded_content):
                        raw_cmd = match.group(1).strip()
                        if raw_cmd:
                            is_safe, _ = is_safe_verification_command(raw_cmd)
                            if is_safe:
                                results.append(
                                    GroundingEvidence(
                                        evidence_id=deterministic_evidence_id(
                                            EvidenceCategory.TEST_COMMAND,
                                            raw_cmd,
                                            ev.provenance,
                                            ev.sequence_no,
                                            sub_id="user_cmd",
                                        ),
                                        category=EvidenceCategory.TEST_COMMAND,
                                        value=raw_cmd,
                                        source_provenance=ev.provenance,
                                        trust=ev.trust,
                                        sequence_no=ev.sequence_no,
                                        is_direct=True,
                                        status=EvidenceStatus.OBSERVED,
                                    )
                                )

        return results
