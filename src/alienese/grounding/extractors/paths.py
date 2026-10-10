"""Deterministic file path extractor."""

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
from alienese.grounding.normalization import normalize_file_path

_PYTHON_TRACEBACK_FILE_RE = re.compile(
    r'File\s+["\']([^"\']+)["\'],\s+line\s+(\d+)(?:,\s+in\s+([a-zA-Z0-9_<>]+))?'
)
_FILE_PATH_TOKEN_RE = re.compile(
    r'(?:^|[\s"\'`(<\[:])((?:[a-zA-Z0-9_\-\.]+/)+[a-zA-Z0-9_\-\.]+\.[a-zA-Z0-9_\-]+)(?:$|[\s"\'`>)\]:,])'
)
_PATH_PARAM_NAMES = frozenset(
    {"path", "filepath", "file_path", "target_file", "filename", "file", "dest", "source"}
)


class PathExtractor:
    """Extracts grounded file paths from events, tracebacks, tool calls, and tool outputs."""

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
            # 1. Traceback file paths in tool results or assistant messages
            bounded_content = ev.content[:MAX_CHARS_PER_OBSERVATION]
            for match in _PYTHON_TRACEBACK_FILE_RE.finditer(bounded_content):
                raw_path = match.group(1)
                line_no = int(match.group(2))
                func_name = match.group(3)
                norm = normalize_file_path(raw_path)
                if norm and not norm.startswith("<"):
                    results.append(
                        GroundingEvidence(
                            evidence_id=deterministic_evidence_id(
                                EvidenceCategory.FILE_PATH,
                                norm,
                                ev.provenance,
                                ev.sequence_no,
                                sub_id=f"tb_{line_no}",
                            ),
                            category=EvidenceCategory.FILE_PATH,
                            value=norm,
                            source_provenance=ev.provenance,
                            trust=ev.trust,
                            sequence_no=ev.sequence_no,
                            is_direct=True,
                            status=EvidenceStatus.FAILED
                            if ev.kind in (EventKind.TOOL_RESULT, EventKind.FAILURE)
                            else EvidenceStatus.OBSERVED,
                            tool_call_id=ev.tool_call_id,
                            line_number=line_no,
                            context_snippet=match.group(0),
                            metadata={"traceback_function": func_name} if func_name else {},
                        )
                    )

            # 2. Previous tool arguments
            if ev.tool_arguments:
                for k, v in ev.tool_arguments.items():
                    if k.lower() in _PATH_PARAM_NAMES and isinstance(v, str):
                        norm = normalize_file_path(v)
                        if norm:
                            results.append(
                                GroundingEvidence(
                                    evidence_id=deterministic_evidence_id(
                                        EvidenceCategory.FILE_PATH,
                                        norm,
                                        ev.provenance,
                                        ev.sequence_no,
                                        sub_id=f"arg_{k}",
                                    ),
                                    category=EvidenceCategory.FILE_PATH,
                                    value=norm,
                                    source_provenance=ev.provenance,
                                    trust=ev.trust,
                                    sequence_no=ev.sequence_no,
                                    is_direct=True,
                                    status=EvidenceStatus.OBSERVED,
                                    tool_call_id=ev.tool_call_id,
                                )
                            )

            # 3. User message explicit paths
            if ev.kind == EventKind.USER_MESSAGE:
                for match in _FILE_PATH_TOKEN_RE.finditer(bounded_content):
                    raw_token = match.group(1)
                    norm = normalize_file_path(raw_token)
                    if norm:
                        results.append(
                            GroundingEvidence(
                                evidence_id=deterministic_evidence_id(
                                    EvidenceCategory.FILE_PATH,
                                    norm,
                                    ev.provenance,
                                    ev.sequence_no,
                                    sub_id="user_token",
                                ),
                                category=EvidenceCategory.FILE_PATH,
                                value=norm,
                                source_provenance=ev.provenance,
                                trust=ev.trust,
                                sequence_no=ev.sequence_no,
                                is_direct=True,
                                status=EvidenceStatus.OBSERVED,
                            )
                        )

            # 4. File listings or confirmed outputs in tool results
            if ev.kind == EventKind.TOOL_RESULT:
                lines = bounded_content.splitlines()
                # If tool output is a list of paths (e.g. ls, find)
                for line in lines[:500]:
                    stripped = line.strip()
                    if not stripped or len(stripped) > 512:
                        continue
                    # Match clean paths on single lines
                    if "/" in stripped or stripped.endswith(
                        (".py", ".json", ".md", ".txt", ".toml", ".yaml", ".yml", ".ts", ".js")
                    ):
                        norm = normalize_file_path(stripped)
                        if norm and not norm.startswith(("-", "total ", "drwx", "-rwx")):
                            results.append(
                                GroundingEvidence(
                                    evidence_id=deterministic_evidence_id(
                                        EvidenceCategory.FILE_PATH,
                                        norm,
                                        ev.provenance,
                                        ev.sequence_no,
                                        sub_id="listing",
                                    ),
                                    category=EvidenceCategory.FILE_PATH,
                                    value=norm,
                                    source_provenance=ev.provenance,
                                    trust=ev.trust,
                                    sequence_no=ev.sequence_no,
                                    is_direct=True,
                                    status=EvidenceStatus.CONFIRMED,
                                    tool_call_id=ev.tool_call_id,
                                )
                            )

        return results
