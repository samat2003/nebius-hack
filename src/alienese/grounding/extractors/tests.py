"""Deterministic test target and test failure extractor."""

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
from alienese.grounding.normalization import normalize_test_target

_PYTEST_FAILED_LINE_RE = re.compile(
    r"(?:FAILED|ERROR)\s+([a-zA-Z0-9_\-\./]+::[a-zA-Z0-9_\-\.\[\]:]+)(?:\s+-\s+(.+))?"
)
_PYTEST_NODE_ID_RE = re.compile(
    r'(?:^|[\s"\'`])([a-zA-Z0-9_\-\./]+(?:test_[a-zA-Z0-9_]+|[a-zA-Z0-9_]+_test)\.py::[a-zA-Z0-9_\-\.\[\]:]+)(?:$|[\s"\'`])'
)
_PYTEST_COMMAND_RE = re.compile(
    r'(?:^|[\s"\'`])((?:pytest|python\s+-m\s+pytest)\s+[^\n\r]+)(?:$|[\s"\'`])'
)
_TEST_FILE_RE = re.compile(
    r'(?:^|[\s"\'`])((?:[a-zA-Z0-9_\-\.]+/)*(?:test_[a-zA-Z0-9_]+|[a-zA-Z0-9_]+_test)\.py)(?:$|[\s"\'`])'
)


class TestExtractor:
    """Extracts test targets, pytest node IDs, test commands, and test failure messages."""

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

            # 1. Pytest FAILED / ERROR output lines
            for match in _PYTEST_FAILED_LINE_RE.finditer(bounded_content):
                raw_node = match.group(1)
                failure_msg = match.group(2)
                norm_node = normalize_test_target(raw_node)
                if norm_node:
                    results.append(
                        GroundingEvidence(
                            evidence_id=deterministic_evidence_id(
                                EvidenceCategory.TEST_TARGET,
                                norm_node,
                                ev.provenance,
                                ev.sequence_no,
                                sub_id="pytest_failed",
                            ),
                            category=EvidenceCategory.TEST_TARGET,
                            value=norm_node,
                            source_provenance=ev.provenance,
                            trust=ev.trust,
                            sequence_no=ev.sequence_no,
                            is_direct=True,
                            status=EvidenceStatus.FAILED,
                            tool_call_id=ev.tool_call_id,
                            context_snippet=match.group(0),
                        )
                    )
                if failure_msg:
                    clean_msg = failure_msg.strip()
                    if clean_msg:
                        results.append(
                            GroundingEvidence(
                                evidence_id=deterministic_evidence_id(
                                    EvidenceCategory.FAILURE_MESSAGE,
                                    clean_msg[:256],
                                    ev.provenance,
                                    ev.sequence_no,
                                    sub_id="pytest_msg",
                                ),
                                category=EvidenceCategory.FAILURE_MESSAGE,
                                value=clean_msg[:256],
                                source_provenance=ev.provenance,
                                trust=ev.trust,
                                sequence_no=ev.sequence_no,
                                is_direct=True,
                                status=EvidenceStatus.FAILED,
                                tool_call_id=ev.tool_call_id,
                            )
                        )

            # 2. Pytest Node IDs anywhere in text
            for match in _PYTEST_NODE_ID_RE.finditer(bounded_content):
                raw_node = match.group(1)
                norm_node = normalize_test_target(raw_node)
                if norm_node:
                    results.append(
                        GroundingEvidence(
                            evidence_id=deterministic_evidence_id(
                                EvidenceCategory.TEST_TARGET,
                                norm_node,
                                ev.provenance,
                                ev.sequence_no,
                                sub_id="pytest_node",
                            ),
                            category=EvidenceCategory.TEST_TARGET,
                            value=norm_node,
                            source_provenance=ev.provenance,
                            trust=ev.trust,
                            sequence_no=ev.sequence_no,
                            is_direct=True,
                            status=EvidenceStatus.OBSERVED,
                            tool_call_id=ev.tool_call_id,
                        )
                    )

            # 3. Test files referenced in user prompts or tool outputs
            for match in _TEST_FILE_RE.finditer(bounded_content):
                raw_file = match.group(1)
                norm_file = normalize_test_target(raw_file)
                if norm_file:
                    results.append(
                        GroundingEvidence(
                            evidence_id=deterministic_evidence_id(
                                EvidenceCategory.TEST_TARGET,
                                norm_file,
                                ev.provenance,
                                ev.sequence_no,
                                sub_id="test_file",
                            ),
                            category=EvidenceCategory.TEST_TARGET,
                            value=norm_file,
                            source_provenance=ev.provenance,
                            trust=ev.trust,
                            sequence_no=ev.sequence_no,
                            is_direct=True,
                            status=EvidenceStatus.OBSERVED,
                            tool_call_id=ev.tool_call_id,
                        )
                    )

            # 4. Explicit observed test commands
            for match in _PYTEST_COMMAND_RE.finditer(bounded_content):
                raw_cmd = match.group(1).strip()
                if raw_cmd and len(raw_cmd) <= 512:
                    results.append(
                        GroundingEvidence(
                            evidence_id=deterministic_evidence_id(
                                EvidenceCategory.TEST_COMMAND,
                                raw_cmd,
                                ev.provenance,
                                ev.sequence_no,
                                sub_id="pytest_cmd",
                            ),
                            category=EvidenceCategory.TEST_COMMAND,
                            value=raw_cmd,
                            source_provenance=ev.provenance,
                            trust=ev.trust,
                            sequence_no=ev.sequence_no,
                            is_direct=True,
                            status=EvidenceStatus.OBSERVED,
                            tool_call_id=ev.tool_call_id,
                        )
                    )

            # 5. Check tool calls for run_test arguments
            if ev.kind == EventKind.TOOL_CALL and ev.tool_arguments:
                for k in ("target", "test_target", "node_id", "test", "tests"):
                    v = ev.tool_arguments.get(k)
                    if isinstance(v, str):
                        norm = normalize_test_target(v)
                        if norm:
                            results.append(
                                GroundingEvidence(
                                    evidence_id=deterministic_evidence_id(
                                        EvidenceCategory.TEST_TARGET,
                                        norm,
                                        ev.provenance,
                                        ev.sequence_no,
                                        sub_id="tool_arg",
                                    ),
                                    category=EvidenceCategory.TEST_TARGET,
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
