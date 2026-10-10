"""Deterministic mutation and verification state extractor."""

from __future__ import annotations

from collections.abc import Sequence

from alienese.contracts.events import EventKind, NormalizedEvent
from alienese.contracts.state import (
    CanonicalCapability,
    ExternalToolBinding,
    WorkingState,
)
from alienese.grounding.evidence import (
    EvidenceCategory,
    EvidenceStatus,
    GroundingEvidence,
    deterministic_evidence_id,
)
from alienese.grounding.extractors.base import MAX_EVENTS_SCANNED
from alienese.grounding.normalization import normalize_file_path

_MUTATION_CAPABILITIES = frozenset(
    {
        CanonicalCapability.APPLY_PATCH,
        CanonicalCapability.WRITE_FILE,
    }
)
_VERIFICATION_CAPABILITIES = frozenset(
    {
        CanonicalCapability.RUN_TEST,
    }
)


class MutationExtractor:
    """Extracts mutation targets and verification results across event history."""

    def extract(
        self,
        events: Sequence[NormalizedEvent],
        state: WorkingState,
        tools: Sequence[ExternalToolBinding],
    ) -> Sequence[GroundingEvidence]:
        results: list[GroundingEvidence] = []
        tools_by_name = {t.external_name: t for t in tools}
        scanned_events = events[-MAX_EVENTS_SCANNED:]

        # Map tool_call_id to tool call events
        call_events_by_id: dict[str, NormalizedEvent] = {}
        for ev in scanned_events:
            if ev.kind == EventKind.TOOL_CALL and ev.tool_call_id:
                call_events_by_id[ev.tool_call_id] = ev

        for ev in scanned_events:
            # 1. Check tool calls for mutations
            if ev.kind == EventKind.TOOL_CALL and ev.tool_name:
                binding = tools_by_name.get(ev.tool_name)
                capability = (
                    binding.canonical_capability if binding else CanonicalCapability.CUSTOM_TOOL
                )
                if capability in _MUTATION_CAPABILITIES and ev.tool_arguments:
                    target = (
                        ev.tool_arguments.get("path")
                        or ev.tool_arguments.get("filepath")
                        or ev.tool_arguments.get("file_path")
                        or ev.tool_arguments.get("target_file")
                    )
                    if isinstance(target, str):
                        norm_target = normalize_file_path(target)
                        if norm_target:
                            results.append(
                                GroundingEvidence(
                                    evidence_id=deterministic_evidence_id(
                                        EvidenceCategory.MUTATION_TARGET,
                                        norm_target,
                                        ev.provenance,
                                        ev.sequence_no,
                                        sub_id="mut_attempt",
                                    ),
                                    category=EvidenceCategory.MUTATION_TARGET,
                                    value=norm_target,
                                    source_provenance=ev.provenance,
                                    trust=ev.trust,
                                    sequence_no=ev.sequence_no,
                                    is_direct=True,
                                    status=EvidenceStatus.OBSERVED,
                                    tool_call_id=ev.tool_call_id,
                                )
                            )

            # 2. Check tool results for verification outcomes
            if ev.kind == EventKind.TOOL_RESULT and ev.tool_call_id:
                call_ev = call_events_by_id.get(ev.tool_call_id)
                tool_name = call_ev.tool_name if call_ev else ev.tool_name
                binding = tools_by_name.get(tool_name) if tool_name else None
                capability = (
                    binding.canonical_capability if binding else CanonicalCapability.CUSTOM_TOOL
                )

                # Check if this tool execution was a verification attempt
                is_verification = capability in _VERIFICATION_CAPABILITIES or (
                    tool_name and "test" in tool_name.lower()
                )
                if is_verification:
                    content_lower = ev.content.lower()
                    has_failure = any(
                        term in content_lower
                        for term in ("failed", "failure", "error", "exit code 1", "exit status 1")
                    )
                    status = EvidenceStatus.FAILED if has_failure else EvidenceStatus.CONFIRMED
                    val = "FAILED" if has_failure else "PASSED"
                    results.append(
                        GroundingEvidence(
                            evidence_id=deterministic_evidence_id(
                                EvidenceCategory.VERIFICATION_RESULT,
                                val,
                                ev.provenance,
                                ev.sequence_no,
                                sub_id="verif_result",
                            ),
                            category=EvidenceCategory.VERIFICATION_RESULT,
                            value=val,
                            source_provenance=ev.provenance,
                            trust=ev.trust,
                            sequence_no=ev.sequence_no,
                            is_direct=True,
                            status=status,
                            tool_call_id=ev.tool_call_id,
                        )
                    )

        return results
